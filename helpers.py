import pandas as pd
from functools import reduce
import numpy as np

def create_categories(tracks_obj):
    # Let's filter by tilt direction and magnitude, velocity magnitude
    # and significant area.
    eastward_motion = (tracks_obj.system_tracks['u_shift'] > 5)
    eastward_motion = eastward_motion.rename('eastward_motion') 
    forward_tilt = ((-45 <= tracks_obj.system_tracks['sys_rel_tilt_dir'])
                    & (tracks_obj.system_tracks['sys_rel_tilt_dir'] <= 45))
    forward_tilt = forward_tilt.rename('forward_tilt')
    categories = pd.merge(eastward_motion,forward_tilt,left_index=True, right_index=True)

    backward_tilt = ((-135 >= tracks_obj.system_tracks['sys_rel_tilt_dir'])
                    | (tracks_obj.system_tracks['sys_rel_tilt_dir'] >= 135))
    backward_tilt = backward_tilt.rename('backward_tilt')
    categories = categories.merge(backward_tilt,left_index=True, right_index=True)

    left_tilt = ((45 <= tracks_obj.system_tracks['sys_rel_tilt_dir'])
                    & (tracks_obj.system_tracks['sys_rel_tilt_dir'] <= 135))
    left_tilt = left_tilt.rename('left_tilt')

    right_tilt = ((-135 <= tracks_obj.system_tracks['sys_rel_tilt_dir'])
                    & (tracks_obj.system_tracks['sys_rel_tilt_dir'] <= -45))
    right_tilt = right_tilt.rename('right_tilt')

    # # downshear_motion = ((-45 <= tracks_obj.system_tracks['shear_rel_sys_dir'])
    # #                 & (tracks_obj.system_tracks['shear_rel_sys_dir'] <= 45))
    # # upshear_motion = ((-135 >= tracks_obj.system_tracks['shear_rel_sys_dir'])
    # #                  | (tracks_obj.system_tracks['shear_rel_sys_dir'] >= 135))
    # sig_shear_mag = (tracks_obj.system_tracks['shear_mag'] >= 2)
    sig_tilt_mag = sig_tilt_mag = (tracks_obj.system_tracks['tilt_mag'] >= 50)
    sig_tilt_mag = sig_tilt_mag.rename('sig_tilt_mag')

    vel_mag = np.sqrt(tracks_obj.system_tracks['u_shift']**2 
                      + tracks_obj.system_tracks['v_shift']**2)
    vel_mag = vel_mag.rename('vel_mag')
    sig_vel_mag = (vel_mag <= 30)
    sig_vel_mag = sig_vel_mag.rename('sig_vel_mag')

    stationary = (vel_mag <= 8) 
    stationary = stationary.rename('stationary')

    linear = (tracks_obj.system_tracks['eccentricity'] > .7)
    linear = linear.rename('linear')

    conv_orient = tracks_obj.system_tracks['orientation']
    vel_dir = tracks_obj.system_tracks['vel_dir']
    conv_align = np.mod(conv_orient - vel_dir + 180, 360)-180

    perp_align = (((45 <= conv_align) & (conv_align <= 130))
                 | ((-135 <= conv_align) & (conv_align <= -45)))
    perp_align = perp_align.rename('perp_align')

    par_align = (((conv_align >= -45) & (conv_align <= 45)) 
                 | (conv_align >= 135) | (conv_align <= -135))
    par_align = par_align.rename('par_align')

    # Note for CPOL 2.5 km, total scan area is only 66052 km^2. This makes 
    # traditional MCS definitions of area > 30000 km^2 difficult to apply, 
    # and still coherently calculate tilt. 

    small_area = (tracks_obj.system_tracks['proj_area'] < 4000)
    small_area = small_area.rename('small_area')

    large_area = (tracks_obj.system_tracks['proj_area'] > 50000)
    large_area = large_area.rename('large_area')

    not_border = (tracks_obj.system_tracks['touch_border']*6.25 / tracks_obj.system_tracks['proj_area']) < 0.01
    not_border = not_border.rename('not_border')

    tracks_0 = tracks_obj.tracks[['touch_border', 'proj_area']].xs(
        0, level='level'
    )
    not_border_0 = (tracks_0['touch_border'] * 6.25 / tracks_0['proj_area']) < 0.01
    not_border_0 = not_border_0.rename('not_border_0')

    dframes = [eastward_motion, forward_tilt, backward_tilt, 
               left_tilt, right_tilt, sig_tilt_mag,
               sig_vel_mag, stationary, linear, small_area,
               large_area, not_border, not_border_0, perp_align, par_align]
    categories = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True), dframes)
    
    return categories
