import pandas as pd
from functools import reduce
import numpy as np


def add_monsoon_regime(tracks_obj):
    pd.read_csv('/g/data/w40/esh563/CPOL_analysis/Pope_regimes.csv')
    pope = pd.read_csv('/g/data/w40/esh563/CPOL_analysis/Pope_regimes.csv', 
                       index_col=0, names=['date', 'regime'])
    pope.index = pd.to_datetime(pope.index)
    regimes = []
    np.datetime64(tracks_obj.system_tracks.index[0][1].date())
    for i in range(len(tracks_obj.system_tracks)):
        d = np.datetime64(tracks_obj.system_tracks.index[i][1].date())
        try:
            regimes.append(pope.loc[d].values[0])
        except:
            regimes.append(np.nan)
    
    tracks_obj.system_tracks['pope_regime']=regimes  
    return tracks_obj
        

def create_categories(tracks_obj):
    # Let's filter by tilt direction and magnitude, velocity magnitude
    # and significant area.
    eastward_motion = (tracks_obj.system_tracks['u_shift'] > 5)
    eastward_motion = eastward_motion.rename('eastward_motion')
    forward_tilt = ((-45 <= tracks_obj.system_tracks['sys_rel_tilt_dir'])
                    & (tracks_obj.system_tracks['sys_rel_tilt_dir'] <= 45))
    forward_tilt = forward_tilt.rename('forward_tilt')
    categories = pd.merge(eastward_motion,forward_tilt,
                          left_index=True, 
                          right_index=True)

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

    linear = (tracks_obj.system_tracks['eccentricity'] > .9)
    linear = linear.rename('linear')

    conv_orient = tracks_obj.system_tracks['orientation']
    vel_dir = tracks_obj.system_tracks['vel_dir']
    conv_align = np.mod(conv_orient-vel_dir+180, 360)-180

    perp_align = (((45 < conv_align) & (conv_align <= 135))
                 | ((-135 < conv_align) & (conv_align <= -45)))
    perp_align = perp_align.rename('perp_align')

    par_align = (((conv_align > -45) & (conv_align <= 45))
                 | (conv_align > 135) | (conv_align <= -135))
    par_align = par_align.rename('par_align')

    # Note for CPOL 2.5 km, total scan area is only 66052 km^2. This makes 
    # traditional MCS definitions of area > 30000 km^2 difficult to apply, 
    # and still coherently calculate tilt. 

    small_area = (tracks_obj.system_tracks['proj_area'] < 4000)
    small_area = small_area.rename('small_area')

    large_area = (tracks_obj.system_tracks['proj_area'] > 50000)
    large_area = large_area.rename('large_area')

    not_border = (tracks_obj.system_tracks['touch_border']*6.25 
                  /tracks_obj.system_tracks['proj_area']) < 0.01
    not_border = not_border.rename('not_border')

    tracks_0 = tracks_obj.tracks[['touch_border', 'proj_area']].xs(
        0, level='level'
    )
    not_border_0 = (tracks_0['touch_border']*6.25/tracks_0['proj_area']) < 0.01
    not_border_0 = not_border_0.rename('not_border_0')

    dframes = [eastward_motion, forward_tilt, backward_tilt, 
               left_tilt, right_tilt, sig_tilt_mag,
               sig_vel_mag, stationary, linear, small_area,
               large_area, not_border, not_border_0, perp_align, par_align]
    categories = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True), dframes)
    
    return categories
    
def get_reanalysis_vars(tracks_obj):
    print('Adding data from Monash Reanalysis')
    # Convert reanalysis dataset to z coords.
    va = xr.open_dataset('/g/data/ua8/Martin/va_analysis/syn599/CPOL_large-scale_forcing.nc')
    va_env = copy.deepcopy(va)
    T_mean = va.T.mean(dim='time')
    integrand = -287.058 * T_mean /(9.807 * va.lev * 100) 
    z = []
    for i in range(len(integrand)):
        z.append(np.trapz(integrand[:i+1], va.lev[:i+1]*100))
    z = np.round(np.array(z))
    p = va.lev.values
    va['lev'] = z
    va = va.rename(lev='z')
    va.z.attrs = {'units': 'm', 'long_name': 
                    'height above ground level', 
                    'axis':'Z'}
    p = np.tile(p,[19674,1])
    va['p'] = (['time', 'z'],p)
    
    # Calculate mean wind
    vel_cl = va_env[['u','v']].sel(lev=[850, 700, 500, 300], method='nearest').mean(dim='lev')
    vel_cl = vel_cl.rename({'u': 'u_cl', 'v': 'v_cl'})
    vel_cl.attrs['units'] = 'm/s'
    vel_cl.attrs['long_name'] = 'mean cloud bearing layer winds'
    
    # Calculate shear
    shear_ds = (va[['u','v']].sel(z=3000, method='nearest') 
                - va[['u','v']].sel(z=0, method='nearest'))
    shear_ds = shear_ds.rename({'u': 'u_shear', 'v': 'v_shear'})
    shear_ds.attrs['units'] = 'm/s'
    shear_ds.attrs['long_name'] = '3 km minus 0 km mean wind shear'
    
    # Restrict tracks_obj to when reanalysis is available
    tmp_tracks = tracks_obj.tracks.reset_index('time')
    tmp_system_tracks = tracks_obj.system_tracks.reset_index('time')
    
    tracks_times = tmp_tracks.time.values.astype(np.datetime64)
    sys_times = tmp_system_tracks.time.values.astype(np.datetime64)
    
    t_cond = tracks_times >= shear_ds.time[0].values
    t_cond &= (tracks_times <= shear_ds.time[-1].values)
    
    t_cond_sys = sys_times >= shear_ds.time[0].values
    t_cond_sys &= (sys_times <= shear_ds.time[-1].values)
    
    tracks_obj.tracks = tracks_obj.tracks[t_cond]
    tracks_obj.system_tracks = tracks_obj.system_tracks[t_cond_sys]
    
    tmp_system_tracks = tmp_system_tracks[t_cond_sys]
    sys_times = tmp_system_tracks.time.values.astype(np.datetime64)

    sys_shears = shear_ds.sel(time = sys_times, method='nearest')
    sys_cl = vel_cl.sel(time = sys_times, method='nearest')
    sys_shears = sys_shears.to_dataframe().reset_index('time', drop=True)
    sys_cl = sys_cl.to_dataframe().reset_index('time', drop=True)

    sys_shears.index = tracks_obj.system_tracks.index
    sys_cl.index = tracks_obj.system_tracks.index
        
    n_lvl = tracks_obj.params['LEVELS'].shape[0]
    
    for var in [sys_shears, sys_cl]:
        tracks_obj.system_tracks = tracks_obj.system_tracks.merge(
            var, left_index=True, right_index=True
        )
        var_alt = system_tracks_to_tracks(var, n_lvl)
        tracks_obj.tracks = tracks_obj.tracks.merge(
            var_alt, left_index=True, right_index=True
        )
    
    # Calculate additional shear related variables
    shear_dir = np.arctan2(tracks_obj.system_tracks['v_shear'], 
                           tracks_obj.system_tracks['u_shear'])
    shear_dir = np.rad2deg(shear_dir)
    shear_dir = shear_dir.rename('shear_dir')
    shear_dir = np.round(shear_dir, 3)
    
    shear_mag = np.sqrt(tracks_obj.system_tracks['u_shear']**2 + 
                        tracks_obj.system_tracks['v_shear']**2)
    shear_mag = shear_mag.rename('shear_mag')
    shear_mag = np.round(shear_mag, 3)
    
    vels = tracks_obj.system_tracks[['u_shift', 'v_shift']]
    vels = vels.rename(columns = {'u_shift':'u_prop', 'v_shift':'v_prop'})
    cl_winds = tracks_obj.system_tracks[['u_cl', 'v_cl']]
    cl_winds = cl_winds.rename(columns = {'u_cl':'u_prop', 'v_cl':'v_prop'})
    prop = vels - cl_winds
    prop_dir = np.arctan2(prop['v_prop'], 
                          prop['u_prop'])
    prop_dir = np.rad2deg(prop_dir)
    prop_dir = prop_dir.rename('prop_dir')
    prop_dir = np.round(prop_dir, 3)
    
    prop_mag = np.sqrt(prop['v_prop'] ** 2 + prop['u_prop'] ** 2)
    prop_mag = prop_mag.rename('prop_mag')
    prop_mag = np.round(prop_mag, 3)
    
    shear_rel_prop_dir = np.mod(prop_dir - shear_dir + 180, 360)-180
    shear_rel_prop_dir = shear_rel_prop_dir.rename('shear_rel_prop_dir')
    shear_rel_prop_dir = np.round(shear_rel_prop_dir, 3)

    for var in [shear_dir, shear_mag, prop_dir, prop_mag, prop, shear_rel_prop_dir]:
        tracks_obj.system_tracks = tracks_obj.system_tracks.merge(
            var, left_index=True, right_index=True
        )
        var_alt = system_tracks_to_tracks(var, n_lvl)
        tracks_obj.tracks = tracks_obj.tracks.merge(
            var_alt, left_index=True, right_index=True
        )
        
    return tracks_obj
