import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import copy
import pickle
import pyart
import tint

def CPOL_files_from_datetime_list(datetimes):
    print('Gathering files.')
    base = '/g/data/rr5/CPOL_radar/CPOL_level_1b/GRIDDED/GRID_150km_2500m/'
    filenames = []
    for i in range(len(datetimes)):
        year = str(datetimes[i])[0:4]
        month = str(datetimes[i])[5:7]
        day = str(datetimes[i])[8:10]
        hour = str(datetimes[i])[11:13]
        minute = str(datetimes[i])[14:16]
        filename = (base + '{0}/{0}{1}{2}/'.format(year, month, day) 
                    + 'CPOL_{0}{1}{2}'.format(year, month, day)
                    + '_{}{}_GRIDS_2500m.nc'.format(hour, minute))
        if os.path.isfile(filename):
            filenames.append(filename)
    
    return sorted(filenames), datetimes[0], datetimes[-1]
    
def CPOL_files_from_TINT_obj(tracks_obj, uid):
    datetimes = tracks_obj.system_tracks.xs(uid, level='uid')
    datetimes = datetimes.reset_index(level='time')['time']
    datetimes = list(datetimes.values)
    [files, start_date, end_date] = CPOL_files_from_datetime_list(datetimes)
    
    return files, start_date, end_date

def get_square_boundary(grid):
    b_ind = set()
    columns = grid.nx
    rows = grid.ny
    for edge in [[0, columns], [rows-1, columns],
                 [rows, 0], [rows, columns-1]]:
        b = np.array([[edge[0]]*edge[1], list(range(edge[1]))])
        b = b.transpose().tolist()
        b = set([tuple(b[i]) for i in range(edge[1])])
        b_ind = b_ind.union(b)
    return b_ind

def get_circular_boundary(grid):
    radius = grid.x['data'][-1]
    # Assume a regular grid
    dx = grid.x['data'][1] - grid.x['data'][0]
    offset = np.sqrt(2*dx**2)
    X, Y = np.meshgrid(grid.x['data'], grid.y['data'], indexing='ij')
    radii = np.sqrt(X**2+Y**2)
    
    b_ind = np.argwhere(np.logical_and(radii >= radius, 
                                       radii < radius + dx))
    b_ind_set = set([tuple(b_ind[i]) for i in range(b_ind.shape[0])])
        
    plt.pcolor(np.logical_and(radii >= radius-offset/2, 
                              radii < radius+offset/2))
    return b_ind_set
    
def load_wet_seasons(years=list(range(1999, 2017))):

    years = set(years) - set([2007, 2008])
    years = sorted(list(years))
    
    filenames = []
    
    for year in years:
        filenames_year = CPOL_files_from_datetime_list(
            np.arange(np.datetime64('{}-11-01 00:00'.format(str(year))), 
                      np.datetime64('{}-04-01 00:00'.format(str(year+1))), 
                      np.timedelta64(10, 'm'))
            )[0]
        filenames += filenames_year
        
    return filenames
    
def system_tracks_to_tracks(sys_var, n_lvl):
    var = sys_var.append([sys_var]*(n_lvl-1)).sort_index(sort_remaining=True)
    levels = np.array(list(range(n_lvl))*len(sys_var))
    try:
        var = var.to_frame().sort_index(sort_remaining=True)
    except:
        print('Input already a dataframe.')
    var.insert(0, 'level', levels)
    var = var.reset_index()
    var = var.set_index(['scan', 'time', 'level', 'uid'])
    var = var.sort_index(sort_remaining=True)
    return var

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
    
def get_CPOL_tracks(year):
    filenames = CPOL_files_from_datetime_list(
        np.arange(np.datetime64('{}-11-01 00:00'.format(str(year))), 
                  np.datetime64('{}-04-01 00:00'.format(str(year+1))), 
                  np.timedelta64(10, 'm'))
    )[0]

    # Generate grid generator 
    # Note generators produce iterators
    # These are alternative to using lists and looping
    grids = (pyart.io.read_grid(fn, include_fields = 'reflectivity')
             for fn in filenames)

    with open('/g/data/w40/esh563/CPOL_analysis/TINT_tracks/circ_b_ind_set.pkl', 
              'rb') as f:
        b_ind_set = pickle.load(f)

    # Define settings for tracking
    settings = {
        'MIN_SIZE' : [40, 400, 800], # square km
        'FIELD_THRESH' : ['convective', 20, 15], # DbZ
        'ISO_THRESH' : [10, 10, 10], # DbZ
        'GS_ALT' : 3000,
        'SEARCH_MARGIN' : 10000, # m. This is just for object matching step:
        # does not affect flow vectors.
        'FLOW_MARGIN' : 40000, # m. Margin around object over which to
        # perform phase correlation.
        'LEVELS' : np.array( # m
            [[3000, 3500], 
             [3500, 7500],
             [7500, 10000]]
        ),
        'TRACK_INTERVAL' : 0,
        'BOUNDARY_GRID_CELLS' : b_ind_set,
        'UPDRAFT_START': 3000
    }

    tracks_obj  = tint.Cell_tracks()

    for parameter in ['MIN_SIZE', 'FIELD_THRESH', 'GS_ALT', 'LEVELS', 
                      'TRACK_INTERVAL', 'ISO_THRESH', 'SEARCH_MARGIN',
                      'FLOW_MARGIN', 'BOUNDARY_GRID_CELLS', 'UPDRAFT_START'
                     ]:
        tracks_obj.params[parameter] = settings[parameter]

    # Calculate tracks
    tracks_obj.get_tracks(grids)
    tracks_obj = get_reanalysis_vars(tracks_obj)

    out_file_name = ('/g/data/w40/esh563/CPOL_analysis/TINT_tracks/'
                     + 'tracks_obj_{}_{}.pkl'.format(str(year), str(year+1)))

    with open(out_file_name, 'wb') as f:
        pickle.dump(tracks_obj, f)
        
    return tracks_obj
    
def combine_tracks(years=list(range(2001, 2015))):
    years = set(years) - set([2007, 2008])
    years = sorted(list(years))
    bp = '/g/data/w40/esh563/CPOL_analysis/TINT_tracks/tracks_obj_'
    max_uid = 0
    tracks=[]
    system_tracks=[]
    for year in years:
        print('Shifting uid for {}'.format(year))
        with open(bp + '{}_{}.pkl'.format(str(year), str(year+1)), 'rb') as f:
            tracks_obj = pickle.load(f)
        tracks_obj.tracks.reset_index(
            level=['scan', 'time', 'level', 'uid'], inplace=True
        )
        tracks_obj.system_tracks.reset_index(
            level=['scan', 'time', 'uid'], inplace=True
        )
        uids = tracks_obj.tracks['uid'].values.astype(int)
        sys_uids = tracks_obj.system_tracks['uid'].values.astype(int)
        uids += max_uid
        sys_uids += max_uid
        
        # Fix tracks merger values
        for i in range(len(tracks_obj.tracks)):
            m_uid = tracks_obj.tracks.mergers.values[i]
            new_m_uid = np.array(list(m_uid)).astype(int) + max_uid
            new_m_uid = set(new_m_uid.astype(str).tolist())
            tracks_obj.tracks.mergers.values[i] = new_m_uid
            
        # Fix system_tracks merger values
        for i in range(len(tracks_obj.system_tracks)):
            m_uid = tracks_obj.system_tracks.mergers.values[i]
            new_m_uid = np.array(list(m_uid)).astype(int) + max_uid
            new_m_uid = set(new_m_uid.astype(str).tolist())
            tracks_obj.system_tracks.mergers.values[i] = new_m_uid
        
        max_uid = uids.max()+1
        
        tracks_obj.tracks['uid'] = uids.astype(str)
        tracks_obj.system_tracks['uid'] = sys_uids.astype(str)
        tracks_obj.tracks.set_index(['scan', 'time', 'level', 'uid'], 
                                    inplace=True)
        tracks_obj.system_tracks.set_index(['scan', 'time', 'uid'],
                                           inplace=True)
        tracks.append(tracks_obj.tracks)
        system_tracks.append(tracks_obj.system_tracks)
        
    print('Concatenating shifted tracks.')    
    tracks_obj.tracks = pd.concat(tracks)
    tracks_obj.system_tracks = pd.concat(system_tracks)
    
    fn = bp + 'com_{}_{}.pkl'.format(str(years[0]), str(years[-1]))
    with open(fn, 'wb') as f:
        pickle.dump(tracks_obj, f)
        
    return tracks_obj
        

            
