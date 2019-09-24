import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import copy
import pickle
import netCDF4

import glob

def caine_files_from_datetime_list(datetimes):
    print('Gathering files.')
    base = '/g/data/w40/esh563/lind04/dbz/'
    filenames = []
    for i in range(len(datetimes)):
        year = str(datetimes[i])[0:4]
        month = str(datetimes[i])[5:7]
        day = str(datetimes[i])[8:10]
        hour = str(datetimes[i])[11:13]
        minute = str(datetimes[i])[14:16]
        filename = ('{0}-{1}-{2}_'.format(year, month, day) 
                    + '{0}:{1}:00'.format(hour, minute)
                    + 'alllevels_zlib.nc')
        if os.path.isfile(filename):
            filenames.append(filename)
    
    return sorted(filenames), datetimes[0], datetimes[-1]
    
    
def caine_files_from_TINT_obj(tracks_obj, uid):
    datetimes = tracks_obj.system_tracks.xs(uid, level='uid')
    datetimes = datetimes.reset_index(level='time')['time']
    datetimes = list(datetimes.values)
    [files, start_date, end_date] = CPOL_files_from_datetime_list(datetimes)
    
    return files, start_date, end_date
    

def calculate_hgt_AGL():
    fn = sorted(glob.glob('/g/data/w40/esh563/d04.dir/*.nc.gz'))
    height_AGL = np.zeros((64, 486, 717))
    for i in range(len(fn[:10])):
        print('Adding file {}.'.format(str(i)))
        ds = xr.open_dataset(fn[i])
        da = (ds.PH+ds.PHB)/9.80665 - ds.HGT
        da = da.sum(axis=0)
        height_AGL += da.values
    height_AGL = height_AGL/(3*10)
    import pdb
    pdb.set_trace()
    height_AGL = height_AGL.mean(axis=(1,2))
    np.save('/g/data/w40/esh563/d04_hgt_AGL.npy', height_AGL)
    return height_AGL
        
           
def wrf_radar_npy_to_nc():
    fn = sorted(glob.glob('/g/data/w40/esh563/lind04/dbz/*.npy'))
    datetimes = np.arange(np.datetime64('2006-02-08 12:00:00'), 
                          np.datetime64('2006-02-13 12:10:00'), 
                          np.timedelta64(10, 'm'))
                          
    z = np.arange(0, 500*41, 500, dtype=float)
    x = np.arange(0, 1250*241, 1250, dtype=float)
    y = np.arange(0, 1250*241, 1250, dtype=float)

    # Open arbitrary CPOL 2500 file from which to extract useful metadata
    CPOL = xr.open_dataset(('/g/data/rr5/CPOL_radar/CPOL_level_1b/' 
                                + 'GRIDDED/GRID_150km_2500m/2006/20060210/'
                                + 'CPOL_20060210_1200_GRIDS_2500m.nc'))
    for i in range(len(fn)):
        print('Creating .nc file {}.'.format(str(i)))
        data = np.load(fn[i])
        surface = np.ones((1,241,241)) * np.nan
        data = np.concatenate([surface, data])
        data[data==-20]=np.nan
        # Could impose radius limiting here... 
        data = np.expand_dims(data, 0)
        da = xr.DataArray(data, dims=('time', 'z', 'x', 'y'), 
                          coords={'time': np.array([datetimes[i]]), 
                                  'z':z, 'x':x, 'y':y})
        
        da.time.encoding['units'] = 'seconds since   '+str(datetimes[i])+'Z'
        da.time.attrs['standard_name'] = 'time'
        da.time.encoding['calendar'] = 'gregorian'
        
        da.coords['x'] -= da.x.mean()
        da.coords['y'] -= da.y.mean()
        
        da.x.attrs = CPOL.x.attrs
        da.y.attrs = CPOL.y.attrs
        da.z.attrs = CPOL.z.attrs
        
        ds = xr.Dataset({'reflectivity':da})
        ds.reflectivity.attrs = CPOL.reflectivity.attrs

        ds = ds.coarsen(x=2, boundary='trim', side='left').mean()
        ds = ds.coarsen(y=2, boundary='trim', side='left').mean()

        # Copy metadata from CPOL dataset
        var_list = ['projection', 'ProjectionCoordinateSystem', 
                    'radar_latitude', 'radar_longitude', 'radar_altitude']
        for var in var_list: 
            ds[var] = CPOL[var]
            
        var_list = ['origin_latitude', 'origin_longitude', 'origin_altitude']
        for var in var_list:
            ds[var] = ('time', CPOL[var].values)
            ds[var].attrs = CPOL[var].attrs
            
        ds['radar_time'] = ('nradar', [datetimes[i]])
        ds.radar_time.attrs = CPOL.radar_time.attrs
        
        ds.to_netcdf(fn[i][:-4] + '.nc', format='NETCDF4')
