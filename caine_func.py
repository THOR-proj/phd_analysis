import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import copy
import pickle
import netCDF4

from scipy.special import gamma

import glob

from numba import jit

'''
def caine_files_from_datetime_list(datetimes):
    print('Gathering files.')
    base = '/g/data/w40/esh563/lind04_ref/'
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
        if os.path.isfile(base + filename):
            filenames.append(base + filename)
    
    return sorted(filenames), datetimes[0], datetimes[-1]
'''   
    
def caine_files_from_datetime_list(datetimes, mp='lin'):
    print('Gathering files.')
    base = '/g/data/w40/esh563/{}d04_ref/'.format(mp)
    filenames = []
    for i in range(len(datetimes)):
        filename = ('{}d04_ref_'.format(mp) 
                    + str(datetimes[i].astype('datetime64[m]')) 
                    + ':00.nc')
        if os.path.isfile(base + filename):
            filenames.append(base + filename)
    
    return sorted(filenames), datetimes[0], datetimes[-1]
    
    
def caine_files_from_TINT_obj(tracks_obj, uid, mp='lin'):
    datetimes = tracks_obj.system_tracks.xs(uid, level='uid')
    datetimes = datetimes.reset_index(level='time')['time']
    datetimes = list(datetimes.values)
    [files, start_date, end_date] = caine_files_from_datetime_list(datetimes, mp=mp)
    
    return files, start_date, end_date
    

def calculate_hgt_AGL():
    fn = sorted(glob.glob('/g/data/w40/esh563/d04.dir/*.nc.gz'))
    height_AGL = np.zeros((64, 486, 717))
    for i in range(len(fn)):
        print('Adding file {}.'.format(str(i)))
        ds = xr.open_dataset(fn[i])
        da = (ds.PH+ds.PHB)/9.80665 - ds.HGT
        da = da.sum(axis=0)
        height_AGL += da.values
    height_AGL = height_AGL/(3*len(fn))
    height_AGL = height_AGL.mean(axis=(1,2))
    np.save('/g/data/w40/esh563/d04_hgt_AGL.npy', height_AGL)
    return height_AGL
    
    
@jit()
def vert_interp(field, z):
    interp = np.ones((41,117,117)) * np.nan
    z_int = np.arange(0.0,20500.0,500.0)

    for i in range(117):
        for j in range(117):
            interp[:,i,j] = np.interp(z_int, z[:,i,j], field[:,i,j], left=np.nan)
    return interp
        
           
def WRF_to_pyart(mp='lin'):
    if mp=='lin':
        fn = sorted(glob.glob('/g/data/w40/esh563/d04.dir/wrfout*.nc.gz'))
        datetimes = np.arange(np.datetime64('2006-02-09 00:00:00'), 
                              np.datetime64('2006-02-13 12:30:00'), 
                              np.timedelta64(30, 'm'))
    else:
        fn = sorted(glob.glob('/g/data/w40/esh563/thompson/d04.dir/wrfout*.nc.gz'))
        datetimes = np.arange(np.datetime64('2006-02-08 18:30:00'), 
                              np.datetime64('2006-02-13 12:30:00'), 
                              np.timedelta64(30, 'm'))                           
                         
    # Open arbitrary CPOL 2500 file from which to extract useful metadata
    CPOL = xr.open_dataset(('/g/data/rr5/CPOL_radar/CPOL_level_1b/' 
                            + 'GRIDDED/GRID_150km_2500m/2006/20060210/'
                            + 'CPOL_20060210_1200_GRIDS_2500m.nc'))
                            
    for i in range(len(fn)):
        print('Starting conversions.')
        
        '''
        if datetimes[i]<np.datetime64('2006-02-13T12:00:00'):
            continue
        
        if datetimes[i]>np.datetime64('2006-02-12T17:50:00'):
            continue
        '''
        WRF = xr.open_dataset(fn[i])
        for j in range(WRF.Times.size):
            wrf = WRF.isel(Time=j)
                
            z = (wrf.PH + wrf.PHB)/9.80665
            
            # Destagger and standardise domains
            if mp=='lin':
                wrf_water = wrf[['QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW',
                                 'QGRAUP', 'T', 'P', 'PB']]
            else:
                wrf_water = wrf[['QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 
                                 'QGRAUP', 'QNRAIN', 'QNICE', 'T', 'P', 'PB']]
            wrf_water['west_east'] = wrf_water.XLONG[0,:]
            wrf_water['south_north'] = wrf_water.XLAT[:,0]

            U = wrf['U']
            U = U.rolling(west_east_stag=2, center=True).mean().dropna('west_east_stag')
            U['west_east_stag'] = wrf_water['west_east'].values
            U = U.rename({'west_east_stag' : 'west_east'})
            U['south_north'] = U.XLAT_U[:,0].values
            U = U.drop(['XLONG_U', 'XLAT_U'])
            U.name = 'U'

            V = wrf['V']
            V = V.rolling(south_north_stag=2, center=True).mean().dropna('south_north_stag')
            V['south_north_stag'] = wrf_water['south_north'].values
            V = V.rename({'south_north_stag' : 'south_north'})
            V['west_east'] = V.XLONG_V[0,:].values
            V = V.drop(['XLONG_V', 'XLAT_V'])
            V.name = 'V'

            wrf_vert = wrf[['W', 'PH', 'PHB']]
            wrf_vert['west_east'] = wrf_vert.XLONG[0,:]
            wrf_vert['south_north'] = wrf_vert.XLAT[:,0]
            wrf_vert = wrf_vert.drop(['XLONG', 'XLAT'])
            wrf_vert = wrf_vert.rolling(bottom_top_stag=2, center=True).mean().dropna('bottom_top_stag')
            wrf_vert = wrf_vert.rename({'bottom_top_stag' : 'bottom_top'})

            wrf = xr.merge([wrf_water, U, V, wrf_vert])
            
            wrf = wrf.coarsen(west_east=2, boundary='trim', side='left').mean()
            wrf = wrf.coarsen(south_north=2, boundary='trim', side='left').mean()
            
            # Restrict to rough CPOL domain
            wrf = wrf.sel(west_east = slice(129.70584, 132.39513))
            wrf = wrf.sel(south_north = slice(-13.55555, -10.931778))
            
            # Interpolate onto standard height levels
            var_list = ['QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 
                        'QGRAUP', 'T', 'P', 'PB', 'U', 'V', 'W']
            if mp=='thompson':
                var_list.append('QNICE')
                var_list.append('QNRAIN')
            ds_list = []
            x = np.arange(-145000, 145000 + 2500, 2500, dtype=float)
            y = np.arange(-145000, 145000 + 2500, 2500, dtype=float)
            z = np.arange(0, 20500, 500, dtype=float)
            t = datetimes[i] + np.timedelta64(10, 'm')*j
            hgt = ((wrf.PH+wrf.PHB)/9.80665).values
            for v in var_list:
                interp_data = vert_interp(wrf[v].values, hgt)
                interp_data = np.expand_dims(interp_data, 0)
                ds = xr.Dataset({v: (['time', 'z', 'y', 'x'],  interp_data),},
                                coords={'time': [t], 'y': y, 'x': x, 'z': z})
                ds_list.append(ds)
                
            # Add latitude longitude values
            longitude = wrf.XLONG.values
            longitude = np.expand_dims(longitude, 0)
            longitude = np.concatenate([longitude] * 41)
            longitude = np.expand_dims(longitude, 0)
            ds = xr.Dataset({'longitude': (['time', 'z', 'y', 'x'],  longitude),},
                            coords={'time': [t], 'y': y, 'x': x, 'z': z})
            ds_list.append(ds)

            latitude = wrf.XLAT.values
            latitude = np.expand_dims(latitude, 0)
            latitude = np.concatenate([latitude] * 41)
            latitude = np.expand_dims(latitude, 0)
            ds = xr.Dataset({'latitude': (['time', 'z', 'y', 'x'],  latitude),},
                            coords={'time': [t], 'y': y, 'x': x, 'z': z})
            ds_list.append(ds)
            
            wrf = xr.merge(ds_list)

            origin_lat = wrf.latitude.mean().values
            origin_lon = wrf.longitude.mean().values
            
            # Create reflectivity field
            rho_rain = 1000
            rho_snow = 100  
            rho_graup = 400
            rho_ice = 890
            
            T = (wrf.T + 300)*(100000/(wrf.PB+wrf.P)) ** (-0.286) - 273.15
            rho_air=(wrf.P+wrf.PB)/(287*(T+273.15)*(0.622+wrf.QVAPOR)/(0.622*(1+wrf.QVAPOR)))
            
            if mp == 'lin':
                N0r = 8*10**6
                N0g = 4*10**6
                N0s = 2*10**7
                rain_ref = 720 * (rho_air * wrf.QRAIN) ** (7/4) / (N0r ** (3/4) * (np.pi * rho_rain) ** (7/4))
                snow_ref = 720 * (rho_air * wrf.QSNOW) ** (7/4) / (N0s ** (3/4) * (np.pi * rho_snow) ** (7/4)) * (rho_snow / rho_rain) ** 2 
                graup_ref = 720 * (rho_air * wrf.QGRAUP) ** (7/4) / (N0g ** (3/4) * (np.pi * rho_graup) ** (7/4)) * (rho_graup / rho_rain) ** 2 
                
                graup_ref.values[T.values<0] = graup_ref.values[T.values<0] * 0.224
                snow_ref.values[T.values<0] = snow_ref.values[T.values<0] * 0.224
            else:
                # Solve for rain reflectivity
                lam_rain = (np.pi*rho_rain*wrf.QNRAIN*gamma(4)/(6*wrf.QRAIN*gamma(1)))**(1/3)
                N0r = wrf.QNRAIN * lam_rain / gamma(1)
                rain_ref = (6/np.pi)**2*(wrf.QRAIN**2/wrf.QNRAIN)*gamma(7)*rho_air/(rho_rain*gamma(4))**2
                rain_ref = rain_ref.fillna(value=0)

                # Solve for graupel reflectivity
                N1 = 10**4
                N2 = 5*10**6
                qg0 = 0.15*10**(-3)
                N0g = ((N1-N2)/2)*np.tanh((qg0-wrf.QGRAUP)/qg0)+(N1+N2)/2
                graup_ref = 720*(rho_air*wrf.QGRAUP)**(7/4)/(N0g**(3/4)*(np.pi*rho_graup)**(7/4))*(rho_graup/rho_ice)**2

                # Solve for snow reflectivity
                a = 5.065339-0.062659*T-3.032362*4+0.029469*T*4-0.000285*T**2+0.312550*4**2+0.00020*T**2*4+0.003199*T*4**2-0.015952*4**3
                a = 10 ** a
                b = 0.476221-0.015896*T+0.165977*4+0.007468*T*4-0.000141*T**2+0.060366*4**2+0.000079*T**2*4+0.000594*T*4**2-0.003577*4**3
                snow_ref = 0.189*(0.069*6/(np.pi*rho_ice))**2*a*(wrf.QSNOW*rho_air/0.069)**b      
                            
            Z = 10 * np.log10(10 ** 18 * (rain_ref+snow_ref+graup_ref))
            Z.values[Z.values<0] = np.nan
            Z.name = 'reflectivity'
            wrf = xr.merge([wrf,Z])
            
            wrf.time.encoding['units'] = 'seconds since   '+str(t)+'Z'
            wrf.time.attrs['standard_name'] = 'time'
            wrf.time.encoding['calendar'] = 'gregorian'

            # Copy metadata from CPOL dataset
            wrf['projection'] = CPOL['projection']
            wrf['ProjectionCoordinateSystem'] = CPOL['ProjectionCoordinateSystem']
            wrf['radar_latitude'] = origin_lat
            wrf['radar_latitude'].attrs = CPOL['radar_latitude'].attrs
            wrf['radar_longitude'] = origin_lon
            wrf['radar_longitude'].attrs = CPOL['radar_longitude'].attrs
            wrf['radar_altitude'] = CPOL['radar_altitude']

            wrf['origin_latitude'] = ('time', [origin_lat])
            wrf['origin_latitude'].attrs = CPOL['origin_latitude'].attrs
            wrf['origin_longitude'] = ('time', [origin_lon])
            wrf['origin_longitude'].attrs = CPOL['origin_longitude'].attrs
            wrf['origin_altitude'] = ('time', CPOL['origin_altitude'].values)
            wrf['origin_altitude'].attrs = CPOL['origin_altitude'].attrs

            wrf['radar_time'] = ('nradar', [t])
            wrf.radar_time.attrs = CPOL.radar_time.attrs
            
            print('Saving file ' + str(t))
            if mp=='lin':
                base = '/g/data/w40/esh563/lind04_ref/'
                wrf.to_netcdf(base + 'lind04_ref_' + str(t) + '.nc', 
                          format='NETCDF4')
            else:
                base = '/g/data/w40/esh563/thompsond04_ref/'
                wrf.to_netcdf(base + 'thompsond04_ref_' + str(t) + '.nc', 
                          format='NETCDF4')
            
        del WRF
           
           
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
        # Create reflectivity data array
        data = np.load(fn[i])
        surface = np.ones((1,241,241)) * np.nan
        data = np.concatenate([surface, data])
        data[data==-20]=np.nan
        # Could impose radius limiting here... 
        data = np.expand_dims(data, 0)
        da_ref = xr.DataArray(data, dims=('time', 'z', 'x', 'y'), 
                              coords={'time': np.array([datetimes[i]]), 
                                      'z':z, 'x':x, 'y':y})

        da_ref.time.encoding['units'] = 'seconds since   '+str(datetimes[i])+'Z'
        da_ref.time.attrs['standard_name'] = 'time'
        da_ref.time.encoding['calendar'] = 'gregorian'

        da_ref.x.attrs = CPOL.x.attrs
        da_ref.y.attrs = CPOL.y.attrs
        da_ref.z.attrs = CPOL.z.attrs

        # Create dataset
        ds = xr.Dataset({'reflectivity':da_ref})
        ds.reflectivity.attrs = CPOL.reflectivity.attrs

        # Add longitude, latitude data
        data_lon = np.load('/g/data/w40/esh563/dbz_lon.npy')
        data_lat = np.load('/g/data/w40/esh563/dbz_lat.npy')

        LON, LAT = np.meshgrid(data_lon, data_lat)
        LON = np.expand_dims(LON, 0)
        LON = np.concatenate([LON]*41)
        LAT = np.expand_dims(LAT, 0)
        LAT = np.concatenate([LAT]*41)
        LON = np.expand_dims(LON, 0)
        LAT = np.expand_dims(LAT, 0)

        ds['latitude'] = (('time', 'z', 'x', 'y'), LAT)
        ds['longitude'] = (('time','z', 'x', 'y'), LON)
        ds.latitude.attrs = CPOL.latitude.attrs
        ds.longitude.attrs = CPOL.longitude.attrs

        ds = ds.coarsen(x=2, boundary='trim', side='left').mean()
        ds = ds.coarsen(y=2, boundary='trim', side='left').mean()

        ds.coords['x'] = np.arange(0, 2500*120, 2500, dtype=float)
        ds.coords['y'] = np.arange(0, 2500*120, 2500, dtype=float)

        ds.coords['x'] -= ds.x.mean()
        ds.coords['y'] -= ds.y.mean()

        origin_lat = ds.latitude.mean().values
        origin_lon = ds.longitude.mean().values

        # Copy metadata from CPOL dataset
        ds['projection'] = CPOL['projection']
        ds['ProjectionCoordinateSystem'] = CPOL['ProjectionCoordinateSystem']
        ds['radar_latitude'] = origin_lat
        ds['radar_latitude'].attrs = CPOL['radar_latitude'].attrs
        ds['radar_longitude'] = origin_lon
        ds['radar_longitude'].attrs = CPOL['radar_longitude'].attrs
        ds['radar_altitude'] = CPOL['radar_altitude']

        ds['origin_latitude'] = ('time', [origin_lat])
        ds['origin_latitude'].attrs = CPOL['origin_latitude'].attrs
        ds['origin_longitude'] = ('time', [origin_lon])
        ds['origin_longitude'].attrs = CPOL['origin_longitude'].attrs
        ds['origin_altitude'] = ('time', CPOL['origin_altitude'].values)
        ds['origin_altitude'].attrs = CPOL['origin_altitude'].attrs

        ds['radar_time'] = ('nradar', [datetimes[i]])
        ds.radar_time.attrs = CPOL.radar_time.attrs

        ds.to_netcdf(fn[i][:-4] + '.nc', format='NETCDF4')
        

def average_winds():
    # Load height field
    hgt_AGL = np.load('/g/data/w40/esh563/d04_hgt_AGL.npy')

    # Create function to de
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
        
    fn = sorted(glob.glob('/g/data/w40/esh563/d04.dir/*.nc.gz'))
    for i in range(len(fn)):
        print('Loading ' + fn[i])
        data = xr.open_dataset(fn[i])

        XLAT = data.XLAT[0,:,0].values
        XLONG = data.XLONG[0,0,:].values
        XLAT_S = data.XLAT_V[0,:,0].values
        XLONG_S = data.XLONG_U[0,0,:].values

        data['bottom_top'] =  np.round(moving_average(hgt_AGL, n=2))
        data['bottom_top_stag'] = np.round(hgt_AGL)

        data = data.drop(['XLAT', 'XLONG', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V'], errors='raise')
        data['south_north'] = XLAT
        data['south_north_stag'] = XLAT_S
        data['west_east'] = XLONG
        data['west_east_stag'] = XLONG_S

        z = np.arange(0,20500,500)

        W = data.W
        # De-stagger
        W = W.rolling(bottom_top_stag=2, center=True).mean().dropna('bottom_top_stag')
        W = W.rename({'bottom_top_stag': 'bottom_top'})
        # Replace coordinates of de-staggered dimension.
        # Note xarray does not do this by default!
        W['bottom_top'] = np.round(moving_average(hgt_AGL, n=2))
        # Now interpolate onto new z values
        W = W.interp({'bottom_top':z})
        # Subset
        W = W.isel({'south_north': slice(63,304), 'west_east': slice(268, 509)})
        # Coarsen changes lat, lon as expected
        W = W.coarsen(west_east=2, boundary='trim', side='left').mean()
        W = W.coarsen(south_north=2, boundary='trim', side='left').mean()
        W.name = 'W'

        U = data.U
        # De-stagger
        U = U.rolling(west_east_stag=2, center=True).mean().dropna('west_east_stag')
        U = U.rename({'west_east_stag': 'west_east'})
        # Replace coordinates of de-staggered dimension.
        # Note xarray does not do this by default!
        U['west_east'] = XLONG
        # Now interpolate onto new z values
        U = U.interp({'bottom_top':z})
        # Subset
        U = U.isel({'south_north': slice(63,304), 'west_east': slice(268, 509)})
        # Coarsen changes lat, lon as expected
        U = U.coarsen(west_east=2, boundary='trim', side='left').mean()
        U = U.coarsen(south_north=2, boundary='trim', side='left').mean()
        U.name = 'U'

        V = data.V
        # De-stagger
        V = V.rolling(south_north_stag=2, center=True).mean()
        V = V.dropna('south_north_stag')
        V = V.rename({'south_north_stag': 'south_north'})
        # Replace coordinates of de-staggered dimension.
        # Note xarray does not do this by default!
        V['south_north'] = XLAT
        # Now interpolate onto new z values
        V = V.interp({'bottom_top':z})
        # Subset
        V = V.isel({'south_north': slice(63,304), 'west_east': slice(268, 509)})
        # Coarsen changes lat, lon as expected
        V = V.coarsen(west_east=2, boundary='trim', side='left').mean()
        V = V.coarsen(south_north=2, boundary='trim', side='left').mean()
        V.name = 'V'

        wind = xr.merge([U,V,W])

        wind = wind.rename({'south_north':'lat', 
                            'west_east':'lon', 
                            'bottom_top': 'alt'})

        base = '/g/data/w40/esh563/lind04_2500_winds/'
        for j in range(3):

            dt = np.datetime64(fn[i][-25:-6].replace('_', 'T'))
            delt = np.timedelta64(10, 'm')

            wind_j = wind.isel({'Time': j})
            print('Saving ' + base + str(dt+j*delt) + '.nc')
            wind_j.to_netcdf(base + str(dt+j*delt) + '.nc', 
                             mode='w', format='NETCDF4')

            # wind.to_netcdf()
