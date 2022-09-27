import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import copy

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

import classification as cl


def get_ACCESS_C(gadi=False):

    if gadi:
        base_dir = '/g/data/wr45/'
    else:
        base_dir = 'https://dapds00.nci.org.au/thredds/dodsC/wr45/'
    base_dir += 'ops_aps3/access-dn/1/'

    datetimes_2020 = np.arange(
        np.datetime64('2020-10-01'),
        np.datetime64('2021-05-01'),
        np.timedelta64(1, 'D'))
    datetimes_2021 = np.arange(
        np.datetime64('2021-10-01'),
        np.datetime64('2022-05-01'),
        np.timedelta64(1, 'D'))

    datetimes = np.concatenate([datetimes_2020, datetimes_2021])

    lon_min = 128
    lon_max = 136
    lat_min = -9
    lat_max = -17

    u_fc_all = [None for i in range(5)]
    v_fc_all = [None for i in range(5)]

    hours = [0, 6, 12, 18, 24]

    dt = datetimes[0]
    date = str(dt)[0:10].replace('-', '')
    hour = '1200'
    u_fc = xr.open_dataset(
        base_dir + date + '/' + hour + '/fc/sfc/uwnd10m.nc')
    v_fc = xr.open_dataset(
        base_dir + date + '/' + hour + '/fc/sfc/vwnd10m.nc')

    u_fc = u_fc.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    v_fc = v_fc.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    u_lon_stable = u_fc.lon
    u_lat_stable = u_fc.lat
    v_lon_stable = v_fc.lon
    v_lat_stable = v_fc.lat

    missing_forecasts = []

    for dt in datetimes:

        fc_data_datetime = dt - np.timedelta64(1, 'D')
        fc_date_str = str(fc_data_datetime)[0:10].replace('-', '')

        try:
            u_fc = xr.open_dataset(
                base_dir + fc_date_str + '/' + '1200' + '/fc/sfc/uwnd10m.nc')
            v_fc = xr.open_dataset(
                base_dir + fc_date_str + '/' + '1200' + '/fc/sfc/vwnd10m.nc')
        except FileNotFoundError:
            print('Forecast data missing at {}'.format(dt))
            missing_forecasts.append(
                base_dir + fc_date_str + '/' + '1200' + '/fc/sfc/uwnd10m.nc')
            continue

        u_fc = u_fc.sel(
            lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        v_fc = v_fc.sel(
            lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

        u_fc['lon'] = u_lon_stable
        u_fc['lat'] = u_lat_stable
        v_fc['lon'] = v_lon_stable
        v_fc['lat'] = v_lat_stable

        # Coarsen access-c
        # access-c ~1.5 km, access-g 12 km
        # so coarsen by 8
        u_fc = u_fc.coarsen({'lat': 8, 'lon': 8}, boundary='trim').mean()
        v_fc = v_fc.coarsen({'lat': 8, 'lon': 8}, boundary='trim').mean()

        for i in range(len(hours)):

            print('Getting {} 00:00 + {} data.'.format(dt, hours[i]))

            an_datetime = dt + np.timedelta64(hours[i], 'h')

            u_fc_i = u_fc.sel(time=an_datetime)
            v_fc_i = v_fc.sel(time=an_datetime)

            u_fc_i['time'] = u_fc_i['time'] - np.timedelta64(hours[i], 'h')
            v_fc_i['time'] = v_fc_i['time'] - np.timedelta64(hours[i], 'h')



            if u_fc_all[i] is None:
                u_fc_all[i] = copy.deepcopy(u_fc_i)
                v_fc_all[i] = copy.deepcopy(v_fc_i)
            else:
                u_fc_all[i] = xr.concat([u_fc_all[i], u_fc_i], dim='time')
                v_fc_all[i] = xr.concat([v_fc_all[i], v_fc_i], dim='time')

    return u_fc_all, v_fc_all
