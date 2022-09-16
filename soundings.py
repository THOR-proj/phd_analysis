import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import classification as cl

def get_ERA5_soundings():
    days_2021 = np.arange(
    np.datetime64('2021-01-01'), np.datetime64('2021-03-01'), np.timedelta64(1, 'D'))
    days_2022 = np.arange(
        np.datetime64('2022-01-01'), np.datetime64('2022-03-01'), np.timedelta64(1, 'D'))
    days_list = [days_2021, days_2022]

    days_hours_2021 = np.arange(
        np.datetime64('2021-01-01'), np.datetime64('2021-03-01'), np.timedelta64(6, 'h'))
    days_hours_2022 = np.arange(
        np.datetime64('2022-01-01'), np.datetime64('2022-03-01'), np.timedelta64(6, 'h'))
    days_hours_list = [days_hours_2021, days_hours_2022]

    pope_df = pd.read_csv(
        'fake_pope_regimes.csv', header=None,
        index_col=0, names=['time', 'pope_regime'])
    pope_df.index = pd.to_datetime(pope_df.index)

    lon = 130.925
    lat = -12.457

    base_dir = '/g/data/w40/esh563/era5/pressure-levels/reanalysis/'

    years = [2021, 2022]

    hours = [0, 6, 12, 18]

    soundings_ds = [[] for i in range(len(hours))]

    for i in range(len(years)):

        datasets = []
        for var in ['u', 'v', 't', 'z']:

            print('Loading {}, {}'.format(years[i], var))

            ds = xr.open_mfdataset(
                [
                    base_dir
                    + '{}/{}/{}_era5_oper_pl_{}0101-{}0131.nc'.format(
                        var, years[i], var, years[i], years[i]),
                    base_dir
                    + '{}/{}/{}_era5_oper_pl_{}0201-{}0228.nc'.format(
                        var, years[i], var, years[i], years[i])])

            ds = ds.sel(
                time=days_hours_list[i], longitude=lon,
                latitude=lat, method='nearest')
            ds = ds.load()

            datasets.append(ds)

        [u, v, t, z] = datasets

        for j in range(len(days_list[i])):

            print('Getting {}'.format(days_list[i][j]))

            for k in range(len(hours)):

                time_jk = days_list[i][j] + np.timedelta64(hours[k], 'h')
                u_t = u.sel(time=time_jk)
                v_t = v.sel(time=time_jk)
                t_t = t.sel(time=time_jk)
                z_t = z.sel(time=time_jk)

                z_t = z_t['z'] / 9.80665

                p_t = copy.deepcopy(u_t)
                p_t = p_t.rename({'u': 'p'})
                p_t['p'] = p_t['level']

                u_t = u_t.rename({'level': 'altitude'})
                v_t = v_t.rename({'level': 'altitude'})
                t_t = t_t.rename({'level': 'altitude'})
                p_t = p_t.rename({'level': 'altitude'})

                u_t = u_t['u'].assign_coords({'altitude': z_t.values})
                v_t = v_t['v'].assign_coords({'altitude': z_t.values})
                t_t = t_t['t'].assign_coords({'altitude': z_t.values})
                p_t = p_t['p'].assign_coords({'altitude': z_t.values})

                ds_t = xr.Dataset({
                    'u': u_t, 'v': v_t,
                    'p': p_t, 't': t_t})

                ds_t = ds_t.sel(altitude=slice(25e3, 0))

                new_alts = np.arange(100, 20100, 100)
                ds_t = ds_t.interp(altitude=new_alts)

                ds_t['time'] = ds_t['time'] - np.timedelta64(hours[k], 'h')

                ds_t['pope_regime'] = pope_df.loc[ds_t['time'].values]

                soundings_ds[k].append(ds_t)

    new_ds = []

    for i in range(len(hours)):
        ds_i = xr.concat(soundings_ds[i], dim='time')
        ds_i = ds_i.drop('dim_0')
        ds_i = ds_i.squeeze()
        new_ds.append(ds_i)

    hours_da = xr.DataArray(hours, name='hour', coords={'hour': hours})

    ERA5_soundings = xr.concat(new_ds, dim=hours_da)

    R = 287.04
    cp = 1005

    ERA5_soundings['theta'] = ERA5_soundings['t']*(
        1e3/ERA5_soundings['p'])**(R/cp)

    return ERA5_soundings
