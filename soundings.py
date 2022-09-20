import xarray as xr
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd
import classification as cl

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units


def plot_skewt(
        soundings, fig=None, figsize=(12, 6), subplots=None,
        legend=False, left_ticks=True, right_ticks=False, title=None):

    if fig is None:
        fig = plt.figure(figsize=figsize)

    p = soundings['p'].values * units.Pa
    T = (soundings['t'].values-273.15) * units.degC
    q = soundings['q']
    # u = soundings['u'] * units.meter_per_second
    # v = soundings['v'] * units.meter_per_second
    heights = soundings['altitude'] * units.meter

    Td = mpcalc.dewpoint_from_specific_humidity(p, T, q)

    skew = SkewT(fig=fig, subplot=subplots, rotation=45)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # colors = [colors[i] for i in [0, 1, 2, 4, 5, 6]]

    cl.init_fonts()

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot.
    skew.plot(p, T, 'r', label='Temperature')
    skew.plot(p, Td, colors[0], label='Dew Point Temperature')
    # skew.plot_barbs(p[::3], u[::3], v[::3])
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-10, 30)

    f = sp.interpolate.interp1d(
        p[::-1], heights[::-1], fill_value='extrapolate')

    secax = skew.ax.secondary_yaxis(location=1)
    secax.set_yticks(np.arange(1000, 0, -100))

    labels = f(np.arange(100000, 0, -10000))/1000
    labels = ['{:.02f}'.format(lbl) for lbl in labels]

    secax.set_yticklabels(labels)
    secax.set_ylabel('Altitude [m]')

    if left_ticks:
        skew.ax.set_ylabel(r'Pressure [hPa]')
    else:
        skew.ax.set_yticklabels([])
        skew.ax.set_ylabel(None)

    skew.ax.set_xlabel(r'Temperature [$^\circ$C]')

    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
    skew.plot(p, prof, 'k', linewidth=2, label='Parcel Profile')

    # Shade areas of CAPE and CIN
    skew.shade_cin(p, T, prof, Td, label='CIN')
    skew.shade_cape(p, T, prof, label='CAPE')

    # Add the relevant special lines
    skew.plot_dry_adiabats(
        label='Dry Adiabat', linewidth=1.25, linestyle='dotted')
    skew.plot_moist_adiabats(
        linewidth=1.25, label='Moist Adiabat', linestyle='dotted')
    skew.plot_mixing_lines(
        linewidth=1, label='Mixing Lines', linestyle='dashed', colors='grey')

    if title is not None:
        skew.ax.set_title(title, fontsize=12)

    if legend:
        skew.ax.legend(
            loc='lower center', bbox_to_anchor=(1.1, -0.3),
            ncol=4, fancybox=True, shadow=True)

    return skew.ax


def get_ERA5_soundings(lon=130.925, lat=-12.457):
    days_2021 = np.arange(
        np.datetime64('2021-01-01'), np.datetime64('2021-03-01'),
        np.timedelta64(1, 'D'))
    days_2022 = np.arange(
        np.datetime64('2022-01-01'), np.datetime64('2022-03-01'),
        np.timedelta64(1, 'D'))
    days_list = [days_2021, days_2022]

    days_hours_2021 = np.arange(
        np.datetime64('2021-01-01'), np.datetime64('2021-03-01'),
        np.timedelta64(6, 'h'))
    days_hours_2022 = np.arange(
        np.datetime64('2022-01-01'), np.datetime64('2022-03-01'),
        np.timedelta64(6, 'h'))
    days_hours_list = [days_hours_2021, days_hours_2022]

    pope_df = pd.read_csv(
        'fake_pope_regimes.csv', header=None,
        index_col=0, names=['time', 'pope_regime'])
    pope_df.index = pd.to_datetime(pope_df.index)

    base_dir = '/g/data/w40/esh563/era5/pressure-levels/reanalysis/'

    years = [2021, 2022]

    hours = [0, 6, 12, 18]

    soundings_ds = [[] for i in range(len(hours))]

    for i in range(len(years)):

        datasets = []
        for var in ['u', 'v', 't', 'q', 'z']:

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

        [u, v, t, q, z] = datasets

        for j in range(len(days_list[i])):

            print('Getting {}'.format(days_list[i][j]))

            for k in range(len(hours)):

                time_jk = days_list[i][j] + np.timedelta64(hours[k], 'h')
                u_t = u.sel(time=time_jk)
                v_t = v.sel(time=time_jk)
                t_t = t.sel(time=time_jk)
                q_t = q.sel(time=time_jk)
                z_t = z.sel(time=time_jk)

                z_t = z_t['z'] / 9.80665

                p_t = copy.deepcopy(u_t)
                p_t = p_t.rename({'u': 'p'})
                p_t['p'] = p_t['level']

                u_t = u_t.rename({'level': 'altitude'})
                v_t = v_t.rename({'level': 'altitude'})
                t_t = t_t.rename({'level': 'altitude'})
                q_t = q_t.rename({'level': 'altitude'})
                p_t = p_t.rename({'level': 'altitude'})

                u_t = u_t['u'].assign_coords({'altitude': z_t.values})
                v_t = v_t['v'].assign_coords({'altitude': z_t.values})
                t_t = t_t['t'].assign_coords({'altitude': z_t.values})
                q_t = q_t['q'].assign_coords({'altitude': z_t.values})
                p_t = p_t['p'].assign_coords({'altitude': z_t.values})

                ds_t = xr.Dataset({
                    'u': u_t, 'v': v_t,
                    'p': p_t, 'q': q_t,
                    't': t_t})

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


def get_ACCESS_G_soundings(lon=130.925, lat=-12.457):

    days_2021 = np.arange(
        np.datetime64('2021-01-01'), np.datetime64('2021-03-01'),
        np.timedelta64(1, 'D'))
    days_2022 = np.arange(
        np.datetime64('2022-01-01'), np.datetime64('2022-03-01'),
        np.timedelta64(1, 'D'))
    days_list = [days_2021, days_2022]

    bad_days_list = [
        np.datetime64('2021-01-12'),
        np.datetime64('2022-01-06'),
        np.datetime64('2022-01-21'),
        np.datetime64('2022-02-17')]

    days = sorted(list(set(np.concatenate(days_list)) - set(bad_days_list)))

    pope_df = pd.read_csv(
        'fake_pope_regimes.csv', header=None,
        index_col=0, names=['time', 'pope_regime'])
    pope_df.index = pd.to_datetime(pope_df.index)

    base_dir = '/g/data/wr45/ops_aps3/access-g/1/'

    hours = [0, 6, 12, 18]

    soundings_ds = [[] for i in range(len(hours))]

    topog = xr.open_dataset(
        base_dir + '20201001/0000/an/sfc/topog.nc')
    topog = topog.interp(lon=lon, lat=lat)

    new_alts = np.arange(100, 20100, 100)

    for i in range(len(days)):

        print('Loading {}'.format(days[i]))

        for j in range(len(hours)):

            datetime_str = days[i].astype(str).replace('-', '')

            year = datetime_str[0:4]
            month = datetime_str[4:6]
            day = datetime_str[6:8]

            date_str = '{}{}{}'.format(year, month, day)
            hour_str = '{:02d}00'.format(hours[j])

            datasets = []
            file_exists = True
            for var in [
                    'wnd_ucmp', 'wnd_vcmp', 'air_temp',
                    'pressure', 'spec_hum']:

                try:
                    ds = xr.open_dataset(
                        base_dir + '{}/{}/an/ml/{}.nc'.format(
                            date_str, hour_str, var))
                except FileNotFoundError:
                    print('Missing File.')
                    file_exists = False
                    continue

                ds = ds.interp(lon=lon, lat=lat)
                ds = ds.load()

                datasets.append(ds)

            if not file_exists:
                continue

            [u_t, v_t, t_t, p_t, q_t] = datasets

            altitude_rho = u_t.A_rho + u_t.B_rho * topog['topog']
            altitude_theta = p_t.A_theta + p_t.B_theta * topog['topog']

            u_t = u_t.rename({'rho_lvl': 'altitude'})
            v_t = v_t.rename({'rho_lvl': 'altitude'})
            t_t = t_t.rename({'theta_lvl': 'altitude'})
            p_t = p_t.rename({'theta_lvl': 'altitude'})
            q_t = q_t.rename({'theta_lvl': 'altitude'})

            u_t = u_t['wnd_ucmp'].assign_coords(
                {'altitude': altitude_rho.squeeze().values})
            v_t = v_t['wnd_vcmp'].assign_coords(
                {'altitude': altitude_rho.squeeze().values})
            t_t = t_t['air_temp'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})
            p_t = p_t['pressure'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})
            q_t = q_t['spec_hum'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})

            u_t = u_t.interp(altitude=new_alts)
            v_t = v_t.interp(altitude=new_alts)
            t_t = t_t.interp(altitude=new_alts)
            p_t = p_t.interp(altitude=new_alts)
            q_t = q_t.interp(altitude=new_alts)

            ds_t = xr.Dataset({
                'u': u_t, 'v': v_t, 'p': p_t, 't': t_t, 'q': q_t})

            ds_t['time'] = ds_t['time'] - np.timedelta64(hours[j], 'h')

            ds_t['pope_regime'] = pope_df.loc[ds_t['time'].values]

            soundings_ds[j].append(ds_t)

    new_ds = []

    for i in range(len(hours)):
        ds_i = xr.concat(soundings_ds[i], dim='time')
        ds_i = ds_i.drop('dim_1')
        ds_i = ds_i.squeeze()
        new_ds.append(ds_i)

    hours_da = xr.DataArray(hours, name='hour', coords={'hour': hours})

    ACCESS_G_soundings = xr.concat(new_ds, dim=hours_da)

    R = 287.04
    cp = 1005

    ACCESS_G_soundings['theta'] = ACCESS_G_soundings['t']*(
        1e5/ACCESS_G_soundings['p'])**(R/cp)

    return ACCESS_G_soundings


def get_ACCESS_C_soundings(lon=130.925, lat=-12.457):

    days_2021 = np.arange(
        np.datetime64('2021-01-01'), np.datetime64('2021-03-01'),
        np.timedelta64(1, 'D'))
    days_2022 = np.arange(
        np.datetime64('2022-01-01'), np.datetime64('2022-03-01'),
        np.timedelta64(1, 'D'))
    days_list = [days_2021, days_2022]

    # bad_days_list = [
    #     np.datetime64('2021-01-12'),
    #     np.datetime64('2022-01-06'),
    #     np.datetime64('2022-01-21'),
    #     np.datetime64('2022-02-17')]

    # days = sorted(list(set(np.concatenate(days_list)) - set(bad_days_list)))
    days = np.concatenate(days_list)

    pope_df = pd.read_csv(
        'fake_pope_regimes.csv', header=None,
        index_col=0, names=['time', 'pope_regime'])
    pope_df.index = pd.to_datetime(pope_df.index)

    base_dir = '/g/data/wr45/ops_aps3/access-dn/1/'

    hours = [0, 6, 12, 18, 24]

    soundings_ds = [[] for i in range(len(hours))]

    topog = xr.open_dataset(
        base_dir + '20201001/0000/an/sfc/topog.nc')
    topog = topog.interp(lon=lon, lat=lat)

    new_alts = np.arange(100, 20100, 100)
    bad_days = []

    for i in range(len(days_2022)):

        print('Loading {}'.format(days[i]))

        datetime_str = (days[i]-np.timedelta64(1, 'D'))
        datetime_str = datetime_str.astype(str).replace('-', '')

        year = datetime_str[0:4]
        month = datetime_str[4:6]
        day = datetime_str[6:8]

        date_str = '{}{}{}'.format(year, month, day)
        hour_str = '1200'

        datasets = []
        file_exists = True

        for var in [
                'wnd_ucmp', 'wnd_vcmp', 'air_temp', 'spec_hum']:

            try:
                ds = xr.open_dataset(
                    base_dir + '{}/{}/fc/ml/{}.nc'.format(
                        date_str, hour_str, var))
                times = np.arange(
                    days[i], days[i]+np.timedelta64(30, 'h'),
                    np.timedelta64(6, 'h'))
                ds = ds.sel(time=times)
                ds = ds.interp(lon=lon, lat=lat)
                ds = ds.load()

            except FileNotFoundError:
                print('Missing File.')
                file_exists = False
                continue
            except ValueError:
                print('Bad Data.')
                file_exists = False
                continue
            except pd.errors.InvalidIndexError:
                print('Bad index.')
                file_exists = False
                continue

            datasets.append(ds)

        try:
            ds = xr.open_dataset(
                base_dir + '{}/{}/fc/pl/geop_ht.nc'.format(
                    date_str, hour_str))
            times = np.arange(
                days[i], days[i]+np.timedelta64(30, 'h'),
                np.timedelta64(6, 'h'))
            ds = ds.sel(time=times)
            ds = ds.interp(lon=lon, lat=lat)
            ds = ds.load()
        except FileNotFoundError:
            print('Missing File.')
            file_exists = False
            ds = None
        except ValueError:
            print('Bad Data.')
            file_exists = False
        except pd.errors.InvalidIndexError:
            print('Bad index.')
            file_exists = False

        if not file_exists:
            bad_days.append(days[i])
            continue

        datasets.append(ds)

        [u, v, t, q, geop] = datasets

        for j in range(len(hours)):

            datasets_t = [
                vble.sel(time=(days[i]+np.timedelta64(hours[j], 'h')))
                for vble in [u, v, t, q, geop]]
            [u_t, v_t, t_t, q_t, geop_t] = datasets_t

            altitude_rho = u_t.A_rho + u_t.B_rho * topog['topog']
            altitude_theta = t_t.A_theta + t_t.B_theta * topog['topog']

            import pdb; pdb.set_trace()
            altitude = geop_t['geop_ht'].values
            pressure = geop_t['lvl'].values

            p = copy.deepcopy(geop_t)
            p = p.drop(['geop_ht', 'lvl'])
            p['pressure'] = pressure
            p['pressure'].assign_coords(
                {'altitude': altitude})

            u_t = u_t.rename({'rho_lvl': 'altitude'})
            v_t = v_t.rename({'rho_lvl': 'altitude'})
            t_t = t_t.rename({'theta_lvl': 'altitude'})
            p_t = p_t.rename({'theta_lvl': 'altitude'})
            q_t = q_t.rename({'theta_lvl': 'altitude'})

            u_t = u_t['wnd_ucmp'].assign_coords(
                {'altitude': altitude_rho.squeeze().values})
            v_t = v_t['wnd_vcmp'].assign_coords(
                {'altitude': altitude_rho.squeeze().values})
            t_t = t_t['air_temp'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})
            p_t = p_t['pressure'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})
            q_t = q_t['spec_hum'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})

            u_t = u_t.interp(altitude=new_alts)
            v_t = v_t.interp(altitude=new_alts)
            t_t = t_t.interp(altitude=new_alts)
            p_t = p_t.interp(altitude=new_alts)
            q_t = q_t.interp(altitude=new_alts)

            ds_t = xr.Dataset({
                'u': u_t, 'v': v_t, 'p': p_t, 't': t_t, 'q': q_t})

            ds_t['time'] = ds_t['time'] - np.timedelta64(hours[j], 'h')

            ds_t['pope_regime'] = pope_df.loc[ds_t['time'].values]

            soundings_ds[j].append(ds_t)

    new_ds = []

    for i in range(len(hours)):
        ds_i = xr.concat(soundings_ds[i], dim='time')
        ds_i = ds_i.drop('dim_1')
        ds_i = ds_i.squeeze()
        new_ds.append(ds_i)

    hours_da = xr.DataArray(hours, name='hour', coords={'hour': hours})

    ACCESS_C_soundings = xr.concat(new_ds, dim=hours_da)

    R = 287.04
    cp = 1005

    ACCESS_C_soundings['theta'] = ACCESS_C_soundings['t']*(
        1e5/ACCESS_C_soundings['p'])**(R/cp)

    return ACCESS_C_soundings


def plot_soundings(ERA5_soundings):

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    cl.init_fonts()

    pope_regime = 2

    min_speed = 0
    max_speed = 0

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 1, 2, 4, 5, 6]]

    ERA5_soundings_pope = ERA5_soundings.where(
        ERA5_soundings['pope_regime'] == pope_regime)

    u_t = ERA5_soundings_pope['u'].mean(dim=['hour', 'time'])
    v_t = ERA5_soundings_pope['v'].mean(dim=['hour', 'time'])
    t_t = ERA5_soundings_pope['theta'].mean(dim=['hour', 'time'])

    u_t_sig = np.sqrt(ERA5_soundings_pope['u'].var(dim=['hour', 'time']))
    v_t_sig = np.sqrt(ERA5_soundings_pope['v'].var(dim=['hour', 'time']))
    t_t_sig = np.sqrt(ERA5_soundings_pope['theta'].var(dim=['hour', 'time']))

    # hour = 18

    # u_t = ERA5_soundings_pope['u'].sel(hour=hour).mean(dim='time')
    # v_t = ERA5_soundings_pope['v'].sel(hour=hour).mean(dim='time')
    # t_t = ERA5_soundings_pope['t'].sel(hour=hour).mean(dim='time')

    # u_t_sig = np.sqrt(ERA5_soundings_pope['u'].sel(hour=hour).var(dim='time'))
    # v_t_sig = np.sqrt(ERA5_soundings_pope['v'].sel(hour=hour).var(dim='time'))

    line1 = ax.plot(
        u_t, u_t.altitude, color=colors[0], label=r'ERA5 $u$',
        linewidth=1.75)
    ax.fill_betweenx(
        u_t.altitude, u_t-u_t_sig, u_t+u_t_sig, color=colors[0],
        alpha=0.1, linewidth=1.75)

    line2 = ax.plot(
        v_t, v_t.altitude, color=colors[0], label=r'ERA5 $v$',
        linestyle='dashed')
    ax.fill_betweenx(
        v_t.altitude, v_t-v_t_sig, v_t+v_t_sig, color=colors[0],
        alpha=0.1, linewidth=1.75, linestyle='--')

    ax.set_xticks(np.arange(-35, 25, 5))
    ax.set_xticks(np.arange(-35, 22.5, 2.5), minor=True)
    ax.set_xlabel(r'Velocity [m/s]')
    ax.set_ylabel(r'Altitude [km]')
    ax.set_yticks(np.arange(0, 22e3, 2e3))
    ax.set_yticks(np.arange(0, 21e3, 1e3), minor=True)
    ax.set_yticklabels(np.arange(0, 22, 2))

    twin0 = ax.twiny()
    twin0.xaxis.set_ticks_position("top")
    twin0.xaxis.set_label_position("top")

    ax.set_xlim([-35, 20])
    twin0.set_xlim([260, 480])
    twin0.set_xticks(np.arange(260, 520, 40))
    twin0.set_xticks(np.arange(260, 500, 20), minor=True)
    twin0.set_xlabel(r'$\theta$ [K]')

    line3 = twin0.plot(
        t_t, t_t.altitude, linestyle='dashdot', color=colors[0],
        label=r'ERA5 $\theta$')
    twin0.fill_betweenx(
        t_t.altitude, t_t-t_t_sig, t_t+t_t_sig, color=colors[0],
        alpha=0.1, linewidth=1.75, linestyle='dashdot')

    min_speed = min(min(np.concatenate([u_t.values, v_t.values])), min_speed)
    max_speed = max(max(np.concatenate([u_t.values, v_t.values])), max_speed)

    max_sig = max(np.max(v_t_sig), np.max(u_t_sig))+5

    line4 = ax.plot(
        [min_speed-max_sig, max_speed+max_sig], [500, 500],
        linestyle='dashed', color='grey', label='MINT Layers')
    ax.plot(
        [min_speed-max_sig, max_speed+max_sig], [3000, 3000],
        linestyle='dashed', color='grey')
    ax.plot(
        [min_speed-max_sig, max_speed+max_sig], [2000, 2000],
        linestyle='dashed', color='grey')
    ax.plot(
        [min_speed-max_sig, max_speed+max_sig], [4000, 4000],
        linestyle='dashed', color='grey')

    lines = line1 + line2 + line3 + line4
    labels = [line.get_label() for line in lines]

    ax.grid(which='major', alpha=0.5, axis='both')
    ax.grid(which='minor', alpha=0.2, axis='both')

    ax.legend(
        lines, labels,
        loc='lower center', bbox_to_anchor=(.5, -0.30),
        ncol=4, fancybox=True, shadow=True)
