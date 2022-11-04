import xarray as xr
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd
import classification as cl
import re

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units


def read_wyoming_sounding_txt(
        loc='darwin', year=2021,
        split_txt='94120 YPDN Darwin Airport Observations at '):
    # Parse txt files
    fn = '{}_sounding_2021_jan.txt'.format(loc)
    with open(fn, 'rb') as f:
        text = f.read().decode()

    fn = '{}_sounding_2021_feb.txt'.format(loc)
    with open(fn, 'rb') as f:
        text += f.read().decode()

    col_names = [
        'p', 'altitude', 't', 'Td', 'RH', 'r', 'drct', 'sknt',
        'theta', 'theta_e', 'theta_v']

    soundings_list = text.split(split_txt)
    soundings_list = [snd for snd in soundings_list if snd != '']

    times = [snd[:15] for snd in soundings_list]
    parse_month = {'Jan': '01', 'Feb': '02'}
    times = [
        np.datetime64(
            '{}-{}-{}T{}'.format(
                snd[11:15], parse_month[snd[7:10]], snd[4:6], snd[0:2]))
        for snd in soundings_list]

    start = 'deg   knot     K      K      K '
    start += '\n------------------------------------------------------------'
    start += '-----------------\n '
    end = 'Station information and sounding indices'
    pattern = '{}(.*?){}'.format(start, end)
    soundings_list_revised = [
        re.search(pattern, snd, re.DOTALL).group(1)
        for snd in soundings_list]

    pattern = '(?<={})(?s)(.*$)'.format(end)
    end_str = 'Description of the data columns or sounding indices.'
    properties_list = [
        re.search(pattern, snd, re.DOTALL).group(1).split(end_str)[0]
        for snd in soundings_list]

    new_alts = np.arange(100, 20100, 100)

    pope_df = pd.read_csv(
        'fake_pope_regimes.csv', header=None,
        index_col=0, names=['time', 'pope_regime'])
    pope_df.index = pd.to_datetime(pope_df.index)

    if loc == 'gove':
        hours = [0]
    else:
        hours = [0, 12]

    da_list = []
    for i in range(len(soundings_list_revised)):
        snd = soundings_list_revised[i]
        df = snd.split('\n')[:-2]
        df = [' '.join(row.split()).split() for row in df]
        df = pd.DataFrame(df, columns=col_names, dtype=float)
        df['u'] = -np.sin(df['drct']*np.pi/180)*df['sknt']*1852/3600
        df['v'] = -np.cos(df['drct']*np.pi/180)*df['sknt']*1852/3600
        df = df.set_index('altitude')
        if loc == 'gove':
            df = df.iloc[1:]

        df['pope_regime'] = pope_df.loc[
            np.datetime64(pd.to_datetime(times[i]).date())].values[0]

        da = xr.Dataset.from_dataframe(df)
        da = da.interp(altitude=new_alts)

        prop = properties_list[i]
        prop = prop.split('\n')[4:-1]
        for k in range(len(prop)):
            try:
                [lab, val] = prop[k].split(':')
            except ValueError:
                continue
            lab = lab.strip()
            val = float(val.strip())
            if lab == 'Convective Available Potential Energy':
                da[lab] = val
        # import pdb; pdb.set_trace()
        da_list.append(da)

    da = xr.concat(da_list, dim='time')
    da = da.assign_coords({'time': times})
    da['pope_regime'] = da['pope_regime'].isel(altitude=0)
    da['p'] = da['p']*1e2
    da['t'] = da['t'] + 273.15
    da['Td'] = da['Td'] + 273.15

    da_t_list = [[] for i in range(len(hours))]
    time_array = np.arange(
        np.datetime64('{:04d}-01-01'.format(year)),
        np.datetime64('{:04d}-03-01'.format(year)),
        np.timedelta64(1, 'D'))
    for time in time_array:
        for j in range(len(hours)):
            # import pdb; pdb.set_trace()
            try:
                da_t = da.sel(time=time+np.timedelta64(hours[j], 'h'))
                da_t['time'] = time
                da_t_list[j].append(da_t)
            except KeyError:
                print('No soundings at {} {}.'.format(time, hours[j]))
                continue

    new_ds = []

    for i in range(len(hours)):
        ds_i = xr.concat(da_t_list[i], dim='time')
        new_ds.append(ds_i)

    obs_soundings = xr.concat(new_ds, dim='hour')
    obs_soundings = obs_soundings.assign_coords({'hour': hours})

    return obs_soundings


# def dl_wyoming(common_times):
#     # Download and parse dataa
#     hours = [0, 12]
#
#     soundings_ds = [[] for i in range(len(hours))]
#
#     for ct in common_times:
#         for hour in hours:
#
#             dt64 = common_times[0] + np.timedelta64(hour, 'h')
#             unix_epoch = np.datetime64(0, 's')
#             one_second = np.timedelta64(1, 's')
#             seconds_since_epoch = (dt64 - unix_epoch) / one_second
#
#             dt = datetime.datetime.utcfromtimestamp(seconds_since_epoch)
#
#             success = False
#             counter = 0
#
#             while success is False or counter < 10:
#                 try:
#                     import pdb; pdb.set_trace()
#                     df = WyomingUpperAir.request_data(dt, 'YPDN')
#                     print('Successfully downloaded {} {}'.format(ct, hour))
#                     success = True
#                 except requests.exceptions.HTTPError:
#
#                     print('Failed download {} {}'.format(ct, hour))
#                     counter += 1
#                     time.sleep(2)
#             da = df.to_xarray().dropna(dim='index')
#
#             da = da.rename({'index': 'altitude'})
#             da = da.assign_coords({'altitude': da['height'].values})
#
#             new_alts = np.arange(100, 20100, 100).astype(np.float64)
#
#             da.interp(altitude=new_alts)


def plot_skewt(
        soundings, fig=None, figsize=(12, 6), subplots=None,
        legend=False, left_ticks=True, right_ticks=False, title=None,
        ylim=(1000, 100), xlim=(-10, 30), label_CAPE=True,
        right_tick_label=True, custom_ticks=False):

    if fig is None:
        fig = plt.figure(figsize=figsize)

    cond = np.logical_not(np.isnan(soundings['p'].values))
    soundings_trunc = soundings.isel({'altitude':cond})

    p = soundings_trunc['p'].values * units.Pa
    T = (soundings_trunc['t'].values-273.15) * units.degC
    try:
        q = soundings_trunc['q']
    except KeyError:
        q = mpcalc.specific_humidity_from_dewpoint(
            p, (soundings_trunc['Td'].values - 273.15) * units.degC)

    # u = soundings['u'] * units.meter_per_second
    # v = soundings['v'] * units.meter_per_second
    heights = soundings_trunc['altitude'] * units.meter
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
    if custom_ticks:
        skew.ax.set_yticks(np.arange(ylim[0], ylim[1]-50, -50)*units['hPa'])
        skew.ax.set_xticks(np.arange(xlim[0]-20, xlim[1]+2, 2)*units['degC'])
    skew.ax.set_ylim(ylim)
    skew.ax.set_xlim(xlim)

    f = sp.interpolate.interp1d(
        p[::-1], heights[::-1], fill_value='extrapolate')

    secax = skew.ax.secondary_yaxis(location=1)

    if custom_ticks:
        secax.set_yticks(np.arange(ylim[0], ylim[1]-50, -50))
        labels = f(np.arange(ylim[0], ylim[1]-50, -50)*100)/1000
    else:
        secax.set_yticks(np.arange(1000, 0, -100))

        labels = f(np.arange(100000, 0, -10000))/1000

    labels = ['{:.02f}'.format(lbl) for lbl in labels]
    secax.set_yticklabels(labels)
    if right_tick_label:
        secax.set_ylabel('Altitude [km]')
    else:
        secax.set_ylabel('')

    if left_ticks:
        skew.ax.set_ylabel(r'Pressure [hPa]')
    else:
        skew.ax.set_yticklabels([])
        skew.ax.set_ylabel(None)

    skew.ax.set_xlabel(r'Temperature [$^\circ$C]')

    parcel_index = 0

    lcl_pressure, lcl_temperature = mpcalc.lcl(
        p[parcel_index], T[parcel_index], Td[parcel_index])
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(
        p, T[parcel_index], Td[parcel_index]).to('degC')
    skew.plot(p, prof, 'k', linewidth=2, label='Parcel Profile')
    # Plot the dew point ascent line

    skew.plot(
        np.array([
            p[parcel_index].magnitude,
            lcl_pressure.magnitude])*units('Pa'),
        np.array([
            Td[parcel_index].magnitude,
            lcl_temperature.magnitude])*units('degC'),
        '-', color='black', linewidth=2)

    # Shade areas of CAPE and CIN
    skew.shade_cin(p, T, prof, label='CIN')
    skew.shade_cape(p, T, prof, label='CAPE')

    # Calculate CAPE/CIN
    CAPE, CIN = mpcalc.surface_based_cape_cin(
        p, T, Td)
    MLCAPE, MLCIN = mpcalc.mixed_layer_cape_cin(
        p, T, Td)

    C_text = 'CAPE = {} J/kg \nMLCAPE = {} J/kg'.format(
        int(round(CAPE.magnitude)), int(round(MLCAPE.magnitude)))

    if label_CAPE:
        skew.ax.text(
            .275, .91, C_text, transform=skew.ax.transAxes,
            # backgroundcolor='white',
            fontsize=10)

    # Add the relevant special lines
    if custom_ticks:

        tmps = (np.arange(xlim[0], xlim[1]+2, 2))*units('degC')

        mixing_ratio = np.array([
            0.0004, 0.001, 0.002, 0.003, 0.004, 0.0055, 0.007, 0.0085,
            0.01, 0.014, 0.016, 0.018, 0.02, 0.022,
            0.024, 0.026, 0.028, 0.03, 0.032]).reshape(-1, 1)

        skew.plot_mixing_lines(
            mixing_ratio=mixing_ratio, linewidth=1,
            label='Mixing Lines', linestyle='dashed', colors='grey')
        skew.plot_dry_adiabats(
            label='Dry Adiabat', linewidth=1.25, linestyle='dotted',
            t0=tmps)
        skew.plot_moist_adiabats(
            linewidth=1.25, label='Moist Adiabat', linestyle='dotted',
            t0=tmps)

    else:
        skew.plot_dry_adiabats(
            label='Dry Adiabat', linewidth=1.25, linestyle='dotted')
        skew.plot_moist_adiabats(
            linewidth=1.25, label='Moist Adiabat', linestyle='dotted')
        skew.plot_mixing_lines(
            linewidth=1, label='Mixing Lines', linestyle='dashed',
            colors='grey')

    if title is not None:
        skew.ax.set_title(title, fontsize=12)

    if legend:
        skew.ax.legend(
            loc='lower center', bbox_to_anchor=(1.1, -0.3),
            ncol=4, fancybox=True, shadow=True)

    return skew.ax, CAPE.magnitude, MLCAPE.magnitude


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
    # days_2022 = np.arange(
    #     np.datetime64('2022-01-01'), np.datetime64('2022-03-01'),
    #     np.timedelta64(1, 'D'))
    # Frustrating - pressure data not archived for ACCESS-C for 2022
    days_list = [days_2021]

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
        base_dir + '20211001/0000/an/sfc/topog.nc')
    topog = topog.interp(lon=lon, lat=lat)

    new_alts = np.arange(100, 20100, 100)
    bad_days = []

    for i in range(len(days)):

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
                'wnd_ucmp', 'wnd_vcmp', 'air_temp', 'spec_hum', 'pressure']:

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

        # try:
        #     ds = xr.open_dataset(
        #         base_dir + '{}/{}/fc/pl/geop_ht.nc'.format(
        #             date_str, hour_str))
        #     times = np.arange(
        #         days[i], days[i]+np.timedelta64(30, 'h'),
        #         np.timedelta64(6, 'h'))
        #     ds = ds.sel(time=times)
        #     ds = ds.interp(lon=lon, lat=lat)
        #     ds = ds.load()
        # except FileNotFoundError:
        #     print('Missing File.')
        #     file_exists = False
        #     ds = None
        # except ValueError:
        #     print('Bad Data.')
        #     file_exists = False
        # except pd.errors.InvalidIndexError:
        #     print('Bad index.')
        #     file_exists = False
        #
        if not file_exists:
            bad_days.append(days[i])
            continue
        #
        # datasets.append(ds)

        [u, v, t, q, p] = datasets

        for j in range(len(hours)):

            datasets_t = [
                vble.sel(time=(days[i]+np.timedelta64(hours[j], 'h')))
                for vble in [u, v, t, q, p]]
            [u_t, v_t, t_t, q_t, p_t] = datasets_t

            altitude_rho = u_t.A_rho + u_t.B_rho * topog['topog']
            altitude_theta = t_t.A_theta + t_t.B_theta * topog['topog']

            # altitude = geop_t['geop_ht'].values
            # pressure = geop_t['lvl'].values
            #
            # p_t = copy.deepcopy(geop_t)
            # p_t['geop_ht'].values = pressure
            # p_t = p_t.assign_coords({'lvl': altitude})
            # p_t = p_t.rename({'lvl': 'altitude', 'geop_ht': 'p'})
            # p_t = p_t['p']

            u_t = u_t.rename({'rho_lvl': 'altitude'})
            v_t = v_t.rename({'rho_lvl': 'altitude'})
            t_t = t_t.rename({'theta_lvl': 'altitude'})
            q_t = q_t.rename({'theta_lvl': 'altitude'})
            p_t = p_t.rename({'theta_lvl': 'altitude'})

            u_t = u_t['wnd_ucmp'].assign_coords(
                {'altitude': altitude_rho.squeeze().values})
            v_t = v_t['wnd_vcmp'].assign_coords(
                {'altitude': altitude_rho.squeeze().values})
            t_t = t_t['air_temp'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})
            q_t = q_t['spec_hum'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})
            p_t = p_t['pressure'].assign_coords(
                {'altitude': altitude_theta.squeeze().values})

            u_t = u_t.interp(altitude=new_alts)
            v_t = v_t.interp(altitude=new_alts)
            t_t = t_t.interp(altitude=new_alts)
            p_t = p_t.interp(altitude=new_alts)
            q_t = q_t.interp(altitude=new_alts)

            # import pdb; pdb.set_trace()

            ds_t = xr.Dataset({
                'u': u_t, 'v': v_t, 'p': p_t, 't': t_t, 'q': q_t})

            ds_t['time'] = ds_t['time'] - np.timedelta64(hours[j], 'h')

            ds_t['pope_regime'] = pope_df.loc[ds_t['time'].values]

            soundings_ds[j].append(ds_t)

    new_ds = []

    for i in range(len(hours)):
        ds_i = xr.concat(soundings_ds[i], dim='time')
        ds_i = ds_i.drop('dim_0')
        ds_i = ds_i.squeeze()
        new_ds.append(ds_i)

    hours_da = xr.DataArray(hours, name='hour', coords={'hour': hours})

    ACCESS_C_soundings = xr.concat(new_ds, dim=hours_da)

    R = 287.04
    cp = 1005

    ACCESS_C_soundings['theta'] = ACCESS_C_soundings['t']*(
        1e5/ACCESS_C_soundings['p'])**(R/cp)

    return ACCESS_C_soundings


def plot_wind_profile(
        soundings_mean, soundings_var, title=None, fig=None, ax=None,
        legend=False, xlim=(-5, 15), ylim=(0, 20e3), ylabel=True,
        dx=4, dy=1e3):

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    cl.init_fonts()

    min_speed = 0
    max_speed = 0

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 1, 2, 4, 5, 6]]

    u_t = soundings_mean['u']
    v_t = soundings_mean['v']

    u_t_sig = np.sqrt(soundings_var['u'])
    v_t_sig = np.sqrt(soundings_var['v'])

    line1 = ax.plot(
        u_t, u_t.altitude, color=colors[0], label=r'$u$ Mean',
        linewidth=1.75)
    ax.fill_betweenx(
        u_t.altitude, u_t-u_t_sig, u_t+u_t_sig, color=colors[0],
        alpha=0.2, label='Standard Deviation')

    line2 = ax.plot(
        v_t, v_t.altitude, color=colors[0], label=r'$v$ Mean',
        linestyle='dashed')
    ax.fill_betweenx(
        v_t.altitude, v_t-v_t_sig, v_t+v_t_sig, color=colors[0],
        alpha=0.2)

    xtick_min = xlim[0]
    xtick_max = xlim[1]

    ax.set_xticks(np.arange(xtick_min, xtick_max+dx, dx))
    ax.set_xticks(np.arange(xtick_min, xtick_max+dx/2, dx/2), minor=True)
    ax.set_xlabel(r'Velocity [m/s]')
    if ylabel:
        ax.set_ylabel(r'Altitude [km]')
    ax.set_yticks(np.arange(0, 20e3+dy, dy))
    ax.set_yticks(np.arange(0, 20e3+dy/2, dy/2), minor=True)
    ax.set_yticklabels(np.arange(0, 20+dy/1e3, dy/1e3).astype(int))
    ax.set_xlim([xtick_min, xtick_max])
    ax.set_ylim(ylim)

    min_speed = min(np.concatenate([u_t.values, v_t.values]))
    max_speed = max(np.concatenate([u_t.values, v_t.values]))

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

    lines = line1 + line2 + line4
    labels = [line.get_label() for line in lines]

    ax.grid(which='major', alpha=0.5, axis='both')
    ax.grid(which='minor', alpha=0.2, axis='both')

    if title is not None:
        ax.set_title(title, fontsize=12)

    if legend:
        ax.legend(
            lines, labels,
            loc='lower center', bbox_to_anchor=(.5, -0.30),
            ncol=4, fancybox=True, shadow=True)
