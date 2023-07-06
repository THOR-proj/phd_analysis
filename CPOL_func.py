import os
import numpy as np
import pandas as pd
import pickle
import pyart
import sys
sys.path.insert(0, '/home/563/esh563/TINT')
sys.path.insert(0, '/home/563/esh563/CPOL_analysis')
import tint
from tint import process_ACCESS as ACC
from tint import process_operational_radar as po
from tint.visualisation import figures
import matplotlib.pyplot as plt
import datetime
import tempfile
import shutil
import classification as cl
from tint.grid_utils import extract_datetimes
from zipfile import BadZipFile


def CPOL_files_from_datetime_list(datetimes, base_dir=None):
    print('Gathering files.')
    if base_dir is None:
        base_dir = '/g/data/hj10/cpol/cpol_level_1b/v2020'
        base_dir += '/gridded/grid_150km_2500m/'
    filenames = []
    for i in range(len(datetimes)):
        year = str(datetimes[i])[0:4]
        month = str(datetimes[i])[5:7]
        day = str(datetimes[i])[8:10]
        hour = str(datetimes[i])[11:13]
        minute = str(datetimes[i])[14:16]
        filename = (base_dir + '{0}/{0}{1}{2}/'.format(year, month, day)
                    + 'twp10cpolgrid150.b2.{0}{1}{2}.'.format(year, month, day)
                    + '{}{}00.nc'.format(hour, minute))
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
    X, Y = np.meshgrid(grid.x['data'], grid.y['data'], indexing='ij')
    radii = np.sqrt(X**2+Y**2)
    b_ind = np.argwhere(
        np.logical_and(radii >= radius, radii < radius + dx))
    b_ind_set = set([tuple(b_ind[i]) for i in range(b_ind.shape[0])])

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


def get_CPOL_season(
        year, base_dir=None, ERA5_dir=None, b_path=None, save_dir=None):
    if base_dir is None:
        base_dir = '/g/data/hj10/cpol/cpol_level_1b/v2020/'
        base_dir += 'gridded/grid_150km_2500m/'
    if ERA5_dir is None:
        ERA5_dir = '/g/data/w40/esh563/era5/pressure-levels/reanalysis/'
    if b_path is None:
        b_path = '/home/563/esh563/CPOL_analysis/circ_b_ind_set.pkl'
    if save_dir is None:
        save_dir = '/g/data/w40/esh563/TINT_tracks/'

    dates = np.arange(
        np.datetime64('{}-10-01 00:00'.format(str(year))),
        np.datetime64('{}-05-01 00:00'.format(str(year+1))),
        np.timedelta64(10, 'm'))

    filenames, start_time, end_time = CPOL_files_from_datetime_list(
        dates, base_dir=base_dir)

    # params = {
    #     'GS_ALT': 2500,  # m
    #     # Layers to identify objects within.
    #     'LEVELS': np.array(
    #         [[500, 4500], [4500, 7500], [7500, 10000]]),  # m
    #     # Interval in the above array used for tracking.
    #     'AMBIENT': 'ERA5', 'AMBIENT_BASE_DIR': ERA5_dir}

    params = {
        'GS_ALT': 1000,
        'LEVELS': np.array(
            [[1000, 1500], [500, 20000]]),
        'WIND_LEVELS': np.array(
            [[500, 3500], [500, 20000]]),
        'FIELD_THRESH': ['convective', 15],
        'MIN_SIZE': [80, 800],
        'ISO_THRESH': [10, 10],
        'AMBIENT': 'ERA5',
        'AMBIENT_BASE_DIR': ERA5_dir,
        'AMBIENT_TIMESTEP': 6,
        'SAVE_DIR': save_dir}

    tracks_obj = tint.Tracks(params=params)

    grids = (
        pyart.io.read_grid(fn, include_fields=['reflectivity'])
        for fn in filenames)

    tracks_obj.get_tracks(grids, b_path=b_path)

    out_file_name = save_dir + '{}1001_{}0501.pkl'.format(year, year+1)
    with open(out_file_name, 'wb') as f:
        pickle.dump(tracks_obj, f)

    return tracks_obj


def get_ACCESS_season(
        radar, year=2020, base_dir=None, b_path=None, save_dir=None):
    if base_dir is None:
        base_dir = '/g/data/hj10/cpol/cpol_level_1b/v2020/'
        base_dir += 'gridded/grid_150km_2500m/'
    if b_path is None:
        b_path = '/home/563/esh563/CPOL_analysis/circ_b_ind_set.pkl'
    if save_dir is None:
        save_dir = '/g/data/w40/esh563/TINT_tracks/'

    common_times = np.loadtxt(
        '/g/data/w40/esh563/ACCESS_radar_common_times.csv',
        dtype=str).astype(np.datetime64)

    start = np.datetime64('{}-10-01T00:00:00'.format(year))
    end = np.datetime64('{}-05-01T00:00:00'.format(year+1))
    datetimes = np.arange(start, end, np.timedelta64(10, 'm'))
    datetimes = sorted([d for d in datetimes if d in common_times])

    tracks_obj = tint.Tracks(params={
        'AMBIENT': 'ERA5',
        'AMBIENT_BASE_DIR': '/g/data/w40/esh563/era5/pressure-levels/reanalysis/',
        'DT': 10,
        'AMBIENT_TIMESTEP': 6,
        'GS_ALT': 0,
        'LEVELS': np.array(
            [[0, 0.5], [1, 1.5]]),
        'WIND_LEVELS': np.array(
            [[500, 2500], [500, 20000]]),
        'FIELD_THRESH': [35, 10],
        'MIN_SIZE': [80, 800],
        'ISO_THRESH': [10, 10],
        'INPUT_TYPE': 'ACCESS_DATETIMES',
        'SAVE_DIR': save_dir,
        'REFERENCE_GRID_FORMAT': 'ODIM',
        'RESET_NEW_DAY': True,
        'REFERENCE_RADAR': radar,
        'REMOTE': True})

    grids = (
        date for date in datetimes)

    tracks_obj.get_tracks(grids, b_path=b_path)

    out_file_name = save_dir + '{}_{}1001_{}0501.pkl'.format(
        radar, year, year+1)
    with open(out_file_name, 'wb') as f:
        pickle.dump(tracks_obj, f)

    return tracks_obj


def get_oper_month(
        radar, year=2020, month=10, base_dir=None, ERA5_dir=None,
        b_path=None, save_dir=None):
    if base_dir is None:
        base_dir = '/g/data/hj10/cpol/cpol_level_1b/v2020/'
        base_dir += 'gridded/grid_150km_2500m/'
    if ERA5_dir is None:
        ERA5_dir = '/g/data/w40/esh563/era5/pressure-levels/reanalysis/'
    if b_path is None:
        b_path = '/home/563/esh563/CPOL_analysis/circ_b_ind_set.pkl'
    if save_dir is None:
        save_dir = '/g/data/w40/esh563/TINT_tracks/'

    # common_datetimes = np.loadtxt(
    #     '/home/563/esh563/CPOL_analysis/radar_common_times.csv',
    #     dtype=str).astype(np.datetime64)
    #
    # datetimes = sorted([
    #     d for d in common_datetimes
    #     if (int(str(d)[0:4]) == year and int(str(d)[5:7]) == month)])

    coverage = pd.read_csv(
        '/home/563/esh563/CPOL_analysis/coverage.csv', index_col=0)
    coverage.index = pd.to_datetime(coverage.index)
    coverage.columns = coverage.columns.astype(int)
    common_datetimes = coverage.loc[:, radar].where(
        coverage.loc[:, radar] == 1).dropna().index.values

    start_datetime = np.datetime64(
        '{:04}-{:02}-01T00:00:00'.format(year, month))
    if month == 12:
        end_datetime = np.datetime64(
            '{:04}-01-01T00:00:00'.format(year+1))
    else:
        end_datetime = np.datetime64(
            '{:04}-{:02}-01T00:00:00'.format(year, month+1))

    days = np.arange(
        start_datetime, end_datetime, np.timedelta64(1, 'D'))

    days = sorted([
        d for d in days
        if np.datetime64(str(d)[:10]) in common_datetimes])

    tracks_obj = tint.Tracks(params={
        'GS_ALT': 1500,  # m
        # Layers to identify objects within.
        'LEVELS': np.array(
            [[500, 3500], [3500, 7500], [7500, 10000]]),  # m
        # Interval in the above array used for tracking.
        'AMBIENT': 'ERA5', 'AMBIENT_BASE_DIR': ERA5_dir,
        'SAVE_DIR': save_dir,
        'REFERENCE_GRID_FORMAT': 'ODIM',
        'INPUT_TYPE': 'OPER_DATETIMES',
        'REFERENCE_RADAR': radar,
        'REMOTE': True,
        'USE_DAY_GRIDS': True,
        'DT': 10})

    # tracks_obj = tint.Tracks(params={
    #     'GS_ALT': 1000,
    #     'LEVELS': np.array(
    #         [[1000, 1500], [500, 20000]]),
    #     'WIND_LEVELS': np.array(
    #         [[500, 2500], [500, 20000]]),
    #     'FIELD_THRESH': [35, 10],
    #     'MIN_SIZE': [80, 800],
    #     'ISO_THRESH': [10, 10],
    #     'AMBIENT': 'ACCESS',
    #     'AMBIENT_BASE_DIR': ERA5_dir,
    #     'AMBIENT_TIMESTEP': 6,
    #     'DT': 10,
    #     'SAVE_DIR': save_dir,
    #     'RESET_NEW_DAY': True,
    #     'REFERENCE_GRID_FORMAT': 'ODIM',
    #     'INPUT_TYPE': 'OPER_DATETIMES',
    #     'REFERENCE_RADAR': radar,
    #     'REMOTE': True})

    if len(days) > 0:

        day_grids = (
            day for day in days)
        good_day = False

        while good_day is False:

            try:
                day_grid = next(day_grids)
            except StopIteration:
                print('All bad data this month: Skipping.')
                return

            try:
                new_grid, file_list = po.get_grid(
                    day_grid, tracks_obj.params, tracks_obj.reference_grid,
                    tracks_obj.tmp_dir, None)
            except BadZipFile:
                print('Bad Zip. Skipping')
                continue

            dt_list = extract_datetimes(file_list)
            grids = (
                dt for dt in dt_list
                if (dt >= start_datetime+np.timedelta64(0, 'h') and dt <= end_datetime))

            if len(dt_list) >= 2:
                tracks_obj.params['DT'] = int(np.argmax(np.bincount((np.diff(
                    np.array(dt_list))).astype(int)))/60)
            if tracks_obj.params['DT'] > 0 and len(file_list) >= 6:
                good_day = True
            else:
                print('Bad day of data, trying next day.')

        if tracks_obj.params['DT'] < 7:
            scale_factor = np.ceil(
                10/tracks_obj.params['DT']).astype(int)
            tracks_obj.params['DT'] = tracks_obj.params['DT']*scale_factor
            new_file_inds = [
                i for i in np.arange(0, len(dt_list))
                if dt_list[i].astype('<M8[m]') in
                np.arange(
                    dt_list[0].astype('<M8[m]'),
                    dt_list[i].astype('<M8[m]')+np.timedelta64(1, 'D'),
                    np.timedelta64(tracks_obj.params['DT'], 'm'))]
            new_file_list = [
                file_list[new_file_inds[i]]
                for i in np.arange(len(new_file_inds))]
            new_dt_list = [
                dt_list[new_file_inds[i]]
                for i in np.arange(len(new_file_inds))]
            file_list = new_file_list
            grids = (
                dt for dt in new_dt_list
                if (dt >= start_datetime+np.timedelta64(0, 'h') and dt <= end_datetime))
            new_datetime = next(grids)

        tracks_obj.get_tracks(grids, day_grids=day_grids, b_path=b_path)

        out_file_name = save_dir + '/{:02d}_{:04d}_{:02d}.pkl'.format(
            radar, year, month)
        with open(out_file_name, 'wb') as f:
            pickle.dump(tracks_obj, f)
    else:
        print('Empty month: Skipping.')

    return tracks_obj


def gen_ACCESS_verification_figures(
        save_dir, fig_dir, radar=63, year=2020, exclusions=None, suffix='',
        start_date=None, end_date=None):

    path = save_dir + 'ACCESS_{}/{}1001_{}0501.pkl'.format(
        radar, year, year+1)
    with open(path, 'rb') as f:
        tracks_obj = pickle.load(f)

    fig_dir = fig_dir + '/ACCESS_{}_{}_verification_scans{}/'.format(
        radar, year, suffix)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print('Creating new directory.')

    if exclusions is None:
        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'duration_cond',
            'small_velocity', 'small_offset']

    excluded = tracks_obj.exclusions[exclusions]
    excluded = excluded.xs(0, level='level')
    excluded = np.any(excluded, 1)
    # excluded = excluded.where(excluded==False).dropna()
    # len(excluded)/3

    included = np.logical_not(excluded)
    included = included.where(included == True).dropna()
    scans = included

    if start_date is not None and end_date is not None:
        scans = scans.loc[:, slice(start_date, end_date), :, :]

    scans = sorted(np.unique(scans.index.get_level_values(1).values))

    for s in scans:

        ACCESS_refl, grid = tint.process_ACCESS.init_ACCESS_C(
            s, tracks_obj.reference_grid, gadi=True)

        current_time = str(datetime.datetime.now())[0:-7]
        current_time = current_time.replace(" ", "_").replace(":", "_")
        current_time = current_time.replace("-", "")

        params = {
            'uid_ind': None, 'line_coords': False, 'center_cell': False,
            'cell_ind': 10, 'winds': False, 'winds_fn': None,
            'crosshair': False, 'fontsize': 18, 'colorbar_flag': True,
            'leg_loc': 2, 'label_type': 'velocities',
            'system_winds': ['shift', 'relative', 'ambient_mean'],
            'boundary': True, 'exclude': True, 'exclusions': exclusions}

        figures.two_level(
            tracks_obj, grid, params=params, alt1=0, alt2=1)
        save_path = fig_dir + '{}.png'.format(s)
        plt.savefig(
            save_path, dpi=200, facecolor='w', edgecolor='white',
            bbox_inches='tight')
    #     plt.show()
        plt.close('all')


def gen_operational_verification_figures(
        save_dir, fig_dir, radar=63, year=2020, exclusions=None, suffix='',
        start_date=None, end_date=None, month=12):

    print('Creating figures for radar {} year {} month {}'.format(
        radar, year, month))

    fig_dir += '/{}_{:04d}_{:02d}{}/'.format(
        radar, year, month, suffix)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print('Creating new directory.')

    if exclusions is None:
        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'duration_cond',
            'small_velocity', 'small_offset']

    path = save_dir + '/{:02}_{:04}_{:02}.pkl'.format(radar, year, month)
    try:
        with open(path, 'rb') as f:
            tracks_obj = pickle.load(f)
    except FileNotFoundError:
        print('File {} does not exist. Confirm tracks were created.'.format(path))
        return

    # tracks_obj = cl.redo_exclusions(tracks_obj)
    if len(tracks_obj.tracks)==0:
        print('Empty tracks file. Quitting.')
        return
    try:
        excluded = tracks_obj.exclusions[exclusions]
        excluded = excluded.xs(0, level='level')
        excluded = np.any(excluded, 1)
        # excluded = excluded.where(excluded==False).dropna()
        # len(excluded)/3

        included = np.logical_not(excluded)
        included = included.where(included == True).dropna()
        scans = included
    except AttributeError:
        print('Small tracks file. Quitting.')
        return

    if start_date is not None and end_date is not None:
        scans = scans.loc[:, slice(start_date, end_date), :, :]
    scans = sorted(np.unique(scans.index.get_level_values(1).values))

    file_list = None
    tmp_dir = tempfile.mkdtemp(dir=save_dir)
    tracks_obj.params['REMOTE'] = True

    for s in scans:

        grid, file_list = tint.process_operational_radar.get_grid(
            s, tracks_obj.params, tracks_obj.reference_grid,
            tmp_dir, file_list)

        current_time = str(datetime.datetime.now())[0:-7]
        current_time = current_time.replace(" ", "_").replace(":", "_")
        current_time = current_time.replace("-", "")

        params = {
            'uid_ind': None, 'line_coords': False, 'center_cell': False,
            'cell_ind': 10, 'winds': False, 'winds_fn': None,
            'crosshair': False, 'fontsize': 18, 'colorbar_flag': True,
            'leg_loc': 2, 'label_type': 'shear',
            'system_winds': ['shear', 'relative'],
            'boundary': True, 'exclude': True, 'exclusions': exclusions}

        figures.two_level(
            tracks_obj, grid, params=params, alt1=2000, alt2=8000)
        save_path = fig_dir + '{}.png'.format(s)
        plt.savefig(
            save_path, dpi=200, facecolor='w', edgecolor='white',
            bbox_inches='tight')
        plt.close('all')

    shutil.rmtree(tmp_dir)


def gen_CPOL_verification_figures(
        save_dir, fig_dir, year=2020, exclusions=None, suffix='',
        start_date=None, end_date=None, month=12):

    print('Creating figures for radar {} year {} month {}'.format(
        year, month))

    fig_dir += '/{}_{:04d}_{:02d}{}/'.format(
        year, month, suffix)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print('Creating new directory.')

    if exclusions is None:
        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'duration_cond',
            'small_velocity', 'small_offset']

    path = save_dir + '/{:02}_{:04}_{:02}.pkl'.format(year, month)
    try:
        with open(path, 'rb') as f:
            tracks_obj = pickle.load(f)
    except FileNotFoundError:
        print('File {} does not exist. Confirm tracks were created.'.format(path))
        return

    # tracks_obj = cl.redo_exclusions(tracks_obj)
    if len(tracks_obj.tracks)==0:
        print('Empty tracks file. Quitting.')
        return
    try:
        excluded = tracks_obj.exclusions[exclusions]
        excluded = excluded.xs(0, level='level')
        excluded = np.any(excluded, 1)
        # excluded = excluded.where(excluded==False).dropna()
        # len(excluded)/3

        included = np.logical_not(excluded)
        included = included.where(included == True).dropna()
        scans = included
    except AttributeError:
        print('Small tracks file. Quitting.')
        return

    if start_date is not None and end_date is not None:
        scans = scans.loc[:, slice(start_date, end_date), :, :]
    scans = sorted(np.unique(scans.index.get_level_values(1).values))

    file_list = None
    tmp_dir = tempfile.mkdtemp(dir=save_dir)
    tracks_obj.params['REMOTE'] = True

    for s in scans:

        grid, file_list = tint.process_operational_radar.get_grid(
            s, tracks_obj.params, tracks_obj.reference_grid,
            tmp_dir, file_list)

        current_time = str(datetime.datetime.now())[0:-7]
        current_time = current_time.replace(" ", "_").replace(":", "_")
        current_time = current_time.replace("-", "")

        params = {
            'uid_ind': None, 'line_coords': False, 'center_cell': False,
            'cell_ind': 10, 'winds': False, 'winds_fn': None,
            'crosshair': False, 'fontsize': 18, 'colorbar_flag': True,
            'leg_loc': 2, 'label_type': 'shear',
            'system_winds': ['shear', 'relative'],
            'boundary': True, 'exclude': True, 'exclusions': exclusions}

        figures.two_level(
            tracks_obj, grid, params=params, alt1=2000, alt2=8000)
        save_path = fig_dir + '{}.png'.format(s)
        plt.savefig(
            save_path, dpi=200, facecolor='w', edgecolor='white',
            bbox_inches='tight')
        plt.close('all')

    shutil.rmtree(tmp_dir)


def combine_tracks(years=list(range(1998, 2016)), base_dir=None):
    if base_dir is None:
        base_dir = '/home/student.unimelb.edu.au/shorte1/'
        base_dir += 'Documents/TINT_tracks/'
    years = set(years) - set([2007, 2008, 2000])
    years = sorted(list(years))
    max_uid = 0
    tracks = []
    system_tracks = []
    tracks_class = []
    exclusions = []

    fields = [
        'grid_x', 'grid_y', 'proj_area',
        'lon', 'lat', 'touch_border', 'semi_major', 'semi_minor',
        'orientation', 'eccentricity', 'u_shift', 'v_shift', 'u_ambient_mean',
        'v_ambient_mean', 'u_relative', 'v_relative', 'u_shear', 'v_shear']
    s_fields = [
        'grid_x', 'grid_y', 'lon',
        'lat', 'u_shift', 'v_shift', 'u_relative', 'v_relative',
        'semi_major', 'semi_minor', 'eccentricity',
        'orientation', 'cells', 'field_max', 'proj_area', 'max_height',
        'touch_border', 'x_vert_disp', 'y_vert_disp', 'tilt_mag', 'vel_dir',
        'tilt_dir', 'sys_rel_tilt_dir']

    for year in years:
        print('Shifting uid for {}'.format(year))
        with open(base_dir + '{}1001_{}0501.pkl'.format(
                str(year), str(year+1)), 'rb') as f:
            tracks_obj = pickle.load(f)
        tracks_obj.tracks = tracks_obj.tracks[fields]
        tracks_obj.system_tracks[s_fields] = tracks_obj.system_tracks[s_fields]
        tracks_obj.tracks.reset_index(
            level=['scan', 'time', 'level', 'uid'], inplace=True)
        tracks_obj.system_tracks.reset_index(
            level=['scan', 'time', 'uid'], inplace=True)
        tracks_obj.tracks_class.reset_index(
            level=['scan', 'time', 'level', 'uid'], inplace=True)
        tracks_obj.exclusions.reset_index(
            level=['scan', 'time', 'level', 'uid'], inplace=True)
        uids = tracks_obj.tracks['uid'].values.astype(int)
        sys_uids = tracks_obj.system_tracks['uid'].values.astype(int)
        class_uids = tracks_obj.tracks_class['uid'].values.astype(int)
        excl_uids = tracks_obj.exclusions['uid'].values.astype(int)
        uids += max_uid
        sys_uids += max_uid
        class_uids += max_uid
        excl_uids += max_uid

        max_uid = uids.max()+1

        tracks_obj.tracks['uid'] = uids.astype(str)
        tracks_obj.system_tracks['uid'] = sys_uids.astype(str)
        tracks_obj.tracks_class['uid'] = class_uids.astype(str)
        tracks_obj.exclusions['uid'] = excl_uids.astype(str)
        tracks_obj.tracks.set_index(
            ['scan', 'time', 'level', 'uid'], inplace=True)
        tracks_obj.system_tracks.set_index(
            ['scan', 'time', 'uid'], inplace=True)
        tracks_obj.tracks_class.set_index(
            ['scan', 'time', 'level', 'uid'], inplace=True)
        tracks_obj.exclusions.set_index(
            ['scan', 'time', 'level', 'uid'], inplace=True)
        tracks.append(tracks_obj.tracks)
        system_tracks.append(tracks_obj.system_tracks)
        tracks_class.append(tracks_obj.tracks_class)
        exclusions.append(tracks_obj.exclusions)

    print('Concatenating shifted tracks.')
    tracks_obj.tracks = pd.concat(tracks)
    tracks_obj.system_tracks = pd.concat(system_tracks)
    tracks_obj.tracks_class = pd.concat(tracks_class)
    tracks_obj.exclusions = pd.concat(exclusions)

    fn = base_dir + 'comb_{}_{}.pkl'.format(str(years[0]), str(years[-1]))
    with open(fn, 'wb') as f:
        pickle.dump(tracks_obj, f)
