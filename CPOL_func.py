import os
import numpy as np
import pandas as pd
import pickle
import pyart
import sys
sys.path.insert(0, '/home/563/esh563/TINT')

import tint


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
        ERA5_dir = '/g/data/rt52/era5/pressure-levels/reanalysis/'
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

    tracks_obj = tint.Tracks(params={
        'AMBIENT': 'ERA5', 'AMBIENT_BASE_DIR': ERA5_dir})

    grids = (
        pyart.io.read_grid(fn, include_fields=['reflectivity'])
        for fn in filenames)

    tracks_obj.get_tracks(grids, b_path=b_path)

    out_file_name = save_dir + '{}1101_{}0401.pkl'.format(year, year+1)
    with open(out_file_name, 'wb') as f:
        pickle.dump(tracks_obj, f)

    return tracks_obj


def combine_tracks(years=list(range(2002, 2015))):
    years = set(years) - set([2007, 2008])
    years = sorted(list(years))
    bp = '/g/data/w40/esh563/CPOL_analysis/TINT_tracks/tracks_obj_'
    max_uid = 0
    tracks = []
    system_tracks = []
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

    fn = bp + 'comb_{}_{}.pkl'.format(str(years[0]), str(years[-1]))
    with open(fn, 'wb') as f:
        pickle.dump(tracks_obj, f)

    return tracks_obj
