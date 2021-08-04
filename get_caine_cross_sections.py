import datetime
import numpy as np
import pyart
import tint
from tint.visualisation.animate import animate

import caine_func as caf

base_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/caine_WRF_data/'
save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_tracks/'


def gen_tracks(filenames, rain=False, micro_physics='thompson'):

    grids = (
        pyart.io.read_grid(fn, include_fields=['reflectivity'])
        for fn in filenames)

    current_time = str(datetime.datetime.now())[0:-7]
    current_time = current_time.replace(" ", "_").replace(":", "_")
    current_time = current_time.replace("-", "")

    tracks_obj = tint.Tracks()
    tracks_obj.get_tracks(grids, b_path=None)

    return tracks_obj


micro_physics = 'thompson'
fig_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_figures/'
fig_dir += 'thompson_objects/'
params = {
    'line_coords': True, 'center_cell': False,
    'cell_ind': 7, 'winds': True,
    'direction': 'perpendicular', 'crosshair': True, 'save_dir': fig_dir,
    'line_average': True, 'streamplot': True, 'relative_winds': True,
    'data_fn': 'angles', 'load_line_coords_winds': False,
    'save_ds': False}

dates = np.arange(
    np.datetime64('2006-02-08 12:00'),
    np.datetime64('2006-02-13 12:30'),
    np.timedelta64(10, 'm'))

filenames, start_time, end_time = caf.caine_files_from_datetime_list(
    dates, micro_physics=micro_physics, base_dir=base_dir)
params['winds_fn'] = filenames

tracks_obj = gen_tracks(filenames, micro_physics=micro_physics)

for uid_ind in np.arange(14).astype(str):

    params['uid_ind'] = uid_ind
    grids = (
        pyart.io.read_grid(fn, include_fields=['reflectivity'])
        for fn in filenames)

    animate(tracks_obj, grids, params)

micro_physics = 'lin'
fig_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_figures/'
fig_dir += 'lin_objects/'

dates = np.arange(
    np.datetime64('2006-02-09 00:00'),
    np.datetime64('2006-02-13 12:10'),
    np.timedelta64(10, 'm'))

filenames, start_time, end_time = caf.caine_files_from_datetime_list(
    dates, micro_physics=micro_physics, base_dir=base_dir)
params['winds_fn'] = filenames

tracks_obj = gen_tracks(filenames, micro_physics=micro_physics)

for uid_ind in np.arange(13).astype(str):

    params['uid_ind'] = uid_ind
    grids = (
        pyart.io.read_grid(fn, include_fields=['reflectivity'])
        for fn in filenames)

    animate(tracks_obj, grids, params)
