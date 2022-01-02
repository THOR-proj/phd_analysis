from matplotlib import rcParams
import pickle
import numpy as np
import matplotlib.pyplot as plt
import classification as cl
from scipy import stats

base_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/CPOL/'
save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_tracks/'
fig_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_figures/'
ERA5_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/era5/'
ERA5_dir += 'pressure-levels/reanalysis/'
WRF_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/caine_WRF_data/'


def init_fonts():
    rcParams.update({'font.family': 'serif'})
    rcParams.update({'font.serif': 'Liberation Serif'})
    rcParams.update({'mathtext.fontset': 'dejavuserif'})
    rcParams.update({'font.size': 12})


def shear_versus_orientation(class_thresh=None, excl_thresh=None):

    years = sorted(list(set(range(1998, 2016)) - {2007, 2008, 2000}))
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    tracks_dir = base_dir + 'TINT_tracks/base/'
    fig_dir = base_dir + 'TINT_figures/'

    shear_angle_list = []
    orientation_list = []
    prop_angle_list = []

    for year in years:

        print('Getting data for year {}.'.format(year))
        fn = '{}1001_{}0501.pkl'.format(year, year+1)

        with open(tracks_dir + fn, 'rb') as f:
            tracks_obj = pickle.load(f)

        if class_thresh is not None and excl_thresh is not None:
            print('Recalculating Exclusions')
            tracks_obj = cl.redo_exclusions(
                tracks_obj, class_thresh, excl_thresh)

        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'duration_cond',
            'small_velocity', 'small_offset']
        excluded = tracks_obj.exclusions[exclusions]
        amb = 'Ambiguous (On Quadrant Boundary)'
        quad_bound = tracks_obj.tracks_class['offset_type'] == amb
        excluded = np.logical_or(np.any(excluded, 1), quad_bound)
        included = np.logical_not(excluded)

        sub_classes = tracks_obj.tracks_class.where(included == True).dropna()

        cols = ['inflow_type', 'offset_type', 'tilt_type', 'propagation_type']
        req_type = (
            'Front Fed', 'Trailing Stratiform',
            'Up-Shear Tilted', 'Down-Shear Propagating')
        FFTS_UST_DSP_cond = np.all(sub_classes[cols] == req_type, axis=1)

        inds = sub_classes.where(FFTS_UST_DSP_cond == True).dropna()
        inds = inds.index.values

        sub_tracks = tracks_obj.tracks.loc[inds]
        sub_tracks = sub_tracks.xs(0, level='level')

        # import pdb; pdb.set_trace()

        u_shear = sub_tracks['u_shear']
        v_shear = sub_tracks['v_shear']
        u_relative = sub_tracks['u_relative']
        v_relative = sub_tracks['v_relative']

        shear_angle = np.arctan2(v_shear, u_shear)
        shear_angle = np.rad2deg(shear_angle)

        prop_angle = np.arctan2(v_relative, u_relative)
        prop_angle = np.rad2deg(prop_angle)

        orientation = sub_tracks['orientation']

        shear_angle_list += shear_angle.tolist()
        orientation_list += orientation.tolist()
        prop_angle_list += prop_angle.tolist()

    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # plt.sca(ax)
    # ax.scatter(
    #     np.mod(shear_angle_list + 90, 180), np.mod(orientation_list, 180),
    #     marker='.', s=2)
    # plt.xticks(np.arange(0, 180+30, 30))
    # plt.yticks(np.arange(0, 180+30, 30))

    return shear_angle_list, orientation_list, prop_angle_list


def plot_shear_angle_versus_orientation(shear_angle_list, orientation_list):

    init_fonts()

    shear = np.mod(np.array(shear_angle_list), 360)
    line_normal = np.mod(np.array(orientation_list)+90, 360)

    np.polyfit(shear, line_normal, deg=1)

    # shear = np.array(shear_angle_list)
    # line_normal = np.array(orientation_list)

    linreg = stats.linregress(shear, line_normal)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # plt.sca(ax)
    ax.scatter(shear, line_normal, marker='.', s=.5)

    dx = 0.5
    x = np.arange(0, 360+dx, dx)

    ax.plot(
        x, linreg.intercept + linreg.slope*x, 'r',
        label='Least Squares')
    stats_lab = 'r = {:.2},   p = {:.2e}'.format(
        linreg.rvalue, linreg.pvalue)
    ax.text(0.05, 1.025, stats_lab, transform=ax.transAxes, size=12)

    plt.xticks(np.arange(0, 360+45, 45))
    plt.yticks(np.arange(0, 360+45, 45))
    plt.xlabel('Shear Direction [Degrees]')
    plt.ylabel('Line Normal Direction [Degrees]')
