from matplotlib import rcParams
import pickle
import numpy as np
import matplotlib.pyplot as plt
import classification as cl
from scipy import stats
import copy
import pandas as pd

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

    all_dic = {
        'shear_angle_list': [], 'orientation_list': [],
        'prop_angle_list': []}

    FFTS_UST_dic = copy.deepcopy(all_dic)
    TS_dic = copy.deepcopy(all_dic)
    LS_dic = copy.deepcopy(all_dic)

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

        # import pdb; pdb.set_trace()

        sub_classes = tracks_obj.tracks_class.where(included == True).dropna()
        inds_all = sub_classes.index.values
        sub_tracks_all = tracks_obj.tracks.loc[inds_all]
        sub_tracks_all = sub_tracks_all.xs(0, level='level')

        sub_tracks_FFTS_UST = get_FFTS_UST(sub_classes, tracks_obj)
        sub_tracks_TS = get_rel_TS(sub_classes, tracks_obj)
        sub_tracks_LS = get_rel_LS(sub_classes, tracks_obj)

        # import pdb; pdb.set_trace()

        sub_tracks = [
            sub_tracks_all, sub_tracks_FFTS_UST,
            sub_tracks_TS, sub_tracks_LS]
        dicts = [
            all_dic, FFTS_UST_dic, TS_dic, LS_dic]

        for i in range(len(sub_tracks)):
            dicts[i] = get_dic_properties(sub_tracks[i], dicts[i])

    return all_dic, FFTS_UST_dic, TS_dic, LS_dic


def shear_versus_orientation_ACCESS(
        class_thresh=None, excl_thresh=None):

    years = [2020, 2021]
    radars = [42, 63, 77]
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    tracks_dir = base_dir + 'TINT_tracks/ACCESS_radar_base/'
    fig_dir = base_dir + 'TINT_figures/'

    all_dic = {
        'shear_angle_list': [], 'orientation_list': [],
        'prop_angle_list': []}

    FFTS_UST_dic = copy.deepcopy(all_dic)
    TS_dic = copy.deepcopy(all_dic)
    LS_dic = copy.deepcopy(all_dic)

    for year in years:
        for radar in radars:

            print('Getting data for radar {}, year {}.'.format(radar, year))
            fn = 'ACCESS_{:02}/{:04}1001_{:04}0501.pkl'.format(
                radar, year, year+1)

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

            # import pdb; pdb.set_trace()

            sub_classes = tracks_obj.tracks_class.where(
                included == True).dropna()
            inds_all = sub_classes.index.values
            sub_tracks_all = tracks_obj.tracks.loc[inds_all]
            sub_tracks_all = sub_tracks_all.xs(0, level='level')

            sub_tracks_FFTS_UST = get_FFTS_UST(sub_classes, tracks_obj)
            sub_tracks_TS = get_rel_TS(sub_classes, tracks_obj)
            sub_tracks_LS = get_rel_LS(sub_classes, tracks_obj)

            # import pdb; pdb.set_trace()

            sub_tracks = [
                sub_tracks_all, sub_tracks_FFTS_UST,
                sub_tracks_TS, sub_tracks_LS]
            dicts = [
                all_dic, FFTS_UST_dic, TS_dic, LS_dic]

            for i in range(len(sub_tracks)):
                dicts[i] = get_dic_properties(sub_tracks[i], dicts[i])

    return all_dic, FFTS_UST_dic, TS_dic, LS_dic


def shear_versus_orientation_radar(
        class_thresh=None, excl_thresh=None):

    years = [2020, 2021]
    radars = [42, 63, 77]
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    tracks_dir = base_dir + 'TINT_tracks/ACCESS_radar_base/'
    fig_dir = base_dir + 'TINT_figures/'

    all_dic = {
        'shear_angle_list': [], 'orientation_list': [],
        'prop_angle_list': []}

    FFTS_UST_dic = copy.deepcopy(all_dic)
    TS_dic = copy.deepcopy(all_dic)
    LS_dic = copy.deepcopy(all_dic)

    for year in years:

        year_months = [
            [year, 10], [year, 11], [year, 12],
            [year+1, 1], [year+1, 2], [year+1, 3],
            [year+1, 4]]

        for radar in radars:
            for year_month in year_months:

                print(
                    'Getting data for radar {}, year {}, month {}.'.format(
                        radar, year_month[0], year_month[1]))
                fn = 'radar_{:02}/{:02}_{:04}_{:02}.pkl'.format(
                    radar, radar, year_month[0], year_month[1])

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

                sub_classes = tracks_obj.tracks_class.where(
                    included == True).dropna()
                inds_all = sub_classes.index.values
                sub_tracks_all = tracks_obj.tracks.loc[inds_all]

                if 0 in sub_tracks_all.index.get_level_values(level='level'):
                    sub_tracks_all = sub_tracks_all.xs(0, level='level')
                else:
                    print('No systems.')
                    continue

                sub_tracks_FFTS_UST = get_FFTS_UST(sub_classes, tracks_obj)
                sub_tracks_TS = get_rel_TS(sub_classes, tracks_obj)
                sub_tracks_LS = get_rel_LS(sub_classes, tracks_obj)

                sub_tracks = [
                    sub_tracks_all, sub_tracks_FFTS_UST,
                    sub_tracks_TS, sub_tracks_LS]
                dicts = [
                    all_dic, FFTS_UST_dic, TS_dic, LS_dic]

                for i in range(len(sub_tracks)):
                    dicts[i] = get_dic_properties(sub_tracks[i], dicts[i])

    return all_dic, FFTS_UST_dic, TS_dic, LS_dic


def get_dic_properties(sub_tracks, dic):
    u_shear = sub_tracks['u_shear']
    v_shear = sub_tracks['v_shear']
    u_relative = sub_tracks['u_relative']
    v_relative = sub_tracks['v_relative']

    shear_angle = np.arctan2(v_shear, u_shear)
    shear_angle = np.rad2deg(shear_angle)

    prop_angle = np.arctan2(v_relative, u_relative)
    prop_angle = np.rad2deg(prop_angle)

    orientation = sub_tracks['orientation_alt']

    dic['shear_angle_list'] += shear_angle.tolist()
    dic['orientation_list'] += orientation.tolist()
    dic['prop_angle_list'] += prop_angle.tolist()

    return dic


def get_FFTS_UST(sub_classes, tracks_obj):

    cols = ['inflow_type', 'offset_type', 'tilt_type']
    req_type = (
        'Front Fed', 'Trailing Stratiform', 'Up-Shear Tilted')
    cond = np.all(sub_classes[cols] == req_type, axis=1)

    inds = sub_classes.where(cond == True).dropna()
    # inds = sub_classes
    inds = inds.index.values

    sub_tracks = tracks_obj.tracks.loc[inds]
    if len(sub_tracks) > 0:
        if 0 in sub_tracks.index.get_level_values(level='level'):
            sub_tracks = sub_tracks.xs(0, level='level')
        else:
            sub_tracks = pd.DataFrame(columns=sub_tracks.columns)

    return sub_tracks


def get_TS(sub_classes, tracks_obj):

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Front Fed', 'Trailing Stratiform')
    cond_1 = np.all(sub_classes[cols] == req_type, axis=1)

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Rear Fed', 'Leading Stratiform')
    cond_2 = np.all(sub_classes[cols] == req_type, axis=1)

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Parallel Fed (Left)', 'Parallel Stratiform (Right)')
    cond_3 = np.all(sub_classes[cols] == req_type, axis=1)

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Parallel Fed (Right)', 'Parallel Stratiform (Left)')
    cond_4 = np.all(sub_classes[cols] == req_type, axis=1)

    cond = cond_1 + cond_2 + cond_3 + cond_4

    inds = sub_classes.where(cond == True).dropna()
    inds = inds.index.values

    sub_tracks = tracks_obj.tracks.loc[inds]
    if len(sub_tracks) > 0:
        if 0 in sub_tracks.index.get_level_values(level='level'):
            sub_tracks = sub_tracks.xs(0, level='level')
        else:
            sub_tracks = pd.DataFrame(columns=sub_tracks.columns)

    return sub_tracks


def get_LS(sub_classes, tracks_obj):

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Front Fed', 'Leading Stratiform')
    cond_1 = np.all(sub_classes[cols] == req_type, axis=1)

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Rear Fed', 'Trailing Stratiform')
    cond_2 = np.all(sub_classes[cols] == req_type, axis=1)

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Parallel Fed (Left)', 'Parallel Stratiform (Left)')
    cond_3 = np.all(sub_classes[cols] == req_type, axis=1)

    cols = ['inflow_type', 'offset_type']
    req_type = (
        'Parallel Fed (Right)', 'Parallel Stratiform (Right)')
    cond_4 = np.all(sub_classes[cols] == req_type, axis=1)

    cond = cond_1 + cond_2 + cond_3 + cond_4

    inds = sub_classes.where(cond == True).dropna()
    # inds = sub_classes
    inds = inds.index.values

    sub_tracks = tracks_obj.tracks.loc[inds]
    if len(sub_tracks) > 0:
        if 0 in sub_tracks.index.get_level_values(level='level'):
            sub_tracks = sub_tracks.xs(0, level='level')
        else:
            sub_tracks = pd.DataFrame(columns=sub_tracks.columns)

    return sub_tracks


def get_rel_TS(sub_classes, tracks_obj):

    cond = sub_classes['rel_offset_type'] == 'Relative Trailing Stratiform'

    inds = sub_classes.where(cond == True).dropna()
    inds = inds.index.values

    sub_tracks = tracks_obj.tracks.loc[inds]
    if len(sub_tracks) > 0:
        if 0 in sub_tracks.index.get_level_values(level='level'):
            sub_tracks = sub_tracks.xs(0, level='level')
        else:
            sub_tracks = pd.DataFrame(columns=sub_tracks.columns)

    return sub_tracks


def get_rel_LS(sub_classes, tracks_obj):

    cond = sub_classes['rel_offset_type'] == 'Relative Leading Stratiform'

    inds = sub_classes.where(cond == True).dropna()
    inds = inds.index.values

    sub_tracks = tracks_obj.tracks.loc[inds]

    if len(sub_tracks) > 0:
        if 0 in sub_tracks.index.get_level_values(level='level'):
            sub_tracks = sub_tracks.xs(0, level='level')
        else:
            sub_tracks = pd.DataFrame(columns=sub_tracks.columns)

    return sub_tracks

#
# def shear_angle_versus_orientation_scatter(shear_angle_list, orientation_list):
#
#     init_fonts()
#
#     shear = np.mod(np.array(shear_angle_list), 360)
#     line_normal = np.mod(np.array(orientation_list)+90, 360)
#
#     for i in range(len(shear)):
#         if shear[i] - line_normal[i] > 180:
#             line_normal[i] += 360
#         elif shear[i] - line_normal[i] < -180:
#             line_normal[i] -= 360
#
#     # shear = np.array(shear_angle_list)
#     # line_normal = np.array(orientation_list)
#
#     linreg = stats.linregress(shear, line_normal)
#     fig, ax = plt.subplots(1, 1, figsize=(4, 4))
#     # plt.sca(ax)
#     ax.scatter(shear, line_normal, marker='.', s=.75)
#
#     dx = 0.5
#     x = np.arange(0, 360+dx, dx)
#
#     ax.plot(
#         x, linreg.intercept + linreg.slope*x, 'r',
#         label='Least Squares')
#     stats_lab = 'slope = {:.2f},   r = {:.2f},   p = {:.2e}'.format(
#         linreg.slope, linreg.rvalue, linreg.pvalue)
#     ax.text(0.05, 1.025, stats_lab, transform=ax.transAxes, size=12)
#
#     plt.xticks(np.arange(0, 360+45, 45))
#     plt.yticks(np.arange(-180, 360+5*45, 45))
#     plt.xlabel('Shear Direction [Degrees]')
#     plt.ylabel('Line Normal Direction [Degrees]')


def shear_angle_versus_orientation_hist(dicts, data='base', titles=None):

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/'

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    init_fonts()

    fig, axes = plt.subplots(
        int(np.ceil(len(dicts) / 2)), 2, figsize=(12, 6))

    first_five = []
    if titles is None:
        titles = ['']*len(dicts)

    for i in range(len(dicts)):
        shear = np.mod(np.array(dicts[i]['shear_angle_list']), 360)
        line_normal = np.mod(np.array(dicts[i]['orientation_list'])+90, 360)
        cosines = np.cos(np.deg2rad(shear-line_normal))
        angles = np.arccos(cosines) * 180 / np.pi

        ax = axes[i // 2, i % 2]
        plt.sca(ax)
        db = 5
        bins = np.arange(0, 180+db, db)

        hist = ax.hist(
            angles, bins=bins, density=True, color=colors[0])
        first_five.append(hist[0][:9].sum()/hist[0].sum())

        minor_ticks = np.arange(0, 180+db, db)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(0, 180+45, 45))
        ax.set_yticks(np.arange(0, .02+.004, .004))
        ax.set_yticks(np.arange(0, .02+.002, .002), minor=True)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # plt.yticks(np.arange(0, 8, 1))
        plt.xlabel('Angle between Shear and Line-Normal [Degrees]')
        plt.ylabel('Density [-]')

        ax.grid(which='major', alpha=0.5, axis='y')
        ax.grid(which='minor', alpha=0.2, axis='y')

        total_lab = 'Total = {}'.format(len(dicts[i]['shear_angle_list']))

        ax.text(
            0.5, 1.035, titles[i],
            transform=ax.transAxes, size=12, ha='center')
        ax.text(
            0.75, .89, total_lab, transform=ax.transAxes, size=12,
            backgroundcolor='1')

    plt.subplots_adjust(hspace=0.35)
    cl.make_subplot_labels(axes.flatten(), size=16, x_shift=-0.175)

    plt.subplots_adjust(hspace=0.45)

    plt.savefig(
        fig_dir + 'shear_versus_orientation_{}.png'.format(data),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

    return first_five


def shear_angle_versus_propagation_hist(dicts, data='base', titles=None):

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/'

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    init_fonts()

    fig, axes = plt.subplots(
        int(np.ceil(len(dicts) / 2)), 2, figsize=(12, 6))

    first_nine = []
    if titles is None:
        titles = ['']*len(dicts)

    for i in range(len(dicts)):
        shear = np.mod(np.array(dicts[i]['shear_angle_list']), 360)
        line_normal = np.mod(np.array(dicts[i]['prop_angle_list']), 360)
        cosines = np.cos(np.deg2rad(shear-line_normal))
        angles = np.arccos(cosines) * 180 / np.pi

        ax = axes[i // 2, i % 2]
        plt.sca(ax)
        db = 5
        bins = np.arange(0, 180+db, db)

        hist = ax.hist(
            angles, bins=bins, density=True, color=colors[0])
        first_nine.append(hist[0][:9].sum()/hist[0].sum())

        # angle_1 = np.random.uniform(low=0, high=np.pi*2, size=1000000)
        # angle_2 = np.random.uniform(low=0, high=np.pi*2, size=1000000)
        # random_cosines = np.cos(angle_2-angle_1)
        # random_angles = np.arccos(random_cosines) * 180 / np.pi
        # ax.hist(
        #     random_angles, bins=bins, density=True, histtype=u'step',
        #     color=colors[3], linewidth=2, alpha=0.75)

        minor_ticks = np.arange(0, 180+db, db)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(0, 180+45, 45))
        ax.set_yticks(np.arange(0, .028+.004, .004))
        ax.set_yticks(np.arange(0, .028+.002, .002), minor=True)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # plt.yticks(np.arange(0, 8, 1))
        plt.xlabel('Angle between Shear and Relative Velocity [Degrees]')
        plt.ylabel('Density [-]')

        ax.grid(which='major', alpha=0.5, axis='y')
        ax.grid(which='minor', alpha=0.2, axis='y')

        ax.text(
            0.5, 1.035, titles[i],
            transform=ax.transAxes, size=12, ha='center')

        total_lab = 'Total = {}'.format(len(dicts[i]['shear_angle_list']))
        ax.text(
            0.75, .89, total_lab, transform=ax.transAxes, size=12,
            backgroundcolor='1')

    plt.subplots_adjust(hspace=0.35)
    cl.make_subplot_labels(axes.flatten(), size=16, x_shift=-0.175)

    plt.subplots_adjust(hspace=0.45)

    plt.savefig(
        fig_dir + 'shear_versus_propagation_{}.png'.format(data),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

    return first_nine


def shear_angle_versus_propagation_hist_compare(
        dicts_radar, dicts_ACCESS, data='base', titles=None):

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/'

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    init_fonts()

    fig, axes = plt.subplots(
        int(np.ceil(len(dicts_radar) / 2)), 2, figsize=(12, 6))

    first_nine = []
    if titles is None:
        titles = ['']*len(dicts_radar)

    hist_maxes = [.024, .044, .036, .02]

    for i in range(len(dicts_radar)):
        shear = np.mod(np.array(dicts_radar[i]['shear_angle_list']), 360)
        line_normal = np.mod(
            np.array(dicts_radar[i]['prop_angle_list']), 360)
        cosines = np.cos(np.deg2rad(shear-line_normal))
        angles_radar = np.arccos(cosines) * 180 / np.pi

        shear = np.mod(
            np.array(dicts_ACCESS[i]['shear_angle_list']), 360)
        line_normal = np.mod(
            np.array(dicts_ACCESS[i]['prop_angle_list']), 360)
        cosines = np.cos(np.deg2rad(shear-line_normal))
        angles_ACCESS = np.arccos(cosines) * 180 / np.pi

        ax = axes[i // 2, i % 2]
        plt.sca(ax)
        db = 5
        bins = np.arange(0, 180+db, db)

        hist = ax.hist(
            [angles_radar, angles_ACCESS], bins=bins, density=True,
            color=[colors[0], colors[1]], rwidth=1,
            label=['Radar', 'ACCESS-C'])
        first_nine.append(hist[0][:9].sum()/hist[0].sum())

        # angle_1 = np.random.uniform(low=0, high=np.pi*2, size=1000000)
        # angle_2 = np.random.uniform(low=0, high=np.pi*2, size=1000000)
        # random_cosines = np.cos(angle_2-angle_1)
        # random_angles = np.arccos(random_cosines) * 180 / np.pi
        # ax.hist(
        #     random_angles, bins=bins, density=True, histtype=u'step',
        #     color=colors[3], linewidth=2, alpha=0.75)

        minor_ticks = np.arange(0, 180+db, db)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(0, 180+45, 45))
        ax.set_yticks(np.arange(0, hist_maxes[i]+.004, .004))
        ax.set_yticks(np.arange(0, hist_maxes[i]+.002, .002), minor=True)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # plt.yticks(np.arange(0, 8, 1))
        plt.xlabel('Angle between Shear and Relative Velocity [Degrees]')
        plt.ylabel('Density [-]')

        ax.grid(which='major', alpha=0.5, axis='y')
        ax.grid(which='minor', alpha=0.2, axis='y')

        ax.text(
            0.5, 1.035, titles[i],
            transform=ax.transAxes, size=12, ha='center')

        total_lab = 'Radar Total = {}            ACCESS-C Total = {}'.format(
            len(dicts_radar[i]['shear_angle_list']),
            len(dicts_ACCESS[i]['shear_angle_list']))

        ax.text(
            0.1, .89, total_lab, transform=ax.transAxes, size=12,
            backgroundcolor='1')

    plt.subplots_adjust(hspace=0.4)
    cl.make_subplot_labels(axes.flatten(), size=16, x_shift=-0.175)

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.5),
        ncol=2, fancybox=True, shadow=True)

    plt.savefig(
        fig_dir + 'shear_versus_propagation_{}_compare.png'.format(data),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

    return first_nine


def shear_angle_versus_orientation_hist_compare(
        dicts_radar, dicts_ACCESS, data='base', titles=None):

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/'

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    init_fonts()

    fig, axes = plt.subplots(
        int(np.ceil(len(dicts_radar) / 2)), 2, figsize=(12, 6))

    first_five = []
    if titles is None:
        titles = ['']*len(dicts_radar)

    hist_maxes = [.02, .024, .02, .02]

    for i in range(len(dicts_radar)):
        shear = np.mod(np.array(dicts_radar[i]['shear_angle_list']), 360)
        line_normal = np.mod(
            np.array(dicts_radar[i]['orientation_list'])+90, 360)
        cosines = np.cos(np.deg2rad(shear-line_normal))
        angles_radar = np.arccos(cosines) * 180 / np.pi

        shear = np.mod(np.array(dicts_ACCESS[i]['shear_angle_list']), 360)
        line_normal = np.mod(
            np.array(dicts_ACCESS[i]['orientation_list'])+90, 360)
        cosines = np.cos(np.deg2rad(shear-line_normal))
        angles_ACCESS = np.arccos(cosines) * 180 / np.pi

        ax = axes[i // 2, i % 2]
        plt.sca(ax)
        db = 5
        bins = np.arange(0, 180+db, db)

        hist = ax.hist(
            [angles_radar, angles_ACCESS], bins=bins, density=True,
            color=[colors[0], colors[1]], rwidth=1,
            label=['Radar', 'ACCESS-C'])
        first_five.append(hist[0][:9].sum()/hist[0].sum())

        minor_ticks = np.arange(0, 180+db, db)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(0, 180+45, 45))
        ax.set_yticks(np.arange(0, hist_maxes[i]+.004, .004))
        ax.set_yticks(np.arange(0, hist_maxes[i]+.002, .002), minor=True)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # plt.yticks(np.arange(0, 8, 1))
        plt.xlabel('Angle between Shear and Line-Normal [Degrees]')
        plt.ylabel('Density [-]')

        ax.grid(which='major', alpha=0.5, axis='y')
        ax.grid(which='minor', alpha=0.2, axis='y')

        total_lab = 'Radar Total = {}            ACCESS-C Total = {}'.format(
            len(dicts_radar[i]['shear_angle_list']),
            len(dicts_ACCESS[i]['shear_angle_list']))

        ax.text(
            0.5, 1.035, titles[i],
            transform=ax.transAxes, size=12, ha='center')
        ax.text(
            0.1, .89, total_lab, transform=ax.transAxes, size=12,
            backgroundcolor='1')

    plt.subplots_adjust(hspace=0.4)
    cl.make_subplot_labels(axes.flatten(), size=16, x_shift=-0.175)

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.5),
        ncol=2, fancybox=True, shadow=True)

    plt.savefig(
        fig_dir + 'shear_versus_orientation_{}_compare.png'.format(data),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

    return first_five
