from matplotlib import rcParams
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from tint.objects import classify_tracks, get_exclusion_categories

base_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/CPOL/'
save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_tracks/'
fig_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_figures/'
ERA5_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/era5/'
ERA5_dir += 'pressure-levels/reanalysis/'
WRF_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/caine_WRF_data/'


def add_monsoon_regime(tracks_obj, base_dir=None):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    pope = pd.read_csv(
        base_dir + 'Pope_regimes.csv', index_col=0, names=['date', 'regime'])
    pope.index = pd.to_datetime(pope.index)
    regimes = []
    np.datetime64(tracks_obj.tracks_class.index[0][1].date())
    for i in range(len(tracks_obj.tracks_class)):
        d = np.datetime64(tracks_obj.tracks_class.index[i][1].date())
        try:
            regimes.append(int(pope.loc[d].values[0]))
        except KeyError:
            regimes.append(0)
    tracks_obj.tracks_class['pope_regime'] = regimes
    return tracks_obj


def init_fonts():
    rcParams.update({'font.family': 'serif'})
    rcParams.update({'font.serif': 'Liberation Serif'})
    rcParams.update({'mathtext.fontset': 'dejavuserif'})
    rcParams.update({'font.size': 12})


def load_year(year, tracks_dir='base'):
    print('Processing year {}'.format(year))
    save_dir = '/home/student.unimelb.edu.au/shorte1/'
    save_dir += 'Documents/TINT_tracks/{}/'.format(tracks_dir)
    filename = save_dir + '{}1001_{}0501.pkl'.format(
        year, year+1)
    with open(filename, 'rb') as f:
        tracks_obj = pickle.load(f)
    return tracks_obj


def get_sub_tracks(tracks_obj, non_linear=False):
    exclusions = [
        'small_area', 'large_area', 'intersect_border',
        'intersect_border_convective', 'duration_cond',
        'small_velocity', 'small_offset']
    if non_linear:
        exclusions += ['non_linear']
    excluded = tracks_obj.exclusions[exclusions]
    amb = 'Ambiguous (On Quadrant Boundary)'
    quad_bound = tracks_obj.tracks_class['offset_type'] == amb
    excluded = np.logical_or(np.any(excluded, 1), quad_bound)
    included = np.logical_not(excluded)
    sub_tracks = tracks_obj.tracks_class.where(included == True).dropna()
    if len(sub_tracks) == 0:
        sub_tracks = None
    else:
        try:
            sub_tracks = sub_tracks.xs(0, level='level')
        except KeyError:
            sub_tracks = None
    return sub_tracks


def get_sub_uids(sub_tracks):
    sub_uids = list(sorted(set(
        [int(sub_tracks.index.values[i][2]) for i in range(len(sub_tracks))])))
    sub_uids = [str(i) for i in sub_uids]
    return sub_uids


def redo_exclusions(tracks_obj, class_thresh, excl_thresh):
    tracks_obj.params['CLASS_THRESH'] = class_thresh
    tracks_obj.params['EXCL_THRESH'] = excl_thresh
    tracks_obj = classify_tracks(tracks_obj)
    tracks_obj = get_exclusion_categories(tracks_obj)
    return tracks_obj


def get_counts(
        base_dir=None, get_exclusions=False, tracks_dir='base',
        non_linear=False, class_thresh=None, excl_thresh=None):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    [
        year_list, uid, time, offset_type, inflow_type,
        tilt_dir, prop_dir, pope_regime] = [
        [] for i in range(8)]
    years = sorted(list(set(range(1998, 2016)) - {2000, 2007, 2008}))
    for year in years:
        tracks_obj = load_year(year, tracks_dir=tracks_dir)
        if get_exclusions:
            print('Getting new exclusions.')
            tracks_obj = redo_exclusions(tracks_obj, class_thresh, excl_thresh)
        print('Adding Pope monsoon regime.')
        tracks_obj = add_monsoon_regime(tracks_obj, base_dir=base_dir)
        sub_tracks = get_sub_tracks(tracks_obj, non_linear=non_linear)
        if sub_tracks is None:
            print('No tracks satisfying conditions. Skipping year.')
            continue
        sub_uids = get_sub_uids(sub_tracks)
        for i in sub_uids:
            obj = sub_tracks.xs(i, level='uid').reset_index(level='time')
            scans = obj.index.values
            scan_label = scans - min(scans)
            offset = sub_tracks['offset_type'].xs(i, level='uid').values
            inflow = sub_tracks['inflow_type'].xs(i, level='uid').values
            tilt = sub_tracks['tilt_type'].xs(i, level='uid').values
            prop = sub_tracks['propagation_type'].xs(i, level='uid').values
            pope = sub_tracks['pope_regime'].xs(i, level='uid').values
            for j in range(len(scan_label)):
                year_list.append(year)
                uid.append(i)
                time.append(scan_label[j]*10)
                offset_type.append(offset[j])
                inflow_type.append(inflow[j])
                tilt_dir.append(tilt[j])
                prop_dir.append(prop[j])
                pope_regime.append(int(pope[j]))
    class_dic = {
        'year': year_list, 'uid': uid, 'time': time,
        'offset_type': offset_type, 'inflow_type': inflow_type,
        'tilt_dir': tilt_dir, 'prop_dir': prop_dir,
        'pope_regime': pope_regime}
    class_df = pd.DataFrame(class_dic)
    class_df.set_index(['year', 'uid', 'time'], inplace=True)
    class_df.sort_index(inplace=True)

    class_df['inflow_type'] = class_df['inflow_type'].str.replace(
        'Ambiguous (Low Relative Velocity)', 'Ambiguous',
        regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Shear Parallel to Stratiform Offset)', 'Ambiguous',
        regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Small Shear)', 'Ambiguous',
        regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Parallel Shear)', 'Ambiguous',
        regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Low Shear)', 'Ambiguous',
        regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Low Relative Velocity)', 'Ambiguous',
        regex=False)

    return class_df


def get_colors():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def set_ticks(ax1, ax2, leg_columns=3):
    plt.sca(ax1)
    plt.xticks(np.arange(30, 310, 30))
    plt.ylabel('Count [-]')
    plt.xlabel('Time since Initiation [m]')
    ax1.legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.475),
        ncol=leg_columns, fancybox=True, shadow=True)
    plt.setp(ax1.lines, linewidth=1.75)

    plt.sca(ax2)
    plt.xticks(np.arange(30, 310, 30))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Ratio [-]')
    plt.xlabel('Time since Initiation [m]')
    plt.setp(ax2.lines, linewidth=1.75)


def initialise_fig():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
    return fig, (ax1, ax2)


def get_time_str():
    current_time = str(datetime.datetime.now())[0:-7]
    current_time = current_time.replace(" ", "_").replace(":", "_")
    current_time = current_time.replace("-", "")
    return current_time


def make_subplot_labels(ax1, ax2):
    ax1.text(-0.15, 1.0, 'a)', transform=ax1.transAxes, size=16)
    ax2.text(-0.15, 1.0, 'b)', transform=ax2.transAxes, size=16)


def get_save_dir(save_dir):
    if save_dir is None:
        save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        save_dir += 'TINT_figures/'
    return save_dir


def plot_offsets(class_df, save_dir=None, append_time=False):
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    offset_types = counts_df.groupby(['time', 'offset_type']).sum()
    TS = offset_types.xs('Trailing Stratiform', level='offset_type')
    LS = offset_types.xs('Leading Stratiform', level='offset_type')
    LeS = offset_types.xs('Parallel Stratiform (Left)', level='offset_type')
    RiS = offset_types.xs('Parallel Stratiform (Right)', level='offset_type')
    max_time = max(
        [max(off_type.index.values) for off_type in [TS, LS, LeS, RiS]])
    new_index = pd.Index(np.arange(0, max_time, 10), name='time')
    [TS, LS, LeS, RiS] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [TS, LS, LeS, RiS]]
    offset_totals = TS + LS + LeS + RiS

    init_fonts()
    fig, (ax1, ax2) = initialise_fig()
    x = np.arange(30, 310, 10)
    ax1.plot(x, TS.loc[x], label='Trailing Stratiform', color=colors[0])
    ax1.plot(x, LS.loc[x], label='Leading Stratiform', color=colors[1])
    ax1.plot(
        x, LeS.loc[x], label='Parallel Stratiform (Left)', color=colors[2])
    ax1.plot(
        x, RiS.loc[x], label='Parallel Stratiform (Right)', color=colors[4])
    ax1.plot(x, offset_totals.loc[x], label='Total', color=colors[3])

    ax2.plot(
        x, (TS/offset_totals).loc[x],
        label='Trailing Stratiform', color=colors[0])
    ax2.plot(
        x, (LeS/offset_totals).loc[x], label='Parallel Stratiform (Left)',
        color=colors[2])
    ax2.plot(
        x, (RiS/offset_totals).loc[x], label='Parallel Stratiform (Right)',
        color=colors[4])
    ax2.plot(
        x, (LS/offset_totals).loc[x],
        label='Leading Stratiform', color=colors[1])

    make_subplot_labels(ax1, ax2)
    set_ticks(ax1, ax2)
    current_time = get_time_str()

    if append_time:
        fn = 'offsets_{}.png'.format(current_time)
    else:
        fn = 'offsets.png'

    plt.savefig(
        save_dir + fn, dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')

    totals = [y.loc[x].sum() for y in [TS, LS, LeS, RiS, offset_totals]]

    return totals


def plot_inflows(class_df, save_dir=None, append_time=False):
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    inflow_types = counts_df.groupby(['time', 'inflow_type']).sum()
    A = inflow_types.xs('Ambiguous', level='inflow_type')
    FF = inflow_types.xs('Front Fed', level='inflow_type')
    RF = inflow_types.xs('Rear Fed', level='inflow_type')
    LeF = inflow_types.xs('Parallel Fed (Left)', level='inflow_type')
    RiF = inflow_types.xs('Parallel Fed (Right)', level='inflow_type')
    max_time = max(
        [max(off_type.index.values) for off_type in [A, FF, RF, LeF, RiF]])
    new_index = pd.Index(np.arange(0, max_time, 10), name='time')
    [A, FF, RF, LeF, RiF] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [A, FF, RF, LeF, RiF]]
    inflow_totals = A + FF + RF + LeF + RiF

    init_fonts()
    fig, (ax1, ax2) = initialise_fig()
    x = np.arange(30, 310, 10)
    ax1.plot(x, FF.loc[x], label='Front Fed', color=colors[0])
    ax1.plot(x, RF.loc[x], label='Rear Fed', color=colors[1])
    ax1.plot(x, LeF.loc[x], label='Parallel Fed (Left)', color=colors[2])
    ax1.plot(x, RiF.loc[x], label='Parallel Fed (Right)', color=colors[4])
    ax1.plot(x, A.loc[x], label='Ambiguous', color=colors[5])
    ax1.plot(x, inflow_totals.loc[x], label='Total', color=colors[3])

    ax2.plot(
        x, (FF/inflow_totals).loc[x], label='Front Fed', color=colors[0])
    ax2.plot(x, (RF/inflow_totals).loc[x], label='Rear Fed', color=colors[1])
    ax2.plot(
        x, (LeF/inflow_totals).loc[x], label='Parallel Fed (Left)',
        color=colors[2])
    ax2.plot(
        x, (RiF/inflow_totals).loc[x], label='Parallel Fed (Right)',
        color=colors[4])
    ax2.plot(x, (A/inflow_totals).loc[x], label='Ambiguous', color=colors[5])
    set_ticks(ax1, ax2)
    make_subplot_labels(ax1, ax2)

    current_time = get_time_str()
    if append_time:
        fn = 'inflows_{}.png'.format(current_time)
    else:
        fn = 'inflows.png'
    plt.savefig(
        save_dir + fn, dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')

    totals = [y.loc[x].sum() for y in [FF, RF, LeF, RiF, A, inflow_totals]]
    return totals


def plot_tilts(class_df, save_dir=None, append_time=False):
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    tilt_types = counts_df.groupby(['time', 'tilt_dir']).sum()
    A = tilt_types.xs('Ambiguous', level='tilt_dir')
    UST = tilt_types.xs('Up-Shear Tilted', level='tilt_dir')
    DST = tilt_types.xs('Down-Shear Tilted', level='tilt_dir')
    max_time = max(
        [max(off_type.index.values) for off_type in [A, UST, DST]])
    new_index = pd.Index(np.arange(0, max_time, 10), name='time')
    [A, UST, DST] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [A, UST, DST]]
    tilt_totals = A + UST + DST

    init_fonts()
    fig, (ax1, ax2) = initialise_fig()
    x = np.arange(30, 310, 10)
    ax1.plot(x, UST.loc[x], label='Up-Shear Tilted', color=colors[0])
    ax1.plot(x, DST.loc[x], label='Down-Shear Tilted', color=colors[1])
    ax1.plot(x, A.loc[x], label='Ambiguous', color=colors[5])
    ax1.plot(x, tilt_totals.loc[x], label='Total', color=colors[3])

    ax2.plot(
        x, (UST/tilt_totals).loc[x], label='Up-Shear Tilted',
        color=colors[0])
    ax2.plot(
        x, (DST/tilt_totals).loc[x], label='Down-Shear Tilted',
        color=colors[1])
    ax2.plot(x, (A/tilt_totals).loc[x], label='Ambiguous', color=colors[5])
    set_ticks(ax1, ax2, leg_columns=2)
    make_subplot_labels(ax1, ax2)

    current_time = get_time_str()
    if append_time:
        fn = 'tilts_{}.png'.format(current_time)
    else:
        fn = 'tilts.png'
    plt.savefig(
        save_dir + fn, dpi=200, facecolor='w', edgecolor='white',
        bbox_inches='tight')

    totals = [y.loc[x].sum() for y in [UST, DST, A, tilt_totals]]
    return totals


def plot_propagations(class_df, save_dir=None, append_time=False):
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    tilt_types = counts_df.groupby(['time', 'prop_dir']).sum()
    A = tilt_types.xs('Ambiguous', level='prop_dir')
    USP = tilt_types.xs('Up-Shear Propagating', level='prop_dir')
    DSP = tilt_types.xs('Down-Shear Propagating', level='prop_dir')
    max_time = max(
        [max(off_type.index.values) for off_type in [A, USP, DSP]])
    new_index = pd.Index(np.arange(0, max_time, 10), name='time')
    [A, USP, DSP] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [A, USP, DSP]]
    prop_totals = A + USP + DSP

    init_fonts()
    fig, (ax1, ax2) = initialise_fig()
    x = np.arange(30, 310, 10)
    ax1.plot(x, DSP.loc[x], label='Down-Shear Propagating', color=colors[0])
    ax1.plot(x, USP.loc[x], label='Up-Shear Propagating', color=colors[1])
    ax1.plot(x, A.loc[x], label='Ambiguous', color=colors[5])
    ax1.plot(
        x, prop_totals.loc[x], label='Total', color=colors[3])

    ax2.plot(
        x, (DSP/prop_totals).loc[x], label='Down-Shear Propagating',
        color=colors[0])
    ax2.plot(
        x, (USP/prop_totals).loc[x], label='Up-Shear Propagating',
        color=colors[1])
    ax2.plot(
        x, (A/prop_totals).loc[x], label='Ambiguous',
        color=colors[5])

    set_ticks(ax1, ax2, leg_columns=2)
    make_subplot_labels(ax1, ax2)

    current_time = get_time_str()
    if append_time:
        fn = 'propagations_{}.png'.format(current_time)
    else:
        fn = 'propagations.png'
    plt.savefig(
        save_dir + fn, dpi=200, facecolor='w', edgecolor='white',
        bbox_inches='tight')

    totals = [y.loc[x].sum() for y in [DSP, USP, A, prop_totals]]
    return totals


def plot_all():
    test_dir = [
        'base', 'lower_conv_level', 'four_levels', 'higher_shear_thresh',
        'higher_rel_vel_thresh', 'higher_theta_e', 'higher_offset_thresh',
        'higher_area_thresh', 'higher_border_thresh', 'linear_50',
        'linear_25']
    test_names = [
        'Base', 'Lower Convective Level', 'Four Levels',
        'Higher Shear Threshold', 'Higher Relative Velocity Threshold',
        'Higher Quadrant Buffer', 'Higher Stratiform Offset Threshold',
        'Higher Minimum Area Threshold',
        'Stricter Border Intersection Threshold',
        '50 km Linearity Threshold',
        '25 km Reduced Axis Ratio Linearity Threshold']

    test = []
    [TS, LS, LeS, RiS, offset_total] = [[] for i in range(5)]
    [FF, RF, LeF, RiF, A_inflow, inflow_total] = [[] for i in range(6)]
    [UST, DST, A_tilt, tilt_total] = [[] for i in range(4)]
    [USP, DSP, A_prop, prop_total] = [[] for i in range(4)]

    for i in range(len(test_dir)):
        base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        class_path = base_dir + 'TINT_tracks/'
        class_path += '{}_classes.pkl'.format(test_dir[i])
        fig_dir = base_dir + 'TINT_figures/'
        fig_dir += test_dir[i] + '/'
        with open(class_path, 'rb') as f:
            class_df = pickle.load(f)
        test.append(test_names[i])

        offset_summary = plot_offsets(class_df, fig_dir)
        TS.append(offset_summary[0].values[0])
        LS.append(offset_summary[1].values[0])
        LeS.append(offset_summary[2].values[0])
        RiS.append(offset_summary[3].values[0])
        offset_total.append(offset_summary[4].values[0])

        inflow_summary = plot_inflows(class_df, fig_dir)
        FF.append(inflow_summary[0].values[0])
        RF.append(inflow_summary[1].values[0])
        LeF.append(inflow_summary[2].values[0])
        RiF.append(inflow_summary[3].values[0])
        A_inflow.append(inflow_summary[4].values[0])
        inflow_total.append(inflow_summary[5].values[0])

        tilt_summary = plot_tilts(class_df, fig_dir)
        UST.append(tilt_summary[0].values[0])
        DST.append(tilt_summary[1].values[0])
        A_tilt.append(tilt_summary[2].values[0])
        tilt_total.append(tilt_summary[3].values[0])

        prop_summary = plot_propagations(class_df, fig_dir)
        USP.append(prop_summary[1].values[0])
        DSP.append(prop_summary[0].values[0])
        A_prop.append(prop_summary[2].values[0])
        prop_total.append(prop_summary[3].values[0])

        plt.close('all')

    offset_sensitivity_df = pd.DataFrame({
        'Test': test, 'Trailing Stratiform': TS,
        'Leading Stratiform': LS, 'Parallel Stratiform (Left)': LeS,
        'Parallel Stratiform (Right)': RiS, 'Total': offset_total})
    offset_sensitivity_df = offset_sensitivity_df.set_index('Test')

    inflow_sensitivity_df = pd.DataFrame({
        'Test': test, 'Front Fed': FF,
        'Rear Fed': RF, 'Parallel Fed (Left)': LeF,
        'Parallel Fed (Right)': RiF, 'Ambiguous': A_inflow,
        'Total': inflow_total})
    inflow_sensitivity_df = inflow_sensitivity_df.set_index('Test')

    tilt_sensitivity_df = pd.DataFrame({
        'Test': test, 'Up-Shear Tilted': UST,
        'Down-Shear Tilted': DST, 'Ambiguous': A_tilt,
        'Total': tilt_total})
    tilt_sensitivity_df = tilt_sensitivity_df.set_index('Test')

    prop_sensitivity_df = pd.DataFrame({
        'Test': test, 'Down-Shear Propagating': DSP,
        'Up-Shear Propagating': USP, 'Ambiguous': A_prop,
        'Total': prop_total})
    prop_sensitivity_df = prop_sensitivity_df.set_index('Test')

    sen_dfs = [
        offset_sensitivity_df, inflow_sensitivity_df,
        tilt_sensitivity_df, prop_sensitivity_df]

    return sen_dfs


def plot_sensitivities(sen_dfs):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    init_fonts()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    offset_c = [colors[i] for i in [0, 1, 2, 4]]
    inflow_c = [colors[i] for i in [0, 1, 2, 4, 5]]
    tilt_c = [colors[i] for i in [0, 1, 5]]
    prop_c = [colors[i] for i in [0, 1, 5]]
    clists = [offset_c, inflow_c, tilt_c, prop_c]
    leg_offset = [-.45, -.55, -.45, -.45]
    labels = ['a)', 'b)', 'c)', 'd)']
    for i in range(len(sen_dfs)):
        base_ratios = sen_dfs[i].drop('Total', axis=1)
        c_list = clists[i]
        for c in base_ratios.columns:
            base_ratios.loc[:, c] = (
                base_ratios.loc[:, c]/sen_dfs[i].loc[:, 'Total'])

        base_ratios = base_ratios.reset_index(drop=True)
        base_ratios.loc[:, 'Test'] = np.array([
            'Base', 'C2', '4LVL', 'S4', 'RV4', 'T15', 'SO15', 'A2',
            'B05', 'L50', 'L252'])
        base_ratios = base_ratios.set_index('Test')
        max_rat = np.ceil(base_ratios.max().max()*10)/10

        ax = axes[i // 2, i % 2]
        ncol = len(base_ratios.columns)
        ax = base_ratios.plot(
            kind='bar', stacked=False, fontsize=12, rot=0, ax=ax,
            yticks=np.arange(0, max_rat+0.1, 0.1), width=0.6*ncol/4,
            color=c_list)
        ax.set_xlabel(None)
        ax.xaxis.set_label_coords(.5, -0.15)
        ax.set_ylabel('Ratio [-]', fontsize=14)
        ax.legend(
            loc='lower center', bbox_to_anchor=(.475, leg_offset[i]),
            ncol=2, fancybox=True, shadow=True)
        ax.set_yticks(np.arange(0, max_rat+0.05, 0.05), minor=True)
        ax.grid(which='minor', alpha=0.2, axis='y')
        ax.grid(which='major', alpha=0.5, axis='y')
        ax.text(-0.15, 1.0, labels[i], transform=ax.transAxes, size=16)
    plt.subplots_adjust(hspace=0.65)
