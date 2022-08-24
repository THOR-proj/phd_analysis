from matplotlib import rcParams
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import datetime
from tint.objects import classify_tracks, get_exclusion_categories
from tint.objects import get_system_tracks
import os


base_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/CPOL/'
save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_tracks/'
fig_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_figures/'
ERA5_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/era5/'
ERA5_dir += 'pressure-levels/reanalysis/'
WRF_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/caine_WRF_data/'


def create_ACCESS_counts(
        save_dir, tracks_base_dir, class_thresh=None,
        excl_thresh=None, non_linear_conds=None, exclusions=None):

    test_names = [
        'ACCESS_63', 'ACCESS_77', 'ACCESS_42']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('Creating new directory.')

    tracks_dir = [
        tracks_base_dir + '/ACCESS_63',
        tracks_base_dir + '/ACCESS_77',
        tracks_base_dir + '/ACCESS_42']

    class_threshes = [
        class_thresh, class_thresh, class_thresh]

    excl_threshes = [
        excl_thresh, excl_thresh, excl_thresh]

    non_linear_conds = [
        non_linear_conds, non_linear_conds, non_linear_conds]

    for i in range(len(tracks_dir)):

        print('Getting classes for test:{}.'.format(test_names[i]))
        class_df = get_counts(
            base_dir='/home/student.unimelb.edu.au/shorte1/Documents/CPOL_analysis/',
            tracks_dir=tracks_dir[i],
            class_thresh=class_threshes[i], excl_thresh=excl_threshes[i],
            non_linear=non_linear_conds[i], years=[2020, 2021], fake_pope=True,
            exclusions=exclusions)

        out_file_name = save_dir + '{}_classes.pkl'.format(test_names[i])
        with open(out_file_name, 'wb') as f:
            pickle.dump(class_df, f)

    radar = [63, 42, 77]
    classes = []
    for r in radar:
        fn = save_dir + 'ACCESS_{}_classes.pkl'.format(r)
        with open(fn, 'rb') as f:
            df = pickle.load(f)
        df = pd.concat({r: df}, names=['radar'])
        classes.append(df)

    combined_classes = pd.concat(classes)
    out_fn = save_dir + 'combined_ACCESS_classes.pkl'
    with open(out_fn, 'wb') as f:
        pickle.dump(combined_classes, f)


def create_oper_radar_counts(
        save_dir, tracks_base_dir, class_thresh=None,
        excl_thresh=None, non_linear_conds=None, exclusions=None):

    test_names = [
        'radar_63', 'radar_77', 'radar_42']

    tracks_dir = [
        tracks_base_dir + '/radar_63',
        tracks_base_dir + '/radar_77',
        tracks_base_dir + '/radar_42']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('Creating new directory.')

    radar = [63, 77, 42]

    class_threshes = [
        class_thresh, class_thresh, class_thresh]

    excl_threshes = [
        excl_thresh, excl_thresh, excl_thresh]

    non_linear_conds = [
        non_linear_conds, non_linear_conds, non_linear_conds]

    for i in range(len(test_names)):

        print('Getting classes for test:{}.'.format(test_names[i]))
        class_df = get_counts_radar(
            base_dir='/home/student.unimelb.edu.au/shorte1/Documents/CPOL_analysis/',
            tracks_dir=tracks_dir[i], class_thresh=class_threshes[i],
            excl_thresh=excl_threshes[i], non_linear=non_linear_conds[i],
            years=[2020, 2021], radar=radar[i], exclusions=exclusions)

        out_file_name = save_dir + '{}_classes.pkl'.format(test_names[i])
        with open(out_file_name, 'wb') as f:
            pickle.dump(class_df, f)

    classes = []
    for r in radar:
        fn = save_dir + 'radar_{}_classes.pkl'.format(r)
        with open(fn, 'rb') as f:
            df = pickle.load(f)
        df = pd.concat({r: df}, names=['radar'])
        classes.append(df)

    combined_classes = pd.concat(classes)
    out_fn = save_dir + 'combined_radar_classes.pkl'
    with open(out_fn, 'wb') as f:
        pickle.dump(combined_classes, f)


def add_monsoon_regime(tracks_obj, base_dir=None, fake_pope=False):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    if fake_pope:
        pope = pd.read_csv(
            base_dir + 'fake_pope_regimes.csv', index_col=0,
            names=['date', 'regime'])
    else:
        pope = pd.read_csv(
            base_dir + 'Pope_regimes.csv', index_col=0,
            names=['date', 'regime'])
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


def init_fonts(fontsize=12):
    rcParams.update({'font.family': 'serif'})
    rcParams.update({'font.serif': 'Liberation Serif'})
    rcParams.update({'mathtext.fontset': 'dejavuserif'})
    rcParams.update({'font.size': fontsize})


def load_year(year, tracks_dir='base'):
    print('Processing year {}'.format(year))
    filename = tracks_dir + '/{}1001_{}0501.pkl'.format(
        year, year+1)
    with open(filename, 'rb') as f:
        tracks_obj = pickle.load(f)
    return tracks_obj


def load_op_month(year, month, radar, tracks_dir='base'):
    print('Processing year {}, month {}, radar, {}'.format(year, month, radar))
    filename = tracks_dir + '/{:02}_{:04}_{:02}.pkl'.format(radar, year, month)
    with open(filename, 'rb') as f:
        tracks_obj = pickle.load(f)
    return tracks_obj


def get_sub_tracks(tracks_obj, non_linear=False, exclusions=None):
    if exclusions is None:
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


def redo_exclusions(tracks_obj, class_thresh=None, excl_thresh=None):
    if class_thresh is not None:
        tracks_obj.params['CLASS_THRESH'] = class_thresh
    if excl_thresh is not None:
        tracks_obj.params['EXCL_THRESH'] = excl_thresh
    tracks_obj = get_system_tracks(tracks_obj)
    tracks_obj = classify_tracks(tracks_obj)
    tracks_obj = get_exclusion_categories(tracks_obj)
    return tracks_obj


def get_counts(
        base_dir=None, tracks_dir='base',
        non_linear=False, class_thresh=None, excl_thresh=None,
        years=sorted(list(set(range(1998, 2016)) - {2000, 2007, 2008})),
        fake_pope=False, exclusions=None):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    [
        year_list, uid, time, offset_type, rel_offset_type, inflow_type,
        tilt_dir, prop_dir, pope_regime, hour] = [
        [] for i in range(10)]
    for year in years:
        tracks_obj = load_year(year, tracks_dir=tracks_dir)
        print('Getting new exclusions.')
        tracks_obj = redo_exclusions(tracks_obj, class_thresh, excl_thresh)
        print('Adding Pope monsoon regime.')
        tracks_obj = add_monsoon_regime(
            tracks_obj, base_dir=base_dir, fake_pope=fake_pope)
        sub_tracks = get_sub_tracks(
            tracks_obj, non_linear=non_linear, exclusions=exclusions)
        if sub_tracks is None:
            print('No tracks satisfying conditions. Skipping year.')
            continue
        sub_uids = get_sub_uids(sub_tracks)
        for i in sub_uids:
            obj = sub_tracks.xs(i, level='uid').reset_index(level='time')
            scans = obj.index.values
            scan_label = scans - min(scans)
            offset = sub_tracks['offset_type'].xs(i, level='uid').values
            rel_offset = sub_tracks['rel_offset_type'].xs(
                i, level='uid').values
            inflow = sub_tracks['inflow_type'].xs(i, level='uid').values
            tilt = sub_tracks['tilt_type'].xs(i, level='uid').values
            prop = sub_tracks['propagation_type'].xs(i, level='uid').values
            pope = sub_tracks['pope_regime'].xs(i, level='uid').values
            hours = obj.time.values
            hours = [int(h.astype(str)[11:13]) for h in hours]

            for j in range(len(scan_label)):
                year_list.append(year)
                uid.append(i)
                time.append(scan_label[j]*10)
                offset_type.append(offset[j])
                rel_offset_type.append(rel_offset[j])
                inflow_type.append(inflow[j])
                tilt_dir.append(tilt[j])
                prop_dir.append(prop[j])
                pope_regime.append(int(pope[j]))
                hour.append(hours[j])
    class_dic = {
        'year': year_list, 'uid': uid, 'time': time, 'hour': hour,
        'offset_type': offset_type, 'rel_offset_type': rel_offset_type,
        'inflow_type': inflow_type,
        'tilt_dir': tilt_dir, 'prop_dir': prop_dir,
        'pope_regime': pope_regime}
    class_df = pd.DataFrame(class_dic)
    class_df.set_index(['year', 'uid', 'time'], inplace=True)
    class_df.sort_index(inplace=True)

    class_df['inflow_type'] = class_df['inflow_type'].str.replace(
        'Ambiguous (Low Relative Velocity)', 'Ambiguous', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Perpendicular Shear)', 'Perpendicular Shear', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Low Shear)', 'Ambiguous', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Small Stratiform Offset)', 'Ambiguous', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (On Quadrant Boundary)', 'Ambiguous', regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Perpendicular Shear)', 'Perpendicular Shear', regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Low Shear)', 'Ambiguous', regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Low Relative Velocity)', 'Ambiguous', regex=False)

    return class_df


def get_counts_radar(
        base_dir=None, tracks_dir='base',
        non_linear=False, class_thresh=None, excl_thresh=None,
        years=[2020, 2021], radar=63, fake_pope=True, exclusions=None):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    [
        year_list, month_list, uid, time, offset_type, rel_offset_type,
        inflow_type, tilt_dir, prop_dir, pope_regime, hour] = [
        [] for i in range(11)]
    for base_year in years:
        year_month = [
            [base_year, 10], [base_year, 11], [base_year, 12],
            [base_year+1, 1], [base_year+1, 2], [base_year+1, 3],
            [base_year+1, 4]]
        for ym in year_month:
            year = ym[0]
            month = ym[1]
            tracks_obj = load_op_month(
                year, month, radar, tracks_dir=tracks_dir)
            print('Getting new exclusions.')
            tracks_obj = redo_exclusions(tracks_obj, class_thresh, excl_thresh)
            print('Adding Pope monsoon regime.')
            tracks_obj = add_monsoon_regime(
                tracks_obj, base_dir=base_dir, fake_pope=fake_pope)
            sub_tracks = get_sub_tracks(
                tracks_obj, non_linear=non_linear, exclusions=exclusions)
            if sub_tracks is None:
                print('No tracks satisfying conditions. Skipping year.')
                continue
            sub_uids = get_sub_uids(sub_tracks)
            for i in sub_uids:
                obj = sub_tracks.xs(i, level='uid').reset_index(level='time')
                scans = obj.index.values
                scan_label = scans - min(scans)
                offset = sub_tracks['offset_type'].xs(i, level='uid').values
                rel_offset = sub_tracks['rel_offset_type'].xs(
                    i, level='uid').values
                inflow = sub_tracks['inflow_type'].xs(i, level='uid').values
                tilt = sub_tracks['tilt_type'].xs(i, level='uid').values
                prop = sub_tracks['propagation_type'].xs(i, level='uid').values
                pope = sub_tracks['pope_regime'].xs(i, level='uid').values
                hours = obj.time.values
                hours = [int(h.astype(str)[11:13]) for h in hours]
                for j in range(len(scan_label)):
                    year_list.append(year)
                    month_list.append(month)
                    uid.append(i)
                    time.append(scan_label[j]*10)
                    offset_type.append(offset[j])
                    rel_offset_type.append(rel_offset[j])
                    inflow_type.append(inflow[j])
                    tilt_dir.append(tilt[j])
                    prop_dir.append(prop[j])
                    pope_regime.append(int(pope[j]))
                    hour.append(hours[j])
    class_dic = {
        'year': year_list, 'month': month_list, 'uid': uid, 'time': time,
        'hour': hour, 'offset_type': offset_type,
        'rel_offset_type': rel_offset_type, 'inflow_type': inflow_type,
        'tilt_dir': tilt_dir, 'prop_dir': prop_dir,
        'pope_regime': pope_regime}
    class_df = pd.DataFrame(class_dic)
    class_df.set_index(['year', 'month', 'uid', 'time'], inplace=True)
    class_df.sort_index(inplace=True)

    class_df['inflow_type'] = class_df['inflow_type'].str.replace(
        'Ambiguous (Low Relative Velocity)', 'Ambiguous', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Perpendicular Shear)', 'Perpendicular Shear', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Low Shear)', 'Ambiguous', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (Small Stratiform Offset)', 'Ambiguous', regex=False)
    class_df['tilt_dir'] = class_df['tilt_dir'].str.replace(
        'Ambiguous (On Quadrant Boundary)', 'Ambiguous', regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Perpendicular Shear)', 'Perpendicular Shear', regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Low Shear)', 'Ambiguous', regex=False)
    class_df['prop_dir'] = class_df['prop_dir'].str.replace(
        'Ambiguous (Low Relative Velocity)', 'Ambiguous', regex=False)

    return class_df


def get_colors():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def set_ticks(
        ax1, ax2, maximum_count, leg_columns=3, legend=True, diurnal=False):
    plt.sca(ax1)
    if diurnal:
        plt.xticks(np.arange(0, 24, 2))
        plt.xlabel('Time of Day [hour UTC]')
    else:
        plt.xticks(np.arange(30, 310, 30))
        plt.xlabel('Time since Detection [min]')

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    if legend:
        ax1.legend(
            by_label.values(), by_label.keys(),
            loc='lower center', bbox_to_anchor=(1.1, -0.685),
            ncol=leg_columns, fancybox=True, shadow=True)
    plt.setp(ax1.lines, linewidth=1.75)

    plt.ylabel('Count [-]')

    max_tick = int(np.ceil(maximum_count / 100) * 100)
    ax1.set_yticks(np.arange(0, max_tick+100, 100), minor=False)
    ax1.set_yticks(np.arange(0, max_tick+50, 50), minor=True)

    plt.sca(ax2)
    if diurnal:
        plt.xticks(np.arange(0, 24, 2))
        plt.xlabel('Time of Day [hour UTC]')
    else:
        plt.xticks(np.arange(30, 310, 30))
        plt.xlabel('Time since Detection [min]')

    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticks(np.arange(0, 1.05, 0.05), minor=True)

    plt.ylabel('Ratio [-]')
    plt.setp(ax2.lines, linewidth=1.75)

    ax1.grid(which='major', alpha=0.5, axis='both')
    ax1.grid(which='minor', alpha=0.2, axis='y')

    ax2.grid(which='major', alpha=0.5, axis='both')
    ax2.grid(which='minor', alpha=0.2, axis='y')


def initialise_fig(length=12, height=3.5, n_subplots=2):
    fig, axes = plt.subplots(
        int(np.ceil(n_subplots/2)), 2, figsize=(length, height))
    return fig, axes


def get_time_str():
    current_time = str(datetime.datetime.now())[0:-7]
    current_time = current_time.replace(" ", "_").replace(":", "_")
    current_time = current_time.replace("-", "")
    return current_time


def make_subplot_labels(axes, size=16, x_shift=-0.15):
    labels = list(string.ascii_lowercase)
    labels = [label + ')' for label in labels]
    for i in range(len(axes)):
        axes[i].text(
            x_shift, 1.0, labels[i], transform=axes[i].transAxes, size=size)


def get_save_dir(save_dir):
    if save_dir is None:
        save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        save_dir += 'TINT_figures/'
    return save_dir


def plot_offsets(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False):
    if (fig is None) or (ax1 is None) or (ax2 is None):
        fig, (ax1, ax2) = initialise_fig()
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    if diurnal:
        x_var = 'hour'
    else:
        x_var = 'time'
    offset_types = counts_df.groupby([x_var, 'offset_type']).sum()
    TS = offset_types.xs('Trailing Stratiform', level='offset_type')
    LS = offset_types.xs('Leading Stratiform', level='offset_type')
    LeS = offset_types.xs('Parallel Stratiform (Left)', level='offset_type')
    RiS = offset_types.xs('Parallel Stratiform (Right)', level='offset_type')
    # max_time = max(
    #     [max(off_type.index.values) for off_type in [TS, LS, LeS, RiS]])
    if diurnal:
        new_index = np.arange(0, 24, 1)
    else:
        max_time = 400
        new_index = pd.Index(np.arange(0, max_time, 10), name='time')
    [TS, LS, LeS, RiS] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [TS, LS, LeS, RiS]]
    offset_totals = TS + LS + LeS + RiS

    init_fonts()
    if diurnal:
        x = new_index
    else:
        x = np.arange(30, 310, 10)
    ax1.plot(
        x, TS.loc[x], label='Trailing Stratiform', color=colors[0],
        linestyle=linestyle)
    ax1.plot(
        x, LS.loc[x], label='Leading Stratiform', color=colors[1],
        linestyle=linestyle)
    ax1.plot(
        x, LeS.loc[x], label='Left Stratiform', color=colors[2],
        linestyle=linestyle)
    ax1.plot(
        x, RiS.loc[x], label='Right Stratiform', color=colors[4],
        linestyle=linestyle)
    ax1.plot(
        x, offset_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle)

    ax2.plot(
        x, (TS/offset_totals).loc[x],
        label='Trailing Stratiform', color=colors[0], linestyle=linestyle)
    ax2.plot(
        x, (LeS/offset_totals).loc[x], label='Left Stratiform',
        color=colors[2], linestyle=linestyle)
    ax2.plot(
        x, (RiS/offset_totals).loc[x], label='Right Stratiform',
        color=colors[4], linestyle=linestyle)
    ax2.plot(
        x, (LS/offset_totals).loc[x],
        label='Leading Stratiform', color=colors[1], linestyle=linestyle)

    # ax2.plot([180, 180], [0, 1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(offset_totals.loc[x].values), maximum),
        legend=legend, diurnal=diurnal)
    totals = [y.loc[x].sum() for y in [TS, LS, LeS, RiS, offset_totals]]
    return totals


def plot_relative_offsets(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False):
    if (fig is None) or (ax1 is None) or (ax2 is None):
        fig, (ax1, ax2) = initialise_fig()
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    if diurnal:
        x_var = 'hour'
    else:
        x_var = 'time'
    offset_types = counts_df.groupby([x_var, 'rel_offset_type']).sum()
    TS = offset_types.xs(
        'Relative Trailing Stratiform', level='rel_offset_type')
    LS = offset_types.xs(
        'Relative Leading Stratiform', level='rel_offset_type')
    LeS = offset_types.xs(
        'Relative Parallel Stratiform (Left)', level='rel_offset_type')
    RiS = offset_types.xs(
        'Relative Parallel Stratiform (Right)', level='rel_offset_type')
    # max_time = max(
    #     [max(off_type.index.values) for off_type in [TS, LS, LeS, RiS]])
    if diurnal:
        new_index = np.arange(0, 24, 1)
    else:
        max_time = 400
        new_index = pd.Index(np.arange(0, max_time, 10), name='time')

    [TS, LS, LeS, RiS] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [TS, LS, LeS, RiS]]
    offset_totals = TS + LS + LeS + RiS

    init_fonts()
    if diurnal:
        x = new_index
    else:
        x = np.arange(30, 310, 10)
    ax1.plot(
        x, TS.loc[x], label='Relative Trailing Stratiform', color=colors[0],
        linestyle=linestyle)
    ax1.plot(
        x, LS.loc[x], label='Relative Leading Stratiform', color=colors[1],
        linestyle=linestyle)
    ax1.plot(
        x, LeS.loc[x], label='Relative Left Stratiform',
        color=colors[2], linestyle=linestyle)
    ax1.plot(
        x, RiS.loc[x], label='Relative Right Stratiform',
        color=colors[4], linestyle=linestyle)
    ax1.plot(
        x, offset_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle)

    ax2.plot(
        x, (TS/offset_totals).loc[x],
        label='Relative Trailing Stratiform', color=colors[0],
        linestyle=linestyle)
    ax2.plot(
        x, (LeS/offset_totals).loc[x],
        label='Relative Left Stratiform',
        color=colors[2], linestyle=linestyle)
    ax2.plot(
        x, (RiS/offset_totals).loc[x],
        label='Relative Right Stratiform',
        color=colors[4], linestyle=linestyle)
    ax2.plot(
        x, (LS/offset_totals).loc[x],
        label='Relative Leading Stratiform', color=colors[1],
        linestyle=linestyle)

    # ax2.plot([120, 120], [0, 1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(offset_totals.loc[x].values), maximum),
        legend=legend, diurnal=diurnal)
    totals = [y.loc[x].sum() for y in [TS, LS, LeS, RiS, offset_totals]]
    return totals


def plot_inflows(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False):
    if (fig is None) or (ax1 is None) or (ax2 is None):
        fig, (ax1, ax2) = initialise_fig()
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    if diurnal:
        x_var = 'hour'
    else:
        x_var = 'time'
    inflow_types = counts_df.groupby([x_var, 'inflow_type']).sum()
    A = inflow_types.xs('Ambiguous', level='inflow_type')
    FF = inflow_types.xs('Front Fed', level='inflow_type')
    RF = inflow_types.xs('Rear Fed', level='inflow_type')
    LeF = inflow_types.xs('Parallel Fed (Left)', level='inflow_type')
    RiF = inflow_types.xs('Parallel Fed (Right)', level='inflow_type')
    # max_time = max(
    #     [max(off_type.index.values) for off_type in [A, FF, RF, LeF, RiF]])

    if diurnal:
        new_index = np.arange(0, 24, 1)
    else:
        max_time = 400
        new_index = pd.Index(np.arange(0, max_time, 10), name='time')

    [A, FF, RF, LeF, RiF] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [A, FF, RF, LeF, RiF]]
    inflow_totals = A + FF + RF + LeF + RiF

    init_fonts()
    if diurnal:
        x = new_index
    else:
        x = np.arange(30, 310, 10)
    ax1.plot(
        x, FF.loc[x], label='Front Fed', color=colors[0],
        linestyle=linestyle)
    ax1.plot(
        x, RF.loc[x], label='Rear Fed', color=colors[1], linestyle=linestyle)
    ax1.plot(
        x, LeF.loc[x], label='Left Fed', color=colors[2],
        linestyle=linestyle)
    ax1.plot(
        x, RiF.loc[x], label='Right Fed', color=colors[4],
        linestyle=linestyle)
    ax1.plot(
        x, A.loc[x], label='Ambiguous', color=colors[5],
        linestyle=linestyle)
    ax1.plot(
        x, inflow_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle)

    ax2.plot(
        x, (FF/inflow_totals).loc[x], label='Front Fed', color=colors[0],
        linestyle=linestyle)
    ax2.plot(
        x, (RF/inflow_totals).loc[x], label='Rear Fed', color=colors[1],
        linestyle=linestyle)
    ax2.plot(
        x, (LeF/inflow_totals).loc[x], label='Left Fed',
        color=colors[2], linestyle=linestyle)
    ax2.plot(
        x, (RiF/inflow_totals).loc[x], label='Right Fed',
        color=colors[4], linestyle=linestyle)
    ax2.plot(
        x, (A/inflow_totals).loc[x], label='Ambiguous', color=colors[5],
        linestyle=linestyle)
    set_ticks(
        ax1, ax2, max(np.max(inflow_totals.loc[x].values), maximum),
        legend=legend, diurnal=diurnal)

    # ax2.plot([120, 120], [0, 1], '--', color='gray')

    totals = [y.loc[x].sum() for y in [FF, RF, LeF, RiF, A, inflow_totals]]
    return totals


def plot_tilts(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False):
    if (fig is None) or (ax1 is None) or (ax2 is None):
        fig, (ax1, ax2) = initialise_fig()
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    if diurnal:
        x_var = 'hour'
    else:
        x_var = 'time'
    tilt_types = counts_df.groupby([x_var, 'tilt_dir']).sum()
    SP = tilt_types.xs('Perpendicular Shear', level='tilt_dir')
    UST = tilt_types.xs('Up-Shear Tilted', level='tilt_dir')
    DST = tilt_types.xs('Down-Shear Tilted', level='tilt_dir')
    # max_time = max(
    #     [max(off_type.index.values) for off_type in [SP, UST, DST]])
    if diurnal:
        new_index = np.arange(0, 24, 1)
    else:
        max_time = 400
        new_index = pd.Index(np.arange(0, max_time, 10), name='time')

    [SP, UST, DST] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [SP, UST, DST]]
    tilt_totals = SP + UST + DST

    init_fonts()
    if diurnal:
        x = new_index
    else:
        x = np.arange(30, 310, 10)
    ax1.plot(
        x, UST.loc[x], label='Up-Shear Tilted', color=colors[0],
        linestyle=linestyle)
    ax1.plot(
        x, DST.loc[x], label='Down-Shear Tilted', color=colors[1],
        linestyle=linestyle)
    ax1.plot(
        x, SP.loc[x], label='Shear Perpendicular', color=colors[5],
        linestyle=linestyle)
    ax1.plot(
        x, tilt_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle)

    ax2.plot(
        x, (UST/tilt_totals).loc[x], label='Up-Shear Tilted',
        color=colors[0], linestyle=linestyle)
    ax2.plot(
        x, (DST/tilt_totals).loc[x], label='Down-Shear Tilted',
        color=colors[1], linestyle=linestyle)
    ax2.plot(
        x, (SP/tilt_totals).loc[x], label='Ambiguous', color=colors[5],
        linestyle=linestyle)

    # ax2.plot([210, 210], [0, 1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(tilt_totals.loc[x].values), maximum),
        leg_columns=2, legend=legend, diurnal=diurnal)

    totals = [y.loc[x].sum() for y in [UST, DST, SP, tilt_totals]]
    return totals


def plot_propagations(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False):
    if (fig is None) or (ax1 is None) or (ax2 is None):
        fig, (ax1, ax2) = initialise_fig()
    save_dir = get_save_dir(save_dir)
    colors = get_colors()
    counts = class_df.reset_index().set_index(['year', 'uid']).value_counts()
    counts = counts.sort_index()
    counts_df = pd.DataFrame({'counts': counts})
    if diurnal:
        x_var = 'hour'
    else:
        x_var = 'time'
    tilt_types = counts_df.groupby([x_var, 'prop_dir']).sum()
    SP = tilt_types.xs('Perpendicular Shear', level='prop_dir')
    USP = tilt_types.xs('Up-Shear Propagating', level='prop_dir')
    DSP = tilt_types.xs('Down-Shear Propagating', level='prop_dir')
    # max_time = max(
    #     [max(off_type.index.values) for off_type in [SP, USP, DSP]])
    if diurnal:
        new_index = np.arange(0, 24, 1)
    else:
        max_time = 400
        new_index = pd.Index(np.arange(0, max_time, 10), name='time')

    [SP, USP, DSP] = [
        off_type.reindex(new_index, fill_value=0)
        for off_type in [SP, USP, DSP]]
    prop_totals = SP + USP + DSP

    init_fonts()
    # fig, (ax1, ax2) = initialise_fig()
    if diurnal:
        x = new_index
    else:
        x = np.arange(30, 310, 10)
    ax1.plot(
        x, DSP.loc[x], label='Down-Shear Propagating', color=colors[0],
        linestyle=linestyle)
    ax1.plot(
        x, USP.loc[x], label='Up-Shear Propagating', color=colors[1],
        linestyle=linestyle)
    ax1.plot(
        x, SP.loc[x], label='Shear Perpendicular', color=colors[5],
        linestyle=linestyle)
    ax1.plot(
        x, prop_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle)

    ax2.plot(
        x, (DSP/prop_totals).loc[x], label='Down-Shear Propagating',
        color=colors[0], linestyle=linestyle)
    ax2.plot(
        x, (USP/prop_totals).loc[x], label='Up-Shear Propagating',
        color=colors[1], linestyle=linestyle)
    ax2.plot(
        x, (SP/prop_totals).loc[x], label='Shear Perpendicular',
        color=colors[5], linestyle=linestyle)

    # ax2.plot([210, 210], [0, 1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(prop_totals.loc[x].values), maximum),
        leg_columns=2, legend=legend, diurnal=diurnal)

    totals = [y.loc[x].sum() for y in [DSP, USP, SP, prop_totals]]
    return totals


def plot_comparison():
    test_dir = ['base', 'two_levels']
    linestyles = ['-', '--']
    legends = [True, False]

    fig_1, axes_1 = initialise_fig(height=10, n_subplots=6)
    fig_2, axes_2 = initialise_fig(height=6, n_subplots=4)

    for i in range(len(test_dir)):
        base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        class_path = base_dir + 'TINT_tracks/'
        class_path += '{}_classes.pkl'.format(test_dir[i])
        fig_dir = base_dir + 'TINT_figures/'
        fig_dir += test_dir[i] + '/'
        with open(class_path, 'rb') as f:
            class_df = pickle.load(f)

        plot_offsets(
            class_df, fig_dir, fig=fig_1, ax1=axes_1[0][0], ax2=axes_1[0][1],
            linestyle=linestyles[i], legend=legends[i], maximum=800)

        # import pdb; pdb.set_trace()

        plot_relative_offsets(
            class_df, fig_dir, fig=fig_1, ax1=axes_1[1][0], ax2=axes_1[1][1],
            linestyle=linestyles[i], legend=legends[i],
            maximum=600)

        plot_inflows(
            class_df, fig_dir, fig=fig_1, ax1=axes_1[2][0], ax2=axes_1[2][1],
            linestyle=linestyles[i], legend=legends[i], maximum=800)

        plt.subplots_adjust(hspace=0.775)
        make_subplot_labels(axes_1.flatten())

        plt.savefig(
            fig_dir + 'offsets_inflows_comparison.png', dpi=200, facecolor='w',
            edgecolor='white', bbox_inches='tight')

        plot_tilts(
            class_df, fig_dir, fig=fig_2, ax1=axes_2[0][0], ax2=axes_2[0][1],
            linestyle=linestyles[i], legend=legends[i], maximum=600)

        plot_propagations(
            class_df, fig_dir, fig=fig_2, ax1=axes_2[1][0], ax2=axes_2[1][1],
            linestyle=linestyles[i], legend=legends[i],
            maximum=600)

        make_subplot_labels(axes_2.flatten())
        plt.subplots_adjust(hspace=0.775)

        plt.savefig(
            fig_dir + 'tilts_propagations_comparison.png', dpi=200,
            facecolor='w',
            edgecolor='white', bbox_inches='tight')

        plt.close('all')


def plot_all(test_dir=None, test_names=None, diurnal=False):

    if (test_dir is None) or (test_names is None):
        test_dir = [
            'base', 'lower_conv_level', 'higher_conv_level',
            'four_levels', 'no_steiner', 'lower_ref_thresh',
            'higher_shear_thresh', 'higher_rel_vel_thresh', 'higher_theta_e',
            'higher_offset_thresh',
            'higher_area_thresh', 'higher_border_thresh', 'linear_50',
            'linear_25', 'combined']
        test_names = [
            'Base', 'Lower Convective Level', 'Higher Convective Level',
            'Four Levels', 'No Steiner', 'Lower Reflectivitiy Thresholds',
            'Higher Shear Threshold', 'Higher Relative Velocity Threshold',
            'Higher Quadrant Buffer', 'Higher Stratiform Offset Threshold',
            'Higher Minimum Area Threshold',
            'Stricter Border Intersection Threshold',
            '50 km Linearity Threshold',
            '25 km Reduced Axis Ratio Linearity Threshold', 'Combined']

    test = []
    [TS, LS, LeS, RiS, offset_total] = [[] for i in range(5)]
    [RTS, RLS, RLeS, RRiS, rel_offset_total] = [[] for i in range(5)]
    [FF, RF, LeF, RiF, A_inflow, inflow_total] = [[] for i in range(6)]
    [UST, DST, A_tilt, tilt_total] = [[] for i in range(4)]
    [USP, DSP, A_prop, prop_total] = [[] for i in range(4)]

    for i in range(len(test_dir)):
    # for i in [0]:
        base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        class_path = base_dir + 'TINT_tracks/'
        class_path += '{}_classes.pkl'.format(test_dir[i])
        fig_dir = base_dir + 'TINT_figures/'
        fig_dir += test_dir[i] + '/'

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            print('Creating new directory.')
        with open(class_path, 'rb') as f:
            class_df = pickle.load(f)
        test.append(test_names[i])

        fig, axes = initialise_fig(height=10, n_subplots=6)

        offset_summary = plot_offsets(
            class_df, fig_dir, fig=fig, ax1=axes[0][0], ax2=axes[0][1],
            diurnal=diurnal)
        TS.append(offset_summary[0].values[0])
        LS.append(offset_summary[1].values[0])
        LeS.append(offset_summary[2].values[0])
        RiS.append(offset_summary[3].values[0])
        offset_total.append(offset_summary[4].values[0])

        rel_offset_summary = plot_relative_offsets(
            class_df, fig_dir, fig=fig, ax1=axes[1][0], ax2=axes[1][1],
            diurnal=diurnal)
        RTS.append(rel_offset_summary[0].values[0])
        RLS.append(rel_offset_summary[1].values[0])
        RLeS.append(rel_offset_summary[2].values[0])
        RRiS.append(rel_offset_summary[3].values[0])
        rel_offset_total.append(rel_offset_summary[4].values[0])

        inflow_summary = plot_inflows(
            class_df, fig_dir, fig=fig, ax1=axes[2][0], ax2=axes[2][1],
            diurnal=diurnal)
        FF.append(inflow_summary[0].values[0])
        RF.append(inflow_summary[1].values[0])
        LeF.append(inflow_summary[2].values[0])
        RiF.append(inflow_summary[3].values[0])
        A_inflow.append(inflow_summary[4].values[0])
        inflow_total.append(inflow_summary[5].values[0])

        plt.subplots_adjust(hspace=0.775)
        make_subplot_labels(axes.flatten())

        plt.savefig(
            fig_dir + 'offsets_inflows' + diurnal*'_diurnal' + '.png',
            dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

        fig, axes = initialise_fig(height=6, n_subplots=4)

        tilt_summary = plot_tilts(
            class_df, fig_dir, fig=fig, ax1=axes[0][0], ax2=axes[0][1],
            diurnal=diurnal)
        UST.append(tilt_summary[0].values[0])
        DST.append(tilt_summary[1].values[0])
        A_tilt.append(tilt_summary[2].values[0])
        tilt_total.append(tilt_summary[3].values[0])

        prop_summary = plot_propagations(
            class_df, fig_dir, fig=fig, ax1=axes[1][0], ax2=axes[1][1],
            diurnal=diurnal)
        USP.append(prop_summary[1].values[0])
        DSP.append(prop_summary[0].values[0])
        A_prop.append(prop_summary[2].values[0])
        prop_total.append(prop_summary[3].values[0])

        make_subplot_labels(axes.flatten())
        plt.subplots_adjust(hspace=0.775)

        plt.savefig(
            fig_dir + 'tilts_propagations' + diurnal*'_diurnal' + '.png',
            dpi=200, facecolor='w',
            edgecolor='white', bbox_inches='tight')

        plt.close('all')

    offset_sensitivity_df = pd.DataFrame({
        'Test': test, 'Trailing Stratiform': TS,
        'Leading Stratiform': LS, 'Left Stratiform': LeS,
        'Right Stratiform': RiS, 'Total': offset_total})
    offset_sensitivity_df = offset_sensitivity_df.set_index('Test')

    inflow_sensitivity_df = pd.DataFrame({
        'Test': test, 'Front Fed': FF,
        'Rear Fed': RF, 'Left Fed': LeF,
        'Right Fed': RiF, 'Ambiguous': A_inflow,
        'Total': inflow_total})
    inflow_sensitivity_df = inflow_sensitivity_df.set_index('Test')

    tilt_sensitivity_df = pd.DataFrame({
        'Test': test, 'Up-Shear Tilted': UST,
        'Down-Shear Tilted': DST, 'Shear Perpendicular': A_tilt,
        'Total': tilt_total})
    tilt_sensitivity_df = tilt_sensitivity_df.set_index('Test')

    prop_sensitivity_df = pd.DataFrame({
        'Test': test, 'Down-Shear Propagating': DSP,
        'Up-Shear Propagating': USP, 'Shear Perpendicular': A_prop,
        'Total': prop_total})
    prop_sensitivity_df = prop_sensitivity_df.set_index('Test')

    rel_offset_sensitivity_df = pd.DataFrame({
        'Test': test, 'Relative Trailing Stratiform': RTS,
        'Relative Leading Stratiform': RLS,
        'Relative Left Stratiform': RLeS,
        'Relative Right Stratiform': RRiS,
        'Total': rel_offset_total})
    rel_offset_sensitivity_df = rel_offset_sensitivity_df.set_index('Test')

    sen_dfs = [
        offset_sensitivity_df, inflow_sensitivity_df,
        tilt_sensitivity_df, prop_sensitivity_df, rel_offset_sensitivity_df]

    return sen_dfs


def plot_sensitivities(sen_dfs, test_dirs, name_abvs=None, suff=''):

    if name_abvs is None:
        name_abvs = [
            'Base', 'C2', 'C4', '4L', 'NS', 'LR', 'S4', 'RV4', 'T15',
            'S15', 'A2', 'B5', 'L50', 'L25', 'C']

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    init_fonts()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    offset_c = [colors[i] for i in [0, 1, 2, 4]]
    inflow_c = [colors[i] for i in [0, 1, 2, 4, 5]]
    tilt_c = [colors[i] for i in [0, 1, 5]]
    prop_c = [colors[i] for i in [0, 1, 5]]
    clists = [offset_c, inflow_c, tilt_c, prop_c, offset_c]
    offset_1 = -.575
    leg_offset = [offset_1, offset_1, offset_1, offset_1, offset_1]
    leg_offset_x = [.475] * 4 + [.475]
    leg_columns = [2, 3, 2, 2, 2]
    for i in range(len(sen_dfs)):
        base_ratios = sen_dfs[i].drop('Total', axis=1)
        c_list = clists[i]
        for c in base_ratios.columns:
            base_ratios.loc[:, c] = (
                base_ratios.loc[:, c]/sen_dfs[i].loc[:, 'Total'])

        base_ratios = base_ratios.reset_index(drop=True)
        base_ratios.loc[:, 'Test'] = np.array(name_abvs)
        base_ratios = base_ratios.set_index('Test')
        max_rat = np.ceil(base_ratios.max().max()*10)/10

        ax = axes[i // 2, i % 2]
        ncol = len(base_ratios.columns)
        ax = base_ratios.plot(
            kind='bar', stacked=False, fontsize=12, rot=0, ax=ax,
            yticks=np.arange(0, max_rat+0.1, 0.1), width=0.625*ncol/4,
            color=c_list)
        ax.set_xlabel(None)
        ax.xaxis.set_label_coords(.5, -0.15)
        ax.set_ylabel('Ratio [-]', fontsize=14)
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(leg_offset_x[i], leg_offset[i]),
            ncol=leg_columns[i], fancybox=True, shadow=True)
        ax.set_yticks(np.arange(0, max_rat+0.05, 0.05), minor=True)
        ax.grid(which='minor', alpha=0.2, axis='y')
        ax.grid(which='major', alpha=0.5, axis='y')

    category_breakdown(
        fig=fig, ax=axes[2, 1], leg_offset_h=-.71, test_dir=test_dirs,
        test_names=name_abvs, name_abvs=name_abvs)
    plt.subplots_adjust(hspace=0.65)
    make_subplot_labels(axes.flatten())

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/'
    plt.savefig(
        fig_dir + 'total_ratio_sensitivities{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def plot_sensitivities_comp(
        sen_dfs_1, sen_dfs_2, test_dirs, name_abvs=None, suff=''):

    if name_abvs is None:
        name_abvs = [
            'Base', 'C2', 'C4', '4L', 'NS', 'LR', 'S4', 'RV4', 'T15',
            'S15', 'A2', 'B5', 'L50', 'L25', 'C']

    fig, axes = plt.subplots(6, 1, figsize=(13, 15))
    init_fonts()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    offset_c = [colors[i] for i in [0, 1, 2, 4]]
    inflow_c = [colors[i] for i in [0, 1, 2, 4, 5]]
    tilt_c = [colors[i] for i in [0, 1, 5]]
    prop_c = [colors[i] for i in [0, 1, 5]]
    clists = [offset_c, inflow_c, tilt_c, prop_c, offset_c]
    offset_1 = -.55
    leg_offset = [offset_1, offset_1, offset_1, offset_1, offset_1]
    leg_offset_x = [.475] * 4 + [.475]
    leg_columns = [4, 5, 3, 3, 4]
    for i in range(len(sen_dfs_1)):
        base_ratios_1 = sen_dfs_1[i].drop('Total', axis=1)
        base_ratios_2 = sen_dfs_2[i].drop('Total', axis=1)
        c_list = clists[i]
        for c in base_ratios_1.columns:
            base_ratios_1.loc[:, c] = (
                base_ratios_1.loc[:, c]/sen_dfs_1[i].loc[:, 'Total'])
            base_ratios_2.loc[:, c] = (
                base_ratios_2.loc[:, c]/sen_dfs_2[i].loc[:, 'Total'])

        base_ratios_1 = base_ratios_1.reset_index(drop=True)
        base_ratios_1.loc[:, 'Test'] = np.array(name_abvs)
        base_ratios_1 = base_ratios_1.set_index('Test')
        max_rat_1 = np.ceil(base_ratios_1.max().max()*10)/10

        base_ratios_2 = base_ratios_2.reset_index(drop=True)
        base_ratios_2.loc[:, 'Test'] = np.array(name_abvs)
        base_ratios_2 = base_ratios_2.set_index('Test')
        max_rat_2 = np.ceil(base_ratios_2.max().max()*10)/10

        max_rat = np.max([max_rat_1, max_rat_2])

        ax = axes.flatten()[i]
        ncol = len(base_ratios_1.columns)

        

        ax = base_ratios_1.plot(
            kind='bar', stacked=False, fontsize=12, rot=0, ax=ax,
            yticks=np.arange(0, max_rat+0.1, 0.1), width=0.625*ncol/8,
            color=c_list, position=1.1)

        ax = base_ratios_2.plot(
            kind='bar', stacked=False, fontsize=12, rot=0, ax=ax,
            yticks=np.arange(0, max_rat+0.1, 0.1), width=0.625*ncol/8,
            color=c_list, position=-.1, edgecolor='black')

        ax.set_xlim(-.5, 15.5)

        ax.set_xlabel(None)
        ax.xaxis.set_label_coords(.5, -0.15)
        ax.set_ylabel('Ratio [-]', fontsize=14)

        lines, labels = ax.get_legend_handles_labels()

        ax.legend(
            lines[:int(len(lines)/2)], labels[:int(len(lines)/2)],
            loc='lower center',
            bbox_to_anchor=(leg_offset_x[i], leg_offset[i]),
            ncol=leg_columns[i], fancybox=True, shadow=True)
        ax.set_yticks(np.arange(0, max_rat+0.05, 0.05), minor=True)
        ax.grid(which='minor', alpha=0.2, axis='y')
        ax.grid(which='major', alpha=0.5, axis='y')

    category_breakdown(
        fig=fig, ax=axes.flatten()[-1], leg_offset_h=-.71, test_dir=test_dirs,
        test_names=name_abvs, name_abvs=name_abvs, ncol=5)
    plt.subplots_adjust(hspace=0.65)
    make_subplot_labels(axes.flatten(), x_shift=-.075)

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/'
    plt.savefig(
        fig_dir + 'total_ratio_sensitivities{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def category_breakdown(
        fig=None, ax=None, leg_offset_h=-0.45, test_dir=None,
        test_names=None, name_abvs=None, ncol=2):

    if test_dir is None:
        test_dir = [
            'base', 'lower_conv_level', 'higher_conv_level', 'two_levels',
            'four_levels',
            'no_steiner', 'lower_ref_thresh', 'higher_shear_thresh',
            'higher_rel_vel_thresh', 'higher_theta_e', 'higher_offset_thresh',
            'higher_area_thresh', 'higher_border_thresh', 'linear_50',
            'linear_25', 'combined']
    if test_names is None:
        test_names = [
            'Base', 'Lower Convective Level', 'Higher Convective Level',
            'Two Levels',
            'Four Levels', 'No Steiner', 'Lower reflectivity Thresholds',
            'Higher Shear Threshold',
            'Higher Relative Velocity Threshold',
            'Higher Quadrant Buffer', 'Higher Stratiform Offset Threshold',
            'Higher Minimum Area Threshold',
            'Stricter Border Intersection Threshold',
            '50 km Linearity Threshold',
            '25 km Linearity Threshold', 'Combined']
    if name_abvs is None:
        name_abvs = [
            'Base', 'C2', 'C4', '2L', '4L', 'NS', 'LR', 'S4', 'RV4', 'T15',
            'S15', 'A2', 'B5', 'L50', 'L25', 'C']

    can_classes = [TS, LS, LeS, RiS, can_A, can_totals] = [
        [] for i in range(6)]
    test = []

    for i in range(len(test_dir)):
        base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        class_path = base_dir + 'TINT_tracks/'
        class_path += '{}_classes.pkl'.format(test_dir[i])

        with open(class_path, 'rb') as f:
            class_df = pickle.load(f)

        test.append(test_names[i])

        can_breakdown = canonical_class_breakdown(class_df)
        [
            can_classes[i].append(can_breakdown[i])
            for i in range(len(can_classes))]

    tilt_sensitivity_df = pd.DataFrame({
        'Test': test, 'Trailing Stratiform Class': TS,
        'Leading Stratiform Class': LS,
        'Left Stratiform Class': LeS,
        'Right Stratiform Class': RiS, 'Ambiguous Inflow': can_A,
        'Total': can_totals})
    tilt_sensitivity_df = tilt_sensitivity_df.set_index('Test')

    tilt_sensitivity_df = tilt_sensitivity_df.drop('Total', axis=1)
    tilt_sensitivity_df = tilt_sensitivity_df.reset_index(drop=True)
    tilt_sensitivity_df.loc[:, 'Test'] = np.array(name_abvs)
    tilt_sensitivity_df = tilt_sensitivity_df.set_index('Test')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 1, 2, 4, 5]]

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    init_fonts()
    tilt_sensitivity_df.plot(
        kind='bar', stacked=False, rot=0,
        yticks=np.arange(0, 0.7, 0.1), width=0.6*5/4,
        fig=fig, ax=ax, color=colors)
    plt.sca(ax)
    plt.ylabel('Ratio [-]', fontsize=14)
    plt.xlabel(None)
    ax.legend(
        loc='lower center', bbox_to_anchor=(0.475, leg_offset_h),
        ncol=ncol, fancybox=True, shadow=True)
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'

    ax.set_yticks(np.arange(0, 0.7, 0.1))
    ax.set_yticks(np.arange(0, .65, 0.05), minor='True')

    ax.grid(which='major', alpha=0.5, axis='y')
    ax.grid(which='minor', alpha=0.2, axis='y')

    # fig_dir = base_dir + 'TINT_figures/'
    # plt.savefig(
    #     fig_dir + 'relative_stratiform_breakdown.png', dpi=200, facecolor='w',
    #     edgecolor='white', bbox_inches='tight')

    return tilt_sensitivity_df


def canonical_class_breakdown(class_df):
    counts_df = pd.DataFrame({'counts': class_df.value_counts()})

    counts_df['ratios'] = counts_df['counts']/(counts_df['counts'].sum())

    q_str = "(offset_type == 'Trailing Stratiform'"
    q_str += "and inflow_type == 'Front Fed')"
    q_str += "or (offset_type == 'Leading Stratiform'"
    q_str += "and inflow_type == 'Rear Fed')"
    q_str += "or (offset_type == 'Parallel Stratiform (Left)'"
    q_str += "and inflow_type == 'Parallel Fed (Right)')"
    q_str += "or (offset_type == 'Parallel Stratiform (Right)'"
    q_str += "and inflow_type == 'Parallel Fed (Left)')"
    TS = counts_df.query(q_str)

    q_str = "(offset_type == 'Leading Stratiform'"
    q_str += "and inflow_type == 'Front Fed')"
    q_str += "or (offset_type == 'Trailing Stratiform'"
    q_str += "and inflow_type == 'Rear Fed')"
    q_str += "or (offset_type == 'Parallel Stratiform (Left)'"
    q_str += "and inflow_type == 'Parallel Fed (Left)')"
    q_str += "or (offset_type == 'Parallel Stratiform (Right)'"
    q_str += "and inflow_type == 'Parallel Fed (Right)')"
    LS = counts_df.query(q_str)

    q_str = "(offset_type == 'Leading Stratiform'"
    q_str += "and inflow_type == 'Parallel Fed (Right)')"
    q_str += "or (offset_type == 'Trailing Stratiform'"
    q_str += "and inflow_type == 'Parallel Fed (Left)')"
    q_str += "or (offset_type == 'Parallel Stratiform (Left)'"
    q_str += "and inflow_type == 'Front Fed')"
    q_str += "or (offset_type == 'Parallel Stratiform (Right)'"
    q_str += "and inflow_type == 'Rear Fed')"
    LeS = counts_df.query(q_str)

    q_str = "(offset_type == 'Leading Stratiform'"
    q_str += "and inflow_type == 'Parallel Fed (Left)')"
    q_str += "or (offset_type == 'Trailing Stratiform'"
    q_str += "and inflow_type == 'Parallel Fed (Right)')"
    q_str += "or (offset_type == 'Parallel Stratiform (Right)'"
    q_str += "and inflow_type == 'Front Fed')"
    q_str += "or (offset_type == 'Parallel Stratiform (Left)'"
    q_str += "and inflow_type == 'Rear Fed')"
    RiS = counts_df.query(q_str)

    q_str = "inflow_type == 'Ambiguous'"
    A = counts_df.query(q_str)

    ratios = [x['ratios'].sum() for x in [TS, LS, LeS, RiS, A]]
    total = counts_df['counts'].sum()
    return ratios + [total]


def plot_categories(
        class_df=None, pope_regime=None, fig=None, ax=None, title=''):

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    if class_df is None:
        class_path = base_dir + 'TINT_tracks/'
        class_path += 'base_classes.pkl'
        with open(class_path, 'rb') as f:
            class_df = pickle.load(f)
    # fig_dir = base_dir + 'TINT_figures/'

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    init_fonts()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 1, 2, 4, 5]]

    counts_df = pd.DataFrame({'count': class_df.value_counts()})
    if pope_regime is not None:
        counts_df = counts_df.xs(pope_regime, level='pope_regime')

    ratio = counts_df['count']/counts_df['count'].sum()
    counts_df['ratio'] = ratio
    grouped_counts = counts_df.groupby(['offset_type', 'inflow_type']).sum()

    pivot_counts = grouped_counts.pivot_table(
        'ratio', ['offset_type'], 'inflow_type')
    pivot_counts.columns.name = None
    pivot_counts = pivot_counts.reindex([
        'Trailing Stratiform', 'Leading Stratiform',
        'Parallel Stratiform (Left)', 'Parallel Stratiform (Right)'])

    columns = pivot_counts.columns.values
    required_columns = [
        'Front Fed', 'Rear Fed',
        'Parallel Fed (Left)', 'Parallel Fed (Right)',
        'Ambiguous']
    for col in required_columns:
        if col not in columns:
            pivot_counts[col] = np.nan

    pivot_counts = pivot_counts[[
        'Front Fed', 'Rear Fed',
        'Parallel Fed (Left)', 'Parallel Fed (Right)',
        'Ambiguous']]
    pivot_counts = pivot_counts.reset_index()
    pivot_counts.loc[:, 'offset_type'] = [
        'Trailing', 'Leading', 'Left', 'Right']
    pivot_counts = pivot_counts.set_index('offset_type')

    pivot_counts = pivot_counts.rename({
        'Parallel Fed (Left)': 'Left Fed',
        'Parallel Fed (Right)': 'Right Fed',
        'Ambiguous': 'Ambiguous Inflow'}, axis=1)

    pivot_counts.plot(
        kind='bar', stacked=False, rot=0, ax=ax,
        yticks=np.arange(0, 0.8, 0.1), width=0.6*5/4,
        color=colors, xlabel=None, legend=False)

    plt.sca(ax)
    plt.ylabel('Ratio [-]')
    plt.xlabel(None)
    ax.set_yticks(np.arange(0, 0.8, 0.05), minor=True)

    # plt.xlabel('Stratiform Offset Category')
    # ax.legend(
    #     loc='upper right',  # bbox_to_anchor=(0.475, -0.575),
    #     ncol=1, fancybox=True, shadow=True)

    ax.grid(which='major', alpha=0.5, axis='y')
    ax.grid(which='minor', alpha=0.2, axis='y')

    total = grouped_counts['count'].sum()

    ax.text(
        0.5, .87, title, ha='center',
        transform=ax.transAxes, size=12, backgroundcolor='1')

    ax.text(
        0.775, .87, 'Total = {}'.format(int(np.round(total))),
        transform=ax.transAxes, size=12, backgroundcolor='1')

    # plt.savefig(
    #     fig_dir + 'categories.png', dpi=200, facecolor='w',
    #     edgecolor='white', bbox_inches='tight')


def pope_comparison(class_df=None):
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    if class_df is None:
        class_path = base_dir + 'TINT_tracks/'
        class_path += 'base_classes.pkl'
        with open(class_path, 'rb') as f:
            class_df = pickle.load(f)
    fig_dir = base_dir + 'TINT_figures/'

    fig, axes = plt.subplots(3, 2, figsize=(12, 7))

    init_fonts()

    titles = [
        'All Regimes', 'Regime 1 (Dry East)', 'Regime 2 (Deep West)',
        'Regime 3 (East)', 'Regime 4 (Shallow West)',
        'Regime 5 (Moist East)']

    plot_categories(
        class_df=class_df, pope_regime=None, fig=fig, ax=axes.flatten()[0],
        title=titles[0])

    for i in range(1, 6):
        plot_categories(
            class_df=class_df, pope_regime=i, fig=fig, ax=axes.flatten()[i],
            title=titles[i])

    # axes.flatten()[0].legend(
    #     loc='upper right',  # bbox_to_anchor=(0.475, -0.575),
    #     ncol=1, fancybox=True, shadow=True)

    make_subplot_labels(axes.flatten())

    plt.sca(axes.flatten()[-2])
    plt.xlabel('Stratiform Offset Category')
    plt.sca(axes.flatten()[-1])
    plt.xlabel('Stratiform Offset Category')

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.6),
        ncol=5, fancybox=True, shadow=True)

    plt.savefig(
        fig_dir + 'pope_breakdown.png', dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')


def pope_comparison_radar(class_df=None, class_path=None):
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    if class_path is None:
        class_path = base_dir + 'TINT_tracks/'
    class_path + 'combined_radar_classes.pkl'
    with open(class_path + 'combined_radar_classes.pkl', 'rb') as f:
        class_df_rad = pickle.load(f)
    with open(class_path + 'combined_ACCESS_classes.pkl', 'rb') as f:
        class_df_ACCESS = pickle.load(f)
    sub_dir = class_path.split('/')[-2]
    fig_dir = base_dir + '/TINT_figures/' + sub_dir + '/'

    fig, axes = plt.subplots(3, 2, figsize=(12, 7))

    init_fonts()

    titles_radar = [
        'Radar: All Regimes', 'Radar: Weak Monsoon',
        'Radar: Active Monsoon']

    titles_ACCESS = [
        'ACCESS-C: All Regimes', 'ACCESS-C: Weak Monsoon',
        'ACCESS-C: Active Monsoon']

    plot_categories(
        class_df=class_df_rad, pope_regime=None, fig=fig, ax=axes[0, 0],
        title=titles_radar[0])

    plot_categories(
        class_df=class_df_ACCESS, pope_regime=None, fig=fig, ax=axes[0, 1],
        title=titles_ACCESS[0])

    for i in range(1, 3):
        plot_categories(
            class_df=class_df_rad, pope_regime=i, fig=fig, ax=axes[i, 0],
            title=titles_radar[i])
        plot_categories(
            class_df=class_df_ACCESS, pope_regime=i, fig=fig,
            ax=axes[i, 1], title=titles_ACCESS[i])

    # axes.flatten()[0].legend(
    #     loc='upper right',  # bbox_to_anchor=(0.475, -0.575),
    #     ncol=1, fancybox=True, shadow=True)

    make_subplot_labels(axes.flatten())

    plt.sca(axes.flatten()[-2])
    plt.xlabel('Stratiform Offset Category')
    plt.sca(axes.flatten()[-1])
    plt.xlabel('Stratiform Offset Category')

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.6),
        ncol=5, fancybox=True, shadow=True)

    plt.savefig(
        fig_dir + 'pope_breakdown_radar.png', dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')


def monsoon_comparison(
        class_df=None, fig=None, ax=None, legend=True, title='',
        required_types=None, titles=None, colors=None):
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    if class_df is None:
        class_path = base_dir + 'TINT_tracks/'
        class_path += 'base_classes.pkl'
        with open(class_path, 'rb') as f:
            class_df = pickle.load(f)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = [colors[i] for i in [0, 1, 2, 4, 6, 7, 8, 5]]

    counts_df = pd.DataFrame({'count': class_df.value_counts()})
    counts_df = counts_df.reset_index('pope_regime')

    pope = counts_df['pope_regime'].values
    pope_dic = {0: 'Not Classified'}
    for i in range(1, 6):
        pope_dic[i] = 'Weak Monsoon'
    pope_dic[2] = 'Active Monsoon'
    monsoon = [pope_dic[p_num] for p_num in pope]
    counts_df['pope_regime'] = monsoon

    counts_df = counts_df.groupby([
        'offset_type', 'inflow_type',
        'tilt_dir', 'prop_dir', 'pope_regime']).sum()
    try:
        counts_df = counts_df.drop('Not Classified', level='pope_regime')
    except KeyError:
        print('No unclassified regime cases.')

    counts_df_inactive = counts_df.xs('Weak Monsoon', level='pope_regime')
    counts_df_active = counts_df.xs('Active Monsoon', level='pope_regime')

    if required_types is None:
        required_types = [
            (
                'Trailing Stratiform', 'Front Fed',
                'Up-Shear Tilted', 'Down-Shear Propagating'),
            (
                'Leading Stratiform', 'Front Fed',
                'Down-Shear Tilted', 'Down-Shear Propagating'),
            (
                'Trailing Stratiform', 'Front Fed',
                'Down-Shear Tilted', 'Up-Shear Propagating'),
            (
                'Trailing Stratiform', 'Rear Fed',
                'Up-Shear Tilted', 'Up-Shear Propagating'),
            (
                'Trailing Stratiform', 'Parallel Fed (Left)',
                'Up-Shear Tilted', 'Down-Shear Propagating'),
            (
                'Parallel Stratiform (Left)', 'Parallel Fed (Left)',
                'Down-Shear Tilted', 'Down-Shear Propagating'),
            (
                'Trailing Stratiform', 'Rear Fed',
                'Down-Shear Tilted', 'Down-Shear Propagating')]
        titles = [
            'Type I: Front-Fed Trailing Stratiform, Up-Shear Tilted, Down-Shear Propagating',
            'Type II: Front-Fed Leading Stratiform, Down-Shear Tilted, Down-Shear Propagating',
            'Type III: Front-Fed Trailing Stratiform, Down-Shear Tilted, Up-Shear Propagating',
            'Type IV: Rear-Fed Trailing Stratiform, Up-Shear Tilted, Up-Shear Propagating',
            'Type V: Left-Fed Trailing Stratiform, Up-Shear Tilted, Down-Shear Propagating',
            'Type VI: Left-Fed Left-Stratiform, Down-Shear Tilted, Down-Shear Propagating',
            'Type VII: Rear-Fed Trailing Stratiform, Down-Shear Tilted, Down-Shear Propagating',
            'All Other Types']

        titles = [
            'Type I', 'Type II', 'Type III', 'Type IV', 'Type V', 'Type VI',
            'Type VII', 'All Other Types']

    total = []
    types = [[] for i in range(len(required_types)+1)]

    for c_df in [counts_df, counts_df_inactive, counts_df_active]:

        c_df = c_df.groupby(
            ['offset_type', 'inflow_type', 'tilt_dir', 'prop_dir']).sum()
        # total_ratio = (c_df['count']/c_df['count'].sum())
        # c_df['total_ratio'] = total_ratio

        c_df = c_df.drop('Ambiguous', level='inflow_type')
        c_df = c_df.drop('Perpendicular Shear', level='tilt_dir')
        c_df = c_df.drop('Ambiguous', level='tilt_dir')
        c_df = c_df.drop('Perpendicular Shear', level='prop_dir')
        c_df = c_df.drop('Ambiguous', level='prop_dir')

        for sys_type in required_types:
            if sys_type not in c_df.index.values.tolist():
                c_df.loc[sys_type, 'count'] = 0

        ratio = c_df['count']/c_df['count'].sum()
        c_df['ratio'] = ratio
        # c_df.sort_values('ratio', axis=0, ascending=False)

        # c_df.sort_values('count', axis=0, ascending=False)
        # import pdb; pdb.set_trace()

        total.append(int(c_df['count'].sum()))

        for i in range(len(required_types)):
            types[i].append(c_df.loc[required_types[i], 'ratio'])

        types[-1].append(
            1 - c_df.loc[[rt for rt in required_types]]['ratio'].sum())

    categories = ['All', 'Weak Monsoon', 'Active Monsoon']
    ratios_dict = {'Wet Season Regime': categories}
    for i in range(len(required_types)):
        ratios_dict[titles[i]] = types[i]
    ratios_dict[titles[-1]] = types[-1]
    ratios_df = pd.DataFrame(ratios_dict)
    ratios_df = ratios_df.set_index('Wet Season Regime')

    ratios_df.plot(
        kind='bar', stacked=False, rot=0, fontsize=12, ax=ax,
        yticks=np.arange(0, 1.1, 0.1), width=0.65*4/4,
        color=colors, legend=False)
    ax.set_xlabel(None)
    # ax.xaxis.set_label_coords(.5, -0.15)
    ax.set_ylabel('Ratio [-]', fontsize=14)
    if legend:
        ax.legend(
            loc='lower center', bbox_to_anchor=(.475, -0.5),
            ncol=2, fancybox=True, shadow=True)
    ax.set_yticks(np.arange(0, 1+0.05, 0.05), minor=True)
    ax.grid(which='minor', alpha=0.2, axis='y')
    ax.grid(which='major', alpha=0.5, axis='y')

    lab_h = 0.90
    tot_lab = ['Total = {}'.format(tot) for tot in total]
    ax.text(
        0.45, 1.04, title, transform=ax.transAxes, size=12, ha='center')
    ax.text(
        0.05, lab_h, tot_lab[0], transform=ax.transAxes, size=12,
        backgroundcolor='1')
    ax.text(
        0.35, lab_h, tot_lab[1], transform=ax.transAxes, size=12,
        backgroundcolor='1')
    ax.text(
        0.7, lab_h, tot_lab[2], transform=ax.transAxes, size=12,
        backgroundcolor='1')

    return ratios_df
