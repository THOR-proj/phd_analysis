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

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

base_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/CPOL/'
save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_tracks/'
fig_dir = '/home/student.unimelb.edu.au/shorte1/Documents/TINT_figures/'
ERA5_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/era5/'
ERA5_dir += 'pressure-levels/reanalysis/'
WRF_dir = '/media/shorte1/Ewan\'s Hard Drive/phd/data/caine_WRF_data/'


def create_oper_radar_counts(
        save_dir, tracks_base_dir, class_thresh=None,
        excl_thresh=None, non_linear_conds=None, exclusions=None,
        morning_only=False, radar=[42, 63, 77], years=range(2012, 2023)):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('Creating new directory.')

    included_radar = [
        2, 3, 4, 5, 6, 8, 9, 14, 16, 17,
        19, 22, 23, 24, 25, 27, 28, 29, 31, 32,
        33, 36, 37, 40, 41, 42, 44, 46, 48,
        49, 50, 52, 53, 54, 55, 56, 63, 64,
        66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77]

    for r in included_radar:
        for y in years:

            dir_ry = tracks_base_dir + '/radar_{}/{}/'.format(r, y)

            print('Getting classes for radar {}, year {}.'.format(r, y))
            class_df = get_counts_radar(
                base_dir='/home/student.unimelb.edu.au/shorte1/Documents/CPOL_analysis/',
                tracks_dir=dir_ry, class_thresh=class_thresh,
                excl_thresh=excl_thresh, non_linear=non_linear_conds,
                years=[y], radar=r, exclusions=exclusions,
                morning_only=morning_only)

            if class_df is not None:
                out_file_name = dir_ry + '/{:02d}_{:04d}_classes.pkl'.format(r, y)
                with open(out_file_name, 'wb') as f:
                    pickle.dump(class_df, f)

    classes = []
    for r in included_radar:
        for y in years:
            dir_ry = tracks_base_dir + '/radar_{}/{}/'.format(r, y)
            fn = dir_ry + '/{:02d}_{:04d}_classes.pkl'.format(r, y)
            try:
                with open(fn, 'rb') as f:
                    df = pickle.load(f)
                df = pd.concat({r: df}, names=['radar'])
                classes.append(df)
            except FileNotFoundError:
                print('Missing file. Skipping.')
                continue

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
        fake_pope=False, exclusions=None, morning_only=False):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    [
        year_list, uid, time, offset_type, rel_offset_type, inflow_type,
        tilt_dir, prop_dir, pope_regime, hour] = [
        [] for i in range(10)]
    for year in years:
        tracks_obj = load_year(year, tracks_dir=tracks_dir)

        # restrict to morning only...
        if morning_only:
            print('Restricting to hour UTC < 12')
            tracks_obj.tracks = tracks_obj.tracks.iloc[
                tracks_obj.tracks.index.get_level_values('time').hour < 12]
            tracks_obj.sysyem_tracks = tracks_obj.system_tracks.iloc[
                tracks_obj.system_tracks.index.get_level_values('time').hour < 12]
            tracks_obj.tracks_class = tracks_obj.tracks_class.iloc[
                tracks_obj.tracks_class.index.get_level_values('time').hour < 12]
            tracks_obj.exclusions = tracks_obj.exclusions.iloc[
                tracks_obj.exclusions.index.get_level_values('time').hour < 12]

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
        years=[2013], radar=63, fake_pope=True, exclusions=None,
        morning_only=False):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    [
        year_list, month_list, uid, time, offset_type, rel_offset_type,
        inflow_type, tilt_dir, prop_dir, pope_regime, hour] = [
        [] for i in range(11)]
    for y in years:
        for m in range(1, 13):
            try:
                # import pdb; pdb.set_trace()
                tracks_obj = load_op_month(
                    y, m, radar, tracks_dir=tracks_dir + '/{}/'.format(y))
            except FileNotFoundError:
                print('No tracks file. Skipping')
                continue

            if len(tracks_obj.tracks) < 6:
                print('No tracks. Skipping')
                continue

            if morning_only:
                print('Restricting to hour UTC < 12')
                tracks_obj.tracks = tracks_obj.tracks.iloc[
                    tracks_obj.tracks.index.get_level_values('time').hour < 12]
                tracks_obj.sysyem_tracks = tracks_obj.system_tracks.iloc[
                    tracks_obj.system_tracks.index.get_level_values('time').hour < 12]
                tracks_obj.tracks_class = tracks_obj.tracks_class.iloc[
                    tracks_obj.tracks_class.index.get_level_values('time').hour < 12]
                tracks_obj.exclusions = tracks_obj.exclusions.iloc[
                    tracks_obj.exclusions.index.get_level_values('time').hour < 12]

            print('Getting new exclusions.')
            try:
                tracks_obj = redo_exclusions(
                    tracks_obj, class_thresh, excl_thresh)
            except KeyError:
                print('No system tracks. Skipping.')
                continue

            print('Adding Pope monsoon regime.')
            tracks_obj = add_monsoon_regime(
                tracks_obj, base_dir=base_dir, fake_pope=fake_pope)

            # Get raw and res sample

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
                    year_list.append(y)
                    month_list.append(m)
                    uid.append(i)
                    time.append(scan_label[j]*10)
                    offset_type.append(offset[j])
                    rel_offset_type.append(rel_offset[j])
                    inflow_type.append(inflow[j])
                    tilt_dir.append(tilt[j])
                    prop_dir.append(prop[j])
                    pope_regime.append(int(pope[j]))
                    hour.append(hours[j])

    if len(year_list) > 0:
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
    else:
        class_df = None

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

    if max_tick < 250:
        ax1.set_yticks(np.arange(0, max_tick+50, 50), minor=False)
        ax1.set_yticks(np.arange(0, max_tick+25, 25), minor=True)
    else:
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


def make_subplot_labels(axes, size=16, x_shift=-0.15, y_shift=0):
    labels = list(string.ascii_lowercase)
    labels = [label + ')' for label in labels]
    for i in range(len(axes)):
        axes[i].text(
            x_shift, 1.0+y_shift, labels[i], transform=axes[i].transAxes,
            size=size)


def get_save_dir(save_dir):
    if save_dir is None:
        save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        save_dir += 'TINT_figures/'
    return save_dir


def plot_offsets(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False, time_thresh=None, linewidth=2):
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
    # import pdb; pdb.set_trace()
    try:
        TS = offset_types.xs('Trailing Stratiform', level='offset_type')
    except KeyError:
        TS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        LS = offset_types.xs('Leading Stratiform', level='offset_type')
    except KeyError:
        LS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        LeS = offset_types.xs('Parallel Stratiform (Left)', level='offset_type')
    except KeyError:
        LeS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        RiS = offset_types.xs('Parallel Stratiform (Right)', level='offset_type')
    except KeyError:
        RiS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
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
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, LS.loc[x], label='Leading Stratiform', color=colors[1],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, LeS.loc[x], label='Left Stratiform', color=colors[2],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, RiS.loc[x], label='Right Stratiform', color=colors[4],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, offset_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle, linewidth=linewidth)

    ax2.plot(
        x, (TS/offset_totals).loc[x],
        label='Trailing Stratiform', color=colors[0], linestyle=linestyle,
        linewidth=linewidth)
    ax2.plot(
        x, (LeS/offset_totals).loc[x], label='Left Stratiform',
        color=colors[2], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (RiS/offset_totals).loc[x], label='Right Stratiform',
        color=colors[4], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (LS/offset_totals).loc[x],
        label='Leading Stratiform', color=colors[1], linestyle=linestyle,
        linewidth=linewidth)

    if time_thresh is not None:
        ax2.plot([time_thresh, time_thresh], [0, 1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(offset_totals.loc[x].values), maximum),
        legend=legend, diurnal=diurnal)
    totals = [y.loc[x].sum() for y in [TS, LS, LeS, RiS, offset_totals]]
    return totals


def plot_relative_offsets(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False, time_thresh=None, linewidth=2):
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
    try:
        TS = offset_types.xs(
            'Relative Trailing Stratiform', level='rel_offset_type')
    except KeyError:
        TS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        LS = offset_types.xs(
        'Relative Leading Stratiform', level='rel_offset_type')
    except KeyError:
        LS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        LeS = offset_types.xs(
            'Relative Parallel Stratiform (Left)', level='rel_offset_type')
    except KeyError:
        LeS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        RiS = offset_types.xs(
            'Relative Parallel Stratiform (Right)', level='rel_offset_type')
    except KeyError:
        RiS = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
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
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, LS.loc[x], label='Relative Leading Stratiform', color=colors[1],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, LeS.loc[x], label='Relative Left Stratiform',
        color=colors[2], linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, RiS.loc[x], label='Relative Right Stratiform',
        color=colors[4], linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, offset_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle, linewidth=linewidth)

    ax2.plot(
        x, (TS/offset_totals).loc[x],
        label='Relative Trailing Stratiform', color=colors[0],
        linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (LeS/offset_totals).loc[x],
        label='Relative Left Stratiform',
        color=colors[2], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (RiS/offset_totals).loc[x],
        label='Relative Right Stratiform',
        color=colors[4], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (LS/offset_totals).loc[x],
        label='Relative Leading Stratiform', color=colors[1],
        linestyle=linestyle, linewidth=linewidth)

    if time_thresh is not None:
        ax2.plot([time_thresh, time_thresh], [0, 1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(offset_totals.loc[x].values), maximum),
        legend=legend, diurnal=diurnal)
    totals = [y.loc[x].sum() for y in [TS, LS, LeS, RiS, offset_totals]]
    return totals


def plot_inflows(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False, time_thresh=None, linewidth=2):
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
    try:
        A = inflow_types.xs('Ambiguous', level='inflow_type')
    except:
        A = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        FF = inflow_types.xs('Front Fed', level='inflow_type')
    except:
        FF = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        RF = inflow_types.xs('Rear Fed', level='inflow_type')
    except KeyError:
        RF = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        LeF = inflow_types.xs('Parallel Fed (Left)', level='inflow_type')
    except KeyError:
        LeF = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        RiF = inflow_types.xs('Parallel Fed (Right)', level='inflow_type')
    except KeyError:
        RiF = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
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
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, RF.loc[x], label='Rear Fed', color=colors[1],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, LeF.loc[x], label='Left Fed', color=colors[2],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, RiF.loc[x], label='Right Fed', color=colors[4],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, A.loc[x], label='Ambiguous', color=colors[5],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, inflow_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle, linewidth=linewidth)

    ax2.plot(
        x, (FF/inflow_totals).loc[x], label='Front Fed', color=colors[0],
        linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (RF/inflow_totals).loc[x], label='Rear Fed', color=colors[1],
        linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (LeF/inflow_totals).loc[x], label='Left Fed',
        color=colors[2], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (RiF/inflow_totals).loc[x], label='Right Fed',
        color=colors[4], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (A/inflow_totals).loc[x], label='Ambiguous', color=colors[5],
        linestyle=linestyle, linewidth=linewidth)
    set_ticks(
        ax1, ax2, max(np.max(inflow_totals.loc[x].values), maximum),
        legend=legend, diurnal=diurnal)

    if time_thresh is not None:
        ax2.plot([time_thresh, time_thresh], [0, 1], '--', color='gray')

    totals = [y.loc[x].sum() for y in [FF, RF, LeF, RiF, A, inflow_totals]]
    return totals


def plot_tilts(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False, time_thresh=None, linewidth=2):
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
    try:
        SP = tilt_types.xs('Perpendicular Shear', level='tilt_dir')
    except KeyError:
        SP = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        UST = tilt_types.xs('Up-Shear Tilted', level='tilt_dir')
    except:
        UST = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        DST = tilt_types.xs('Down-Shear Tilted', level='tilt_dir')
    except:
        DST = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
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
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, DST.loc[x], label='Down-Shear Tilted', color=colors[1],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, SP.loc[x], label='Shear Perpendicular', color=colors[5],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, tilt_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle, linewidth=linewidth)

    ax2.plot(
        x, (UST/tilt_totals).loc[x], label='Up-Shear Tilted',
        color=colors[0], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (DST/tilt_totals).loc[x], label='Down-Shear Tilted',
        color=colors[1], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (SP/tilt_totals).loc[x], label='Ambiguous', color=colors[5],
        linestyle=linestyle, linewidth=linewidth)

    if time_thresh is not None:
        ax2.plot([time_thresh, time_thresh], [0, 1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(tilt_totals.loc[x].values), maximum),
        leg_columns=2, legend=legend, diurnal=diurnal)

    totals = [y.loc[x].sum() for y in [UST, DST, SP, tilt_totals]]
    return totals


def plot_propagations(
        class_df, save_dir=None, append_time=False, fig=None,
        ax1=None, ax2=None, linestyle='-', legend=True, maximum=0,
        diurnal=False, time_thresh=None, linewidth=2):
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
    try:
        SP = tilt_types.xs('Perpendicular Shear', level='prop_dir')
    except:
        SP = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        USP = tilt_types.xs('Up-Shear Propagating', level='prop_dir')
    except:
        USP = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
    try:
        DSP = tilt_types.xs('Down-Shear Propagating', level='prop_dir')
    except:
        DSP = pd.DataFrame({'time': [0], 'counts': [0]}).set_index('time')
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
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, USP.loc[x], label='Up-Shear Propagating', color=colors[1],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, SP.loc[x], label='Shear Perpendicular', color=colors[5],
        linestyle=linestyle, linewidth=linewidth)
    ax1.plot(
        x, prop_totals.loc[x], label='Total', color=colors[3],
        linestyle=linestyle, linewidth=linewidth)

    ax2.plot(
        x, (DSP/prop_totals).loc[x], label='Down-Shear Propagating',
        color=colors[0], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (USP/prop_totals).loc[x], label='Up-Shear Propagating',
        color=colors[1], linestyle=linestyle, linewidth=linewidth)
    ax2.plot(
        x, (SP/prop_totals).loc[x], label='Shear Perpendicular',
        color=colors[5], linestyle=linestyle, linewidth=linewidth)

    if time_thresh is not None:
        ax2.plot([time_thresh, time_thresh], [0, 1.1], '--', color='gray')

    set_ticks(
        ax1, ax2, max(np.max(prop_totals.loc[x].values), maximum),
        leg_columns=2, legend=legend, diurnal=diurnal)

    totals = [y.loc[x].sum() for y in [DSP, USP, SP, prop_totals]]
    return totals


def plot_comparison(
        test_dir=None, suffix='', maximums=None, title=None,
        time_threshes=None):

    if test_dir is None:
        test_dir = ['base', 'two_levels']
    if maximums is None:
        maximums = [800, 600, 800, 600, 600]
    if time_threshes is None:
        time_thresh_ACCESS = [180, 120, 120, 150, 120]
        time_thresh_radar = [180, 120, 120, 150, 120]
        time_threshes = [time_thresh_ACCESS, time_thresh_radar]

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
            linestyle=linestyles[i], legend=legends[i], maximum=maximums[0],
            time_thresh=time_thresh_radar[0])

        # import pdb; pdb.set_trace()

        plot_relative_offsets(
            class_df, fig_dir, fig=fig_1, ax1=axes_1[1][0], ax2=axes_1[1][1],
            linestyle=linestyles[i], legend=legends[i],
            maximum=maximums[1], time_thresh=time_thresh_radar[1])

        plot_inflows(
            class_df, fig_dir, fig=fig_1, ax1=axes_1[2][0], ax2=axes_1[2][1],
            linestyle=linestyles[i], legend=legends[i], maximum=maximums[2],
            time_thresh=time_thresh_radar[2])

        plt.subplots_adjust(hspace=0.775)
        make_subplot_labels(axes_1.flatten())

        if title is not None:
            plt.suptitle(title, y=.925)

        plt.savefig(
            fig_dir + 'offsets_inflows_comparison{}.png'.format(suffix),
            dpi=200, facecolor='w',
            edgecolor='white', bbox_inches='tight')

        plot_tilts(
            class_df, fig_dir, fig=fig_2, ax1=axes_2[0][0], ax2=axes_2[0][1],
            linestyle=linestyles[i], legend=legends[i], maximum=maximums[3],
            time_thresh=time_thresh_radar[3])

        plot_propagations(
            class_df, fig_dir, fig=fig_2, ax1=axes_2[1][0], ax2=axes_2[1][1],
            linestyle=linestyles[i], legend=legends[i],
            maximum=maximums[4], time_thresh=time_thresh_radar[4])

        make_subplot_labels(axes_2.flatten())
        plt.subplots_adjust(hspace=0.775)

        if title is not None:
            plt.suptitle(title, y=.945)

        plt.savefig(
            fig_dir + 'tilts_propagations_comparison{}.png'.format(suffix),
            dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

        plt.close('all')


def plot_all_oldschool(
        test_dir=None, test_names=None, diurnal=False,
        time_threshes=None, fig_style='paper'):

    # import pdb; pdb.set_trace()

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
    if time_threshes is None:
        time_threshes = [[None]*5]*len(test_names)

    test = []
    [TS, LS, LeS, RiS, offset_total] = [[] for i in range(5)]
    [RTS, RLS, RLeS, RRiS, rel_offset_total] = [[] for i in range(5)]
    [FF, RF, LeF, RiF, A_inflow, inflow_total] = [[] for i in range(6)]
    [UST, DST, A_tilt, tilt_total] = [[] for i in range(4)]
    [USP, DSP, A_prop, prop_total] = [[] for i in range(4)]

    if fig_style == 'paper':
        print('Paper style. Using defaults.')
        plt.style.use('default')
        init_fonts()
        fc = 'w'
        linewidth = 1.75
    else:
        print('Dark Mode.')
        plt.style.use("dark_background")
        fc = 'k'
        linewidth = 2.5

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

        # import pdb; pdb.set_trace()

        if fig_style == 'paper':
            fig, axes = initialise_fig(height=10, n_subplots=6)
        else:
            fig, axes = initialise_fig(height=12, n_subplots=6)

        offset_summary = plot_offsets(
            class_df, fig_dir, fig=fig, ax1=axes[0][0], ax2=axes[0][1],
            diurnal=diurnal, time_thresh=time_threshes[i][0],
            linewidth=linewidth)
        TS.append(offset_summary[0].values[0])
        LS.append(offset_summary[1].values[0])
        LeS.append(offset_summary[2].values[0])
        RiS.append(offset_summary[3].values[0])
        offset_total.append(offset_summary[4].values[0])

        rel_offset_summary = plot_relative_offsets(
            class_df, fig_dir, fig=fig, ax1=axes[1][0], ax2=axes[1][1],
            diurnal=diurnal, time_thresh=time_threshes[i][1],
            linewidth=linewidth)
        RTS.append(rel_offset_summary[0].values[0])
        RLS.append(rel_offset_summary[1].values[0])
        RLeS.append(rel_offset_summary[2].values[0])
        RRiS.append(rel_offset_summary[3].values[0])
        rel_offset_total.append(rel_offset_summary[4].values[0])

        inflow_summary = plot_inflows(
            class_df, fig_dir, fig=fig, ax1=axes[2][0], ax2=axes[2][1],
            diurnal=diurnal, time_thresh=time_threshes[i][2],
            linewidth=linewidth)
        FF.append(inflow_summary[0].values[0])
        RF.append(inflow_summary[1].values[0])
        LeF.append(inflow_summary[2].values[0])
        RiF.append(inflow_summary[3].values[0])
        A_inflow.append(inflow_summary[4].values[0])
        inflow_total.append(inflow_summary[5].values[0])

        plt.subplots_adjust(hspace=0.775)
        if fig_style == 'paper':
            make_subplot_labels(axes.flatten())

        plt.savefig(
            fig_dir + 'offsets_inflows' + diurnal*'_diurnal' + '.svg',
            dpi=200, facecolor=fc, edgecolor=fc, bbox_inches='tight')

        fig, axes = initialise_fig(height=6, n_subplots=4)

        if fig_style == 'paper':
            fig, axes = initialise_fig(height=6, n_subplots=4)
        else:
            fig, axes = initialise_fig(height=7, n_subplots=4)

        tilt_summary = plot_tilts(
            class_df, fig_dir, fig=fig, ax1=axes[0][0], ax2=axes[0][1],
            diurnal=diurnal, time_thresh=time_threshes[i][3],
            linewidth=linewidth)
        UST.append(tilt_summary[0].values[0])
        DST.append(tilt_summary[1].values[0])
        A_tilt.append(tilt_summary[2].values[0])
        tilt_total.append(tilt_summary[3].values[0])

        prop_summary = plot_propagations(
            class_df, fig_dir, fig=fig, ax1=axes[1][0], ax2=axes[1][1],
            diurnal=diurnal, time_thresh=time_threshes[i][4],
            linewidth=linewidth)
        USP.append(prop_summary[1].values[0])
        DSP.append(prop_summary[0].values[0])
        A_prop.append(prop_summary[2].values[0])
        prop_total.append(prop_summary[3].values[0])

        if fig_style == 'paper':
            make_subplot_labels(axes.flatten())
        plt.subplots_adjust(hspace=0.775)

        plt.savefig(
            fig_dir + 'tilts_propagations' + diurnal*'_diurnal' + '.svg',
            dpi=200, facecolor=fc,
            edgecolor=fc, bbox_inches='tight')

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


def plot_all(
        diurnal=False, time_threshes=None, radars=[42, 63, 77]):

    if time_threshes is None:
        time_threshes = [[None]*5]

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    class_path = base_dir + 'TINT_tracks/national/combined_radar_classes.pkl'

    [TS, LS, LeS, RiS, offset_total] = [[] for i in range(5)]
    [RTS, RLS, RLeS, RRiS, rel_offset_total] = [[] for i in range(5)]
    [FF, RF, LeF, RiF, A_inflow, inflow_total] = [[] for i in range(6)]
    [UST, DST, A_tilt, tilt_total] = [[] for i in range(4)]
    [USP, DSP, A_prop, prop_total] = [[] for i in range(4)]

    with open(class_path, 'rb') as f:
        class_df_all = pickle.load(f)

    # for r in radars:
    for radar in radars:

        print('Plotting radar {}.'.format(radar))

        fig_dir = base_dir + 'TINT_figures/national/radar_{}/'.format(radar)

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            print('Creating new directory.')

        try:
            # import pdb; pdb.set_trace()
            class_df = class_df_all.loc[radar, :, :, :, :]

        except KeyError:
            print('Missing data. Skipping')
            return

        fig, axes = initialise_fig(height=10, n_subplots=6)

        offset_summary = plot_offsets(
            class_df, fig_dir, fig=fig, ax1=axes[0][0], ax2=axes[0][1],
            diurnal=diurnal, time_thresh=None)
        TS.append(offset_summary[0].values[0])
        LS.append(offset_summary[1].values[0])
        LeS.append(offset_summary[2].values[0])
        RiS.append(offset_summary[3].values[0])
        offset_total.append(offset_summary[4].values[0])

        rel_offset_summary = plot_relative_offsets(
            class_df, fig_dir, fig=fig, ax1=axes[1][0], ax2=axes[1][1],
            diurnal=diurnal, time_thresh=None)
        RTS.append(rel_offset_summary[0].values[0])
        RLS.append(rel_offset_summary[1].values[0])
        RLeS.append(rel_offset_summary[2].values[0])
        RRiS.append(rel_offset_summary[3].values[0])
        rel_offset_total.append(rel_offset_summary[4].values[0])

        inflow_summary = plot_inflows(
            class_df, fig_dir, fig=fig, ax1=axes[2][0], ax2=axes[2][1],
            diurnal=diurnal, time_thresh=None)
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
            diurnal=diurnal, time_thresh=None)
        UST.append(tilt_summary[0].values[0])
        DST.append(tilt_summary[1].values[0])
        A_tilt.append(tilt_summary[2].values[0])
        tilt_total.append(tilt_summary[3].values[0])

        prop_summary = plot_propagations(
            class_df, fig_dir, fig=fig, ax1=axes[1][0], ax2=axes[1][1],
            diurnal=diurnal, time_thresh=None)
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
        'Radar': radars, 'Trailing Stratiform': TS,
        'Leading Stratiform': LS, 'Left Stratiform': LeS,
        'Right Stratiform': RiS, 'Total': offset_total})
    offset_sensitivity_df = offset_sensitivity_df.set_index('Radar')

    inflow_sensitivity_df = pd.DataFrame({
        'Radar': radars, 'Front Fed': FF,
        'Rear Fed': RF, 'Left Fed': LeF,
        'Right Fed': RiF, 'Ambiguous': A_inflow,
        'Total': inflow_total})
    inflow_sensitivity_df = inflow_sensitivity_df.set_index('Radar')

    tilt_sensitivity_df = pd.DataFrame({
        'Radar': radars, 'Up-Shear Tilted': UST,
        'Down-Shear Tilted': DST, 'Shear-Perpendicular Tilted': A_tilt,
        'Total': tilt_total})
    tilt_sensitivity_df = tilt_sensitivity_df.set_index('Radar')

    prop_sensitivity_df = pd.DataFrame({
        'Radar': radars, 'Down-Shear Propagating': DSP,
        'Up-Shear Propagating': USP, 'Shear-Perpendicular Propagating': A_prop,
        'Total': prop_total})
    prop_sensitivity_df = prop_sensitivity_df.set_index('Radar')

    rel_offset_sensitivity_df = pd.DataFrame({
        'Radar': radars, 'Relative Trailing Stratiform': RTS,
        'Relative Leading Stratiform': RLS,
        'Relative Left Stratiform': RLeS,
        'Relative Right Stratiform': RRiS,
        'Total': rel_offset_total})
    rel_offset_sensitivity_df = rel_offset_sensitivity_df.set_index('Radar')

    radar_dfs = [
        offset_sensitivity_df, inflow_sensitivity_df,
        tilt_sensitivity_df, prop_sensitivity_df, rel_offset_sensitivity_df]

    return radar_dfs


def plot_sensitivities_oldschool(
        sen_dfs, test_dirs, raw, res, name_abvs=None, suff='', alt_layout=False,
        fig_style='paper', raw_max=40000, res_max=1000, raw_step=4000, res_step=100):

    if name_abvs is None:
        name_abvs = [
            'Base', 'C2', 'C4', '4L', 'NS', 'LR', 'S4', 'RV4', 'T15',
            'S15', 'A2', 'B5', 'L50', 'L25', 'C']

    ss_df = pd.DataFrame({'Raw Sample': raw, 'Restricted Sample': res}, index=sen_dfs[0].index)

    if fig_style == 'paper':
        print('Paper style. Using defaults.')
        plt.style.use('default')
        fc = 'w'
        width = 12
        init_fonts()
        wspace = .3
    else:
        print('Dark Mode.')
        plt.style.use("dark_background")
        fc = 'k'
        width = 16
        wspace = .2

    if alt_layout:
        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    else:
        fig, axes = plt.subplots(3, 2, figsize=(width, 9))
    colors = get_colors()
    offset_c = [colors[i] for i in [0, 1, 2, 4]]
    inflow_c = [colors[i] for i in [0, 1, 2, 4, 5]]
    tilt_c = [colors[i] for i in [0, 1, 5]]
    prop_c = [colors[i] for i in [0, 1, 5]]
    clists = [offset_c, inflow_c, tilt_c, prop_c, offset_c]
    offset_1 = -.575
    leg_offset = [offset_1, offset_1, offset_1, offset_1, offset_1]
    leg_offset_x = [.45] * 4 + [.45]
    leg_columns = [2, 3, 2, 2, 2]

    # plot raw and res
    # import pdb; pdb.set_trace()
    ncol = 2
    ax = axes[0, 0]
    x = np.arange(0, 15)

    b1 = ax.bar(
        x-0.7*ncol/12, ss_df['Raw Sample'].values,
        width=0.7*ncol/6, label='Raw Sample', zorder=1)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, raw_max+raw_step/2, raw_step))
    ax.set_yticks(np.arange(0, raw_max+raw_step/4, raw_step/2), minor=True)
    ax.set_xticklabels(name_abvs)
    ax.set_ylabel('Raw Sample Size [-]')
    ax.set_xlim([-.5, 14.5])
    ax.set_ylim([0, raw_max])
    for k in range(14):
        ax.plot(
            [.5+k, .5+k], [0, raw_max], 'grey', alpha=.6, linewidth=1, zorder=1)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax_alt = ax.twinx()
    b2 = ax_alt.bar(
        x+0.7*ncol/12, ss_df['Restricted Sample'].values,
        width=0.7*ncol/6, color='Tab:orange',
        label='Restricted Sample', zorder=1)

    ax_alt.set_yticks(np.arange(0, res_max+res_step/2, res_step))
    ax_alt.set_yticks(np.arange(0, res_max+res_step/4, res_step/2), minor=True)
    
    ax_alt.set_ylabel('Restricted Sample Size [-]')

    # added these three lines
    leg_h = [b1[0], b2[0]]
    labs = ['Raw Sample', 'Restricted Sample']

    # ax.grid(which='major', alpha=.8, axis='y', color='grey', linewidth=1, zorder=2)
    ax_alt.grid(
        which='major', alpha=.8, axis='y', color='grey', linewidth=1, zorder=2)

    ax_alt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.legend(
        leg_h, labs, loc='lower center',
        bbox_to_anchor=(leg_offset_x[0], -0.45),
        ncol=2, fancybox=True, shadow=True)

    for i in range(0, len(sen_dfs)):

        base_ratios = sen_dfs[i].drop('Total', axis=1)
        c_list = clists[i]
        for c in base_ratios.columns:
            base_ratios.loc[:, c] = (
                base_ratios.loc[:, c]/sen_dfs[i].loc[:, 'Total'])

        base_ratios = base_ratios.reset_index(drop=True)
        base_ratios.loc[:, 'Test'] = np.array(name_abvs)
        base_ratios = base_ratios.set_index('Test')
        max_rat = np.ceil(base_ratios.max().max()*10)/10

        ax = axes[(i+1) // 2, (i+1) % 2]
        ncol = len(base_ratios.columns)
        ax = base_ratios.plot(
            kind='bar', stacked=False, fontsize=12, rot=0, ax=ax,
            yticks=np.arange(0, max_rat+0.2, 0.1), width=0.7*ncol/4,
            color=c_list)
        ax.set_xlabel(None)
        ax.xaxis.set_label_coords(.5, -0.15)

        for k in range(14):
            ax.plot(
                [.5+k, .5+k], [0, max_rat+.1], 'grey', alpha=.6, linewidth=1)
        ax.set_xlim([-.5, 14.5])
        ax.set_ylim([0, max_rat+.1])

        ax.set_ylabel('Ratio [-]', fontsize=12)
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(leg_offset_x[i], leg_offset[i]),
            ncol=leg_columns[i], fancybox=True, shadow=True)
        ax.set_yticks(np.arange(0, max_rat+0.05, 0.05), minor=True)
        ax.tick_params(axis='x', length=0)
        # ax.grid(which='minor', alpha=0.2, axis='y', color='w')
        ax.grid(which='major', alpha=.8, axis='y', color='grey', linewidth=1)
    #
    # category_breakdown(
    #     fig=fig, ax=axes[2, 1], leg_offset_h=-.71, test_dir=test_dirs,
    #     test_names=name_abvs, name_abvs=name_abvs)
    plt.subplots_adjust(hspace=0.65)

    plt.subplots_adjust(wspace=wspace)
    make_subplot_labels(axes.flatten())

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/'
    plt.savefig(
        fig_dir + 'total_ratio_sensitivities{}.png'.format(suff),
        dpi=200, facecolor=fc, edgecolor=fc, bbox_inches='tight')


def plot_sensitivities(
        sen_dfs, suff='', alt_layout=False):

    if alt_layout:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(6, 1, figsize=(13, 20))
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
    #import pdb; pdb.set_trace()
    for i in range(len(sen_dfs)):
        base_ratios = sen_dfs[i].drop('Total', axis=1)
        c_list = clists[i]
        for c in base_ratios.columns:
            base_ratios.loc[:, c] = (
                base_ratios.loc[:, c]/sen_dfs[i].loc[:, 'Total'])

        max_rat = np.ceil(base_ratios.max().max()*10)/10

        ax = axes[i]
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

    # category_breakdown(
    #     fig=fig, ax=axes[2, 1], leg_offset_h=-.71, test_dir=test_dirs,
    #     test_names=name_abvs, name_abvs=name_abvs)
    plt.subplots_adjust(hspace=0.65)
    make_subplot_labels(axes.flatten())

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/national/'
    plt.savefig(
        fig_dir + 'total_ratio_sensitivities{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def plot_pie_map(radar_dfs):

    radar_info = pd.read_csv(
        '~/Documents/phd/AURA_analysis/radar_site_list.csv', index_col=0)

    radars = [
        2, 3, 4, 5, 6, 8, 9, 14, 16, 17, 19, 22, 23, 24, 25, 27, 28, 29, 31, 32,
        33, 36, 37, 40, 41, 42, 44, 46, 48, 49, 50, 52, 53, 54, 55, 56, 63, 64,
        66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77]

    offset_dic = dict(zip(
        radars, [[0, 0] for i in range(len(radars))]))
    offset_dic[24] = [-.5, 2]
    offset_dic[66] = [3, 1]
    offset_dic[50] = [-2, 1.5]
    offset_dic[71] = [-2.25, 1]
    offset_dic[54] = [3.25, -.5]
    offset_dic[3] = [2, -3.5]
    offset_dic[76] = [2, 1.5]
    offset_dic[37] = [-2, -1.5]
    offset_dic[64] = [1.5, 1.75]
    offset_dic[46] = [-1.5, -1.75]
    offset_dic[49] = [-3, 2]
    offset_dic[68] = [-1.5, -3.5]
    offset_dic[55] = [-1.25, 1.25]
    offset_dic[69] = [-3.75, .75]
    offset_dic[32] = [2.5, 1.5]
    offset_dic[40] = [.5, -4]
    offset_dic[28] = [2.5, -1]
    offset_dic[8] = [2, 2.5]
    offset_dic[23] = [1.25, 2.5]
    offset_dic[41] = [0, 2.5]
    offset_dic[19] = [-1, 2]
    offset_dic[29] = [1, 2]
    offset_dic[73] = [-3, 0]
    offset_dic[22] = [1, 2]
    offset_dic[72] = [-1.75, 1.25]
    offset_dic[56] = [-2, 0]
    offset_dic[63] = [-2, 0]
    offset_dic[42] = [-2, -2]

    good_radar = radar_info.loc[radars]
    # import pdb; pdb.set_trace()
    # new_order = [0, 4, 1, 2, 3]
    # radar_dfs_alt = [radar_dfs[i] for i in new_order]

    init_fonts()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    offset_c = [colors[i] for i in [0, 1, 2, 4]]
    inflow_c = [colors[i] for i in [0, 1, 2, 4, 5]]
    tilt_c = [colors[i] for i in [0, 1, 5]]
    prop_c = [colors[i] for i in [0, 1, 5]]
    clists = [offset_c, inflow_c, tilt_c, prop_c, offset_c]
    
    names = [
        'Stratiform-Offset', 'Inflow', 'Tilt', 'Propagation',
        'Relative Stratiform-Offset']
    offset_1 = -.575
    # leg_offset = [offset_1, offset_1, offset_1, offset_1, offset_1]
    # leg_offset_x = [.475] * 4 + [.475]
    # leg_columns = [2, 3, 2, 2, 2]
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    fig_dir = base_dir + 'TINT_figures/national/'

    def pct_fmt(pct):
        if pct > 15:
            return "{:1.0f}".format(pct)
        else:
            return ''

    # import pdb; pdb.set_trace()

    def plot_pie_inset(
            data, ilon, ilat, ax, width, total, radar, offsets, colors):
        width = 5*width
        ax_sub = ax.inset_axes(
            bounds=[
                ilon-width/2+offsets[0], ilat-width/2+offsets[1],
                width, width],
            transform=ax.transData,
            zorder=1)

        # patches, texts, other = ax_sub.pie(
        #     data, colors=colors, autopct=lambda pct: pct_fmt(pct),
        #     normalize=True, pctdistance=.6,
        #     textprops={'fontsize': 10},
        #     wedgeprops={
        #         'edgecolor' : 'black',
        #         'linewidth': 0.75,
        #         'antialiased': True})
        patches, texts = ax_sub.pie(
            data, colors=colors, normalize=True,
            wedgeprops={
                'edgecolor' : 'black',
                'linewidth': 1,
                'antialiased': True})

        [p.set_zorder(2) for p in patches]

        # ax_sub.text(
        #     ilon+offsets[0], ilat+offsets[1]-width/2,
        #     str(total), transform=ax.transData, fontsize=10,
        #     horizontalalignment='center', backgroundcolor='1',
        #     bbox=dict(facecolor='white', alpha=0.5, linewidth=0, pad=-.3),
        #     zorder=1)
        return patches, texts

    leg_col = [5, 6, 4, 4, 2]
    offsets = [-.115, -.115, -.115, -.115, -.15]

    for i in [0, 1, 4, 2, 3]:
    # for i in [0]:
        base_ratios = radar_dfs[i].drop('Total', axis=1)
        c_list = clists[i]
        for c in base_ratios.columns:
            base_ratios.loc[:, c] = (
                base_ratios.loc[:, c]/radar_dfs[i].loc[:, 'Total'])

        max_rat = np.ceil(base_ratios.max().max()*10)/10
        ncol = len(base_ratios.columns)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_title(names[i])
        ax.coastlines(resolution='50m', zorder=1)

        ax.set_extent([110, 160, -9, -44], crs=ccrs.PlateCarree())
        grid = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=1, color='gray', alpha=0.4, linestyle='--')

        grid.xlocator = mticker.FixedLocator(np.arange(105, 160, 5))
        grid.ylocator = mticker.FixedLocator(np.arange(-50, -5, 5))

        states_provinces_50m = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        ax.add_feature(states_provinces_50m, edgecolor='grey', zorder=1)

        c_land = tuple(np.array([249.0, 246.0, 216.0])/256)
        c_water = tuple(np.array([252.0, 252.0, 256.0])/256)

        land_50m = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m',
            edgecolor='face',
            facecolor=c_land)
        ax.add_feature(land_50m, zorder=0)

        ocean_50m = cfeature.NaturalEarthFeature(
            'physical', 'ocean', '50m',
            edgecolor='face',
            facecolor=c_water)
        ax.add_feature(ocean_50m, zorder=0)

        grid.right_labels = False
        grid.top_labels = False

        # import pdb; pdb.set_trace()
        arrow = False
        max_tot = np.max(radar_dfs[i].loc[:, 'Total'].values)
        min = 50

        for radar in radars:
            # Plot pie chart
            # import pdb; pdb.set_trace()
            if radar_dfs[i].loc[radar, 'Total'] < min:
                continue

            pie_data = base_ratios.loc[radar]
            good_radar_i = good_radar.loc[[radar]].iloc[0]

            lat, lon = good_radar_i[['site_lat', 'site_lon']]
            total = radar_dfs[i].loc[radar, 'Total']

            lin_scale = .6
            width = 1*lin_scale + (total-min)/(max_tot-min)*(1-lin_scale)

            patches, other = plot_pie_inset(
                pie_data, lon, lat, ax, width,
                total, radar, offset_dic[radar], colors=c_list)

            if offset_dic[radar] != [0, 0]:

                ax.quiver(
                    lon+offset_dic[radar][0], lat+offset_dic[radar][1],
                    -offset_dic[radar][0], -offset_dic[radar][1],
                    edgecolor='black', zorder=3, angles='xy',
                    scale_units='xy', scale=1, width=.004,
                    facecolor=(1, 1, 1, 0.4),
                    linewidth=1)
            dot = ax.scatter(
                [lon], [lat], s=30, marker='o', color=colors[3],
                zorder=4, linewidth=1, edgecolor='black')

            ax.text(
                lon+offset_dic[radar][0],
                lat+offset_dic[radar][1]+.95*5*width/2,
                '#' + str(radar), fontsize=10,
                horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=0.85, linewidth=0, pad=-.3),
                zorder=3)
            # ax.arrow(
            #     lon+offset_dic[radar][0], lat+offset_dic[radar][1],
            #     -offset_dic[radar][0], -offset_dic[radar][1],
            #     width=.1, color='k', zorder=2, alpha=.4)

        plt.legend(
            patches, list(pie_data.index),
            loc='lower center',
            bbox_to_anchor=(.5, offsets[i]),
            ncol=leg_col[i], fancybox=True, shadow=True)
        plt.savefig(
            fig_dir + 'pie_chart_{}.png'.format(
                names[i].replace('-', '_',).replace(' ', '_').lower()),
            dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

def plot_sensitivities_comp(
        sen_dfs_1, sen_dfs_2, test_dirs_1, test_dirs_2,
        name_abvs=None, suff=''):

    # import pdb; pdb.set_trace()
    new_order = [0, 4, 1, 2, 3]
    sen_dfs_1 = [sen_dfs_1[i] for i in new_order]
    sen_dfs_2 = [sen_dfs_2[i] for i in new_order]

    if name_abvs is None:
        name_abvs = [
            'Base', 'C2', 'C4', '4L', 'NS', 'LR', 'S4', 'RV4', 'T15',
            'S15', 'A2', 'B5', 'L50', 'L25', 'C']

    fig, axes = plt.subplots(5, 1, figsize=(12, 12))
    init_fonts()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    offset_c = [colors[i] for i in [0, 0, 1, 1, 2, 2, 4, 4]]
    inflow_c = [colors[i] for i in [0, 0, 1, 1, 2, 2, 4, 4, 5, 5]]
    tilt_c = [colors[i] for i in [0, 0, 1, 1, 5, 5]]
    prop_c = [colors[i] for i in [0, 0, 1, 1, 5, 5]]
    clists = [offset_c, offset_c, inflow_c, tilt_c, prop_c]
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

        col_names = base_ratios_1.columns.values.tolist()
        new_col_names_1 = [
            col_names[i] + ' Radar' for i in range(len(col_names))]
        rename_dict_1 = {
            col_names[i]: new_col_names_1[i] for i in range(len(col_names))}
        base_ratios_1 = base_ratios_1.rename(columns=rename_dict_1)
        new_col_names_2 = [
            col_names[i] + ' ACCESS-C' for i in range(len(col_names))]
        rename_dict_2 = {
            col_names[i]: new_col_names_2[i] for i in range(len(col_names))}
        base_ratios_2 = base_ratios_2.rename(columns=rename_dict_2)

        base_ratios = pd.concat([base_ratios_1, base_ratios_2], axis=1)
        # col_names = base_ratios.columns.values.tolist()

        # import pdb; pdb.set_trace()
        new_col_order = []
        for i in range(len(new_col_names_1)):
            new_col_order.append(new_col_names_1[i])
            new_col_order.append(new_col_names_2[i])
        base_ratios = base_ratios[new_col_order]

        ax = base_ratios.plot(
            kind='bar', stacked=False, fontsize=12, rot=0, ax=ax,
            yticks=np.arange(0, np.ceil(max_rat*10)/10+0.1, 0.1),
            width=0.75*ncol/4,
            color=c_list)

        ax.set_xlim(-.5, len(name_abvs)-.5)

        ax.set_xlabel(None)
        ax.xaxis.set_label_coords(.5, -0.15)
        ax.set_ylabel('Ratio [-]', fontsize=14)

        lines, labels = ax.get_legend_handles_labels()

        ax.legend(
            lines[::2], col_names,
            loc='lower center',
            bbox_to_anchor=(leg_offset_x[i], leg_offset[i]),
            ncol=leg_columns[i]+1, fancybox=True, shadow=True)
        ax.set_yticks(
            np.arange(0, np.ceil(max_rat*10)/10+0.1, 0.05), minor=True)
        ax.grid(which='minor', alpha=0.2, axis='y')
        ax.grid(which='major', alpha=0.5, axis='y')

    plt.suptitle(
        'Category Ratios for Radar (Left Bars) and ACCESS-C (Right Bars)',
        fontsize=14, y=.90)
    # category_breakdown_comp(
    #     fig=fig, ax=axes.flatten()[-1], leg_offset_h=-.71,
    #     test_dir_1=test_dirs_1, test_dir_2=test_dirs_2,
    #     test_names=name_abvs, name_abvs=name_abvs, ncol=5)
    plt.subplots_adjust(hspace=0.65)
    make_subplot_labels(axes.flatten(), x_shift=-.07, y_shift=-.04)

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


def category_breakdown_comp(
        fig=None, ax=None, leg_offset_h=-0.45, test_dir_1=None,
        test_dir_2=None, test_names=None, name_abvs=None, ncol=2):

    if test_dir_1 is None or test_dir_2 is None:
        test_dir_base = [
            'ACCESS_radar_base/',
            'ACCESS_radar_ambient_swapped/',
            'ACCESS_radar_lower_wind_level/',
            'ACCESS_radar_higher_wind_level/',
            'ACCESS_radar_no_steiner/',
            'ACCESS_radar_lower_ref_thresh/',
            'ACCESS_radar_higher_shear_thresh/',
            'ACCESS_radar_higher_rel_vel_thresh/',
            'ACCESS_radar_higher_theta_e/',
            'ACCESS_radar_higher_offset_thresh/',
            'ACCESS_radar_higher_area_thresh/',
            'ACCESS_radar_higher_conv_area_thresh/',
            'ACCESS_radar_higher_border_thresh/',
            'ACCESS_radar_linear_50/',
            'ACCESS_radar_linear_25/',
            'ACCESS_radar_combined_sensitivity/']
        test_dir_1 = [t + 'combined_radar' for t in test_dir_base]
        test_dir_2 = [t + 'combined_ACCESS' for t in test_dir_base]
    if name_abvs is None:
        name_abvs = [
            'Base', 'SA', 'W2', 'W4', 'NS', 'LR', 'S4', 'RV4', 'T15',
            'S15', 'A2', 'CA', 'B5', 'L50', 'L25', 'C']
    if test_names is None:
        test_names = name_abvs

    can_classes_1 = [TS_1, LS_1, LeS_1, RiS_1, can_A_1, can_totals_1] = [
        [] for i in range(6)]
    can_classes_2 = [TS_2, LS_2, LeS_2, RiS_2, can_A_2, can_totals_2] = [
        [] for i in range(6)]
    test = []

    for i in range(len(test_dir_1)):
        base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        class_path = base_dir + 'TINT_tracks/'
        class_path += '{}_classes.pkl'.format(test_dir_1[i])

        with open(class_path, 'rb') as f:
            class_df_1 = pickle.load(f)

        base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
        class_path = base_dir + 'TINT_tracks/'
        class_path += '{}_classes.pkl'.format(test_dir_2[i])

        with open(class_path, 'rb') as f:
            class_df_2 = pickle.load(f)

        test.append(test_names[i])

        can_breakdown_1 = canonical_class_breakdown(class_df_1)
        [
            can_classes_1[i].append(can_breakdown_1[i])
            for i in range(len(can_classes_1))]

        can_breakdown_2 = canonical_class_breakdown(class_df_2)
        [
            can_classes_2[i].append(can_breakdown_2[i])
            for i in range(len(can_classes_2))]

    tilt_sensitivity_df_1 = pd.DataFrame({
        'Test': test, 'Trailing Stratiform Class': TS_1,
        'Leading Stratiform Class': LS_1,
        'Left Stratiform Class': LeS_1,
        'Right Stratiform Class': RiS_1, 'Ambiguous Inflow': can_A_1,
        'Total': can_totals_1})
    tilt_sensitivity_df_1 = tilt_sensitivity_df_1.set_index('Test')

    tilt_sensitivity_df_1 = tilt_sensitivity_df_1.drop('Total', axis=1)
    tilt_sensitivity_df_1 = tilt_sensitivity_df_1.reset_index(drop=True)
    tilt_sensitivity_df_1.loc[:, 'Test'] = np.array(name_abvs)
    tilt_sensitivity_df_1 = tilt_sensitivity_df_1.set_index('Test')

    tilt_sensitivity_df_2 = pd.DataFrame({
        'Test': test, 'Trailing Stratiform Class': TS_2,
        'Leading Stratiform Class': LS_2,
        'Left Stratiform Class': LeS_2,
        'Right Stratiform Class': RiS_2, 'Ambiguous Inflow': can_A_2,
        'Total': can_totals_2})
    tilt_sensitivity_df_2 = tilt_sensitivity_df_2.set_index('Test')

    tilt_sensitivity_df_2 = tilt_sensitivity_df_2.drop('Total', axis=1)
    tilt_sensitivity_df_2 = tilt_sensitivity_df_2.reset_index(drop=True)
    tilt_sensitivity_df_2.loc[:, 'Test'] = np.array(name_abvs)
    tilt_sensitivity_df_2 = tilt_sensitivity_df_2.set_index('Test')

    col_names = tilt_sensitivity_df_1.columns.values.tolist()
    new_col_names_1 = [
        col_names[i] + ' Radar' for i in range(len(col_names))]
    rename_dict_1 = {
        col_names[i]: new_col_names_1[i] for i in range(len(col_names))}
    tilt_sensitivity_df_1 = tilt_sensitivity_df_1.rename(columns=rename_dict_1)
    new_col_names_2 = [
        col_names[i] + ' ACCESS-C' for i in range(len(col_names))]
    rename_dict_2 = {
        col_names[i]: new_col_names_2[i] for i in range(len(col_names))}
    tilt_sensitivity_df_2 = tilt_sensitivity_df_2.rename(columns=rename_dict_2)

    tilt_sensitivity_df = pd.concat(
        [tilt_sensitivity_df_1, tilt_sensitivity_df_2], axis=1)

    new_col_order = []
    for i in range(len(new_col_names_1)):
        new_col_order.append(new_col_names_1[i])
        new_col_order.append(new_col_names_2[i])
    tilt_sensitivity_df = tilt_sensitivity_df[new_col_order]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 0, 1, 1, 2, 2, 4, 4, 5, 5]]

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    init_fonts()

    n_col = len(tilt_sensitivity_df_1.columns)

    tilt_sensitivity_df.plot(
        kind='bar', stacked=False, rot=0,
        yticks=np.arange(0, 0.8, 0.1), width=0.75*n_col/4,
        fig=fig, ax=ax, color=colors)
    plt.sca(ax)
    plt.ylabel('Ratio [-]', fontsize=14)
    plt.xlabel(None)

    lines, labels = ax.get_legend_handles_labels()

    ax.legend(
        lines[::2], col_names,
        loc='lower center', bbox_to_anchor=(0.475, leg_offset_h),
        ncol=ncol, fancybox=True, shadow=True)

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'

    ax.set_yticks(np.arange(0, 0.8, 0.1))
    ax.set_yticks(np.arange(0, .75, 0.05), minor='True')

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


def compare_categories(
        class_df_1=None, class_df_2=None, pope_regime=None,
        fig=None, ax=None, title=''):

    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    if class_df_1 is None:
        class_path_1 = base_dir + 'TINT_tracks/'
        class_path_1 += 'base_classes.pkl'
        with open(class_path_1, 'rb') as f:
            class_df_1 = pickle.load(f)

        class_path_2 = base_dir + 'TINT_tracks/'
        class_path_2 += 'base_classes.pkl'
        with open(class_path_2, 'rb') as f:
            class_df_2 = pickle.load(f)
    # fig_dir = base_dir + 'TINT_figures/'

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    init_fonts()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 0, 1, 1, 2, 2, 4, 4, 5, 5]]

    counts_df_1 = pd.DataFrame({'count': class_df_1.value_counts()})
    counts_df_2 = pd.DataFrame({'count': class_df_2.value_counts()})
    if pope_regime is not None:
        counts_df_1 = counts_df_1.xs(pope_regime, level='pope_regime')
        counts_df_2 = counts_df_2.xs(pope_regime, level='pope_regime')

    pivot_counts_list = []
    grouped_counts_list = []

    for counts_df in [counts_df_1, counts_df_2]:
        ratio = counts_df['count']/counts_df['count'].sum()
        counts_df['ratio'] = ratio
        grouped_counts = counts_df.groupby(
            ['offset_type', 'inflow_type']).sum()

        grouped_counts_list.append(grouped_counts)

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

        pivot_counts_list.append(pivot_counts)

    col_names = pivot_counts_list[0].columns.values.tolist()
    new_col_names_1 = [
        col_names[i] + ' Radar' for i in range(len(col_names))]
    rename_dict_1 = {
        col_names[i]: new_col_names_1[i] for i in range(len(col_names))}
    pivot_counts_list[0] = pivot_counts_list[0].rename(columns=rename_dict_1)
    new_col_names_2 = [
        col_names[i] + ' ACCESS-C' for i in range(len(col_names))]
    rename_dict_2 = {
        col_names[i]: new_col_names_2[i] for i in range(len(col_names))}
    pivot_counts_list[1] = pivot_counts_list[1].rename(columns=rename_dict_2)

    pivot_counts = pd.concat(pivot_counts_list, axis=1)
    new_col_order = []
    for i in range(len(new_col_names_1)):
        new_col_order.append(new_col_names_1[i])
        new_col_order.append(new_col_names_2[i])
    pivot_counts = pivot_counts[new_col_order]

    pivot_counts.plot(
        kind='bar', stacked=False, rot=0, ax=ax,
        yticks=np.arange(0, 0.7, 0.1), width=0.6*5/4,
        color=colors, xlabel=None, legend=False)

    plt.sca(ax)
    plt.ylabel('Ratio [-]')
    plt.xlabel(None)
    ax.set_yticks(np.arange(0, 0.65, 0.05), minor=True)

    # plt.xlabel('Stratiform Offset Category')
    # ax.legend(
    #     loc='upper right',  # bbox_to_anchor=(0.475, -0.575),
    #     ncol=1, fancybox=True, shadow=True)

    ax.grid(which='major', alpha=0.5, axis='y')
    ax.grid(which='minor', alpha=0.2, axis='y')

    total_1 = grouped_counts_list[0]['count'].sum()
    total_2 = grouped_counts_list[1]['count'].sum()

    ax.text(
        0.52, .87, title, ha='center',
        transform=ax.transAxes, size=12, backgroundcolor='1')

    total_text = 'Radar Total = {}        ACCESS-C Total = {}'.format(
        int(np.round(total_1)), int(np.round(total_2)))

    ax.text(
        0.15, .72, total_text,
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


def pope_comparison_radar_sensitivity(
        class_df_1=None, class_path_1=None,
        class_df_2=None, class_path_2=None, titles=None):
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    if class_path_1 is None:
        class_path_1 = base_dir + 'TINT_tracks/ACCESS_radar_base/'
        class_path_2 = base_dir + 'TINT_tracks/ACCESS_radar_ambient_swapped/'
    if titles is None:
        titles = ['Base', 'Swapped Ambient']
    with open(class_path_1 + 'combined_radar_classes.pkl', 'rb') as f:
        class_df_1_rad = pickle.load(f)
    with open(class_path_1 + 'combined_ACCESS_classes.pkl', 'rb') as f:
        class_df_1_ACCESS = pickle.load(f)
    with open(class_path_2 + 'combined_radar_classes.pkl', 'rb') as f:
        class_df_2_rad = pickle.load(f)
    with open(class_path_2 + 'combined_ACCESS_classes.pkl', 'rb') as f:
        class_df_2_ACCESS = pickle.load(f)

    sub_dir = class_path_1.split('/')[-2]
    fig_dir = base_dir + '/TINT_figures/' + sub_dir + '/'

    fig, axes = plt.subplots(3, 2, figsize=(12, 7))

    init_fonts()

    titles_1 = [
        '{}: All Regimes'.format(titles[0]),
        '{}: Weak Monsoon'.format(titles[0]),
        '{}: Active Monsoon'.format(titles[0])]

    titles_2 = [
        '{}: All Regimes'.format(titles[1]),
        '{}: Weak Monsoon'.format(titles[1]),
        '{}: Active Monsoon'.format(titles[1])]

    compare_categories(
        class_df_1=class_df_1_rad, class_df_2=class_df_1_ACCESS,
        pope_regime=None, fig=fig, ax=axes[0, 0],
        title=titles_1[0])

    compare_categories(
        class_df_1=class_df_2_rad, class_df_2=class_df_2_ACCESS,
        pope_regime=None, fig=fig, ax=axes[0, 1],
        title=titles_2[0])

    for i in range(1, 3):
        compare_categories(
            class_df_1=class_df_1_rad, class_df_2=class_df_1_ACCESS,
            pope_regime=i, fig=fig, ax=axes[i, 0],
            title=titles_1[i])
        compare_categories(
            class_df_1=class_df_2_rad, class_df_2=class_df_2_ACCESS,
            pope_regime=i, fig=fig,
            ax=axes[i, 1], title=titles_2[i])

    # axes.flatten()[0].legend(
    #     loc='upper right',  # bbox_to_anchor=(0.475, -0.575),
    #     ncol=1, fancybox=True, shadow=True)

    make_subplot_labels(axes.flatten())

    plt.sca(axes.flatten()[-2])
    plt.xlabel('Stratiform Offset Category')
    plt.sca(axes.flatten()[-1])
    plt.xlabel('Stratiform Offset Category')

    lines, labels = axes.flatten()[-2].get_legend_handles_labels()
    col_names = [
        'Front Fed', 'Rear Fed', 'Left Fed', 'Right Fed',
        'Ambiguous Inflow']

    axes.flatten()[-2].legend(
        lines[::2], col_names,
        loc='lower center', bbox_to_anchor=(1.1, -0.6),
        ncol=5, fancybox=True, shadow=True)

    plt.suptitle(
        'Category Ratios for Radar (Left Bars) and ACCESS-C (Right Bars)',
        fontsize=14, y=.94)

    plt.savefig(
        fig_dir + 'pope_breakdown_radar_sensitivity.png', dpi=200, facecolor='w',
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


def monsoon_comparison_ACCESS_radar(
        class_df_1=None, class_df_2=None, fig=None, ax=None, legend=True,
        title='', required_types=None, titles=None, colors=None):
    base_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    if class_df_1 is None:
        class_path = base_dir + 'TINT_tracks/ACCESS_radar_base/'
        class_path += 'combined_radar_classes.pkl'
        with open(class_path, 'rb') as f:
            class_df_1 = pickle.load(f)
        class_path = base_dir + 'TINT_tracks/ACCESS_radar_base/'
        class_path += 'combined_ACCESS_classes.pkl'
        with open(class_path, 'rb') as f:
            class_df_2 = pickle.load(f)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = [
            colors[i]
            for i in [0, 0, 1, 1, 2, 2, 4, 4, 6, 6, 7, 7, 5, 5]]

    ratios_df_list = []
    totals_list = []
    for class_df in [class_df_1, class_df_2]:
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
                    'Leading Stratiform', 'Rear Fed',
                    'Down-Shear Tilted', 'Up-Shear Propagating'),
                (
                    'Parallel Stratiform (Right)', 'Parallel Fed (Right)',
                    'Down-Shear Tilted', 'Down-Shear Propagating')]
            titles = [
                'Type I: Front-Fed Trailing Stratiform, Up-Shear Tilted, Down-Shear Propagating',
                'Type II: Front-Fed Leading Stratiform, Down-Shear Tilted, Down-Shear Propagating',
                'Type III: Front-Fed Trailing Stratiform, Down-Shear Tilted, Up-Shear Propagating',
                'Type IV: Rear-Fed Trailing Stratiform, Up-Shear Tilted, Up-Shear Propagating',
                'Type VIII: Rear-Fed Leading Stratiform, Down-Shear Tilted, Up-Shear Propagating',
                'Type IX: Right-Fed Right Stratiform, Down-Shear Tilted, Down-Shear Propagating',
                'All Other Types']

            titles = [
                'Type I', 'Type II', 'Type III', 'Type IV', 'Type VIII',
                'Type IX', 'All Other Types']

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

        totals_list.append(total)

        categories = ['All', 'Weak Monsoon', 'Active Monsoon']
        ratios_dict = {'Wet Season Regime': categories}
        for i in range(len(required_types)):
            ratios_dict[titles[i]] = types[i]
        ratios_dict[titles[-1]] = types[-1]
        ratios_df = pd.DataFrame(ratios_dict)
        ratios_df = ratios_df.set_index('Wet Season Regime')
        ratios_df_list.append(ratios_df)

    # import pdb; pdb.set_trace()

    col_names = ratios_df_list[0].columns.values.tolist()
    new_col_names_1 = [
        col_names[i] + ' Radar' for i in range(len(col_names))]
    rename_dict_1 = {
        col_names[i]: new_col_names_1[i] for i in range(len(col_names))}
    ratios_df_list[0] = ratios_df_list[0].rename(columns=rename_dict_1)
    new_col_names_2 = [
        col_names[i] + ' ACCESS-C' for i in range(len(col_names))]
    rename_dict_2 = {
        col_names[i]: new_col_names_2[i] for i in range(len(col_names))}
    ratios_df_list[1] = ratios_df_list[1].rename(columns=rename_dict_2)

    ratios_df = pd.concat(ratios_df_list, axis=1)
    new_col_order = []
    for i in range(len(new_col_names_1)):
        new_col_order.append(new_col_names_1[i])
        new_col_order.append(new_col_names_2[i])
    ratios_df = ratios_df[new_col_order]

    ratios_df.plot(
        kind='bar', stacked=False, rot=0, fontsize=12, ax=ax,
        yticks=np.arange(0, 1.1, 0.1), width=0.7*4/4,
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

    lab_h = 0.81
    fs = 12

    tot_lab = [
        'Totals\n {} / {}'.format(
            totals_list[0][i], totals_list[1][i])
        for i in range(len(totals_list[0]))]

    ax.text(
        0.45, 1.04, title, transform=ax.transAxes, size=12, ha='center')

    ax.text(
        0.21, lab_h, tot_lab[0], transform=ax.transAxes, size=fs,
        backgroundcolor='1', ha='center')
    ax.text(
        0.52, lab_h, tot_lab[1], transform=ax.transAxes, size=fs,
        backgroundcolor='1', ha='center')
    ax.text(
        0.8, lab_h, tot_lab[2], transform=ax.transAxes, size=fs,
        backgroundcolor='1', ha='center')

    return ratios_df
