from matplotlib import rcParams
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def load_year(year, subscript=''):
    print('Processing year {}'.format(year))
    save_dir = '/home/student.unimelb.edu.au/shorte1/'
    save_dir += 'Documents//TINT_tracks/base/'
    filename = save_dir + '{}1001_{}0501{}.pkl'.format(
        year, year+1, subscript)
    with open(filename, 'rb') as f:
        tracks_obj = pickle.load(f)
    return tracks_obj


def get_sub_tracks(tracks_obj):
    exclusions = [
        'small_area', 'large_area', 'intersect_border',
        'intersect_border_convective', 'duration_cond',
        'small_velocity', 'small_offset']
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


def get_counts(base_dir=None):
    if base_dir is None:
        base_dir = '/g/data/w40/esh563/CPOL_analysis/'
    [
        year_list, uid, time, offset_type, inflow_type,
        tilt_dir, prop_dir, pope_regime] = [
        [] for i in range(8)]
    years = sorted(list(set(range(1998, 2016)) - {2000, 2007, 2008}))
    for year in years:
        tracks_obj = load_year(year)
        print('Adding Pope monsoon regime.')
        tracks_obj = add_monsoon_regime(tracks_obj, base_dir=base_dir)
        sub_tracks = get_sub_tracks(tracks_obj)
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


def plot_offsets(class_df):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(30, 310, 10)
    ax1.plot(x, TS.loc[x], label='Trailing Stratiform')
    ax1.plot(x, LS.loc[x], label='Leading Stratiform')
    ax1.plot(x, LeS.loc[x], label='Parallel Stratiform (Left)')
    ax1.plot(x, RiS.loc[x], label='Parallel Stratiform (Right)')
    ax1.plot(x, offset_totals.loc[x], label='Total')
    plt.sca(ax1)
    plt.xticks(np.arange(30, 310, 30))
    plt.ylabel('Count [-]')
    plt.xlabel('Time since Initiation [m]')
    ax1.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)

    ax2.plot(x, (TS/offset_totals).loc[x], label='Trailing Stratiform')
    ax2.plot(x, (LS/offset_totals).loc[x], label='Leading Stratiform')
    ax2.plot(x, (LeS/offset_totals).loc[x], label='Parallel Stratiform (Left)')
    ax2.plot(
        x, (RiS/offset_totals).loc[x], label='Parallel Stratiform (Right)')
    plt.sca(ax2)
    plt.xticks(np.arange(30, 300, 30))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Ratio [-]')
    plt.xlabel('Time since Initiation [m]')
    ax2.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)


def plot_inflows(class_df):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(30, 310, 10)
    ax1.plot(x, FF.loc[x], label='Front Fed')
    ax1.plot(x, RF.loc[x], label='Rear Fed')
    ax1.plot(x, LeF.loc[x], label='Parallel Fed (Left)')
    ax1.plot(x, RiF.loc[x], label='Parallel Fed (Right)')
    ax1.plot(x, inflow_totals.loc[x], label='Total')
    ax1.plot(x, A.loc[x], label='Ambiguous')
    plt.sca(ax1)
    plt.xticks(np.arange(30, 310, 30))
    plt.ylabel('Count [-]')
    plt.xlabel('Time since Initiation [m]')
    ax1.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)

    ax2.plot(x, (FF/inflow_totals).loc[x], label='Front Fed')
    ax2.plot(x, (RF/inflow_totals).loc[x], label='Rear Fed')
    ax2.plot(x, (LeF/inflow_totals).loc[x], label='Parallel Fed (Left)')
    ax2.plot(x, (RiF/inflow_totals).loc[x], label='Parallel Fed (Right)')
    ax2.plot(x, (A/inflow_totals).loc[x], label='Ambiguous')
    plt.sca(ax2)
    plt.xticks(np.arange(30, 300, 30))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Ratio [-]')
    plt.xlabel('Time since Initiation [m]')
    ax2.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)


def plot_tilts(class_df):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(30, 310, 10)
    ax1.plot(x, UST.loc[x], label='Up-Shear Tilted')
    ax1.plot(x, DST.loc[x], label='Down-Shear Tilted')
    ax1.plot(x, A.loc[x], label='Ambiguous')
    ax1.plot(x, tilt_totals.loc[x], label='Total', color='red')
    plt.sca(ax1)
    plt.xticks(np.arange(30, 310, 30))
    plt.ylabel('Count [-]')
    plt.xlabel('Time since Initiation [m]')
    ax1.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)

    ax2.plot(x, (UST/tilt_totals).loc[x], label='Up-Shear Tilted')
    ax2.plot(x, (DST/tilt_totals).loc[x], label='Down-Shear Tilted')
    ax2.plot(x, (A/tilt_totals).loc[x], label='Ambiguous')
    plt.sca(ax2)
    plt.xticks(np.arange(30, 300, 30))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Ratio [-]')
    plt.xlabel('Time since Initiation [m]')
    ax2.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)


def plot_propagations(class_df):
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
    tilt_totals = A + USP + DSP

    init_fonts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(30, 310, 10)
    ax1.plot(x, USP.loc[x], label='Up-Shear Propagating')
    ax1.plot(x, DSP.loc[x], label='Down-Shear Propagating')
    ax1.plot(x, A.loc[x], label='Ambiguous')
    ax1.plot(x, tilt_totals.loc[x], label='Total', color='red')
    plt.sca(ax1)
    plt.xticks(np.arange(30, 310, 30))
    plt.ylabel('Count [-]')
    plt.xlabel('Time since Initiation [m]')
    ax1.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)

    ax2.plot(x, (USP/tilt_totals).loc[x], label='Up-Shear Propagating')
    ax2.plot(x, (DSP/tilt_totals).loc[x], label='Down-Shear Propagating')
    ax2.plot(x, (A/tilt_totals).loc[x], label='Ambiguous')
    plt.sca(ax2)
    plt.xticks(np.arange(30, 300, 30))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Ratio [-]')
    plt.xlabel('Time since Initiation [m]')
    ax2.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.6),
        ncol=2, fancybox=True, shadow=True)
