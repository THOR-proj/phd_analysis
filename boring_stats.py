import sys
sys.path.insert(0, '/home/student.unimelb.edu.au/shorte1/Documents/TINT')
sys.path.insert(0, '/home/563/esh563/TINT')

import pickle

import numpy as np
import classification as cl
import matplotlib.pyplot as plt


def get_boring_radar_stats(
        save_dir, exclusions=None, regime=None, pope_dir=None,
        radars=[63, 42, 77]):

    if pope_dir is None:
        pope_dir = '/home/student.unimelb.edu.au/shorte1/'
        pope_dir += 'Documents/CPOL_analysis/'

    conv_area = []
    strat_area = []
    times = []
    u_shift = []
    v_shift = []
    orientation = []
    eccentricity = []
    offset_mag = []
    durations = []

    count = 0
    system_count = 0

    if exclusions is None:
        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'duration_cond',
            'small_velocity', 'small_offset']

    for radar in radars:
        for year in [2020, 2021]:
            print('Radar {}, year {}.'.format(radar, year))
            years_months = [
                [year, 10], [year, 11], [year, 12],
                [year+1, 1], [year+1, 2], [year+1, 3],
                [year+1, 4]]
            for year_month in years_months:
                path = save_dir + 'radar_{}/{}_{}_{:02}.pkl'.format(
                    radar, radar, year_month[0], year_month[1])
                with open(path, 'rb') as f:
                    tracks_obj = pickle.load(f)

                tracks_obj = cl.redo_exclusions(tracks_obj)
                tracks_obj = cl.add_monsoon_regime(
                    tracks_obj, base_dir=pope_dir, fake_pope=True)

                excluded = tracks_obj.exclusions[exclusions]
                excluded = np.any(excluded, 1)
                included = np.logical_not(excluded)

                if regime is None:
                    cond = (included == True)
                else:
                    cond = np.logical_and(
                        tracks_obj.tracks_class['pope_regime'] == regime,
                        included == True)

                sub_classes = tracks_obj.tracks_class.where(cond).dropna()

                inds_all = sub_classes.index.values
                sub_tracks_all = tracks_obj.tracks.loc[inds_all]
                try:
                    sub_tracks_conv = sub_tracks_all.xs(0, level='level')
                    sub_tracks_strat = sub_tracks_all.xs(1, level='level')
                    conv_area += list(sub_tracks_conv['proj_area'].values)
                    strat_area += list(sub_tracks_strat['proj_area'].values)

                    pos_0 = sub_tracks_conv[['grid_x', 'grid_y']]
                    pos_1 = sub_tracks_strat[['grid_x', 'grid_y']]
                    mag = pos_1-pos_0
                    mag_num = np.sqrt(
                        mag['grid_x'].values**2 + mag['grid_y'].values**2)
                    offset_mag += list(mag_num)

                    u_shift += list(sub_tracks_conv['u_shift'].values)
                    v_shift += list(sub_tracks_conv['v_shift'].values)
                    orientation += list(
                        sub_tracks_conv['orientation'].values)
                    eccentricity += list(
                        sub_tracks_conv['eccentricity'].values)

                    times += list(
                        sub_tracks_conv.reset_index()['time'].values)
                    count += len(sub_tracks_conv)
                    uids = set([ind[2] for ind in inds_all])
                    system_count += len(uids)

                    for uid in uids:
                        times_uid = sub_tracks_conv.xs(
                            uid, level='uid').reset_index()['time']
                        duration = times_uid.values[-1] - times_uid.values[0]
                        duration = duration.astype('timedelta64[m]')
                        durations.append(duration.astype(int))

                except KeyError:
                    print('No included observations.')

    out = [
        conv_area, strat_area, times, count, system_count,
        u_shift, v_shift, orientation, eccentricity, offset_mag, durations]

    return out


def get_boring_ACCESS_stats(
        save_dir, exclusions=None, regime=None, pope_dir=None,
        radars=[63, 42, 77]):

    if pope_dir is None:
        pope_dir = '/home/student.unimelb.edu.au/shorte1/'
        pope_dir += 'Documents/CPOL_analysis/'

    conv_area = []
    strat_area = []
    times = []
    u_shift = []
    v_shift = []
    orientation = []
    eccentricity = []
    offset_mag = []
    durations = []

    count = 0
    system_count = 0

    if exclusions is None:
        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'duration_cond',
            'small_velocity', 'small_offset']

    for radar in radars:
        for year in [2020, 2021]:
            print('Radar {}, year {}.'.format(radar, year))
            path = save_dir + 'ACCESS_{}/{}1001_{}0501.pkl'.format(
                radar, year, year+1)
            with open(path, 'rb') as f:
                tracks_obj = pickle.load(f)

            tracks_obj = cl.redo_exclusions(tracks_obj)
            tracks_obj = cl.add_monsoon_regime(
                tracks_obj, base_dir=pope_dir, fake_pope=True)

            excluded = tracks_obj.exclusions[exclusions]
            excluded = np.any(excluded, 1)
            included = np.logical_not(excluded)

            if regime is None:
                cond = (included == True)
            else:
                cond = np.logical_and(
                    tracks_obj.tracks_class['pope_regime'] == regime,
                    included == True)

            sub_classes = tracks_obj.tracks_class.where(cond).dropna()

            inds_all = sub_classes.index.values
            sub_tracks_all = tracks_obj.tracks.loc[inds_all]
            try:
                sub_tracks_conv = sub_tracks_all.xs(0, level='level')
                sub_tracks_strat = sub_tracks_all.xs(1, level='level')
                conv_area += list(sub_tracks_conv['proj_area'].values)
                strat_area += list(sub_tracks_strat['proj_area'].values)

                pos_0 = sub_tracks_conv[['grid_x', 'grid_y']]
                pos_1 = sub_tracks_strat[['grid_x', 'grid_y']]
                mag = pos_1-pos_0
                mag_num = np.sqrt(
                    mag['grid_x'].values**2 + mag['grid_y'].values**2)
                offset_mag += list(mag_num)

                u_shift += list(sub_tracks_conv['u_shift'].values)
                v_shift += list(sub_tracks_conv['v_shift'].values)
                orientation += list(
                    sub_tracks_conv['orientation'].values)
                eccentricity += list(
                    sub_tracks_conv['eccentricity'].values)

                times += list(
                    sub_tracks_conv.reset_index()['time'].values)
                count += len(sub_tracks_conv)
                uids = set([ind[2] for ind in inds_all])
                system_count += len(uids)

                for uid in uids:
                    times_uid = sub_tracks_conv.xs(
                        uid, level='uid').reset_index()['time']
                    duration = times_uid.values[-1] - times_uid.values[0]
                    duration = duration.astype('timedelta64[m]')
                    durations.append(duration.astype(int))

            except KeyError:
                print('No included observations.')

    out = [
        conv_area, strat_area, times, count, system_count,
        u_shift, v_shift, orientation, eccentricity, offset_mag, durations]

    return out


def count_radar_exclusions(
        save_dir, regime=None, pope_dir=None,
        radars=[63, 42, 77]):
    if pope_dir is None:
        pope_dir = '/home/student.unimelb.edu.au/shorte1/'
        pope_dir += 'Documents/CPOL_analysis/'

    exclusions = ['simple_duration_cond']
    exclusions_list = [
        'intersect_border', 'intersect_border_convective',
        'duration_cond', 'small_velocity', 'small_offset']

    exclusions_counts = [0 for i in range(len(exclusions_list)+1)]

    for radar in radars:
        for year in [2020, 2021]:
            print('Radar {}, year {}.'.format(radar, year))
            years_months = [
                [year, 10], [year, 11], [year, 12],
                [year+1, 1], [year+1, 2], [year+1, 3],
                [year+1, 4]]
            for year_month in years_months:
                path = save_dir + 'radar_{}/{}_{}_{:02}.pkl'.format(
                    radar, radar, year_month[0], year_month[1])
                with open(path, 'rb') as f:
                    tracks_obj = pickle.load(f)

                tracks_obj = cl.redo_exclusions(tracks_obj)
                tracks_obj = cl.add_monsoon_regime(
                    tracks_obj, base_dir=pope_dir, fake_pope=True)

                excluded = tracks_obj.exclusions[exclusions]
                excluded = np.any(excluded, 1)
                included = np.logical_not(excluded)

                if regime is None:
                    cond = (included == True)
                else:
                    cond = np.logical_and(
                        tracks_obj.tracks_class['pope_regime'] == regime,
                        included == True)

                sub_classes = tracks_obj.tracks_class.where(cond).dropna()

                inds_all = sub_classes.index.values
                sub_exclusions = tracks_obj.exclusions.loc[inds_all]

                if len(sub_exclusions) > 0:
                    exclusions_counts[0] += len(
                        sub_exclusions.xs(0, level='level'))
                    for i in range(len(exclusions_list)):
                        excl = sub_exclusions[exclusions_list[i]]
                        excl = excl.xs(0, level='level')
                        excl = excl.where(excl == True).dropna()
                        exclusions_counts[i+1] += len(excl)

    return exclusions_counts


def count_ACCESS_exclusions(
        save_dir, regime=None, pope_dir=None,
        radars=[63, 42, 77]):
    if pope_dir is None:
        pope_dir = '/home/student.unimelb.edu.au/shorte1/'
        pope_dir += 'Documents/CPOL_analysis/'

    exclusions = ['simple_duration_cond']
    exclusions_list = [
        'intersect_border', 'intersect_border_convective',
        'duration_cond', 'small_velocity', 'small_offset']

    exclusions_counts = [0 for i in range(len(exclusions_list)+1)]

    for radar in radars:
        for year in [2020, 2021]:
            print('Radar {}, year {}.'.format(radar, year))
            path = save_dir + 'ACCESS_{}/{}1001_{}0501.pkl'.format(
                radar, year, year+1)
            with open(path, 'rb') as f:
                tracks_obj = pickle.load(f)

            tracks_obj = cl.redo_exclusions(tracks_obj)
            tracks_obj = cl.add_monsoon_regime(
                tracks_obj, base_dir=pope_dir, fake_pope=True)

            excluded = tracks_obj.exclusions[exclusions]
            excluded = np.any(excluded, 1)
            included = np.logical_not(excluded)

            if regime is None:
                cond = (included == True)
            else:
                cond = np.logical_and(
                    tracks_obj.tracks_class['pope_regime'] == regime,
                    included == True)

            sub_classes = tracks_obj.tracks_class.where(cond).dropna()

            inds_all = sub_classes.index.values
            sub_exclusions = tracks_obj.exclusions.loc[inds_all]

            if len(sub_exclusions) > 0:
                exclusions_counts[0] += len(
                    sub_exclusions.xs(0, level='level'))
                for i in range(len(exclusions_list)):
                    excl = sub_exclusions[exclusions_list[i]]
                    excl = excl.xs(0, level='level')
                    excl = excl.where(excl == True).dropna()
                    exclusions_counts[i+1] += len(excl)

    return exclusions_counts


def compare_exclusions(all_excl_radar, all_excl_ACCESS, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    cl.init_fonts()

    labels = [
        'All', 'Any Layer Intersect Border', 'Convective Layer Intersect Border',
        'Duration Cond', 'Small Velocity', 'Small Stratiform Offset']
    x = np.arange(len(labels))
    width = 0.35

    axes.flatten()[0].barh(x-width/2, all_excl_radar, width, label='Radar')
    axes.flatten()[0].barh(x+width/2, all_excl_ACCESS, width, label='ACCESS-C')
    # axes.flatten()[0].set_xticks(np.arange(0, 12e4, 2e4))
    axes.flatten()[0].set_yticks(x)
    axes.flatten()[0].set_yticklabels(labels)
    axes.flatten()[0].invert_yaxis()
    axes.flatten()[0].set_xlabel('Count [-]')
    axes.flatten()[0].set_title('Exclusion Criteria Counts')
    axes.flatten()[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    axes.flatten()[1].barh(
        x-width/2, np.array(all_excl_radar)/all_excl_radar[0],
        width, label='Radar')
    axes.flatten()[1].barh(
        x+width/2, np.array(all_excl_ACCESS)/all_excl_ACCESS[0],
        width, label='ACCESS-C')
    axes.flatten()[1].set_yticks(x)
    axes.flatten()[1].set_yticklabels(labels)
    axes.flatten()[1].invert_yaxis()
    axes.flatten()[0].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.45),
        ncol=2, fancybox=True, shadow=True)
    axes.flatten()[1].set_xlabel('Ratio [-]')
    axes.flatten()[1].set_title('Exclusion Criteria Ratios')

    plt.subplots_adjust(wspace=.75)
    cl.make_subplot_labels(axes.flatten(), x_shift=-.15)

    if title is not None:
        plt.suptitle(title)


def plot_counts(
        all_obs_radar, all_obs_weak_radar, all_obs_active_radar,
        all_obs_ACCESS, all_obs_weak_ACCESS, all_obs_active_ACCESS,
        QC_obs_radar, QC_obs_weak_radar, QC_obs_active_radar,
        QC_obs_ACCESS, QC_obs_weak_ACCESS, QC_obs_active_ACCESS):

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    cl.init_fonts()

    labels = ['All', 'Weak Monsoon', 'Active Monsoon']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    radar = [all_obs_radar[3], all_obs_weak_radar[3], all_obs_active_radar[3]]
    ACCESS = [
        all_obs_ACCESS[3], all_obs_weak_ACCESS[3], all_obs_active_ACCESS[3]]

    QC_radar = [QC_obs_radar[3], QC_obs_weak_radar[3], QC_obs_active_radar[3]]
    QC_ACCESS = [
        QC_obs_ACCESS[3], QC_obs_weak_ACCESS[3], QC_obs_active_ACCESS[3]]

    ax.flatten()[0].bar(x-width/2, radar, width, label='Radar')
    ax.flatten()[0].bar(x+width/2, ACCESS, width, label='ACCESS-C')

    ax.flatten()[1].bar(x-width/2, QC_radar, width, label='Radar')
    ax.flatten()[1].bar(x+width/2, QC_ACCESS, width, label='ACCESS-C')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.flatten()[0].set_ylabel('Count [-]')
    ax.flatten()[0].set_title('Raw Observation Count')
    ax.flatten()[0].set_xticks(x)
    ax.flatten()[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[0].set_yticks(np.arange(0, 120000, 20000))
    ax.flatten()[0].set_xticklabels(labels)
    ax.flatten()[0].grid(which='major', alpha=0.5, axis='y')

    ax.flatten()[1].set_ylabel('Count [-]')
    ax.flatten()[1].set_title('Restricted Sample Observation Count')
    ax.flatten()[1].set_xticks(x)
    ax.flatten()[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[1].set_yticks(np.arange(0, 7000, 1000))

    ax.flatten()[1].set_xticklabels(labels)
    ax.flatten()[1].grid(which='major', alpha=0.5, axis='y')

    radar = [all_obs_radar[4], all_obs_weak_radar[4], all_obs_active_radar[4]]
    ACCESS = [
        all_obs_ACCESS[4], all_obs_weak_ACCESS[4], all_obs_active_ACCESS[4]]

    QC_radar = [QC_obs_radar[4], QC_obs_weak_radar[4], QC_obs_active_radar[4]]
    QC_ACCESS = [
        QC_obs_ACCESS[4], QC_obs_weak_ACCESS[4], QC_obs_active_ACCESS[4]]

    ax.flatten()[2].bar(x - width/2, radar, width, label='Radar')
    ax.flatten()[2].bar(x + width/2, ACCESS, width, label='ACCESS-C')

    ax.flatten()[3].bar(x - width/2, QC_radar, width, label='Radar')
    ax.flatten()[3].bar(x + width/2, QC_ACCESS, width, label='ACCESS-C')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.flatten()[2].set_ylabel('Count [-]')
    ax.flatten()[2].set_title('Raw System Count')
    ax.flatten()[2].set_xticks(x)
    ax.flatten()[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[2].set_yticks(np.arange(0, 8000, 1000))
    ax.flatten()[2].set_xticklabels(labels)
    ax.flatten()[2].grid(which='major', alpha=0.5, axis='y')

    ax.flatten()[3].set_ylabel('Count [-]')
    ax.flatten()[3].set_title('Restricted Sample System Count')
    ax.flatten()[3].set_xticks(x)
    ax.flatten()[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[3].set_yticks(np.arange(0, 900, 100))
    ax.flatten()[3].set_xticklabels(labels)
    ax.flatten()[3].grid(which='major', alpha=0.5, axis='y')

    ax.flatten()[2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.45),
        ncol=2, fancybox=True, shadow=True)

    cl.make_subplot_labels(ax.flatten(), x_shift=-.15)

    plt.subplots_adjust(hspace=.4)


def plot_counts_regional(
        all_obs_radar_42, all_obs_radar_63, all_obs_radar_77,
        all_obs_ACCESS_42, all_obs_ACCESS_63, all_obs_ACCESS_77,
        QC_obs_radar_42, QC_obs_radar_63, QC_obs_radar_77,
        QC_obs_ACCESS_42, QC_obs_ACCESS_63, QC_obs_ACCESS_77):

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    cl.init_fonts()

    labels = ['42: Katherine', '63: Berrimah', '77: Arafura']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    radar = [all_obs_radar_42[3], all_obs_radar_63[3], all_obs_radar_77[3]]
    ACCESS = [all_obs_ACCESS_42[3], all_obs_ACCESS_63[3], all_obs_ACCESS_77[3]]

    QC_radar = [QC_obs_radar_42[3], QC_obs_radar_63[3], QC_obs_radar_77[3]]
    QC_ACCESS = [QC_obs_ACCESS_42[3], QC_obs_ACCESS_63[3], QC_obs_ACCESS_77[3]]

    ax.flatten()[0].bar(x-width/2, radar, width, label='Radar')
    ax.flatten()[0].bar(x+width/2, ACCESS, width, label='ACCESS-C')

    ax.flatten()[1].bar(x-width/2, QC_radar, width, label='Radar')
    ax.flatten()[1].bar(x+width/2, QC_ACCESS, width, label='ACCESS-C')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.flatten()[0].set_ylabel('Count [-]')
    ax.flatten()[0].set_title('Raw Observation Count')
    ax.flatten()[0].set_xticks(x)
    ax.flatten()[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[0].set_yticks(np.arange(0, 45000, 5000))
    ax.flatten()[0].set_xticklabels(labels)
    ax.flatten()[0].grid(which='major', alpha=0.5, axis='y')

    ax.flatten()[1].set_ylabel('Count [-]')
    ax.flatten()[1].set_title('Restricted Sample Observation Count')
    ax.flatten()[1].set_xticks(x)
    ax.flatten()[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[1].set_yticks(np.arange(0, 3500, 500))

    ax.flatten()[1].set_xticklabels(labels)
    ax.flatten()[1].grid(which='major', alpha=0.5, axis='y')

    radar = [all_obs_radar_42[4], all_obs_radar_63[4], all_obs_radar_77[4]]
    ACCESS = [all_obs_ACCESS_42[4], all_obs_ACCESS_63[4], all_obs_ACCESS_77[4]]

    QC_radar = [QC_obs_radar_42[4], QC_obs_radar_63[4], QC_obs_radar_77[4]]
    QC_ACCESS = [QC_obs_ACCESS_42[4], QC_obs_ACCESS_63[4], QC_obs_ACCESS_77[4]]

    ax.flatten()[2].bar(x - width/2, radar, width, label='Radar')
    ax.flatten()[2].bar(x + width/2, ACCESS, width, label='ACCESS-C')

    ax.flatten()[3].bar(x - width/2, QC_radar, width, label='Radar')
    ax.flatten()[3].bar(x + width/2, QC_ACCESS, width, label='ACCESS-C')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.flatten()[2].set_ylabel('Count [-]')
    ax.flatten()[2].set_title('Raw System Count')
    ax.flatten()[2].set_xticks(x)
    ax.flatten()[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[2].set_yticks(np.arange(0, 3000, 500))
    ax.flatten()[2].set_xticklabels(labels)
    ax.flatten()[2].grid(which='major', alpha=0.5, axis='y')

    ax.flatten()[3].set_ylabel('Count [-]')
    ax.flatten()[3].set_title('Restricted Sample System Count')
    ax.flatten()[3].set_xticks(x)
    ax.flatten()[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[3].set_yticks(np.arange(0, 350, 50))
    ax.flatten()[3].set_xticklabels(labels)
    ax.flatten()[3].grid(which='major', alpha=0.5, axis='y')

    ax.flatten()[2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.45),
        ncol=2, fancybox=True, shadow=True)

    cl.make_subplot_labels(ax.flatten(), x_shift=-.15)

    plt.subplots_adjust(hspace=.4)


def compare_sizes(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None):

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    cl.init_fonts()

    ax.flatten()[0].hist(
        [all_obs_radar[0], all_obs_ACCESS[0]],
        bins=np.arange(80, 2500, 160), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[0].set_title('Raw Convective Areas')

    ax.flatten()[1].hist(
        [QC_obs_radar[0], QC_obs_ACCESS[0]],
        bins=np.arange(80, 2500, 160), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[1].set_title('Restricted Sample Convective Areas')

    ax.flatten()[2].hist(
        [all_obs_radar[1], all_obs_ACCESS[1]],
        bins=np.arange(800, 15000, 800), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[2].set_title('Raw Stratiform Areas')

    ax.flatten()[3].hist(
        [QC_obs_radar[1], QC_obs_ACCESS[1]],
        bins=np.arange(800, 15000, 800), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[3].set_title('Restricted Sample Stratiform Areas')

    ax.flatten()[2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.4),
        ncol=2, fancybox=True, shadow=True)

    for i in range(len(ax.flatten())):
        ax.flatten()[i].ticklabel_format(
            axis='y', style='sci', scilimits=(0, 0))
        ax.flatten()[i].grid(which='major', alpha=0.5, axis='y')
        if density:
            ax.flatten()[i].set_ylabel('Density [-]')
        else:
            ax.flatten()[i].set_ylabel('Count [-]')
        ax.flatten()[i].set_xlabel('Area [km$^2$]', labelpad=0)

    cl.make_subplot_labels(ax.flatten(), x_shift=-.1)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        plt.suptitle(title, fontsize=14)


def compare_velocities(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None):

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    cl.init_fonts()

    ax.flatten()[0].hist(
        [all_obs_radar[5], all_obs_ACCESS[5]],
        bins=np.arange(-20, 22, 2), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[0].set_title('Raw Zonal Velocities')

    ax.flatten()[1].hist(
        [QC_obs_radar[5], QC_obs_ACCESS[5]],
        bins=np.arange(-20, 22, 2), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[1].set_title('Restricted Sample Zonal Velocities')

    ax.flatten()[2].hist(
        [all_obs_radar[6], all_obs_ACCESS[6]],
        bins=np.arange(-20, 22, 2), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[2].set_title('Raw Meridional Velocities')

    ax.flatten()[3].hist(
        [QC_obs_radar[6], QC_obs_ACCESS[6]],
        bins=np.arange(-20, 22, 2), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[3].set_title('Restricted Sample Meridional Velocities')

    ax.flatten()[2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.5),
        ncol=2, fancybox=True, shadow=True)

    for i in range(len(ax.flatten())):
        ax.flatten()[i].ticklabel_format(
            axis='y', style='sci', scilimits=(0, 0))
        ax.flatten()[i].grid(which='major', alpha=0.5, axis='y')
        if density:
            ax.flatten()[i].set_ylabel('Density [-]')
        else:
            ax.flatten()[i].set_ylabel('Count [-]')
        ax.flatten()[i].set_xlabel('Velocity [m/s]', labelpad=0)

    cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        plt.suptitle(title, fontsize=14)


def compare_shape(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None):

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    cl.init_fonts()

    ax.flatten()[0].hist(
        [all_obs_radar[7], all_obs_ACCESS[7]],
        bins=np.arange(0, 391.25, 11.25), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[0].set_title('Raw Orientations')
    ax.flatten()[0].set_xticks(np.arange(0, 405, 45))

    ax.flatten()[1].hist(
        [QC_obs_radar[7], QC_obs_ACCESS[7]],
        bins=np.arange(0, 391.25, 11.25), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[1].set_title('Restricted Sample Orientations')
    ax.flatten()[1].set_xticks(np.arange(0, 405, 45))

    ax.flatten()[2].hist(
        [all_obs_radar[8], all_obs_ACCESS[8]],
        bins=np.arange(0, 1.1, .05), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[2].set_title('Raw Eccentricities')

    ax.flatten()[3].hist(
        [QC_obs_radar[8], QC_obs_ACCESS[8]],
        bins=np.arange(0, 1.1, .05), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[3].set_title('Restricted Sample Eccentricities')

    ax.flatten()[2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.5),
        ncol=2, fancybox=True, shadow=True)

    for i in range(len(ax.flatten())):
        ax.flatten()[i].ticklabel_format(
            axis='y', style='sci', scilimits=(0, 0))
        ax.flatten()[i].grid(which='major', alpha=0.5, axis='y')
        if density:
            ax.flatten()[i].set_ylabel('Density [-]')
        else:
            ax.flatten()[i].set_ylabel('Count [-]')

    ax.flatten()[2].set_xlabel('Eccentricity [-]', labelpad=0)
    ax.flatten()[3].set_xlabel('Eccentricity [-]', labelpad=0)
    ax.flatten()[0].set_xlabel('Orientation [degrees]', labelpad=0)
    ax.flatten()[1].set_xlabel('Orientation [degrees]', labelpad=0)

    cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        plt.suptitle(title, fontsize=14)


def compare_offset(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None):

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cl.init_fonts()

    ax.flatten()[0].hist(
        [np.array(all_obs_radar[9])/1e3, np.array(all_obs_ACCESS[9])/1e3],
        bins=np.arange(0, 105, 5), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[0].set_title('Raw Stratiform Offset Length')
    ax.flatten()[0].set_xticks(np.arange(0, 110, 10))

    ax.flatten()[1].hist(
        [np.array(QC_obs_radar[9])/1e3, np.array(QC_obs_ACCESS[9])/1e3],
        bins=np.arange(0, 105, 5), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[1].set_title('Restricted Sample Stratiform Offset Length')
    ax.flatten()[1].set_xticks(np.arange(0, 110, 10))

    ax.flatten()[0].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.3),
        ncol=2, fancybox=True, shadow=True)

    for i in range(len(ax.flatten())):
        ax.flatten()[i].ticklabel_format(
            axis='y', style='sci', scilimits=(0, 0))
        ax.flatten()[i].grid(which='major', alpha=0.5, axis='y')
        if density:
            ax.flatten()[i].set_ylabel('Density [-]')
        else:
            ax.flatten()[i].set_ylabel('Count [-]')

    ax.flatten()[0].set_xlabel('Stratiform Offset Magnitude [m]', labelpad=0)
    ax.flatten()[1].set_xlabel('Stratiform Offset Magnitude [m]', labelpad=0)

    cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        plt.suptitle(title, fontsize=14)


def compare_time(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None):

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cl.init_fonts()

    a_hour = [int(s.astype(str)[11:13]) for s in all_obs_radar[2]]
    b_hour = [int(s.astype(str)[11:13]) for s in all_obs_ACCESS[2]]
    c_hour = [int(s.astype(str)[11:13]) for s in QC_obs_radar[2]]
    d_hour = [int(s.astype(str)[11:13]) for s in QC_obs_ACCESS[2]]

    ax.flatten()[0].hist(
        [a_hour, b_hour],
        bins=np.arange(0, 25, 1), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[0].set_title('Raw Observation Count')
    ax.flatten()[0].set_xticks(np.arange(0, 25, 2))

    ax.flatten()[1].hist(
        [c_hour, d_hour],
        bins=np.arange(0, 25, 1), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[1].set_title('Restricted Observation Count')
    ax.flatten()[1].set_xticks(np.arange(0, 25, 2))

    ax.flatten()[0].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.3),
        ncol=2, fancybox=True, shadow=True)

    for i in range(len(ax.flatten())):
        ax.flatten()[i].ticklabel_format(
            axis='y', style='sci', scilimits=(0, 0))
        ax.flatten()[i].grid(which='major', alpha=0.5, axis='y')
        if density:
            ax.flatten()[i].set_ylabel('Density [-]')
        else:
            ax.flatten()[i].set_ylabel('Count [-]')

    ax.flatten()[0].set_xlabel('Time [hour UST]', labelpad=0)
    ax.flatten()[1].set_xlabel('Time [hour UST]', labelpad=0)

    cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        plt.suptitle(title, fontsize=14)
