import sys
sys.path.insert(0, '/home/student.unimelb.edu.au/shorte1/Documents/TINT')
sys.path.insert(0, '/home/563/esh563/TINT')

import pickle
import numpy as np
import classification as cl
import matplotlib.pyplot as plt


def get_all_and_QC_radar_stats(
        save_dir, class_thresh=None, excl_thresh=None,
        exclusions=None, morning_only=False, radars=[42, 63, 77]):
    all_obs_radar = get_boring_radar_stats(
        save_dir, class_thresh=None, excl_thresh=None,
        exclusions=['simple_duration_cond'], regime=None,
        radars=radars, morning_only=morning_only)
    QC_obs_radar = get_boring_radar_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=radars,
        morning_only=morning_only)

    all_obs_weak_radar = get_boring_radar_stats(
        save_dir, exclusions=['simple_duration_cond'],
        class_thresh=None, excl_thresh=None,
        regime=1, radars=radars, morning_only=morning_only)
    QC_obs_weak_radar = get_boring_radar_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=1,
        radars=radars, morning_only=morning_only)

    all_obs_active_radar = get_boring_radar_stats(
        save_dir, exclusions=['simple_duration_cond'],
        class_thresh=None, excl_thresh=None, regime=2,
        radars=radars, morning_only=morning_only)
    QC_obs_active_radar = get_boring_radar_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=2, radars=radars,
        morning_only=morning_only)

    all_radar = [
        all_obs_radar, QC_obs_radar, all_obs_weak_radar, QC_obs_weak_radar,
        all_obs_active_radar, QC_obs_active_radar]
    return all_radar


def get_all_and_QC_ACCESS_stats(
        save_dir, exclusions=None, class_thresh=None,
        excl_thresh=None, morning_only=False, radars=[42, 63, 77]):
    all_obs_ACCESS = get_boring_ACCESS_stats(
        save_dir, class_thresh=None, excl_thresh=None,
        exclusions=['simple_duration_cond'], regime=None,
        radars=radars, morning_only=morning_only)
    QC_obs_ACCESS = get_boring_ACCESS_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None,
        radars=radars, morning_only=morning_only)

    all_obs_weak_ACCESS = get_boring_ACCESS_stats(
        save_dir, class_thresh=None, excl_thresh=None,
        exclusions=['simple_duration_cond'], regime=1,
        radars=radars, morning_only=morning_only)
    QC_obs_weak_ACCESS = get_boring_ACCESS_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=1,
        radars=radars, morning_only=morning_only)

    all_obs_active_ACCESS = get_boring_ACCESS_stats(
        save_dir, class_thresh=None, excl_thresh=None,
        exclusions=['simple_duration_cond'], regime=2,
        radars=radars, morning_only=morning_only)
    QC_obs_active_ACCESS = get_boring_ACCESS_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=2,
        radars=radars, morning_only=morning_only)

    all_ACCESS = [
        all_obs_ACCESS, QC_obs_ACCESS, all_obs_weak_ACCESS, QC_obs_weak_ACCESS,
        all_obs_active_ACCESS, QC_obs_active_ACCESS]

    return all_ACCESS


def get_all_time_series(
        save_dir, exclusions=None, class_thresh=None,
        excl_thresh=None, morning_only=False, radars=[42, 63, 77]):

    time_series_radar = get_radar_prop_so_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=radars)

    time_series_weak_radar = get_radar_prop_so_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=1, radars=radars)

    time_series_active_radar = get_radar_prop_so_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=2, radars=radars)

    time_series_ACCESS = get_ACCESS_prop_so_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=radars)

    time_series_weak_ACCESS = get_ACCESS_prop_so_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=1, radars=radars)

    time_series_active_ACCESS = get_ACCESS_prop_so_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=2, radars=radars)

    time_series_all = [
        time_series_radar, time_series_weak_radar, time_series_active_radar,
        time_series_ACCESS, time_series_weak_ACCESS, time_series_active_ACCESS]

    return time_series_all


def get_all_regional(
        save_dir, exclusions=None,
        class_thresh=None, excl_thresh=None):

    all_obs_ACCESS_42 = get_boring_ACCESS_stats(
        save_dir, exclusions=['simple_duration_cond'], class_thresh=None,
        excl_thresh=None, regime=None, radars=[42])
    QC_obs_ACCESS_42 = get_boring_ACCESS_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=[42])

    all_obs_ACCESS_63 = get_boring_ACCESS_stats(
        save_dir,  exclusions=['simple_duration_cond'], class_thresh=None,
        excl_thresh=None, regime=None, radars=[63])
    QC_obs_ACCESS_63 = get_boring_ACCESS_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=[63])

    all_obs_ACCESS_77 = get_boring_ACCESS_stats(
        save_dir,  exclusions=['simple_duration_cond'], class_thresh=None,
        excl_thresh=None, regime=None, radars=[77])
    QC_obs_ACCESS_77 = get_boring_ACCESS_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=[77])

    all_obs_radar_42 = get_boring_radar_stats(
        save_dir,  exclusions=['simple_duration_cond'], class_thresh=None,
        excl_thresh=None, regime=None, radars=[42])
    QC_obs_radar_42 = get_boring_radar_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=[42])

    all_obs_radar_63 = get_boring_radar_stats(
        save_dir,  exclusions=['simple_duration_cond'], class_thresh=None,
        excl_thresh=None, regime=None, radars=[63])
    QC_obs_radar_63 = get_boring_radar_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=[63])

    all_obs_radar_77 = get_boring_radar_stats(
        save_dir,  exclusions=['simple_duration_cond'], class_thresh=None,
        excl_thresh=None, regime=None, radars=[77])
    QC_obs_radar_77 = get_boring_radar_stats(
        save_dir, exclusions=exclusions, class_thresh=class_thresh,
        excl_thresh=excl_thresh, regime=None, radars=[77])

    all_obs_regional = [
        all_obs_ACCESS_42, QC_obs_ACCESS_42, all_obs_ACCESS_63, QC_obs_ACCESS_63,
        all_obs_ACCESS_77, QC_obs_ACCESS_77, all_obs_radar_42, QC_obs_radar_42,
        all_obs_radar_63, QC_obs_radar_63, all_obs_radar_77, QC_obs_radar_77]

    return all_obs_regional


def plot_regional_seasonal_and_so(
        all_obs_regional, all_radar, all_ACCESS, fig_dir, suff):
    density = False

    [
        all_obs_radar, QC_obs_radar, all_obs_weak_radar, QC_obs_weak_radar,
        all_obs_active_radar, QC_obs_active_radar] = all_radar

    [
        all_obs_ACCESS, QC_obs_ACCESS, all_obs_weak_ACCESS, QC_obs_weak_ACCESS,
        all_obs_active_ACCESS, QC_obs_active_ACCESS] = all_ACCESS

    [
        all_obs_ACCESS_42, QC_obs_ACCESS_42,
        all_obs_ACCESS_63, QC_obs_ACCESS_63,
        all_obs_ACCESS_77, QC_obs_ACCESS_77,
        all_obs_radar_42, QC_obs_radar_42,
        all_obs_radar_63, QC_obs_radar_63,
        all_obs_radar_77, QC_obs_radar_77] = all_obs_regional

    fig, axes = plt.subplots(3, 2, figsize=(12, 7))

    plot_counts_regional_seasonal(
        all_obs_radar, all_obs_weak_radar, all_obs_active_radar,
        all_obs_ACCESS, all_obs_weak_ACCESS, all_obs_active_ACCESS,
        QC_obs_radar, QC_obs_weak_radar, QC_obs_active_radar,
        QC_obs_ACCESS, QC_obs_weak_ACCESS, QC_obs_active_ACCESS,
        all_obs_radar_42, all_obs_radar_63, all_obs_radar_77,
        all_obs_ACCESS_42, all_obs_ACCESS_63, all_obs_ACCESS_77,
        QC_obs_radar_42, QC_obs_radar_63, QC_obs_radar_77,
        QC_obs_ACCESS_42, QC_obs_ACCESS_63, QC_obs_ACCESS_77,
        fig=fig, ax=axes[:2, :], legend=False, sp_labels=False)

    compare_offset(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=density,
        fig=fig, ax=axes[2, :])

    cl.make_subplot_labels(axes.flatten(), x_shift=-.13)

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.65),
        ncol=2, fancybox=True, shadow=True)

    plt.savefig(
        fig_dir + '/regional_seasonal_counts_offset_{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def plot_all_diurnal(all_radar, all_ACCESS, fig_dir, suff):

    [
        all_obs_radar, QC_obs_radar, all_obs_weak_radar, QC_obs_weak_radar,
        all_obs_active_radar, QC_obs_active_radar] = all_radar

    [
        all_obs_ACCESS, QC_obs_ACCESS, all_obs_weak_ACCESS, QC_obs_weak_ACCESS,
        all_obs_active_ACCESS, QC_obs_active_ACCESS] = all_ACCESS

    density = False

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    compare_time(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=density,
        title='All Monsoon Regimes', fig=fig, ax=axes[0, :])

    compare_time(
        all_obs_weak_radar, all_obs_weak_ACCESS,
        QC_obs_weak_radar, QC_obs_weak_ACCESS, density=density,
        title='Weak Monsoon', fig=fig, ax=axes[1, :])

    compare_time(
        all_obs_active_radar, all_obs_active_ACCESS,
        QC_obs_active_radar, QC_obs_active_ACCESS, density=density,
        title='Active Monsoon', fig=fig, ax=axes[2, :])

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -1.10),
        ncol=2, fancybox=True, shadow=True)

    plt.subplots_adjust(hspace=1.1)

    cl.make_subplot_labels(axes.flatten(), x_shift=-.14)

    maximums = [10e3, 4e2, 8e3, 3e2, 2e3, 2.5e2]
    dy = [2e3, 1e2, 2e3, .5e2, .5e3, .5e2]

    for i in range(len(axes.flatten())):
        axes.flatten()[i].set_yticks(np.arange(0, maximums[i]+dy[i], dy[i]))
        axes.flatten()[i].set_yticks(
            np.arange(0, maximums[i]+dy[i]/2, dy[i]/2), minor=True)
        axes.flatten()[i].grid(which='minor', alpha=0.2, axis='y')
        axes.flatten()[i].grid(which='major', alpha=0.5, axis='y')

    plt.savefig(
        fig_dir + '/time_ACCESS_radar_active_compare_{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def plot_all_orientations(all_radar, all_ACCESS, fig_dir, suff):
    fig, axes = plt.subplots(3, 2, figsize=(12, 7))

    density = False

    [
        all_obs_radar, QC_obs_radar, all_obs_weak_radar, QC_obs_weak_radar,
        all_obs_active_radar, QC_obs_active_radar] = all_radar

    [
        all_obs_ACCESS, QC_obs_ACCESS, all_obs_weak_ACCESS, QC_obs_weak_ACCESS,
        all_obs_active_ACCESS, QC_obs_active_ACCESS] = all_ACCESS

    compare_orientation(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=density,
        title='All Monsoon Regimes', fig=fig, ax=axes[0, :])

    compare_orientation(
        all_obs_weak_radar, all_obs_weak_ACCESS,
        QC_obs_weak_radar, QC_obs_weak_ACCESS, density=density,
        title='Weak Monsoon', fig=fig, ax=axes[1, :])

    compare_orientation(
        all_obs_active_radar, all_obs_active_ACCESS,
        QC_obs_active_radar, QC_obs_active_ACCESS, density=density,
        title='Active Monsoon', fig=fig, ax=axes[2, :])

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.75),
        ncol=2, fancybox=True, shadow=True)

    plt.subplots_adjust(hspace=.65)

    cl.make_subplot_labels(axes.flatten(), x_shift=-.14)

    maximums = [8e3, 4e2, 8e3, 2.5e2, 2e3, 2e2]
    dy = [2e3, 1e2, 2e3, .5e2, .5e3, .5e2]

    for i in range(len(axes.flatten())):
        axes.flatten()[i].set_yticks(np.arange(0, maximums[i]+dy[i], dy[i]))
        axes.flatten()[i].set_yticks(
            np.arange(0, maximums[i]+dy[i]/2, dy[i]/2), minor=True)
        axes.flatten()[i].grid(which='minor', alpha=0.2, axis='y')
        axes.flatten()[i].grid(which='major', alpha=0.5, axis='y')

    plt.savefig(
        fig_dir + '/orientation_ACCESS_radar_active_compare_{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def plot_all_time_series(time_series_all, fig_dir, suff):

    [
        time_series_radar, time_series_weak_radar,
        time_series_active_radar, time_series_ACCESS,
        time_series_weak_ACCESS, time_series_active_ACCESS] = time_series_all

    fig, axes = plt.subplots(5, 3, figsize=(12, 12))

    cl.init_fonts()

    x = np.arange(0, 24*60+10, 10)

    radar_d_list = [
        time_series_radar, time_series_weak_radar, time_series_active_radar]
    ACCESS_d_list = [
        time_series_ACCESS, time_series_weak_ACCESS, time_series_active_ACCESS]

    y_lims = [(0, 4000), (0, 40), (0, 8), (0, 2000), (0, 10000)]
    dy = [1000, 10, 2, 500, 2000]
    # y_labels = [
    #     r'$|\mathrm{\mathbf{s}}|$ [km]',
    #     r'$|\mathrm{\mathbf{v}}_r|$ [m/s]',
    #     r'Convective Area [km$^2$]']
    y_labels = [
        r'Observation Count [-]',
        r'Stratiform Offset [km]',
        r'Flow-Relative Speed [m/s]',
        r'Convective Area [km$^2$]',
        r'Stratiform Area [km$^2$]']

    for k in range(axes.shape[1]):
        j = 0
        axes[j, k].plot(x, radar_d_list[k][0], '', label='Radar Count')

        axes[j, k].plot(x, ACCESS_d_list[k][0], label='ACCESS Mean')

        y_lim = y_lims[j]

        axes[j, k].set_xlim([0, 240])
        axes[j, k].set_xticks(np.arange(0, 270, 30))
        axes[j, k].set_yticks(
            np.arange(y_lim[0], y_lim[1]+dy[j], dy[j]))
        axes[j, k].set_yticks(
            np.arange(y_lim[0], y_lim[1]+int(dy[j]/2), int(dy[j]/2)),
            minor=True)

        axes[j, k].set_ylim(y_lim)
        axes[j, k].set_ylabel(y_labels[j])
        axes[j, k].grid(which='major', alpha=0.5, axis='both')
        axes[j, k].grid(which='minor', alpha=0.2, axis='y')

        axes[j, k].ticklabel_format(
            axis='y', style='sci', scilimits=(0, 0))

    for j in range(1, axes.shape[0]):
        for k in range(axes.shape[1]):

            index = j

            radar_d = radar_d_list[k]
            ACCESS_d = ACCESS_d_list[k]

            [
                means_radar, sig_radar, means_ACCESS, sig_ACCESS] = [
                np.zeros(len(radar_d[index])) for i in range(4)]

            lens_radar = np.array(
                [len(radar_d[index][i]) for i in range(len(radar_d[index]))])
            lens_ACCESS = np.array(
                [len(ACCESS_d[index][i]) for i in range(len(ACCESS_d[index]))])

            for i in range(len(radar_d[index])):
                if lens_radar[i] > 0:
                    means_radar[i] = np.nanmean(radar_d[index][i])
                    sig_radar[i] = np.sqrt(np.nanvar(radar_d[index][i]))
                if lens_ACCESS[i] > 0:
                    means_ACCESS[i] = np.array(np.nanmean(ACCESS_d[index][i]))
                    sig_ACCESS[i] = np.sqrt(np.nanvar(ACCESS_d[index][i]))

            if j == 1:
                means_radar = means_radar/1000
                sig_radar = sig_radar/1000
                means_ACCESS = means_ACCESS/1000
                sig_ACCESS = sig_ACCESS/1000

            axes[j, k].plot(x, means_radar, '', label='Radar Mean')
            axes[j, k].fill_between(
                x, means_radar+sig_radar, means_radar-sig_radar, alpha=0.2,
                label='Radar Standard Deviation')

            axes[j, k].plot(x, means_ACCESS, label='ACCESS Mean')
            axes[j, k].fill_between(
                x, means_ACCESS+sig_ACCESS, means_ACCESS-sig_ACCESS, alpha=0.2,
                label='ACCESS Standard Deviation')

            y_lim = y_lims[j]

            axes[j, k].set_xlim([0, 240])
            axes[j, k].set_xticks(np.arange(0, 270, 30))
            axes[j, k].set_yticks(
                np.arange(y_lim[0], y_lim[1]+dy[j], dy[j]))
            axes[j, k].set_yticks(
                np.arange(y_lim[0], y_lim[1]+int(dy[j]/2), int(dy[j]/2)),
                minor=True)

            axes[j, k].set_ylim(y_lim)
            axes[j, k].set_ylabel(y_labels[j])
            axes[j, k].grid(which='major', alpha=0.5, axis='both')
            axes[j, k].grid(which='minor', alpha=0.2, axis='y')

            if j == 3 or j == 4:
                axes[j, k].ticklabel_format(
                    axis='y', style='sci', scilimits=(0, 0))

    for j in range(3):
        axes[4, j].set_xlabel('Time since Detection [min]')

    titles = ['All Regimes', 'Weak Monsoon', 'Active Monsoon']

    for j in range(3):
        axes[0, j].set_title(titles[j], fontsize=12)

    cl.make_subplot_labels(axes.flatten(), x_shift=-.3, y_shift=.07)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    lines, labels = axes[-1, 0].get_legend_handles_labels()

    lines = [lines[i] for i in [0, 2, 1, 3]]
    labels = [labels[i] for i in [0, 2, 1, 3]]

    axes[-1, 0].legend(
        loc='lower center', bbox_to_anchor=(1.7, -0.75),
        ncol=4, fancybox=True, shadow=True)

    plt.savefig(
        fig_dir + '/time_series_{}.png'.format(suff),
        dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')


def plot_all_eccentricities(all_radar, all_ACCESS, fig_dir, suff):

    density = False

    [
        all_obs_radar, QC_obs_radar, all_obs_weak_radar, QC_obs_weak_radar,
        all_obs_active_radar, QC_obs_active_radar] = all_radar

    [
        all_obs_ACCESS, QC_obs_ACCESS, all_obs_weak_ACCESS, QC_obs_weak_ACCESS,
        all_obs_active_ACCESS, QC_obs_active_ACCESS] = all_ACCESS

    compare_eccentricity(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=density,
        title='All Monsoon Regimes')

    plt.savefig(
        fig_dir + '/shape_ACCESS_radar_all_{}.png'.format(suff),
        dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')

    compare_eccentricity(
        all_obs_weak_radar, all_obs_weak_ACCESS,
        QC_obs_weak_radar, QC_obs_weak_ACCESS, density=density,
        title='Weak Monsoon')

    plt.savefig(
        fig_dir + '/shape_ACCESS_radar_weak_{}.png'.format(suff),
        dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')

    compare_eccentricity(
        all_obs_active_radar, all_obs_active_ACCESS,
        QC_obs_active_radar, QC_obs_active_ACCESS, density=density,
        title='Active Monsoon')

    plt.savefig(
        fig_dir + '/shape_ACCESS_radar_active_{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def plot_all_velocities(all_radar, all_ACCESS, fig_dir, suff):

    density = False

    [
        all_obs_radar, QC_obs_radar, all_obs_weak_radar, QC_obs_weak_radar,
        all_obs_active_radar, QC_obs_active_radar] = all_radar

    [
        all_obs_ACCESS, QC_obs_ACCESS, all_obs_weak_ACCESS, QC_obs_weak_ACCESS,
        all_obs_active_ACCESS, QC_obs_active_ACCESS] = all_ACCESS

    fig, axes = plt.subplots(3, 2, figsize=(12, 7))

    compare_velocities(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=density,
        title='All Monsoon Regimes', labels=False,
        legend=False, fig=fig, ax=axes[0, :])

    compare_velocities(
        all_obs_weak_radar, all_obs_weak_ACCESS,
        QC_obs_weak_radar, QC_obs_weak_ACCESS, density=density, labels=False,
        title='Weak Monsoon', legend=False, fig=fig, ax=axes[1, :])

    compare_velocities(
        all_obs_active_radar, all_obs_active_ACCESS,
        QC_obs_active_radar, QC_obs_active_ACCESS, density=density,
        labels=False, title='Active Monsoon', fig=fig, ax=axes[2, :])

    plt.subplots_adjust(hspace=.65)

    cl.make_subplot_labels(axes.flatten(), x_shift=-.14)

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.75),
        ncol=2, fancybox=True, shadow=True)

    maximums = [2.5e4, 10e2, 2e4, 10e2, 8e3, 8e2]
    dy = [.5e4, 2e2, .5e4, 2e2, 2e3, 2e2]

    for i in range(len(axes.flatten())):
        axes.flatten()[i].set_yticks(np.arange(0, maximums[i]+dy[i], dy[i]))
        axes.flatten()[i].set_yticks(
            np.arange(0, maximums[i]+dy[i]/2, dy[i]/2), minor=True)
        axes.flatten()[i].grid(which='minor', alpha=0.2, axis='y')
        axes.flatten()[i].grid(which='major', alpha=0.5, axis='y')

    plt.savefig(
        fig_dir + '/zonal_ACCESS_radar_active_compare_{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')

    fig, axes = plt.subplots(3, 2, figsize=(12, 7))

    direction='meri'

    compare_velocities(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=density,
        title='All Monsoon Regimes', labels=False,
        legend=False, fig=fig, ax=axes[0, :], direction=direction)

    compare_velocities(
        all_obs_weak_radar, all_obs_weak_ACCESS,
        QC_obs_weak_radar, QC_obs_weak_ACCESS, density=density, labels=False,
        title='Weak Monsoon', legend=False, fig=fig, ax=axes[1, :],
        direction=direction)

    compare_velocities(
        all_obs_active_radar, all_obs_active_ACCESS,
        QC_obs_active_radar, QC_obs_active_ACCESS, density=density,
        labels=False, title='Active Monsoon', fig=fig, ax=axes[2, :],
        direction=direction)

    plt.subplots_adjust(hspace=.65)

    cl.make_subplot_labels(axes.flatten(), x_shift=-.14)

    axes.flatten()[-2].legend(
        loc='lower center', bbox_to_anchor=(1.1, -0.75),
        ncol=2, fancybox=True, shadow=True)

    maximums = [2.5e4, 10e2, 2e4, 10e2, 8e3, 8e2]
    dy = [.5e4, 2e2, .5e4, 2e2, 2e3, 2e2]

    for i in range(len(axes.flatten())):
        axes.flatten()[i].set_yticks(np.arange(0, maximums[i]+dy[i], dy[i]))
        axes.flatten()[i].set_yticks(
            np.arange(0, maximums[i]+dy[i]/2, dy[i]/2), minor=True)
        axes.flatten()[i].grid(which='minor', alpha=0.2, axis='y')
        axes.flatten()[i].grid(which='major', alpha=0.5, axis='y')

    plt.savefig(
        fig_dir + '/meridional_ACCESS_radar_active_compare_{}.png'.format(suff),
        dpi=200, facecolor='w', edgecolor='white', bbox_inches='tight')


def get_boring_radar_stats(
        save_dir, exclusions=None, class_thresh=None, excl_thresh=None,
        regime=None, pope_dir=None, radars=[63, 42, 77], morning_only=False):

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

                tracks_obj = cl.redo_exclusions(
                    tracks_obj, class_thresh=class_thresh,
                    excl_thresh=excl_thresh)
                tracks_obj = cl.add_monsoon_regime(
                    tracks_obj, base_dir=pope_dir, fake_pope=True)

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
        save_dir, class_thresh=None, excl_thresh=None,
        exclusions=None, regime=None, pope_dir=None,
        radars=[63, 42, 77], morning_only=False):

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

            tracks_obj = cl.redo_exclusions(
                tracks_obj, class_thresh=class_thresh,
                excl_thresh=excl_thresh)
            tracks_obj = cl.add_monsoon_regime(
                tracks_obj, base_dir=pope_dir, fake_pope=True)

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


def get_boring_CPOL_stats(
        save_dir, exclusions=None, regime=None, pope_dir=None):

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

    years = sorted(list(set(range(1998, 2016)) - {2007, 2008, 2000}))
    for year in years:
        print('Year {}.'.format(year))
        path = save_dir + '{}1001_{}0501.pkl'.format(year, year+1)
        with open(path, 'rb') as f:
            tracks_obj = pickle.load(f)

        tracks_obj = cl.redo_exclusions(tracks_obj)
        tracks_obj = cl.add_monsoon_regime(
            tracks_obj, base_dir=pope_dir, fake_pope=True)

        excluded = tracks_obj.exclusions[exclusions]
        excluded = np.any(excluded, 1)
        included = np.logical_not(excluded)

        ind_1 = len(tracks_obj.params['LEVELS'])-1

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
            sub_tracks_strat = sub_tracks_all.xs(ind_1, level='level')
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


def count_CPOL_exclusions(
        save_dir, regime=None, pope_dir=None):
    if pope_dir is None:
        pope_dir = '/home/student.unimelb.edu.au/shorte1/'
        pope_dir += 'Documents/CPOL_analysis/'

    exclusions = ['simple_duration_cond']
    exclusions_list = [
        'intersect_border', 'intersect_border_convective',
        'duration_cond', 'small_velocity', 'small_offset']

    exclusions_counts = [0 for i in range(len(exclusions_list)+1)]

    years = sorted(list(set(range(1998, 2016)) - {2007, 2008, 2000}))
    for year in years:
        print('Year {}.'.format(year))
        path = save_dir + '{}1001_{}0501.pkl'.format(year, year+1)
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

    # Add some text for labels, title and cUTCom x-axis tick labels, etc.
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

    # Add some text for labels, title and cUTCom x-axis tick labels, etc.
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

    # Add some text for labels, title and cUTCom x-axis tick labels, etc.
    ax.flatten()[0].set_ylabel('Count [-]')
    ax.flatten()[0].set_title('Raw Observation Count')
    ax.flatten()[0].set_xticks(x)
    ax.flatten()[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[0].set_yticks(np.arange(0, 45000, 5000))
    ax.flatten()[0].set_yticks(np.arange(0, 45000, 2500), minor=True)
    ax.flatten()[0].set_xticklabels(labels)
    ax.flatten()[0].grid(which='major', alpha=0.5, axis='y')

    ax.flatten()[1].set_ylabel('Count [-]')
    ax.flatten()[1].set_title('Restricted Sample Observation Count')
    ax.flatten()[1].set_xticks(x)
    ax.flatten()[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[1].set_yticks(np.arange(0, 3500, 500))
    ax.flatten()[1].set_yticks(np.arange(0, 3500, 250), minor=True)

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


def plot_counts_regional_seasonal(
        all_obs_radar, all_obs_weak_radar, all_obs_active_radar,
        all_obs_ACCESS, all_obs_weak_ACCESS, all_obs_active_ACCESS,
        QC_obs_radar, QC_obs_weak_radar, QC_obs_active_radar,
        QC_obs_ACCESS, QC_obs_weak_ACCESS, QC_obs_active_ACCESS,
        all_obs_radar_42, all_obs_radar_63, all_obs_radar_77,
        all_obs_ACCESS_42, all_obs_ACCESS_63, all_obs_ACCESS_77,
        QC_obs_radar_42, QC_obs_radar_63, QC_obs_radar_77,
        QC_obs_ACCESS_42, QC_obs_ACCESS_63, QC_obs_ACCESS_77,
        fig=None, ax=None, sp_labels=False, legend=False):

    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 5))
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
    ax.flatten()[0].set_yticks(np.arange(0, 110000, 20000/2), minor=True)
    ax.flatten()[0].set_xticklabels(labels)
    ax.flatten()[0].grid(which='major', alpha=0.5, axis='y')
    ax.flatten()[0].grid(which='minor', alpha=0.2, axis='y')

    ax.flatten()[1].set_ylabel('Count [-]')
    ax.flatten()[1].set_title('Restricted Sample Observation Count')
    ax.flatten()[1].set_xticks(x)
    ax.flatten()[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[1].set_yticks(np.arange(0, 7000, 1000))
    ax.flatten()[1].set_yticks(np.arange(0, 6500, 500), minor=True)

    ax.flatten()[1].set_xticklabels(labels)
    ax.flatten()[1].grid(which='major', alpha=0.5, axis='y')
    ax.flatten()[1].grid(which='minor', alpha=0.2, axis='y')

    labels = ['Tindal', 'Berrimah', 'Arafura']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    radar = [all_obs_radar_42[3], all_obs_radar_63[3], all_obs_radar_77[3]]
    ACCESS = [all_obs_ACCESS_42[3], all_obs_ACCESS_63[3], all_obs_ACCESS_77[3]]

    QC_radar = [QC_obs_radar_42[3], QC_obs_radar_63[3], QC_obs_radar_77[3]]
    QC_ACCESS = [QC_obs_ACCESS_42[3], QC_obs_ACCESS_63[3], QC_obs_ACCESS_77[3]]

    ax.flatten()[2].bar(x-width/2, radar, width, label='Radar')
    ax.flatten()[2].bar(x+width/2, ACCESS, width, label='ACCESS-C')

    ax.flatten()[3].bar(x-width/2, QC_radar, width, label='Radar')
    ax.flatten()[3].bar(x+width/2, QC_ACCESS, width, label='ACCESS-C')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.flatten()[2].set_ylabel('Count [-]')
    ax.flatten()[2].set_title('Raw Observation Count')
    ax.flatten()[2].set_xticks(x)
    ax.flatten()[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[2].set_yticks(np.arange(0, 50000, 10000))
    ax.flatten()[2].set_yticks(np.arange(0, 45000, 10000/2), minor=True)
    ax.flatten()[2].set_xticklabels(labels)
    ax.flatten()[2].grid(which='major', alpha=0.5, axis='y')
    ax.flatten()[2].grid(which='minor', alpha=0.2, axis='y')

    ax.flatten()[3].set_ylabel('Count [-]')
    ax.flatten()[3].set_title('Restricted Sample Observation Count')
    ax.flatten()[3].set_xticks(x)
    ax.flatten()[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.flatten()[3].set_yticks(np.arange(0, 3500, 500))
    ax.flatten()[3].set_yticks(np.arange(0, 3250, 500/2), minor=True)

    ax.flatten()[3].set_xticklabels(labels)
    ax.flatten()[3].grid(which='major', alpha=0.5, axis='y')
    ax.flatten()[3].grid(which='minor', alpha=0.2, axis='y')

    if legend:
        ax.flatten()[2].legend(
            loc='lower center', bbox_to_anchor=(1.1, -0.45),
            ncol=2, fancybox=True, shadow=True)

    if sp_labels:
        cl.make_subplot_labels(ax.flatten(), x_shift=-.15)

    plt.subplots_adjust(hspace=.4)


def compare_all_sizes(all_radar, all_ACCESS, fig_dir, suff):

    density = False

    [
        all_obs_radar, QC_obs_radar, all_obs_weak_radar, QC_obs_weak_radar,
        all_obs_active_radar, QC_obs_active_radar] = all_radar

    [
        all_obs_ACCESS, QC_obs_ACCESS, all_obs_weak_ACCESS, QC_obs_weak_ACCESS,
        all_obs_active_ACCESS, QC_obs_active_ACCESS] = all_ACCESS

    compare_sizes(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=density,
        title='All Monsoon Regimes')
    plt.savefig(
        fig_dir + '/sizes_ACCESS_radar_all_{}.png'.format(suff),
        dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')

    compare_sizes(
        all_obs_weak_radar, all_obs_weak_ACCESS,
        QC_obs_weak_radar, QC_obs_weak_ACCESS, density=density,
        title='Weak Monsoon')
    plt.savefig(
        fig_dir + '/sizes_ACCESS_radar_weak_{}.png'.format(suff),
        dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')

    compare_sizes(
        all_obs_active_radar, all_obs_active_ACCESS,
        QC_obs_active_radar, QC_obs_active_ACCESS, density=density,
        title='Active Monsoon')
    plt.savefig(
        fig_dir + '/sizes_ACCESS_radar_active_{}.png'.format(suff),
        dpi=200, facecolor='w',
        edgecolor='white', bbox_inches='tight')


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
        QC_obs_radar, QC_obs_ACCESS,
        fig=None, ax=None, density=True, title=None,
        labels=False, legend=False, direction='Zonal'):

    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 4))
    cl.init_fonts()

    if direction == 'Zonal':
        ind = 5
    else:
        ind = 6

    ax.flatten()[0].hist(
        [all_obs_radar[ind], all_obs_ACCESS[ind]],
        bins=np.arange(-20, 22, 2), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[0].set_title('Raw {} Velocities'.format(direction))

    ax.flatten()[1].hist(
        [QC_obs_radar[ind], QC_obs_ACCESS[ind]],
        bins=np.arange(-20, 22, 2), label=['Radar', 'ACCESS-C'],
        density=density)

    ax.flatten()[1].set_title(
        'Restricted Sample {} Velocities'.format(direction))

    if legend:
        ax.flatten()[0].legend(
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

    if labels:
        cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        ax.flatten()[0].text(
            1.08, 1.25, title, size=14, ha='center',
            transform=ax.flatten()[0].transAxes)


def compare_orientation(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None,
        legend=False, labels=False, fig=None, ax=None):

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 2))
    cl.init_fonts()

    ax.flatten()[0].hist(
        [all_obs_radar[7], all_obs_ACCESS[7]],
        bins=np.arange(0, 391.25, 11.25), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[0].set_title('Raw Orientations')
    ax.flatten()[0].set_xticks(np.arange(0, 405, 45))

    ax.flatten()[1].hist(
        [QC_obs_radar[7], QC_obs_ACCESS[7]],
        bins=np.arange(0, 391.25, 11.25), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[1].set_title('Restricted Sample Orientations')
    ax.flatten()[1].set_xticks(np.arange(0, 405, 45))

    if legend:
        ax.flatten()[0].legend(
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

    ax.flatten()[0].set_xlabel('Orientation [degrees]', labelpad=0)
    ax.flatten()[1].set_xlabel('Orientation [degrees]', labelpad=0)

    if labels:
        cl.make_subplot_labels(ax.flatten(), x_shift=-.15)

    plt.subplots_adjust(hspace=.45)

    if title is not None:
        ax.flatten()[0].text(
            1.08, 1.25, title, size=14, ha='center',
            transform=ax.flatten()[0].transAxes)


def compare_eccentricity(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None,
        legend=False, labels=False):

    fig, ax = plt.subplots(1, 2, figsize=(12, 2))
    cl.init_fonts()

    ax.flatten()[0].hist(
        [all_obs_radar[8], all_obs_ACCESS[8]],
        bins=np.arange(0, 1.1, .05), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[0].set_title('Raw Eccentricities')

    ax.flatten()[1].hist(
        [QC_obs_radar[8], QC_obs_ACCESS[8]],
        bins=np.arange(0, 1.1, .05), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[1].set_title('Restricted Sample Eccentricities')

    if legend:
        ax.flatten()[0].legend(
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

    ax.flatten()[0].set_xlabel('Eccentricity [-]', labelpad=0)
    ax.flatten()[1].set_xlabel('Eccentricity [-]', labelpad=0)

    if labels:
        cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        ax.flatten()[0].text(
            1.08, 1.25, title, size=14, ha='center',
            transform=ax.flatten()[0].transAxes)


def compare_offset(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS, density=True, title=None,
        fig=None, ax=None, legend=False, labels=False):

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 2))
    cl.init_fonts()

    ax.flatten()[0].hist(
        [np.array(all_obs_radar[9])/1e3, np.array(all_obs_ACCESS[9])/1e3],
        bins=np.arange(0, 52, 2), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[0].set_title('Raw Stratiform Offset Lengths')
    ax.flatten()[0].set_xticks(np.arange(0, 54, 4))
    ax.flatten()[0].set_xticks(np.arange(0, 52, 2), minor=True)

    ax.flatten()[1].hist(
        [np.array(QC_obs_radar[9])/1e3, np.array(QC_obs_ACCESS[9])/1e3],
        bins=np.arange(0, 52, 2), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[1].set_title('Restricted Sample Stratiform Offset Lengths')
    ax.flatten()[1].set_xticks(np.arange(0, 54, 4))
    ax.flatten()[1].set_xticks(np.arange(0, 52, 2), minor=True)

    if legend:
        ax.flatten()[0].legend(
            loc='lower center', bbox_to_anchor=(1.1, -0.3),
            ncol=2, fancybox=True, shadow=True)

    for i in range(len(ax.flatten())):
        ax.flatten()[i].ticklabel_format(
            axis='y', style='sci', scilimits=(0, 0))
        ax.flatten()[i].grid(which='major', alpha=0.5, axis='y')
        ax.flatten()[i].grid(which='minor', alpha=0.2, axis='y')
        if density:
            ax.flatten()[i].set_ylabel('Density [-]')
        else:
            ax.flatten()[0].set_yticks(np.arange(0, 2e4, .5e4))
            ax.flatten()[0].set_yticks(np.arange(0, 1.75e4, .25e4), minor=True)
            ax.flatten()[1].set_yticks(np.arange(0, 12e2, 2e2))
            ax.flatten()[1].set_yticks(np.arange(0, 11e2, 1e2), minor=True)
            ax.flatten()[i].set_ylabel('Count [-]')

    ax.flatten()[0].set_xlabel('Stratiform Offset Magnitude [km]', labelpad=0)
    ax.flatten()[1].set_xlabel('Stratiform Offset Magnitude [km]', labelpad=0)

    if labels:
        cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        plt.suptitle(title, fontsize=14)


def compare_time(
        all_obs_radar, all_obs_ACCESS,
        QC_obs_radar, QC_obs_ACCESS,
        fig=None, ax=None, density=True, title=None,
        labels=False, legend=False):

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 2))
    cl.init_fonts()

    a_hour = [int(s.astype(str)[11:13]) for s in all_obs_radar[2]]
    b_hour = [int(s.astype(str)[11:13]) for s in all_obs_ACCESS[2]]
    c_hour = [int(s.astype(str)[11:13]) for s in QC_obs_radar[2]]
    d_hour = [int(s.astype(str)[11:13]) for s in QC_obs_ACCESS[2]]

    ax.flatten()[0].hist(
        [a_hour, b_hour],
        bins=np.arange(0, 25, 1), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[0].set_title('Raw Observation Count')
    ax.flatten()[0].set_xticks(np.arange(0, 25, 2))

    spine_offset = -.395

    twin0 = ax.flatten()[0].twiny()
    twin0.spines.bottom.set_position(("axes", spine_offset))
    twin0.xaxis.set_ticks_position("bottom")
    twin0.xaxis.set_label_position("bottom")

    ax.flatten()[0].set_xlim([-1, 25])
    twin0.set_xlim([8, 34])
    twin0.set_xticks(np.arange(0, 25, 2)+9)
    twin0.set_xticklabels((np.arange(0, 25, 2)+9) % 24)
    twin0.set_xlabel('Time [hour LST]')

    ax.flatten()[1].hist(
        [c_hour, d_hour],
        bins=np.arange(0, 25, 1), label=['Radar', 'ACCESS-C'],
        density=density, rwidth=1)

    ax.flatten()[1].set_title('Restricted Observation Count')
    ax.flatten()[1].set_xticks(np.arange(0, 25, 2))

    twin1 = ax.flatten()[1].twiny()
    twin1.spines.bottom.set_position(("axes", spine_offset))
    twin1.xaxis.set_ticks_position("bottom")
    twin1.xaxis.set_label_position("bottom")
    ax.flatten()[1].set_xlim([-1, 25])
    twin1.set_xlim([8, 34])
    twin1.set_xticks(np.arange(0, 25, 2)+9)
    twin1.set_xticklabels((np.arange(0, 25, 2)+9) % 24)
    twin1.set_xlabel('Time [hour LST]')

    if legend:
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

    ax.flatten()[0].set_xlabel('Time [hour UTC]', labelpad=0)
    ax.flatten()[1].set_xlabel('Time [hour UTC]', labelpad=0)

    if labels:
        cl.make_subplot_labels(ax.flatten(), x_shift=-.15)
    plt.subplots_adjust(hspace=.45)

    if title is not None:
        ax.flatten()[0].text(
            1.08, 1.25, title, size=14, ha='center',
            transform=ax.flatten()[0].transAxes)


def gen_error_model_plot(
        z_s=10000, fig=None, ax=None, legend=False,
        closest='convective', tau=252, r_max=20e3, dr=5e3,
        t_min=0, t_max=None, r_min=0):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))

    cl.init_fonts()

    if t_max == None:
        t_max = tau

    z_c = 1000

    s_mags = [0, 5e3, 10e3, 20e3, 30e3, 40e3]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 1, 2, 4, 5, 6]]

    base_label = r"$\Delta r="

    del_ts = []

    for j in range(len(s_mags)):

        if closest == 'convective':
            r_c = np.arange(r_min, r_max+10, 10)
            r_s = r_c + s_mags[j]
        elif closest == 'stratiform':
            r_s = np.arange(r_min, r_max+10, 10)
            r_c = r_s + s_mags[j]

        elav_s = np.zeros(len(r_c))
        elav_c = np.zeros(len(r_c))

        for i in range(len(r_c)):

            if r_s[i] == 0:
                elav_s[i] = np.pi/2
            else:
                elav_s[i] = np.abs(np.arctan(z_s/np.abs(r_s[i])))

            if r_c[i] == 0:
                elav_c[i] = np.pi/2
            else:
                elav_c[i] = np.abs(np.arctan(z_c/np.abs(r_c[i])))

        # import pdb; pdb.set_trace()

        # t1 = 252*((elav_c*180/np.pi-.5)/31.5)**(1/3)
        # t2 = 252*((elav_s*180/np.pi-.5)/31.5)**(1/3)

        c = 62.57
        a = 31.5/np.sinh(252/c)

        t1 = np.arcsinh((elav_c*180/np.pi-0.5)/a)*c
        t2 = np.arcsinh((elav_s*180/np.pi-0.5)/a)*c

        del_t = t2-t1
        # import pdb; pdb.set_trace()
        del_t[
            np.logical_or(
                elav_c > 32*np.pi/180,
                elav_s > 32*np.pi/180)] = np.nan
        del_ts.append(del_t)

        lab = base_label + '{}'.format(int(s_mags[j]/1000)) + '$ km'

        if closest == 'convective':
            ax.plot(
                r_c, del_t, color=colors[j], label=lab, linewidth=1.75)
        else:
            ax.plot(
                r_s, del_t, color=colors[j], label=lab, linewidth=1.75)

    ax.set_xticks(np.arange(r_min, r_max+dr, dr))
    ax.set_xticks(np.arange(r_min, r_max+dr/2, dr/2), minor=True)
    ax.set_xticklabels(np.arange(r_min/1e3, r_max/1e3+dr/1e3, dr/1e3))
    if closest == 'convective':
        ax.set_xlabel(r'$r_c$ [km]')
    else:
        ax.set_xlabel(r'$r_s$ [km]')

    ax.set_yticks(np.arange(t_min, t_max+60, 60))
    ax.set_yticks(np.arange(t_min, t_max+30, 30), minor=True)
    ax.set_ylabel(r'$\Delta t$ [s]')

    ax.grid(which='minor', alpha=0.2, axis='both')
    ax.grid(which='major', alpha=0.5, axis='both')

    ax.set_ylim([t_min-20, t_max])

    lines, labels = ax.get_legend_handles_labels()
    lines = [lines[i] for i in [0, 3, 1, 4, 2]]
    labels = [labels[i] for i in [0, 3, 1, 4, 2]]

    if legend:
        ax.legend(
            lines, labels,
            loc='lower center',
            bbox_to_anchor=(.5, -.55),
            ncol=3, fancybox=True, shadow=True)

    if closest == 'convective':
        ax.set_title(
            r'$z_s = ${} km, $r_s = r_c+\Delta r$'.format(int(z_s/1000)))
    elif closest == 'stratiform':
        ax.set_title(
            r'$z_s = ${} km, $r_c = r_s+\Delta r$'.format(int(z_s/1000)))


def plot_deformation(
        z_c=1e3, z_s=7.5e3, tau=252, fig=None, ax=None,
        conv_radius=10e3, conv_centroid_x=5e3, conv_centroid_y=0,
        strat_radius=16e3, strat_centroid_x=10e3, plot=True,
        strat_centroid_y=0, u=10, v=0, extent=None, dx=None):

    if (fig is None or ax is None) and plot is True:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cl.init_fonts()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in [0, 1, 2, 4, 5, 6]]

    theta = np.arange(0, 2*np.pi+2*np.pi/1000, 2*np.pi/1000)

    r_b = z_s/np.tan(32*np.pi/180)

    blind_spot_x = r_b*np.cos(theta)
    blind_spot_y = r_b*np.sin(theta)

    conv_border_x = conv_centroid_x + conv_radius*np.cos(theta)
    conv_border_y = conv_centroid_y + conv_radius*np.sin(theta)

    strat_border_x = strat_centroid_x + strat_radius*np.cos(theta)
    strat_border_y = strat_centroid_y + strat_radius*np.sin(theta)

    r_c = np.sqrt(conv_border_x**2 + conv_border_y**2)
    r_s = np.sqrt(strat_border_x**2 + strat_border_y**2)

    elav_s = np.zeros(len(r_c))
    elav_c = np.zeros(len(r_c))

    for i in range(len(r_s)):
        if r_s[i] == 0:
            elav_s[i] = np.pi/2
        else:
            elav_s[i] = np.abs(np.arctan(z_s/np.abs(r_s[i])))

        if r_c[i] == 0:
            elav_c[i] = np.pi/2
        else:
            elav_c[i] = np.abs(np.arctan(z_c/np.abs(r_c[i])))

    c = 62.57
    a = 31.5/np.sinh(252/c)

    t_c = np.arcsinh((elav_c*180/np.pi-0.5)/a)*c
    t_s = np.arcsinh((elav_s*180/np.pi-0.5)/a)*c

    error_strat_border_x = strat_border_x + u*t_s
    error_conv_border_x = conv_border_x + u*t_c
    error_strat_border_y = strat_border_y + v*t_s
    error_conv_border_y = conv_border_y + v*t_c

    c_conv_error = get_centroid(error_conv_border_x, error_conv_border_y)
    c_strat_error = get_centroid(error_strat_border_x, error_strat_border_y)

    so_mag = np.sqrt(
        (conv_centroid_x - strat_centroid_x)**2
        + (conv_centroid_y - strat_centroid_y)**2)
    so_error_mag = np.sqrt(
        (c_conv_error[1] - c_strat_error[1])**2
        + (c_conv_error[2] - c_strat_error[2])**2)

    if plot is True:

        ax.plot(
            [conv_centroid_x, strat_centroid_x],
            [conv_centroid_y, strat_centroid_y], linestyle='-',
            color=prop_cycle.by_key()['color'][3],
            linewidth=1.75, label='Stratiform Offset', alpha=.6,)

        ax.plot(
            [c_conv_error[1], c_strat_error[1]],
            [c_conv_error[2], c_strat_error[2]], linestyle='--',
            color=prop_cycle.by_key()['color'][3],
            linewidth=1.75, label='Deformed Stratiform Offset',)

        ax.plot(
            blind_spot_x, blind_spot_y, '--', linewidth=1.75, color='k',
            alpha=.6, label='Stratiform Blind-Spot Boundary')

        ax.plot(
            conv_border_x, conv_border_y, linewidth=1.75, color=colors[0],
            label='Convective Boundary', alpha=.6)

        ax.plot(
            error_conv_border_x, error_conv_border_y, '--', linewidth=1.75,
            color=colors[0], label='Deformed Convective Boundary')
        ax.plot(
            strat_border_x, strat_border_y, linewidth=1.75, color=colors[1],
            label='Stratiform Boundary', alpha=.6)
        ax.plot(
            error_strat_border_x, error_strat_border_y, '--',
            linewidth=1.75, color=colors[1],
            label='Deformed Stratiform Boundary')

        ax.scatter(
            [0], [0], marker='*', s=300, color='red',
            edgecolors='k', label='Radar')
        # ax.text(
        #     -4e3, -5e3, 'Radar', fontsize=12)

        title = r'$|\mathrm{\mathbf{s}}|=' + '${:.02f} km, '.format(so_mag/1e3)
        title += r"$|\mathrm{\mathbf{s}}'|=" + '${:.02f} km'.format(
            so_error_mag/1e3)

        ax.set_title(title)

        if extent is not None and dx is not None:
            ax.set_xlim([extent[0], extent[1]])
            ax.set_ylim([extent[2], extent[3]])
            ax.set_xticks(np.arange(extent[0], extent[1]+dx, dx))
            ax.set_xticks(
                np.arange(extent[0], extent[1]+dx/2, dx/2), minor=True)
            ax.set_xticklabels(
                (np.arange(extent[0], extent[1]+dx, dx)/1e3).astype(int))
            ax.set_yticks(np.arange(extent[2], extent[3]+dx, dx))
            ax.set_yticks(
                np.arange(extent[2], extent[3]+dx/2, dx/2), minor=True)
            ax.set_yticklabels((np.arange(
                extent[2], extent[3]+dx, dx)/1e3).astype(int))

        ax.grid(which='minor', alpha=0.2, axis='both')
        ax.grid(which='major', alpha=0.5, axis='both')
        ax.set_xlabel(r'$x$ [km]')
        ax.set_ylabel(r'$y$ [km]')

    return so_error_mag


def get_centroid(x, y):
    A = .5*np.array([
        x[i]*y[i+1] - x[i+1]*y[i]
        for i in range(len(x)-1)]).sum()

    cx = 1/(6*A)*np.array([
        (x[i]+x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        for i in range(len(x)-1)]).sum()

    cy = 1/(6*A)*np.array([
        (y[i]+y[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        for i in range(len(x)-1)]).sum()

    return A, cx, cy


def get_radar_prop_so_stats(
        save_dir, exclusions=None, class_thresh=None, excl_thresh=None,
        regime=None, pope_dir=None, radars=[63, 42, 77]):

    if pope_dir is None:
        pope_dir = '/home/student.unimelb.edu.au/shorte1/'
        pope_dir += 'Documents/CPOL_analysis/'

    times = np.arange(0, 24*60+10, 10)
    offset_mag_list = [[] for i in range(len(times))]
    prop_mag_list = [[] for i in range(len(times))]
    conv_area_list = [[] for i in range(len(times))]
    strat_area_list = [[] for i in range(len(times))]
    eccentricity_list = [[] for i in range(len(times))]
    shear_normal_list = [[] for i in range(len(times))]
    prop_normal_list = [[] for i in range(len(times))]
    count_list = [0 for i in range(len(times))]

    if exclusions is None:
        # exclusions = [
        #     'small_area', 'large_area', 'intersect_border',
        #     'intersect_border_convective', 'duration_cond',
        #     'small_velocity', 'small_offset']
        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'simple_duration_cond']

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

                tracks_obj = cl.redo_exclusions(
                    tracks_obj, class_thresh=class_thresh,
                    excl_thresh=excl_thresh)
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

                uids = set([ind[2] for ind in inds_all])

                for uid in uids:

                    sub_tracks_uid = sub_tracks_all.xs(uid, level='uid')
                    uid_scan_index = np.array(sorted(list(set([
                        ind[0] for ind in sub_tracks_uid.index.values]))))
                    uid_scan_index = uid_scan_index - uid_scan_index[0]

                    sub_tracks_conv = sub_tracks_uid.xs(0, level='level')
                    sub_tracks_strat = sub_tracks_uid.xs(1, level='level')
                    conv_area = list(sub_tracks_conv['proj_area'].values)
                    strat_area = list(sub_tracks_strat['proj_area'].values)

                    pos_0 = sub_tracks_conv[['grid_x', 'grid_y']]
                    pos_1 = sub_tracks_strat[['grid_x', 'grid_y']]
                    mag = pos_1-pos_0
                    mag_num = np.sqrt(
                        mag['grid_x'].values**2 + mag['grid_y'].values**2)
                    offset_mag = list(mag_num)

                    eccentricity = sub_tracks_conv['eccentricity'].values

                    u_shear = sub_tracks_conv['u_shear'].values
                    v_shear = sub_tracks_conv['v_shear'].values
                    u_relative = sub_tracks_conv['u_relative'].values
                    v_relative = sub_tracks_conv['v_relative'].values

                    shear_angle = np.arctan2(v_shear, u_shear)
                    shear_angle = np.rad2deg(shear_angle)

                    prop_angle = np.arctan2(v_relative, u_relative)
                    prop_angle = np.rad2deg(prop_angle)

                    orientation = sub_tracks_conv['orientation_alt'].values

                    prop_mag = np.sqrt(u_relative**2 + v_relative**2)

                    shear_angle = np.mod(shear_angle, 360)
                    prop_angle = np.mod(prop_angle, 360)
                    line_normal = np.mod(orientation+90, 360)

                    cosines_shear = np.cos(np.deg2rad(shear_angle-line_normal))
                    angles_shear = np.arccos(cosines_shear) * 180 / np.pi

                    cosines_prop = np.cos(np.deg2rad(prop_angle-line_normal))
                    angles_prop = np.arccos(cosines_prop) * 180 / np.pi

                    for i in range(len(uid_scan_index)):
                        offset_mag_list[uid_scan_index[i]].append(
                            offset_mag[i])
                        prop_mag_list[uid_scan_index[i]].append(
                            prop_mag[i])
                        conv_area_list[uid_scan_index[i]].append(
                            conv_area[i])
                        strat_area_list[uid_scan_index[i]].append(
                            strat_area[i])
                        eccentricity_list[uid_scan_index[i]].append(
                            eccentricity[i])
                        shear_normal_list[uid_scan_index[i]].append(
                            angles_shear[i])
                        prop_normal_list[uid_scan_index[i]].append(
                            angles_prop[i])
                        count_list[uid_scan_index[i]] += 1

    out = [
        count_list,
        offset_mag_list, prop_mag_list, conv_area_list, strat_area_list,
        eccentricity_list, shear_normal_list, prop_normal_list]

    return out


def get_ACCESS_prop_so_stats(
        save_dir, exclusions=None, class_thresh=None,
        excl_thresh=None, regime=None, pope_dir=None,
        radars=[63, 42, 77]):

    if pope_dir is None:
        pope_dir = '/home/student.unimelb.edu.au/shorte1/'
        pope_dir += 'Documents/CPOL_analysis/'

    times = np.arange(0, 24*60+10, 10)
    offset_mag_list = [[] for i in range(len(times))]
    prop_mag_list = [[] for i in range(len(times))]
    conv_area_list = [[] for i in range(len(times))]
    strat_area_list = [[] for i in range(len(times))]
    eccentricity_list = [[] for i in range(len(times))]
    shear_normal_list = [[] for i in range(len(times))]
    prop_normal_list = [[] for i in range(len(times))]
    count_list = [0 for i in range(len(times))]

    if exclusions is None:
        # exclusions = [
        #     'small_area', 'large_area', 'intersect_border',
        #     'intersect_border_convective', 'duration_cond',
        #     'small_velocity', 'small_offset']
        exclusions = [
            'small_area', 'large_area', 'intersect_border',
            'intersect_border_convective', 'simple_duration_cond']

    for radar in radars:
        for year in [2020, 2021]:
            print('Radar {}, year {}.'.format(radar, year))
            path = save_dir + 'ACCESS_{}/{}1001_{}0501.pkl'.format(
                radar, year, year+1)
            with open(path, 'rb') as f:
                tracks_obj = pickle.load(f)

            tracks_obj = cl.redo_exclusions(
                tracks_obj, class_thresh=class_thresh, excl_thresh=excl_thresh)
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

            uids = set([ind[2] for ind in inds_all])

            for uid in uids:

                sub_tracks_uid = sub_tracks_all.xs(uid, level='uid')
                uid_scan_index = np.array(sorted(list(set([
                    ind[0] for ind in sub_tracks_uid.index.values]))))
                uid_scan_index = uid_scan_index - uid_scan_index[0]

                # uid_time_index = uid_time_index - uid_time_index[0]

                # import pdb; pdb.set_trace()

                sub_tracks_conv = sub_tracks_uid.xs(0, level='level')
                sub_tracks_strat = sub_tracks_uid.xs(1, level='level')
                conv_area = list(sub_tracks_conv['proj_area'].values)
                strat_area = list(sub_tracks_strat['proj_area'].values)

                pos_0 = sub_tracks_conv[['grid_x', 'grid_y']]
                pos_1 = sub_tracks_strat[['grid_x', 'grid_y']]
                mag = pos_1-pos_0
                mag_num = np.sqrt(
                    mag['grid_x'].values**2 + mag['grid_y'].values**2)
                offset_mag = list(mag_num)

                prop_mag = np.sqrt(
                    sub_tracks_conv['u_relative'].values**2
                    + sub_tracks_conv['v_relative'].values**2)
                eccentricity = sub_tracks_conv['eccentricity'].values

                u_shear = sub_tracks_conv['u_shear'].values
                v_shear = sub_tracks_conv['v_shear'].values
                u_relative = sub_tracks_conv['u_relative'].values
                v_relative = sub_tracks_conv['v_relative'].values

                shear_angle = np.arctan2(v_shear, u_shear)
                shear_angle = np.rad2deg(shear_angle)

                prop_angle = np.arctan2(v_relative, u_relative)
                prop_angle = np.rad2deg(prop_angle)

                orientation = sub_tracks_conv['orientation_alt'].values

                prop_mag = np.sqrt(u_relative**2 + v_relative**2)

                shear_angle = np.mod(shear_angle, 360)
                prop_angle = np.mod(prop_angle, 360)
                line_normal = np.mod(orientation+90, 360)

                cosines_shear = np.cos(np.deg2rad(shear_angle-line_normal))
                angles_shear = np.arccos(cosines_shear) * 180 / np.pi

                cosines_prop = np.cos(np.deg2rad(prop_angle-line_normal))
                angles_prop = np.arccos(cosines_prop) * 180 / np.pi

                for i in range(len(uid_scan_index)):
                    offset_mag_list[uid_scan_index[i]].append(
                        offset_mag[i])
                    prop_mag_list[uid_scan_index[i]].append(
                        prop_mag[i])
                    conv_area_list[uid_scan_index[i]].append(
                        conv_area[i])
                    strat_area_list[uid_scan_index[i]].append(
                        strat_area[i])
                    eccentricity_list[uid_scan_index[i]].append(
                        eccentricity[i])
                    shear_normal_list[uid_scan_index[i]].append(
                        angles_shear[i])
                    prop_normal_list[uid_scan_index[i]].append(
                        angles_prop[i])
                    count_list[uid_scan_index[i]] += 1

    out = [
        count_list,
        offset_mag_list, prop_mag_list, conv_area_list, strat_area_list,
        eccentricity_list, shear_normal_list, prop_normal_list]

    return out
