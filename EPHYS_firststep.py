#!/usr/bin/env python
# coding: utf-8

# ## EPHYS analysis
from utilsJ.Behavior import ComPipe
# Load modules and data
import scipy.signal as ss
import glob
# Import all needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def iti_clean(times, min_ev_dur, bef_aft):
    if bef_aft == 'bef':
        times_ = np.insert(times, obj=0, values=0)
    elif bef_aft == 'aft':
        times_ = np.append(times, times[-1]+2*min_ev_dur)
    iti = np.diff(times_)
    times = times[iti > min_ev_dur]
    return times


def plot_events(evs, label='', color='k', lnstl='-'):
    for i in evs:
        label = label if i == evs[0] else ''
        plt.plot(np.array([i, i]), [0, 1], color=color, label=label,
                 linestyle=lnstl)


def plot_psths(spike_times, sel_clstrs, events, s_rate, spikes_offset,
               margin_spks_plot=1, bin_size=.1, name=''):
    bins = np.linspace(-margin_spks_plot, margin_spks_plot-bin_size,
                       int(2*margin_spks_plot/bin_size))
    f, ax = plt.subplots(nrows=3, ncols=5, figsize=(15, 12))
    ax = ax.flatten()
    for i_cl, cl in enumerate(sel_clstrs):
        spks_cl = spike_times[spike_clusters == cl]/s_rate+spikes_offset
        spks_mat = np.tile(spks_cl, (1, len(events)))-events[None, :]
        hists = np.array([np.histogram(spks_mat[:, i], bins)[0]
                          for i in range(spks_mat.shape[1])])
        hists = hists/bin_size
        psth = np.mean(hists, axis=0)
        # hist, _ = np.histogram(spks_cl, bins=bins)
        # hist = hist/step
        ax[i_cl].plot(bins[:-1]+bin_size/2, psth)
        ax[i_cl].set_title(clstrs_qlt[i_cl])
    ax[10].set_xlabel('Time (s)')
    ax[10].set_ylabel('Mean firing rate (Hz)')
    f.savefig('/home/molano/Dropbox/psths_'+name+'.png')


def get_behavior(main_folder):
    # BEHAVIOR
    p = ComPipe.chom('LE113',  # sujeto (nombre de la carpeta under parentpath)
                     parentpath=main_folder,
                     analyze_trajectories=False)  # precarga sesiones disponibles
    p.load_available()  # just in case, refresh
    print(p.available[2])  # exmpl target sess / filename string is the actual arg
    p.load(p.available[2])
    p.process()
    p.trial_sess.head()  # preprocessed df stored in attr. trial_sess
    return p.sess


if __name__ == '__main__':
    plt.close('all')
    main_folder = '/home/manuel/fof_data/'
    df = get_behavior(main_folder=main_folder)

    # get behavior events
    # XXX: I changed to using BPOD-INITIAL-TIME instead of PC-TIME. However, there
    # seems to be a missmatch between the two that grows throughout the session
    # StartSound. THERE MIGHT BE SOMETHING YOU'RE NOT BE TAKING INTO ACCOUNT
    # csv_strt_snd_times = df.loc[(df['MSG'] == 'StartSound') &
    #                             (df.TYPE == 'TRANSITION'), 'PC-TIME']
    # Outcome
    csv_strt_outc_times = df.loc[((df['MSG'] == 'Reward') |
                                  (df['MSG'] == 'Punish')) &
                                 (df.TYPE == 'TRANSITION'),
                                 'BPOD-INITIAL-TIME'].values
    # StartSound
    csv_strt_snd_times = df.loc[(df['MSG'] == 'StartSound') &
                                (df.TYPE == 'TRANSITION'),
                                'BPOD-INITIAL-TIME'].values
    # Trial start time
    csv_trial_bpod_time = df.loc[(df['MSG'] == 'TRIAL-BPOD-TIME') &
                                 (df.TYPE == 'INFO'),
                                 'BPOD-INITIAL-TIME'].values

    # csv_trial_bpod_pctime = df.loc[(df['MSG'] == 'New trial') &
    #                                (df.TYPE == 'TRIAL'), 'PC-TIME']
    # csv_trial_bpod_pctime = np.array([60*60*x.hour+60*x.minute+x.second +
    #                                   x.microsecond/1e6
    #                                   for x in csv_trial_bpod_pctime])
    # csv_trial_bpod_pctime = csv_trial_bpod_pctime - csv_trial_bpod_pctime[0]

    csv_ss_sec = csv_strt_snd_times + csv_trial_bpod_time
    csv_so_sec = csv_strt_outc_times + csv_trial_bpod_time
    # Transform date to seconds
    # csv_ss_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
    #                        for x in csv_strt_snd_times])
    # csv_so_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
    #                        for x in csv_strt_outc_times])
    # csv_so_sec = csv_so_sec-csv_ss_sec[0]
    # csv_ss_sec = csv_ss_sec-csv_ss_sec[0]
    csv_so_sec = csv_so_sec - csv_ss_sec[0]
    csv_ss_sec = csv_ss_sec - csv_ss_sec[0]
    import sys
    sys.exit()
    # ELECTRO
    # sampling rate
    s_rate = 3e4
    min_ev_dur = 0.01*s_rate  # 50ms
    # Importing the data from a session
    path = main_folder+'/LE113/electro/LE113_2021-06-05_12-38-09/'
    # Load spike sorted data
    # Times of the spikes, array of lists
    spike_times = np.load(path+'spike_times.npy')
    # cluster number of each of the spikes, same length as before
    spike_clusters = np.load(path+'spike_clusters.npy')
    # Cluster labels (good, noise, mua) for the previous two arrays
    df_labels = pd.read_csv(path+"cluster_group.tsv", sep='\t')
    # sel_clltrs = df_labels.loc[df_labels.group == 'good', 'cluster_id'].values
    sel_clstrs = df_labels['cluster_id'].values
    clstrs_qlt = df_labels['group']
    # load channels (continuous) data
    data_files = glob.glob(path+'/*.dat')
    data_files = [f for f in data_files if 'temp' not in f]
    assert len(data_files) == 1, 'Number of .dat files is different from 0'
    data = np.memmap(data_files[0], dtype='int16')
    if len(data) % 40 == 0:
        num_ch = 40
        samples = data.reshape((len(data) // num_ch, num_ch))
        assert len(data) % num_ch-1 != 0
    elif len(data) % 39 == 0:
        num_ch = 39
        samples = data.reshape((len(data) // num_ch, num_ch))
        assert len(data) % num_ch+1 != 0
    # load and med-filter TTL channels
    trace1 = samples[:, 35]
    trace1 = trace1/np.max(trace1)
    trace1_filt = ss.medfilt(trace1, 3)
    trace2 = samples[:, 36]
    trace2 = trace2/np.max(trace2)
    trace2_filt = ss.medfilt(trace2, 3)
    # stimulus corresponds to ch36=high and ch35=low
    stim = 1*((trace2_filt-trace1_filt) > 0.5)
    # stimulus corresponds to ch36=high and ch35=high
    outcome = 1*((trace2_filt+trace1_filt) > 1.9)
    assert np.sum(1*(stim+outcome > 1)) == 0
    # stim starts/ends
    stim_starts = np.where(np.diff(stim) > 0.9)[0]
    stim_ends = np.where(np.diff(stim) < -0.9)[0]
    ttl_stim_strt = stim_starts/s_rate
    # outcome starts/ends
    outc_starts = np.where(np.diff(outcome) > 0.9)[0]
    outc_ends = np.where(np.diff(outcome) < -0.9)[0]
    ttl_outc_strt = outc_starts/s_rate

    # compute spikes offset from stimulus start
    spikes_offset = -ttl_stim_strt[0]
    ttl_stim_strt = ttl_stim_strt+spikes_offset
    ttl_outc_strt = ttl_outc_strt+spikes_offset
    assert len(csv_ss_sec) == len(ttl_stim_strt)
    assert np.max(csv_ss_sec-ttl_stim_strt) < 0.05, print(np.max(csv_ss_sec -
                                                          ttl_stim_strt))
    # assert len(csv_so_sec) == len(ttl_outc_strt)
    # assert np.max(csv_so_sec-ttl_outc_strt) < 0.05, print(np.max(csv_so_sec -
    #                                                    ttl_outc_strt))

    # import sys
    # sys.exit()
    offset = 29550000
    num_samples = 200000
    events = {'stim_starts': stim_starts, 'outc_starts': outc_starts,
              'samples': samples[offset:offset+num_samples, 35:39]}
    np.savez(path+'/events.npz', **events)

    # plot PSTHs
    plot_psths(spike_times=spike_times, sel_clstrs=sel_clstrs, events=csv_ss_sec,
               s_rate=s_rate, spikes_offset=spikes_offset, margin_spks_plot=1,
               bin_size=.1, name='stim')
    plot_psths(spike_times=spike_times, sel_clstrs=sel_clstrs, events=csv_so_sec,
               s_rate=s_rate, spikes_offset=spikes_offset, margin_spks_plot=1,
               bin_size=.1, name='outcome')

    # f = plt.figure()
    # plot_events(ttl_stim_strt, label='ttl-stim', color='m')
    # plot_events(csv_ss_sec, label='start-sound', color='c', lnstl='--')
    # f.savefig('/home/molano/Dropbox/stim_check.png')

    # f = plt.figure()
    # plot_events(ttl_outc_strt, label='ttl-outcome', color='m')
    # plot_events(csv_so_sec, label='outcome', color='c', lnstl='--')
    # f.savefig('/home/molano/Dropbox/outcome_check.png')
