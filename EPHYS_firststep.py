#!/usr/bin/env python
# coding: utf-8

# ## EPHYS analysis
from utilsJ.Behavior import ComPipe
# Load modules and data
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import glob
# Import all needed libraries
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import itertools
from scipy import stats
from ast import literal_eval
from open_ephys.analysis import Session
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


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


if __name__ == '__main__':
    plt.close('all')
    main_folder = '/home/molano/fof_data/'
    # BEHAVIOR
    p = ComPipe.chom('LE113',  # sujeto (nombre de la carpeta under parentpath)
                     parentpath=main_folder,
                     analyze_trajectories=False)  # precarga sesiones disponibles
    p.load_available()  # just in case, refresh
    print(p.available[2])  # exmpl target sess / filename string is the actual arg
    p.load(p.available[2])
    p.process()
    p.trial_sess.head()  # preprocessed df stored in attr. trial_sess
    df = p.sess
    csv_strt_snd_times = df.loc[(df['MSG'] == 'StartSound') &
                                (df.TYPE == 'TRANSITION'), 'PC-TIME']
    csv_ss_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
                           for x in csv_strt_snd_times])
    csv_ss_sec = csv_ss_sec-csv_ss_sec[0]
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
    trace2 = samples[:, 36]
    trace2 = trace2/np.max(trace2)
    import scipy.signal as ss
    trace2_filt = ss.medfilt(trace2, 3)
    # stimulus corresponds to ch36=high and ch35=low
    stim = 1*((trace2_filt-trace1) > 0.5)
    starts = np.where(np.diff(stim) > 0.9)[0]
    # starts = iti_clean(times=starts, min_ev_dur=min_ev_dur, bef_aft='bef')
    ends = np.where(np.diff(stim) < -0.9)[0]
    ttl_ev_strt = starts/s_rate
    # compute spikes offset
    spikes_offset = -ttl_ev_strt[0]
    ttl_ev_strt = ttl_ev_strt+spikes_offset
    assert len(csv_ss_sec) == len(ttl_ev_strt)
    assert np.max(csv_ss_sec-ttl_ev_strt) < 0.05, print(np.max(csv_ss_sec -
                                                               ttl_ev_strt))
    offset = 52320000
    num_samples = 100000
    events = {'ttl_ev_strt': starts, 'ev_end': ends,
              'samples': samples[offset:offset+300000, 35:39]}
    print(len(events['ttl_ev_strt']))
    np.savez(path+'/events.npz', **events)

    # plot stuff
    margin_spks_plot = 50
    bin_size = 0.2
    bins = np.linspace(-margin_spks_plot, margin_spks_plot,
                       2*margin_spks_plot//bin_size)
    step = np.diff(bins)[0]
    f, ax = plt.subplots(nrows=3, ncols=5)
    ax = ax.flatten()
    for i_cl, cl in enumerate(sel_clstrs):
        spks_cl = spike_times[spike_clusters == cl]/s_rate+spikes_offset
        spks_mat = np.tile(spks_cl, (1, len(ttl_ev_strt)))-ttl_ev_strt[None, :]
        hists = np.array([np.histogram(spks_mat[:, i], bins)[0]
                          for i in range(spks_mat.shape[1])])
        psth = np.mean(hists, axis=0)
        # hist, _ = np.histogram(spks_cl, bins=bins)
        # hist = hist/step
        ax[i_cl].plot(bins[:-1]+step/2, psth)
        ax[i_cl].set_title(clstrs_qlt[i_cl])
    f.savefig('/home/molano/Dropbox/psths.png')
    # plot_events(ttl_ev_strt, label='ttl-stim', color='m')
    # plot_events(csv_ss_sec, label='start-sound', color='c', lnstl='--')
