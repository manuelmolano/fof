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


if __name__ == '__main__':
    plt.close('all')
    # BEHAVIOR
    p = ComPipe.chom('LE113',  # sujeto (nombre de la carpeta under parentpath)
                     parentpath='/home/molano/fof_data/',
                     analyze_trajectories=False)  # precarga sesiones disponibles
    p.load_available()  # just in case, refresh
    print(p.available[2])  # exmpl target sess / filename string is the actual arg
    p.load(p.available[2])
    p.process()
    p.trial_sess.head()  # preprocessed df stored in attr. trial_sess
    # stop = True
    # if stop:
    #     import sys
    #     sys.exit()

    # ELECTRO
    # sampling rate
    s_rate = 3e4
    min_ev_dur = 0.01*s_rate  # 50ms
    # Importing the data from a session
    path = '/home/molano/fof_data/LE113/electro/LE113_2021-06-05_12-38-09/'
    # Load spike sorted data
    # Times of the spikes, array of lists
    spike_times = np.load(path+'spike_times.npy')
    # cluster number of each of the spikes, same length as before
    spike_clusters = np.load(path+'spike_clusters.npy')
    # Cluster labels (good, noise, mua) for the previous two arrays
    df_labels = pd.read_csv(path+"cluster_group.tsv", sep='\t')

    data_files = glob.glob(path+'/*.dat')
    data_files = [f for f in data_files if 'temp' not in f]
    assert len(data_files) == 1
    data = np.memmap(data_files[0], dtype='int16')

    if len(data) % 40 == 0:
        num_ch = 40
        samples = data.reshape((len(data) // num_ch, num_ch))
        assert len(data) % num_ch-1 != 0
    elif len(data) % 39 == 0:
        num_ch = 39
        samples = data.reshape((len(data) // num_ch, num_ch))
        assert len(data) % num_ch+1 != 0

    num_samples = 1500000
    plot = False
    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ln_stl = ['-', '--']
    ev_strt = []
    ev_end = []
    for i_ch, ch in enumerate(range(35, 37)):
        print('--------')
        print(ch)
        trace = samples[:, ch]
        trace = trace/np.max(trace)
        abv_th = 1*(trace > 0.8)
        starts = np.where(np.diff(abv_th) > 0.5)[0]
        # starts = iti_clean(times=starts, min_ev_dur=min_ev_dur, bef_aft='bef')
        ends = np.where(np.diff(abv_th) < -0.5)[0]
        # ends = iti_clean(times=ends, min_ev_dur=min_ev_dur, bef_aft='aft')
        ev_strt.append(starts)
        ev_end.append(ends)
        print(len(starts))
        print(len(ends))
        ev_durs = ends-starts
        comp_starts_durs = np.diff(starts) > ev_durs[:-1]
        assert comp_starts_durs.all(), np.where(~comp_starts_durs)[0]
        if plot:
            trace_plt = trace[:num_samples]
            abv_th_plt = abv_th[:num_samples]
            ax[i_ch//2].plot(np.arange(len(trace_plt))/s_rate, trace_plt,
                             label='ch '+str(ch+1), linestyle=ln_stl[i_ch//2])

            ax[i_ch//2].plot(np.arange(len(trace_plt))/s_rate, abv_th_plt,
                             label='ch '+str(ch+1), linestyle=ln_stl[i_ch//2])

            strts_plt = starts[starts < num_samples]
            for strt in strts_plt:
                ax[i_ch//2].axvline(x=strt/s_rate, linestyle='--', color='k')
            ends_plt = ends[ends < num_samples]
            for end in ends_plt:
                ax[i_ch//2].axvline(x=end/s_rate, linestyle='--',
                                    color=(.5, .5, .5))

            ax[i_ch//2].legend()
    if plot:
        ax[1].set_xlabel('Time (s)')
        ax[0].set_ylabel('Normalized values')
        ax[1].set_ylabel('Normalized values')
    # get stim only which corresponds to ch-35==0 and ch-36==1
    trace = samples[:, 35]
    trace = trace/np.max(trace)
    abv_th1 = trace > 0.8
    trace = samples[:, 36]
    trace = trace/np.max(trace)
    abv_th2 = trace > 0.8
    stim = 1*np.logical_and(~abv_th1, abv_th2)
    starts = np.where(np.diff(stim) > 0.5)[0]
    # starts = iti_clean(times=starts, min_ev_dur=min_ev_dur, bef_aft='bef')
    ends = np.where(np.diff(stim) < -0.5)[0]
    ev_strt.append(starts)
    ev_end.append(ends)
    events = {'ev_strt': ev_strt, 'ev_end': ev_end}
    print(len(events['ev_strt']))
    np.savez(path+'/events.npz', **events)
    stop = True
    if stop:
        import sys
        sys.exit()
    # Transforms array of lists into array of ints
    spike_times_df = [item for sublist in spike_times for item in sublist]

    # Put the data in a single dataframe, one colum spikes, one colum clusters
    s1 = pd.Series(spike_times_df, name='times')
    s2 = pd.Series(spike_clusters, name='cluster_id')
    df_temp = pd.concat([s1, s2], axis=1)

    # Merge with cluster labels, use cluster ID to associate each one
    df = pd.merge(df_temp, df_labels, on=['cluster_id'])

    # Select only clusters labelled good, which are presumably single units
    df = df.loc[df.group == 'good']

    # Transform the values per session to seconds. This takes into account the
    # frame rate of the recordings, 30000Hz for virtually all the sessions.
    df['fixed_times'] = (df.times/s_rate)
    print(min(df['fixed_times']), max(df['fixed_times']))

    # Plot the first minute to have an impression of how it looks like
    sns.scatterplot('fixed_times', 'cluster_id',
                    data=df.loc[(df['fixed_times'] < 60) &
                                (df.group == 'good')],
                    s=30, color='black')
    session = Session(path)
    # print(session)
    samples = session.recordings[0].continuous[0].samples[:, -8]
    timestamps = session.recordings[0].continuous[0].timestamps

    # samples = session.recordings[0].continuous[0].samples[:,-7]

    # Put the data in a single dataframe
    s1 = pd.Series(samples, name='samples')
    s2 = pd.Series(timestamps, name='timestamps')
    df_ttl = pd.concat([s1, s2], axis=1)

    # Transform the analog channel to boolean.
    df_ttl.loc[df_ttl['samples'] >= 1000, 'ttl'] = 1
    df_ttl.loc[df_ttl['samples'] < 1000, 'ttl'] = 0

    # Look for the places where there is a change and it goes from 0 to 1.
    # 1 for on delay and -1 for off delay
    df_ttl['diff'] = df_ttl.ttl.diff()
    df_ttl['diff'].unique()

    # On is a 1 and -1 is an off of the delay period

    # Recover the first timestamp of the session. This is important because
    # it is not 0 but when the recording was started.
    initial = df_ttl.timestamps.iloc[0]

    # Substract the first timestamp and divide by sampling frequency
    df_ttl.timestamps = (df_ttl.timestamps - initial)/30000

    # Remove the rest of the values that does not have a change.
    df_ttl = df_ttl.loc[(df_ttl['diff'] == 1) | (df_ttl['diff'] == -1)]

    # Create a new colum with the delay duration
    df_ttl['delay'] =\
        np.around(df_ttl.loc[(df_ttl['diff'] == 1) |
                             (df_ttl['diff'] == -1)]['timestamps'].diff(), 2)

    # Check that we recover all the delays that were used.
    df_ttl.delay.unique()

    # Save the data in a new csv.
    os.getcwd()
    os.chdir(path)
    df_ttl.to_csv(path+'timestamps.csv')

    # Remove all the colums which don't show the right delay lengths.
    df_ttl = df_ttl.loc[(df_ttl['delay'] == 1) | (df_ttl['delay'] == 10) |
                        (df_ttl['delay'] == 3.00) | (df_ttl['delay'] == 0.1)]

    np.array_equal(df.delay_times, df_ttl.delay)
