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
# import inventory as inv
from scipy.stats import norm
from scipy import stats


def append_features(features, new_data):
    for k in features.keys():
        features[k].append(new_data[k])


def iti_clean(times, min_ev_dur, bef_aft):
    if bef_aft == 'bef':
        times_ = np.insert(times, obj=0, values=0)
    elif bef_aft == 'aft':
        times_ = np.append(times, times[-1]+2*min_ev_dur)
    iti = np.diff(times_)
    times = times[iti > min_ev_dur]
    return times


def plot_events(evs, ev_strt=0, ev_end=1e6, s_rate=3e4, label='', color='k',
                lnstl='-'):
    evs_plt = s_rate*evs.copy()
    evs_plt = evs_plt[evs_plt < ev_end]
    evs_plt = evs_plt[evs_plt > ev_strt]
    for i in evs_plt:
        label = label if i == evs_plt[0] else ''
        plt.plot(np.array([i, i]), [0, 1.1], color=color, label=label,
                 linestyle=lnstl)


def plot_psths(spike_times, sel_clstrs, events, s_rate, spikes_offset,
               clstrs_qlt, spike_clusters, margin_spks_plot=1, bin_size=.1,
               name=''):
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
        ax[i_cl].set_title(str(cl)+' / #spks: '+str(len(spks_cl)))
    ax[10].set_xlabel('Time (s)')
    ax[10].set_ylabel('Mean firing rate (Hz)')
    f.savefig('/home/molano/Dropbox/psths_'+name+'.png')


def rm_top_right_lines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def histogram_psth(spk_times, events, bins, bin_size):
    spks_mat = np.tile(spk_times, (1, len(events)))-events.T
    hists = np.array([np.histogram(spks_mat[:, i], bins)[0]
                      for i in range(spks_mat.shape[1])])
    hists = hists/bin_size
    psth = np.mean(hists, axis=0)
    return psth


def significance(mat, window=100):
    edges = np.linspace(0, mat[0].shape[1], int(mat[0].shape[1]/window)+1)
    edges = edges.astype(int)
    pvalues = []
    for i_e in range(len(edges)-1):
        fr1 = np.sum(mat[0][:, edges[i_e]:edges[i_e+1]], axis=1)
        fr2 = np.sum(mat[1][:, edges[i_e]:edges[i_e+1]], axis=1)
        _, pvalue = stats.ranksums(fr1, fr2)
        pvalues.append(pvalue)
    return pvalues


def scatter(spk_tms, evs, margin_psth, ax=None, color='k', offset=0, alpha=1,
            plot=True):
    spk_raster = 1000*spk_tms
    evs = 1000*evs
    feats_spks = {'aligned_spks': []}
    for i_ev, ev in enumerate(evs):
        spk_tmp = spk_raster-ev
        spk_tmp = spk_tmp[np.logical_and(spk_tmp > -margin_psth,
                                         spk_tmp < margin_psth)]
        feats_spks['aligned_spks'].append(spk_tmp)
        if plot:
            ax.scatter(spk_tmp, np.ones((len(spk_tmp)))+i_ev+offset,
                       color=color, s=1, alpha=alpha)
    return feats_spks


def convolve_psth(spk_times, events, std=20, margin=1000):
    if len(events) > 0:
        krnl_len = 5*std
        # pass spikes to ms
        spk_times = 1000*spk_times.flatten()
        spk_times = spk_times.astype(int)
        spk_dist = spk_times[-1] - spk_times[0]
        # build spike vector
        spk_vct = np.zeros((spk_dist+2*margin, ))
        spk_vct[spk_times-spk_times[0]+margin] = 1
        # convolve
        x = np.linspace(norm.ppf(1e-5, scale=std),
                        norm.ppf(1-1e-5, scale=std), krnl_len)
        kernel = norm.pdf(x, scale=std)
        kernel = kernel/np.sum(kernel)  # XXX: why isn't kernel already normalized?
        spk_conv = np.convolve(spk_vct, kernel)
        # spk_conv = gaussian_filter1d(spk_vct, std)
        # pass events to ms
        events = 1000*events
        # offset events
        events = events.astype(int)-spk_times[0]+margin
        events = events[np.logical_and(events >= margin, events < spk_dist-margin)]
        peri_evs = np.array([spk_conv[x-margin:x+margin] for x in events])
        if len(events) > 0:
            psth = np.mean(peri_evs, axis=0)*1000
        else:
            psth = []
    else:
        psth = []
        peri_evs = []
    return psth, peri_evs


def get_behavior(main_folder, subject):
    # BEHAVIOR
    p = ComPipe.chom(subject,  # sujeto (nombre de la carpeta under parentpath)
                     parentpath=main_folder,
                     analyze_trajectories=False)  # precarga sesiones disponibles
    return p


def get_startSound_times(df):
    # STIM INITITAL PC-TIMES
    csv_strt_snd_times = df.loc[(df['MSG'] == 'StartSound') &
                                (df.TYPE == 'TRANSITION'), 'PC-TIME']
    csv_ss_sec = date_2_secs(csv_date=csv_strt_snd_times)
    return csv_ss_sec

    # # STIM FINAL PC-TIMES (animal's response)
    # csv_resp_times = df.loc[(df['MSG'] == 'WaitResponse') &
    #                         (df.TYPE == 'TRANSITION'), 'PC-TIME']
    # csv_r_sec, _ = date_2_secs(csv_date=csv_resp_times)

    # sil_tr_pctime = df.loc[(df['MSG'] == 'silence_trial') &
    #                        (df.TYPE == 'VAL'), 'PC-TIME']
    # csv_sltr_sec, _ = date_2_secs(csv_date=sil_tr_pctime)
    # # GET ALSO BPOD TIMES
    # # StartSound
    # csv_strt_snd_times = df.loc[(df['MSG'] == 'StartSound') &
    #                             (df.TYPE == 'TRANSITION'),
    #                             'BPOD-INITIAL-TIME'].values
    # # Trial start time
    # csv_trial_bpod_time = df.loc[(df['MSG'] == 'TRIAL-BPOD-TIME') &
    #                              (df.TYPE == 'INFO'),
    #                              'BPOD-INITIAL-TIME'].values
    # csv_ss_bp_sec = csv_strt_snd_times + csv_trial_bpod_time

    # csv_sltr_sec = csv_sltr_sec-csv_ss_sec[0]
    # csv_ss_sec = csv_ss_sec-csv_ss_sec[0]
    # csv_tmplt = get_template(events=csv_ss_sec, factor=tmplt_factor)
    # csv_ss_bp_sec = csv_ss_bp_sec - csv_ss_bp_sec[0]

    # return csv_ss_sec


def date_2_secs(csv_date):
    csv_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
                        for x in csv_date])
    return csv_sec


def get_electro(path, s_rate=3e4, s_rate_eff=2e3):
    # ELECTRO
    # sampling rate
    sampling = int(s_rate/s_rate_eff)
    # Importing the data from a session
    # path = main_folder+'/LE113/electro/LE113_2021-06-05_12-38-09/'
    # load channels (continuous) data
    data_files = glob.glob(path+'/*.dat')
    data_files = [f for f in data_files if 'temp' not in f]
    # assert len(data_files) == 1, 'Number of .dat files is '+str(len(data_files))
    print('Number of .dat files is '+str(len(data_files)))
    print('Loading '+data_files[0])
    data = np.memmap(data_files[0], dtype='int16', mode='r')
    if len(data) % 40 == 0:
        num_ch = 40
        samples = data.reshape((len(data) // num_ch, num_ch))
        assert len(data) % num_ch-1 != 0
    elif len(data) % 39 == 0:
        num_ch = 39
        samples = data.reshape((len(data) // num_ch, num_ch))
        assert len(data) % num_ch+1 != 0
    # subsample data
    samples = samples[0::sampling, :]
    return samples


def find_events(samples, chnls=[35, 36], s_rate=3e4, events='stim_ttl',
                fltr_k=None):
    # load and med-filter TTL channels
    trace1 = samples[:, chnls[0]]
    # normalize
    trace1 = trace1/np.max(trace1)
    # filter
    trace1 = ss.medfilt(trace1, fltr_k) if fltr_k is not None else trace1
    trace2 = samples[:, chnls[1]]
    # normalize
    trace2 = trace2/np.max(trace2)
    # filter
    trace2 = ss.medfilt(trace2, fltr_k) if fltr_k is not None else trace2

    if events == 'stim_ttl':
        # stimulus corresponds to ch36=high and ch35=low
        assert chnls[0] == 35 and chnls[1] == 36
        signal = 1*((trace2-trace1) > 0.5)
    elif events == 'fix':
        # fixation corresponds to ch35=high and ch36=low
        assert chnls[0] == 35 and chnls[1] == 36
        signal = 1*((trace1-trace2) > 0.5)
    elif events == 'outcome':
        # outcome corresponds to ch36=high and ch35=high
        assert chnls[0] == 35 and chnls[1] == 36
        signal = 1*((trace2+trace1) > 1.5)
    elif events == 'stim_analogue':
        assert chnls[0] == 37 and chnls[1] == 38
        signal = 1*((trace2+trace1) > .9)
    # stim starts/ends
    stim_starts = np.where(np.diff(signal) > 0.9)[0]
    stim_ends = np.where(np.diff(signal) < -0.9)[0]
    # to seconds
    ttl_stim_strt = stim_starts/s_rate
    ttl_stim_end = stim_ends/s_rate
    return ttl_stim_strt, ttl_stim_end, signal


def get_template(events, factor=1):
    tmplt = np.zeros((int(factor*events[-1])+1,))
    tmplt[np.round(factor*events.astype(int))] = 1
    tmplt -= np.mean(tmplt)
    return tmplt


def get_spikes(path):
    # Load spike sorted data
    # Times of the spikes, array of lists
    spike_times = np.load(path+'/spike_times.npy')
    # cluster number of each of the spikes, same length as before
    spike_clusters = np.load(path+'/spike_clusters.npy')
    # Cluster labels (good, noise, mua) for the previous two arrays
    df_labels = pd.read_csv(path+'/cluster_group.tsv', sep='\t')
    # sel_clltrs = df_labels.loc[df_labels.group == 'good', 'cluster_id'].values
    sel_clstrs = df_labels['cluster_id'].values
    clstrs_qlt = df_labels['group']
    return spike_times, spike_clusters, sel_clstrs, clstrs_qlt


if __name__ == '__main__':
    plot_stuff = True
    if plot_stuff:
        import matplotlib.pyplot as plt
        plt.close('all')
    s_rate = 3e4
    s_rate_eff = 2e3
    tmplt_factor = 10
    main_folder = '/home/molano/fof_data/behavioral_data/'
    # sbj = 'LE101'
    sbj = 'LE113'
    p = get_behavior(main_folder=main_folder, subject=sbj)
    p.load(p.available[0])
    p.process()
    p.trial_sess.head()  # preprocessed df stored in attr. trial_sess
    df = p.sess
    csv_ss_sec = get_startSound_times(df=df)
    csv_tmplt = get_template(events=csv_ss_sec, factor=tmplt_factor)
    csv_offset = csv_ss_sec[0]
    # csv_ss_sec -= csv_offset
    # get behavior events
    # I changed to using BPOD-INITIAL-TIME instead of PC-TIME. However, there
    # seems to be a missmatch between the two that grows throughout the session
    # StartSound. Apparently, there is a period of time between trials during which
    # the BPOD is switched off and that produces a missmatch between BPOD and TTL
    # times
    path = '/home/molano/fof_data/AfterClustering/LE113/LE113_2021-06-05_12-38-09/'
    # path = '/home/molano/fof_data/LE101/electro/LE101_2021-06-08_10-50-06/'
    samples = get_electro(path=path, s_rate=s_rate, s_rate_eff=s_rate_eff)
    # get stim ttl starts/ends
    ttl_stim_strt, ttl_stim_end, signal = find_events(samples=samples,
                                                      chnls=[35, 36],
                                                      s_rate=s_rate_eff,
                                                      events='stim_ttl',
                                                      fltr_k=3)
    offset = ttl_stim_strt[0] - csv_offset
    ttl_stim_strt -= offset
    inventory = {'rat': [], 'session': [], 'bhv_session': [], 'sgnl_stts': [],
                 'state': [], 'date': [],  'sil_per': [], 'offset': [],
                 'num_stms_csv': [], 'num_stms_anlg': [],
                 'num_stms_ttl': [], 'num_fx_ttl': [], 'num_outc_ttl': [],
                 'stms_dists_med': [], 'stms_dists_max': []}
    for v in inventory.values():
        v.append(np.nan)
    inventory['rat'].append(sbj)
    inventory['session'].append(path)
    inventory['date'].append('')
    inventory['bhv_session'].append('')
    evs_comp = csv_ss_sec
    if len(ttl_stim_strt) > 0:
        inventory['offset'][-1] = ttl_stim_strt[0]
        inventory['num_stms_csv'][-1] = len(evs_comp)
        inventory['num_stms_ttl'][-1] = len(ttl_stim_strt)
        if len(evs_comp) > len(ttl_stim_strt):
            dists = np.array([np.min(np.abs(evs_comp-ttl))
                              for ttl in ttl_stim_strt])
        elif len(evs_comp) < len(ttl_stim_strt):
            dists = np.array([np.min(np.abs(ttl_stim_strt-evs))
                              for evs in evs_comp])
        else:
            dists = np.abs(evs_comp-ttl_stim_strt)
        inventory['stms_dists_med'][-1] = np.median(dists)
        inventory['stms_dists_max'][-1] = np.max(dists)
        inventory['state'].append('ok')
        print('Median difference between start sounds')
        print(np.median(dists))
        print('Max difference between start sounds')
        print(np.max(dists))
    else:
        inventory['state'].append('no_ttls')
    csv_times = date_2_secs(df['PC-TIME'])
    ttl_indx = np.searchsorted(csv_times, ttl_stim_strt)
    df['ttl_stim_strt'] = np.nan
    df['ttl_stim_strt'][ttl_indx] = 1

    # LOAD SPIKES
    spike_times, spike_clusters, sel_clstrs, clstrs_qlt = get_spikes(path=path)
    # plot PSTHs
    plot_psths(spike_times=spike_times, sel_clstrs=sel_clstrs,
               spike_clusters=spike_clusters, clstrs_qlt=clstrs_qlt,
               s_rate=s_rate, spikes_offset=-offset, events=csv_ss_sec,
               margin_spks_plot=1, bin_size=.1, name='stim')
    import sys
    sys.exit()

    ttl_tmplt = get_template(events=ttl_stim_strt, factor=tmplt_factor)
    ttl_stim_strt -= offset

    # get original stim starts/ends
    # ttl_stim_analogue_strt, ttl_stim_analogue_end, _ =\
    #     find_events(samples=samples, chnls=[37, 38], s_rate=s_rate_eff,
    #                 events='stim_analogue')
    aux = np.array([(np.min(np.abs(csv_ss_sec-ttl_ss)),
                     np.argmin(np.abs(csv_ss_sec-ttl_ss)))
                    for ttl_ss in ttl_stim_strt])
    ttl_ref = ttl_stim_strt
    csv_ref = csv_ss_sec
    if plot_stuff:
        plt.figure()
        plt.hist(aux[:, 0])
        plt.figure()
        ev_strt = int(12500000*s_rate_eff/s_rate)
        ev_end = int(13500000*s_rate_eff/s_rate)
        samples_plt = samples[ev_strt:ev_end, :]
        max_samples = np.max(samples_plt, axis=0)
        samples_plt = samples_plt/max_samples
        plt.figure()
        plt.plot(samples_plt)
        plt.plot(np.arange(ev_strt, ev_end)-offset*s_rate_eff,
                 signal[ev_strt:ev_end], label='signal')
        # plt.plot(np.arange(ev_strt, ev_end)-offset*s_rate_eff,
        #          samples[ev_strt:ev_end, 35]/3e4, label='35')
        # plt.plot(np.arange(ev_strt, ev_end)-offset*s_rate_eff,
        #          samples[ev_strt:ev_end, 36]/3e4, label='36', linestyle='--')
        # plt.plot(np.arange(ev_strt, ev_end)-offset*s_rate_eff,
        #          samples[ev_strt:ev_end, 37]/3e4, label='37')
        # plt.plot(np.arange(ev_strt, ev_end)-offset*s_rate_eff,
        #          samples[ev_strt:ev_end, 38]/3e4, label='38', linestyle='--')

        # plt.plot(np.arange(ev_strt, ev_end)-offset*s_rate_eff,
        #          samples[ev_strt:ev_end, 20]/1e4, label='20', linestyle='--')
        # plt.plot(np.arange(ev_strt, ev_end)-offset*s_rate_eff,
        #          samples[ev_strt:ev_end, 21]/1e4, label='21', linestyle='--')

        ev_strt = ev_strt-offset*s_rate_eff
        ev_end = ev_end-offset*s_rate_eff
        plot_events(evs=ttl_stim_strt, ev_strt=ev_strt, ev_end=ev_end,
                    label='ttl', s_rate=s_rate_eff)
        plot_events(evs=csv_ss_sec, ev_strt=ev_strt, ev_end=ev_end, color='m',
                    label='csv-stim', s_rate=s_rate_eff)
        plt.legend()

    # asdasd
    # ttl_ori_tmplt = get_template(events=ttl_stim_analogue_strt,
    #                              factor=tmplt_factor)
    conv_w = 200*tmplt_factor
    conv = np.convolve(csv_tmplt, np.flip(ttl_tmplt[:conv_w]), mode='same')
    offset = np.argmax(conv)-conv_w/2
    assert len(csv_ref) == len(ttl_ref), str(len(csv_ref))+'  '+str(len(ttl_ref))

    # plt.figure()
    # plt.plot(conv)
    # plt.figure()
    # plt.plot(ttl_ori_tmplt)
    # plt.plot(ttl_tmplt, '--')
    # plt.plot(np.arange(len(csv_tmplt))-offset, csv_tmplt-0.1)
    # asdasd
    # plt.figure()
    # strt = 0
    # end = 1e10
    # analog_stim1 = samples[:, 37]
    # analog_stim1 = analog_stim1/np.max(analog_stim1)
    # analog_stim2 = samples[:, 38]
    # analog_stim2 = analog_stim2/np.max(analog_stim2)
    # aux = signal - (analog_stim1+analog_stim2)/2
    # krnl = np.zeros((1000,))
    # krnl[-10:] = -1
    # krnl = krnl - np.mean(krnl)
    # aux = aux - np.mean(aux)
    # plt.plot(aux)
    # plt.plot(np.convolve(aux, np.flip(krnl), mode='same'))
    # plt.plot(samples[int(strt*3e4):int(end*3e4), 35], label='35')
    # plt.plot(samples[int(strt*3e4):int(end*3e4), 36], label='36')
    # plt.plot(samples[int(strt*3e4):int(end*3e4), 37]-10, label='37')
    # plt.plot(samples[int(strt*3e4):int(end*3e4), 38]-10, label='38')
    # plt.legend()
    # ttl_ref = ttl_stim_analogue_strt[0]
    # outcome starts/ends
    # ttl_outc_strt, ttl_outc_end, _ = find_events(samples=samples, chnls=[35, 36],
    #                                              s_rate=s_rate_eff,
    #                                              events='outcome')

    # compute spikes offset from stimulus start
    # spikes_offset = -ttl_ref[0]
    # ttl_ref = ttl_ref+spikes_offset
    # ttl_outc_strt = ttl_outc_strt+spikes_offset
    # f, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    # ax = ax.flatten()
    # ax[0].plot(csv_ref-ttl_ref)
    # ax[0].set_ylabel('CSV_PC_TIME - TTL (s)')
    # ax[1].hist(csv_ref-ttl_ref, 50)
    # ax[1].set_ylabel('Count')
    # ax[1].set_xlabel('CSV_PC_TIME - TTL (s)')
    # ax[2].plot(csv_ss_bp_sec-ttl_ref)
    # ax[2].set_ylabel('CSV_BPOD_TIME - TTL (s)')
    # ax[2].set_xlabel('Trial number')
    # ax[3].hist(csv_ss_bp_sec-ttl_ref, 50)
    # ax[3].set_ylabel('Count')
    # ax[3].set_xlabel('CSV_BPOD_TIME - TTL (s)')
    # f.savefig('/home/molano/Dropbox/csv_ttl_diff.png')
    # assert np.max(np.abs(csv_ref-ttl_ref)) < 0.05,\
    #     np.argmax(np.abs(csv_ref-ttl_ref))
    # assert len(csv_so_sec) == len(ttl_outc_strt)
    # assert np.max(csv_so_sec-ttl_outc_strt) < 0.05, print(np.max(csv_so_sec -
    #                                                    ttl_outc_strt))

    # import sys
    # sys.exit()
    # offset = 29550000
    # num_samples = 200000
    # events = {'stim_starts': ttl_stim_strt, 'outc_starts': ttl_outc_strt,
    #           'samples': samples[offset:offset+num_samples, 35:39]}
    # np.savez(path+'/events.npz', **events)

    # f = plt.figure()
    # plot_events(ttl_stim_strt, label='ttl-stim', color='m')
    # plot_events(csv_ss_sec, label='start-sound', color='c', lnstl='--')
    # f.savefig('/home/molano/Dropbox/stim_check.png')

    # f = plt.figure()
    # plot_events(ttl_outc_strt, label='ttl-outcome', color='m')
    # plot_events(csv_so_sec, label='outcome', color='c', lnstl='--')
    # f.savefig('/home/molano/Dropbox/outcome_check.png')

