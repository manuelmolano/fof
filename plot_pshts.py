#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:26:32 2021

@author: molano
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns
# import utils as ut
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')
import utils as ut
rojo = np.array((228, 26, 28))/255
azul = np.array((55, 126, 184))/255
verde = np.array((77, 175, 74))/255
morado = np.array((152, 78, 163))/255
naranja = np.array((255, 127, 0))/255
marron = np.array((166, 86, 40))/255
amarillo = np.array((155, 155, 51))/255
rosa = np.array((247, 129, 191))/255
cyan = np.array((0, 1, 1))
gris = np.array((.5, .5, 0.5))
azul_2 = np.array([56, 108, 176])/255
rojo_2 = np.array([240, 2, 127])/255

grad_colors = sns.diverging_palette(145, 300, n=7)
grad_colors = [[0.8*g for g in gc] for gc in grad_colors]
PLOT = False


def plt_psths(spk_tms, evs, ax=None, margin_psth=1000, std_conv=20, lbl='',
              color='k', alpha=1):
    psth_cnv, peri_ev = ut.convolve_psth(spk_times=spk_tms, events=evs,
                                         std=std_conv, margin=margin_psth)
    if PLOT and len(psth_cnv) > 0:
        ax.axvline(x=0, linestyle='--', color=(.7, .7, .7))
        xs = np.arange(2*margin_psth)-margin_psth
        xs = xs/1000
        ax.plot(xs, psth_cnv, label=lbl+' (n: '+str(len(evs))+')', color=color,
                alpha=alpha)
    features = {}
    features['conv_psth'] = psth_cnv
    features['peri_ev'] = peri_ev
    return features


def plot_scatter(ax, spk_tms, evs, margin_psth, color='k', offset=0, alpha=1):
    spk_raster = 1000*spk_tms
    evs = 1000*evs
    feats_spks = {'aligned_spks': []}
    for i_ev, ev in enumerate(evs):
        spk_tmp = spk_raster-ev
        spk_tmp = spk_tmp[np.logical_and(spk_tmp > -margin_psth,
                                         spk_tmp < margin_psth)]
        feats_spks['aligned_spks'].append(spk_tmp)
        if PLOT:
            ax.scatter(spk_tmp, np.ones((len(spk_tmp)))+i_ev+offset,
                       color=color, s=1, alpha=alpha)
    return feats_spks


def find_repeated_evs(evs_dists, indx_evs):
    """
    Find repeated events in indx_evs and only keeps the ones with the minimum
    distance to the associated TTL ev.

    Both vectors have the length equal to the number of csv events.

    Parameters
    ----------
    evs_dists : array
        vector containing, for each csv fixation event, the min distance to TTL ev.
    indx_evs : array
        vector containing, for each csv fix event, the index of the closest TTL ev.

    Returns
    -------
    idx_assoc_tr : array
        mask indicating which indexes of the csv vector that should be discarded
        because there is not associated TTL event.

    """
    # remove repeated events (when 2 csv evs have the same ttl event associated)
    evs_unq, indx, counts = np.unique(indx_evs, return_index=1,
                                      return_counts=1)
    idx_assoc_tr = np.ones((len(evs_dists),)) == 1
    indx_rep = np.where(counts > 1)[0]
    if len(indx_rep) > 0:  # if true, there are 2 csv evs w/ the same ttl ev assoc
        for i_r in indx_rep:
            # get indx with the same associated ttl ev
            idx_i_rep = np.where(indx_evs == evs_unq[i_r])[0]
            assert len(idx_i_rep) > 1
            # find the csv ev that is closer to the ttl ev
            min_dist_idx = np.argmin(evs_dists[idx_i_rep])
            # remove from mask the rest of indxes
            idx_assoc_tr[np.delete(idx_i_rep, min_dist_idx)] = False
    return idx_assoc_tr


def preprocess_events(b_data, e_data, ev, evs_mrgn, fixtn_time):
    # pass times to seconds
    trial_times = ut.date_2_secs(b_data.fix_onset_dt)
    # get events
    events = e_data[ev]
    # remove special trials and trials when sound failed
    idx_good_trs = np.logical_and(b_data['special_trial'].values == 0,
                                  b_data['soundrfail'].values == 0)
    # look for ttl event associated to each csv fixation event
    aux = np.array([(np.min(np.abs(events-tr)), np.argmin(np.abs(events-tr)))
                    for tr in trial_times])
    evs_dists = aux[:, 0]
    indx_evs = aux[:, 1].astype(int)
    filt_evs = events[indx_evs]
    # find repeated evs
    idx_assoc_tr = find_repeated_evs(evs_dists, indx_evs)
    # get indexes of ttl events that are close to csv fixation events
    if ev == 'fix_strt':
        synchr_cond = evs_dists < evs_mrgn
    elif ev == 'stim_ttl_strt' or ev == 'stim_anlg_strt':
        synchr_cond = np.abs(evs_dists-fixtn_time) < evs_mrgn
    elif ev == 'outc_strt':
        synchr_cond = np.ones((len(evs_dists),)) == 1
    # indx of regular trials
    indx_good_evs = np.logical_and.reduce((synchr_cond, idx_good_trs,
                                           idx_assoc_tr))
    return filt_evs, indx_good_evs


def get_label(cs):
    ch_lbl = ['Right', 'Left', '-']
    prev_ch_lbl = ['Prev. right', 'Prev left', '-']
    outc_lbl = ['Error', 'Correct', '-']
    prev_outc_lbl = ['Prev. error', 'Prev. correct', '-']
    prev_tr_lbl = ['Prev. Alt.', 'Prev. Rep.', '-']
    return ch_lbl[cs[0]]+' / '+prev_ch_lbl[cs[1]]+' / '+outc_lbl[cs[2]]+' / ' +\
        prev_outc_lbl[cs[3]]+' / '+prev_tr_lbl[cs[4]]


def psth_binary_cond(mat, cl, e_data, b_data, ax, ev, lbls, clrs, spk_offset=0,
                     std_conv=20, margin_psth=1000, fixtn_time=.3,
                     evs_mrgn=1e-2, mask=None, alpha=1,
                     sign_w=100):
    """
    Plot raster-plots and psths conditioned on (prev) choice.

    Parameters
    ----------
    mat : array
        array with values for conditioning.
    cl : int
        cluster to plot.
    e_data : dict
        dictionary containing spikes info (times, clusters, quality..).
    b_data : dataframe
        contains behavioral info.
    ax : axis
        where to plot the rasterplots (ax[0]) and the psths (ax[1]).
    ev : str
        event to align to: fix_strt, stim_ttl_strt, outc_strt, stim_anlg_strt
    lbls : list
        list with labels for legend.
    clrs : list
        list with colors for traces.
    spk_offset : int, optional
        offset to plot spikes in the raster-plot (0)
    std_conv : float, optional
        std for gaussian used to produce firing rates (20 ms)
    margin_psth : int, optional
        pre and post event time to plot rasters and psths (1000 ms)
    fixtn_time : float, optional
        fixation time (.3 s)
    evs_mrgn : float, optional
        max missmatch allowed between csv and ttl events (1e-2)
    mask : array, optional
        mask to further filter trials (None)
    alpha : float, optional
        alpha for plotting (1)
    sign_w : int, optional
        window to use to calculate significances (100ms)

    Returns
    -------
    spk_offset : int
        spikes offset for next raster-plots
    features : dict
        dictionary containing the info to store.

    """
    # get spikes
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    # select trials
    filt_evs, indx_good_evs = preprocess_events(b_data=b_data, e_data=e_data,
                                                ev=ev, evs_mrgn=evs_mrgn,
                                                fixtn_time=fixtn_time)
    # further filtering
    if mask is not None:
        indx_good_evs = np.logical_and(indx_good_evs, mask)
    # get choices
    features = {'conv_psth': [], 'peri_ev': []}
    feats_spks = {'aligned_spks': []}
    vals, counts = np.unique(mat, return_counts=1)
    vals = vals[counts > np.sum(counts)/20]
    assert len(vals) == 2, str(vals)
    # vals = vals[~ np.isnan(vals)]
    for i_v, v in enumerate(vals):
        # plot psth for right trials
        indx_ch = np.logical_and(mat == v, indx_good_evs)
        evs = filt_evs[indx_ch]
        if len(evs) > 0:
            assert len(np.unique(evs)) == len(evs), 'Repeated events!'
            f_spks = plot_scatter(ax=ax[0], spk_tms=spk_tms, evs=evs,
                                  color=clrs[i_v], margin_psth=margin_psth,
                                  alpha=alpha, offset=spk_offset)
            ut.append_features(features=feats_spks, new_data=f_spks)
            feats = plt_psths(spk_tms=spk_tms, evs=evs, ax=ax[1],
                              std_conv=std_conv, margin_psth=margin_psth,
                              lbl=lbls[i_v], color=clrs[i_v], alpha=alpha)
            ut.append_features(features=features, new_data=feats)
        else:
            return 0, {}
        spk_offset += len(evs)
    features.update(feats_spks)
    # assess significance
    resps = features['peri_ev']
    if (np.array([len(x) for x in resps]) > 0).all():
        sign_mat = ut.significance(mat=resps, window=sign_w)
        features['sign_mat'] = sign_mat
        if PLOT:  # plot significance
            ylim = ax[1].get_ylim()
            edges = np.linspace(0, resps[0].shape[1], int(resps[0].shape[1]/sign_w)+1)
            for i_e in range(len(edges)-1):
                if sign_mat[i_e] < 0.01:
                    sign_per = (np.array([edges[i_e], edges[i_e+1]])-margin_psth)/1e3
                    ax[1].plot(sign_per, [ylim[1], ylim[1]], 'k')
    else:
        features['sign_mat'] = []
    return spk_offset, features


def get_responses(e_data, b_data, cl, cl_qlt, session, sv_folder, cond,
                  std_conv=20, margin_psth=1000):
    if PLOT:
        f, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 10), sharey='row')
    else:
        ax = np.zeros((2, 3))
    ev_keys = ['fix_strt', 'stim_ttl_strt', 'outc_strt']
    features = {'conv_psth': [], 'aligned_spks': [], 'peri_ev': [], 'sign_mat': []}
    for i_e, ev in enumerate(ev_keys):
        if 'ch' in cond:
            prev_choice = (cond == 'prev_ch')
            choice = b_data['R_response'].shift(periods=1*prev_choice).values
            lbls = ['Prev. Right', 'Prev. Left']\
                if prev_choice else ['Right', 'Left']
            clrs = [verde, morado]
            _, feats = psth_binary_cond(cl=cl, mat=choice, e_data=e_data,
                                        b_data=b_data, ev=ev, ax=ax[:, i_e],
                                        std_conv=std_conv, margin_psth=margin_psth,
                                        lbls=lbls, clrs=clrs)
        elif 'outc' in cond:
            prev_outc = (cond == 'prev_outc')
            outcome = b_data['hithistory'].shift(periods=1*prev_outc).values
            lbls = ['Prev. error', 'Prev. correct']\
                if prev_outc else ['Error', 'Correct']
            clrs = ['k', naranja]
            _, feats = psth_binary_cond(cl=cl, mat=outcome, e_data=e_data,
                                        b_data=b_data, ev=ev, ax=ax[:, i_e],
                                        std_conv=std_conv, margin_psth=margin_psth,
                                        lbls=lbls, clrs=clrs)
        elif cond == 'prev_tr':
            prev_tr_mat = b_data['rep_response'].shift(periods=1).values
            lbls = ['Prev. Alt.', 'Prev. Rep.']
            clrs = [rojo, azul]
            _, feats = psth_binary_cond(cl=cl, mat=prev_tr_mat, e_data=e_data,
                                        b_data=b_data, ev=ev, ax=ax[:, i_e],
                                        std_conv=std_conv, margin_psth=margin_psth,
                                        lbls=lbls, clrs=clrs)
        if len(feats) > 0:
            ut.append_features(features=features, new_data=feats)
    if PLOT:
        ax[0, 0].set_ylabel('Trial')
        ax[1, 0].set_ylabel('Firing rate (Hz)')
        ax[1, 0].set_xlabel('Peri-fixation time (s)')
        ax[1, 1].set_xlabel('Peri-stim time (s)')
        ax[1, 2].set_xlabel('Peri-outcome time (s)')
        for a in ax.flatten():
            ut.rm_top_right_lines(a)
            a.legend()
        num_spks = np.sum(e_data['clsts'] == cl)
        f.suptitle(str(cl)+' / #spks: '+str(num_spks)+' / qlt: '+cl_qlt)
        if not os.path.exists(sv_folder):
            os.makedirs(sv_folder)
        print(sv_folder+'/'+cl_qlt+'_'+str(cl)+'_'+session+'_'+'.png')
        f.savefig(sv_folder+'/'+cl_qlt+'_'+str(cl)+'_'+session+'_' +
                  '.png')
        plt.close(f)
    return features

# TODO: separate computing from plotting


def batch_plot(inv, main_folder, sv_folder, cond, std_conv=20, margin_psth=1000,
               sel_sess=[], sel_rats=[], name='ch', sel_qlts=['good', 'mua']):
    rats = glob.glob(main_folder+'LE*')
    features = {'sign_mat': []}
    for r in rats:
        rat = os.path.basename(r)
        sessions = glob.glob(r+'/LE*')
        for sess in sessions:
            session = os.path.basename(sess)
            print('----')
            print(session)
            if session not in sel_sess and rat not in sel_rats and\
               (len(sel_sess) != 0 or len(sel_rats) != 0):
                continue
            idx = [i for i, x in enumerate(inv['session']) if x.endswith(session)]
            if len(idx) != 1:
                print(str(idx))
                continue
            e_file = sess+'/e_data.npz'
            e_data = np.load(e_file, allow_pickle=1)
            sel_clstrs = e_data['sel_clstrs']
            print(inv['sess_class'][idx[0]])
            print('Number of cluster: ', len(sel_clstrs))
            if inv['sess_class'][idx[0]] == 'good' and len(sel_clstrs) > 0:
                b_file = sess+'/df_trials'
                b_data = pd.read_pickle(b_file)
                for i_cl, cl in enumerate(sel_clstrs):
                    cl_qlt = e_data['clstrs_qlt'][i_cl]
                    if cl_qlt in sel_qlts:
                        feats = get_responses(e_data=e_data, b_data=b_data, cl=cl,
                                              cl_qlt=cl_qlt, session=session,
                                              std_conv=std_conv, cond=cond,
                                              margin_psth=margin_psth,
                                              sv_folder=sv_folder)
                        ut.append_features(features=features, new_data=feats)

    np.savez(sv_folder+'/feats.npz', **features)
    return features


if __name__ == '__main__':
    plt.close('all')
    analysis_type = 'psth'
    std_conv = 50
    margin_psth = 1000
    home = 'molano'
    main_folder = '/home/'+home+'/fof_data/'
    if home == 'manuel':
        sv_folder = main_folder+'/psths/'
    elif home == 'molano':
        sv_folder = '/home/molano/Dropbox/project_Barna/FOF_project/psths/'
    inv = np.load('/home/'+home+'/fof_data/sess_inv_extended.npz', allow_pickle=1)
    sel_rats = []  # ['LE113']  # 'LE101'
    sel_sess = []  # ['LE104_2021-04-12_11-38-47']  # ['LE104_2021-06-02_13-14-24']
    # ['LE77_2020-12-04_08-27-33']  # ['LE113_2021-06-05_12-38-09']
    # file = main_folder+'/'+rat+'/sessions/'+session+'/extended_df'
    # ['no_cond', 'prev_ch_and_context', 'context' 'prev_outc',
    # 'prev_outc_and_ch', 'coh', 'prev_ch', 'ch', 'outc']
    conditions = ['ch', 'prev_ch', 'outc', 'prev_outc', 'prev_tr']
    for cond in conditions:
        sv_f = sv_folder+'/'+cond+'/'
        if not os.path.exists(sv_f):
            os.makedirs(sv_f)
        batch_plot(inv=inv, main_folder=main_folder, cond=cond,
                   std_conv=std_conv, margin_psth=margin_psth,
                   sel_sess=sel_sess, sv_folder=sv_f, sel_rats=sel_rats)
