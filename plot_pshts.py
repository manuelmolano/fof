#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:26:32 2021

@author: molano
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import glob
import seaborn as sns
from dPCA import dPCA
from copy import deepcopy as dpcp
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


def plt_psths(spk_tms, evs, ax=None, margin_psth=1000, std_conv=20, lbl='',
              color='k', alpha=1, plot=True):
    psth_cnv, peri_ev = ut.convolve_psth(spk_times=spk_tms, events=evs,
                                         std=std_conv, margin=margin_psth)
    if plot and len(psth_cnv) > 0:
        ax.axvline(x=0, linestyle='--', color=(.7, .7, .7))
        xs = np.arange(2*margin_psth)-margin_psth
        xs = xs/1000
        ax.plot(xs, psth_cnv, label=lbl+' (n: '+str(len(evs))+')', color=color,
                alpha=alpha)
    features = {}
    features['conv_psth'] = psth_cnv
    features['peri_ev'] = peri_ev
    features['aligned_spks'] = [np.where(pe)[0]-margin_psth for pe in peri_ev]
    return features


def plot_scatter(ax, spk_tms, evs, margin_psth, color='k', offset=0, alpha=1):
    spk_raster = 1000*spk_tms
    evs = 1000*evs
    for i_ev, ev in enumerate(evs):
        spk_tmp = spk_raster-ev
        spk_tmp = spk_tmp[np.logical_and(spk_tmp > -margin_psth,
                                         spk_tmp < margin_psth)]
        ax.scatter(spk_tmp, np.ones((len(spk_tmp)))+i_ev+offset,
                   color=color, s=1, alpha=alpha)


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


def get_cond_trials(b_data, e_data, ev, cl, conditions, evs_mrgn=1e-2,
                    fixtn_time=.3, margin_psth=1000, std_conv=20):
    # get spikes
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    # select trials
    filt_evs, indx_good_evs = preprocess_events(b_data=b_data, e_data=e_data,
                                                ev=ev, evs_mrgn=evs_mrgn,
                                                fixtn_time=fixtn_time)
    num_tr = len(b_data)
    if cond['ch']:
        ch_mat = b_data['R_response'].values
    else:
        ch_mat = np.zeros((num_tr))-1
    if cond['prev_ch']:
        prev_ch_mat = b_data['R_response'].shift(periods=1).values
    else:
        prev_ch_mat = np.zeros((num_tr))-1
    if cond['outc']:
        outc_mat = b_data['hithistory'].values
    else:
        outc_mat = np.zeros((num_tr))-1
    if cond['prev_outc']:
        prev_outc_mat = b_data['hithistory'].shift(periods=1).values
    else:
        prev_outc_mat = np.zeros((num_tr))-1
    if cond['prev_tr']:
        prev_tr_mat = b_data['rep_response'].shift(periods=1).values
    else:
        prev_tr_mat = np.zeros((num_tr))-1
    active_idx = [i for i, v in enumerate(cond.values()) if v]
    shape = [200]+len(active_idx)*[2]+[2*margin_psth]
    trialR = np.empty(shape)
    trialR[:] = np.nan
    min_num_tr = 1e6
    for i_c, case in enumerate(conditions):
        choice = case[0]
        prev_choice = case[1]
        outcome = case[2]
        prev_outcome = case[3]
        prev_trans = case[4]
        mask = np.logical_and.reduce((indx_good_evs,
                                      ch_mat == choice,
                                      prev_ch_mat == prev_choice,
                                      outc_mat == outcome,
                                      prev_outc_mat == prev_outcome,
                                      prev_tr_mat == prev_trans))
        evs = filt_evs[mask]
        psth, peri_ev = plt_psths(spk_tms=spk_tms, evs=evs, std_conv=std_conv,
                                  margin_psth=margin_psth, plot=False)
        idx = [np.arange(len(peri_ev))]+[case[i] for i in active_idx]
        if len(peri_ev) > 0:
            trialR[idx] = peri_ev
        min_num_tr = min(min_num_tr, len(peri_ev))
    return trialR, min_num_tr


def psth_binary_cond(mat, cl, e_data, b_data, ax, ev, lbls, clrs, spk_offset=0,
                     std_conv=20, margin_psth=1000, fixtn_time=.3,
                     evs_mrgn=1e-2, prev_choice=False, mask=None, alpha=1,
                     sign_w=100):
    """
    Plot raster-plots and psths conditioned on (prev) choice.

    Parameters
    ----------
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
    prev_choice : boolean, optional
        whether to condition on previous instead of current choice (False)
    mask : array, optional
        mask to further filter trials (None)
    alpha : float, optional
        alpha for plotting (1)

    Returns
    -------
    spk_offset : int
        spikes offset for next raster-plots

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
    features = {'conv_psth': [], 'aligned_spks': [], 'peri_ev': []}
    for i_v, v in enumerate(np.unique(mat)):
        # plot psth for right trials
        indx_ch = np.logical_and(mat == v, indx_good_evs)
        evs = filt_evs[indx_ch]
        if len(evs) > 0:
            assert len(np.unique(evs)) == len(evs), 'Repeated events!'
            plot_scatter(ax=ax[0], spk_tms=spk_tms, evs=evs, color=clrs[i_v],
                         margin_psth=margin_psth, alpha=alpha, offset=spk_offset)
            feats = plt_psths(spk_tms=spk_tms, evs=evs, ax=ax[1],
                              std_conv=std_conv, margin_psth=margin_psth,
                              lbl=lbls[i_v], color=clrs[i_v], alpha=alpha)
            ut.append_features(features=features, new_data=feats)
        spk_offset += len(evs)
    # assess significance
    ylim = ax[1].get_ylim()
    resps = features['peri_ev']
    sign_mat = ut.significance(mat=resps, window=sign_w)
    edges = np.linspace(0, resps[0].shape[1], int(resps[0].shape[1]/sign_w)+1)
    for i_e in range(len(edges)-1):
        if sign_mat[i_e] < 0.01:
            sign_per = (np.array([edges[i_e], edges[i_e+1]])-margin_psth)/1e3
            print(sign_per)
            ax[1].plot(sign_per, [ylim[1], ylim[1]], 'k')
    features['sign_mat'] = sign_mat
    return spk_offset, features


def plot_figure(e_data, b_data, cl, cl_qlt, session, sv_folder, cond,
                std_conv=20, margin_psth=1000):
    f, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 10), sharey='row')
    ev_keys = ['fix_strt', 'stim_ttl_strt', 'outc_strt']
    features = {'conv_psth': [], 'aligned_spks': [], 'peri_ev': [], 'sign_mat': []}
    for i_e, ev in enumerate(ev_keys):
        prev_choice = (cond == 'prev_ch')
        choice = b_data['R_response'].shift(periods=1*prev_choice).values
        lbls = ['Right', 'Left']
        clrs = [verde, morado]
        _, feats = psth_binary_cond(cl=cl, mat=choice, e_data=e_data,
                                    b_data=b_data, ev=ev, ax=ax[:, i_e],
                                    std_conv=std_conv, margin_psth=margin_psth,
                                    prev_choice=prev_choice, lbls=lbls, clrs=clrs)
        ut.append_features(features=features, new_data=feats)

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
    for r in rats:
        rat = os.path.basename(r)
        all_traces = {}
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
            else:
                i = idx[0]
                print(str(np.round(inv['num_stms_csv'][i], 3))+' / ' +
                      str(np.round(inv['sil_per'][i], 3))+' /// ' +
                      str(np.round(inv['num_stim_ttl'][i], 3))+' / ' +
                      str(np.round(inv['stim_ttl_dists_med'][i], 3))+' / ' +
                      str(np.round(inv['stim_ttl_dists_max'][i], 3))+' /// ' +
                      str(np.round(inv['num_stim_analogue'][i], 3))+' / ' +
                      str(np.round(inv['stim_analogue_dists_med'][i], 3))+' / ' +
                      str(np.round(inv['stim_analogue_dists_max'][i], 3)))
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
                        features = plot_figure(e_data=e_data, b_data=b_data, cl=cl,
                                               cl_qlt=cl_qlt, session=session,
                                               std_conv=std_conv, cond=cond,
                                               margin_psth=margin_psth,
                                               sv_folder=sv_folder)
                        all_traces[str(cl)+'_'+session] = features

        np.savez(sv_folder+'/'+rat+'_traces.npz', **all_traces)


def compute_dPCA(main_folder, sel_sess, sel_rats, inv, lbls_cps, std_conv=20,
                 margin_psth=1000, sel_qlts=['good'], conditioning={},
                 sv_folder=''):
    def get_conds(conditioning={}):
        cond = {'ch': True, 'prev_ch': False, 'outc': False, 'prev_outc': True,
                'prev_tr': False}
        cond.update(conditioning)
        ch = [0, 1] if cond['ch'] else [-1]
        prev_ch = [0, 1] if cond['prev_ch'] else [-1]
        outc = [0, 1] if cond['outc'] else [-1]
        prev_outc = [0, 1] if cond['prev_outc'] else [-1]
        prev_tr = [0, 1] if cond['prev_tr'] else [-1]
        cases = itertools.product(ch, prev_ch, outc, prev_outc, prev_tr)
        return cases

    conditions = get_conds(conditioning=conditioning)
    time = np.arange(2*margin_psth)
    num_comps = 2
    num_cols = len(lbls_cps)+1
    ev_keys = ['fix_strt', 'stim_ttl_strt', 'outc_strt']
    rats = glob.glob(main_folder+'LE*')
    for r in rats:
        rat = os.path.basename(r)
        sessions = glob.glob(r+'/LE*')
        all_trR = []
        min_num_tr = 1e6
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
                        for i_e, ev in enumerate(ev_keys):
                            trR, min_n_tr =\
                                get_cond_trials(b_data=b_data, e_data=e_data,
                                                ev=ev, cl=cl, std_conv=std_conv,
                                                margin_psth=margin_psth,
                                                conditions=dpcp(conditions))
                            if min_n_tr > 10:
                                all_trR.append(trR)
                                min_num_tr = min(min_num_tr, min_n_tr)
        if len(all_trR) > 0:
            conditions = get_conds(conditioning=conditioning)
            all_trR = np.array(all_trR)
            all_trR = np.swapaxes(all_trR, 0, 1)
            # trial-average data
            R = np.nanmean(all_trR, 0)
            # center data
            mean_acr_tr = np.mean(R.reshape((R.shape[0], -1)), 1)
            for l in range(len(lbls_cps)):
                mean_acr_tr = mean_acr_tr[:, None]
            R -= mean_acr_tr
            dpca = dPCA.dPCA(labels=lbls_cps, regularizer='auto')
            dpca.protect = ['t']
            all_trR = all_trR[:min_num_tr]
            Z = dpca.fit_transform(R, all_trR)
            var_exp = dpca.explained_variance_ratio_
            f, ax = plt.subplots(nrows=num_comps, ncols=num_cols, figsize=(16, 7))
            lbls_to_plot = list(lbls_cps)+[lbls_cps]
            for i_c in range(num_comps):
                for i_d, dim in enumerate(lbls_to_plot):
                    for cond in dpcp(conditions):
                        i_cond = [c for c in cond if c != -1]
                        idx = [i_c]+i_cond+[np.arange(2*margin_psth)]
                        ax[i_c, i_d].plot(time, Z[dim][idx], label=str(i_cond))
                    ax[i_c, i_d].set_title(dim+' C' + str(i_c+1) + ' v. expl.: ' +
                                           str(np.round(var_exp[dim][i_c], 2)))
            ax[i_c, i_d].legend()
            name = ''.join([i[0]+str(i[1]) for i in conditioning.items()])
            f.savefig(sv_folder+rat+'_'+name+'.png')


def units_stats(inv, main_folder, sv_folder, name='ch'):
    rats = glob.glob(main_folder+'LE*')
    rats = [x for x in rats if x[-4] != '.']
    f, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))  # ,sharex=1, sharey=1)
    ax = ax.flatten()
    total_sums = np.zeros((2))
    for i_r, r in enumerate(rats):
        rat = os.path.basename(r)
        sessions = glob.glob(r+'/LE*')
        num_unts = []
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
            else:
                i = idx[0]
                print(str(np.round(inv['num_stms_csv'][i], 3))+' / ' +
                      str(np.round(inv['sil_per'][i], 3))+' /// ' +
                      str(np.round(inv['num_stim_ttl'][i], 3))+' / ' +
                      str(np.round(inv['stim_ttl_dists_med'][i], 3))+' / ' +
                      str(np.round(inv['stim_ttl_dists_max'][i], 3))+' /// ' +
                      str(np.round(inv['num_stim_analogue'][i], 3))+' / ' +
                      str(np.round(inv['stim_analogue_dists_med'][i], 3))+' / ' +
                      str(np.round(inv['stim_analogue_dists_max'][i], 3)))
            e_file = sess+'/e_data.npz'
            e_data = np.load(e_file, allow_pickle=1)
            sel_clstrs = e_data['sel_clstrs']
            clstrs_qlt = e_data['clstrs_qlt']
            print(inv['sess_class'][idx[0]])
            print('Total number of cluster: ', len(sel_clstrs))
            print('Number single units: ', np.sum(clstrs_qlt == 'good'))
            print('Number MUAs: ', np.sum(clstrs_qlt == 'mua'))
            num_unts.append([np.sum(clstrs_qlt == 'good'),
                             np.sum(clstrs_qlt == 'mua')])
        num_unts = np.array(num_unts)
        means = np.mean(num_unts, axis=0)
        sums = np.sum(num_unts, axis=0)
        total_sums += sums
        maxs = np.max(num_unts)
        print('Numbers and counts of single units:')
        print(np.unique(num_unts[:, 0], return_counts=1))
        print('Numbers and counts of multi-units:')
        print(np.unique(num_unts[:, 1], return_counts=1))
        if maxs > 0:
            ax[i_r].hist(num_unts, np.arange(maxs+2)-0.5)
            ax[i_r].set_title(rat)
            ax[i_r].legend(['SU ('+str(np.round(means[0], 2)) +
                            ', '+str(sums[0])+')',
                            'MUA ('+str(np.round(means[1], 2)) +
                            ', '+str(sums[1])+')'])
            if i_r == 0 or i_r == 3:
                ax[i_r].set_ylabel('Number of sessions')
            if i_r > 2:
                ax[i_r].set_xlabel('Number of units per session')
    print('Total number of SU and MUA:')
    print(total_sums)
    f.savefig(sv_folder+'num_units_per_sess_and_rat.png')
    # if inv['sess_class'][idx[0]] == 'good' and len(sel_clstrs) > 0:


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
    sel_sess = []  # ['LE104_2021-06-02_13-14-24']  # ['LE104_2021-05-17_12-02-40']
    # ['LE77_2020-12-04_08-27-33']  # ['LE113_2021-06-05_12-38-09']
    if analysis_type == 'dpca':
        cond = {'ch': False, 'prev_ch': True, 'outc': False, 'prev_outc': True,
                'prev_tr': True}
        lbls_cps = 'cprt'
        compute_dPCA(inv=inv, main_folder=main_folder, std_conv=std_conv,
                     margin_psth=margin_psth, sel_sess=sel_sess, sel_rats=sel_rats,
                     conditioning=cond, lbls_cps=lbls_cps, sv_folder=sv_folder)
    elif analysis_type == 'psth':
        # file = main_folder+'/'+rat+'/sessions/'+session+'/extended_df'
        # ['no_cond', 'prev_ch_and_context', 'context' 'prev_outc',
        # 'prev_outc_and_ch', 'coh', 'prev_ch', 'ch', 'outc']
        conditions = ['ch']
        for cond in conditions:
            sv_f = sv_folder+'/'+cond+'/'
            if not os.path.exists(sv_f):
                os.makedirs(sv_f)
            batch_plot(inv=inv, main_folder=main_folder, cond=cond,
                       std_conv=std_conv, margin_psth=margin_psth,
                       sel_sess=sel_sess, sv_folder=sv_f, sel_rats=sel_rats)
    elif analysis_type == 'stats':
        units_stats(inv=inv, main_folder=main_folder, sv_folder=sv_folder)
