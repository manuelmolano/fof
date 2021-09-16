#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:06:14 2021

@author: molano
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import glob
from dPCA import dPCA
from copy import deepcopy as dpcp
import plot_pshts as pp
from numpy import logical_and as and_
# this is for PCA plots
lbls_perf = ['error', 'correct', '-']
lbls_cont = ['alt', 'rep', '-']
lbls_last_trans = ['alt', 'rep', '-']
lbls_side = ['left', 'right', '-']
lbls_ev = ['low', 'high', '-']


def get_fig(display_mode=None, font=6, figsize=(8, 8)):
    import matplotlib
    if display_mode is not None:
        if display_mode:
            matplotlib.use('Qt5Agg')
        else:
            matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.4  # width reserved for blank space between subplots
    hspace = 0.4  # height reserved for white space between subplots
    f = plt.figure(figsize=figsize, dpi=250)
    matplotlib.rcParams.update({'font.size': font, 'lines.linewidth': 0.5,
                                'axes.titlepad': 1, 'lines.markersize': 3})
    plt.subplots_adjust(left=left, bottom=bottom, right=right,
                        top=top, wspace=wspace, hspace=hspace)
    return f


def plot_masks(trans, choice, mask, p_hist):
    st = 0
    num_stps = 200
    get_fig(display_mode=True)
    plt.plot(trans[st:st+num_stps], '+-',
             label='trans (0: altR, 1: RepR, 2: AltL, 3: RepL)')
    plt.plot(choice[st:st+num_stps], '+-', label='choice')
    plt.plot(mask[st:st+num_stps]-2, '+-', label='final mask')
    plt.plot(p_hist[st:st+num_stps], '+-', label='outcome')
    for ind in range(num_stps):
        plt.plot([ind, ind], [-3, 3], '--',
                 color=(.7, .7, .7))
    plt.legend()
    import sys
    sys.exit()


def get_repetitions(mat):
    """
    Return mask indicating the repetitions in mat.

    Makes diff of the input vector, mat, to obtain the repetition vector X,
    i.e. X will be 1 at t if the value of mat at t is equal to that at t-1
    Parameters
    ----------
    mat : array
        array of elements.

    Returns
    -------
    repeats : array
        mask indicating the repetitions in mat.

    """
    mat = mat.flatten()
    values = np.unique(mat)
    # We need to account for size reduction of np.diff()
    rand_ch = np.array(np.random.choice(values, size=(1,)))
    repeat_choice = np.concatenate((rand_ch, mat))
    diff = np.diff(repeat_choice)
    repeats = (diff == 0)*1.
    repeats[np.isnan(diff)] = np.nan
    return repeats


def get_transition_mat(choice, conv_w=5):
    """
    Return array indicating the number of repetitions in the last conv_w trials.

    convolves the repetition vector to get a count of the number of repetitions
    in the last conv_w trials

    Parameters
    ----------
    choice : array
        it is expected to be a repetition vector obtained from get_repetitions fn.
    conv_w : str, optional
        window to consider past trials (5)

    Returns
    -------
    np.array
        array is equal to conv_w/2 when there have been conv_w repetitions
        and -conv_w/2 when there have been 0 repetitions.

    """
    # selectivity to transition probability
    limit = -conv_w+1 if conv_w > 1 else len(choice)
    repeat = get_repetitions(choice)
    transition = np.convolve(repeat, np.ones((conv_w,)),
                             mode='full')[0:limit]
    transition_ev = np.concatenate((np.array([0]), transition[:-1]))
    return transition_ev


def get_cond_trials(b_data, cond, cond_list, margin=1000, exp_data={},
                    conv_w=2, exp_nets='exps', max_n_evs=500):
    if exp_nets == 'exps':
        assert 'e_data' in exp_data.keys(), 'Please provide e_data dictionary'
        assert 'cl' in exp_data.keys(), 'Please provide cluster to analyze'
        assert 'ev' in exp_data.keys(), 'Please provide event for aligning'

        prms = {'evs_mrgn': 1e-2, 'fixtn_time': .3, 'std_conv': 20}
        prms.update(exp_data)
        e_data = prms['e_data']
        # get spikes
        spk_tms = e_data['spks'][e_data['clsts'] == prms['cl']][:, None]
        # select trials
        evs, indx_good_evs = pp.preprocess_events(b_data=b_data, e_data=e_data,
                                                  ev=prms['ev'],
                                                  evs_mrgn=prms['evs_mrgn'],
                                                  fixtn_time=prms['fixtn_time'])
        choices = b_data['R_response'].values
        prev_choices = b_data['R_response'].shift(periods=1).values
        outcome = b_data['hithistory'].values
        prev_outcome = b_data['hithistory'].shift(periods=1).values
        rep_resp = b_data['rep_response'].shift(periods=1).values
        outc_2_tr_bck = b_data['hithistory'].shift(periods=2).values
        num_tr = len(b_data)
    elif exp_nets == 'nets':
        fix_tms = np.where(data['stimulus'][:, 0])[0]
        states = data['states']
        states = states[:, int(states.shape[1]/2):]
        choices = data['choice'][fix_tms]-1
        gt = data['gt'][fix_tms]-1
        prev_choices = np.roll(choices, shift=1)
        outcome = data['perf'][fix_tms]
        assert ((choices == gt) == outcome).all()
        prev_outcome = np.roll(outcome, shift=1)
        rep_resp = get_repetitions(choices)
        outc_2_tr_bck = np.roll(outcome, shift=2)
        num_tr = len(choices)
        indx_good_evs = choices != -1
    if cond['ch'] != -1:
        ch_mat = choices
    else:
        ch_mat = np.zeros((num_tr))-1
    if cond['prev_ch'] != -1:
        prev_ch_mat = prev_choices
    else:
        prev_ch_mat = np.zeros((num_tr))-1
    if cond['outc'] != -1:
        outc_mat = outcome
    else:
        outc_mat = np.zeros((num_tr))-1
    if cond['prev_outc'] != -1:
        prev_outc_mat = prev_outcome
    else:
        prev_outc_mat = np.zeros((num_tr))-1
    if cond['prev_tr'] != -1:
        prev_tr_mat = rep_resp
    else:
        prev_tr_mat = np.zeros((num_tr))-1
    if cond['ctxt'] != -1:
        trans = get_transition_mat(prev_choices, conv_w=conv_w)
        trans /= conv_w
        p_hist = np.convolve(outc_2_tr_bck, np.ones((conv_w+1,)),
                             mode='full')[0:-conv_w]
        p_hist /= (conv_w+1)
        trans_mask = np.logical_or(and_(trans != 0, trans != 1), p_hist != 1)
        trans_prev_ch = trans + 2*prev_choices
        trans_prev_ch[trans_mask] = -1
        # plt.figure()
        # plt.plot(p_hist, '-+', label='perf. hist', lw=2)
        # plt.plot(trans, '-+', label='ctxt', lw=2)
        # plt.plot(trans_prev_ch, '-+', label='ctxt and prev ch', lw=2)
        # plt.plot(2*b_data['rep_response'].shift(periods=1).values, '-+',
        #          label='prev ch', lw=2)
        # plt.plot(choices-3, '-+', label='choice', lw=2)
        # plt.plot(outcome-3, '-+', label='outcome', lw=2)
        # for ind in range(200):
        #     plt.plot([ind, ind], [-3, 3], '--', color=(.7, .7, .7))
        # plt.plot(trans_prev_ch, '-+', label='ctxt and prev ch cond', lw=2)
        # plt.legend()
        # asdasd
    else:
        trans_prev_ch = np.zeros((num_tr))-1
    active_idx = [i for i, v in enumerate(cond.values()) if v != -1]
    feats_shape = [v for v in cond.values() if v != -1]
    if exp_nets == 'exps':
        shape = [500]+feats_shape+[2*margin]
    elif exp_nets == 'nets':
        n_unts = states.shape[1]
        shape = [n_unts, max_n_evs]+feats_shape+[np.sum(margin)]
    trialR = np.empty(shape)
    trialR[:] = np.nan
    min_num_tr = 1e6
    max_num_tr = -1e6
    for i_c, case in enumerate(cond_list):
        choice = case[0]
        prev_choice = case[1]
        outcome = case[2]
        prev_outcome = case[3]
        prev_trans = case[4]
        ctxt = case[5]
        mask = and_.reduce((indx_good_evs, ch_mat == choice, outc_mat == outcome,
                            prev_ch_mat == prev_choice, prev_tr_mat == prev_trans,
                            trans_prev_ch == ctxt, prev_outc_mat == prev_outcome,))

        if (False and choice == -1 and prev_choice == -1 and outcome == -1
           and prev_outcome == 1 and prev_trans == -1 and ctxt == 0):
            plot_masks(trans=trans_prev_ch, mask=mask,
                       choice=choices,
                       p_hist=outcome)
        if exp_nets == 'exps':
            filt_evs = evs[mask]
            feats = pp.plt_psths(spk_tms=spk_tms, evs=filt_evs, plot=False,
                                 std_conv=prms['std_conv'], margin_psth=margin)
            peri_ev = feats['peri_ev']
            idx = [np.arange(len(peri_ev))]+[case[i] for i in active_idx]
            n_evs = len(peri_ev)
            if n_evs > 0:
                trialR[idx] = peri_ev

        elif exp_nets == 'nets':
            # XXX: there might be a better way to do this. The difference with the
            # experimental data is that here we storethe activity of all neurons
            # at the same time.This didn't work:
            # [np.arnge(n_unts),np.arnge(len(peri_ev))]+[case[i] for i in act_idx]
            n_evs = np.sum(mask)
            if n_evs == 0:
                print(1)
            if n_evs > max_n_evs:
                sel_tms = fix_tms[mask][np.random.randint(0, n_evs, (max_n_evs))]
                n_evs = max_n_evs
            else:
                sel_tms = fix_tms[mask]
            for i_tr, tr in enumerate(sel_tms):
                idx = [np.arange(n_unts), i_tr]+[case[i] for i in active_idx]
                trialR[idx] = states[tr-margin[0]:tr+margin[1], :].T
        min_num_tr = min(min_num_tr, n_evs)
        max_num_tr = max(max_num_tr, n_evs)
    return trialR, min_num_tr

def get_conds(conditioning={}):
    cond = {'ch': -1, 'prev_ch': -1, 'outc': -1, 'prev_outc': 2, 'prev_tr': -1,
            'ctxt': 4}
    cond.update(conditioning)
    ch = [0, 1] if cond['ch'] != -1 else [-1]
    prev_ch = [0, 1] if cond['prev_ch'] != -1 else [-1]
    outc = [0, 1] if cond['outc'] != -1 else [-1]
    prev_outc = [0, 1] if cond['prev_outc'] != -1 else [-1]
    prev_tr = [0, 1] if cond['prev_tr'] != -1 else [-1]
    ctxt = [0, 1, 2, 3] if cond['ctxt'] != -1 else [-1]
    cases = itertools.product(ch, prev_ch, outc, prev_outc, prev_tr, ctxt)
    return cases

def compute_dPCA_exps(main_folder, sel_sess, sel_rats, inv, lbls_cps, std_conv=20,
                      margin_psth=1000, sel_qlts=['mua', 'good'], conditioning={},
                      sv_folder=''):
    exp_data = {'evs_mrgn': 1e-2, 'fixtn_time': .3,
                'std_conv': std_conv}

    cond_list = get_conds(conditioning=conditioning)
    time = np.arange(2*margin_psth)
    num_comps = 2
    num_cols = len(lbls_cps)+1
    ev_keys = ['fix_strt', 'stim_ttl_strt', 'outc_strt']
    rats = glob.glob(main_folder+'LE*')
    all_trR = []
    min_num_tr = 1e6
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
                        for i_e, ev in enumerate(ev_keys):
                            exp_data['e_data'] = e_data
                            exp_data['cl'] = cl
                            exp_data['ev'] = ev
                            trR, min_n_tr =\
                                get_cond_trials(b_data=b_data, exp_data=exp_data,
                                                margin=margin_psth,
                                                cond_list=dpcp(cond_list),
                                                cond=conditioning)
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
                    idx = [i_c]+i_cond+[time]
                    ax[i_c, i_d].plot(time, Z[dim][idx], label=str(i_cond))
                ax[i_c, i_d].set_title(dim+' C' + str(i_c+1) + ' v. expl.: ' +
                                       str(np.round(var_exp[dim][i_c], 2)))
        ax[i_c, i_d].legend()
        name = ''.join([i[0]+str(i[1]) for i in conditioning.items()])
        f.savefig(sv_folder+name+'.png')


if __name__ == '__main__':
    # cond = {'ch': 2, 'prev_ch': 2, 'outc': 2, 'prev_outc': 2, 'prev_tr': 2,
    #         'ctxt': 4}
    plt.close('all')
    home = 'manuel'
    exp_nets = 'nets'
    if exp_nets == 'nets':
        # Simulations
        main_folder = '/home/manuel/priors_analysis/annaK/sims_21/' +\
            'alg_ACER_seed_0_n_ch_2_BiasCorr/test_2AFC_activity/'
        data = np.load(main_folder+'data.npz', allow_pickle=1)
        cond = {'ch': 2, 'prev_ch': -1, 'outc': -1, 'prev_outc': 2, 'prev_tr': -1,
                'ctxt': 4}
        cond_list = get_conds(conditioning=cond)
        margin = [0, 4]
        trialR, min_num_tr = get_cond_trials(b_data=data, cond=cond,
                                             cond_list=cond_list,
                                             exp_nets='nets',
                                             margin=margin)
        lbls_cps = 'doct'
        num_comps = 3
        num_cols = len(lbls_cps)+1
        time = np.arange(np.sum(margin))
        if len(trialR) > 0:
            conditions = get_conds(conditioning=cond)
            all_trR = np.swapaxes(trialR, 0, 1)
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
            f, ax = plt.subplots(nrows=num_comps, ncols=num_cols,
                                 figsize=(16, 7))
            lbls_to_plot = list(lbls_cps)+[lbls_cps]
            for i_c in range(num_comps):
                for i_d, dim in enumerate(lbls_to_plot):
                    for cond in dpcp(conditions):
                        i_cond = [c for c in cond if c != -1]
                        idx = [i_c]+i_cond+[time]
                        ax[i_c, i_d].plot(time, Z[dim][idx], label=str(i_cond))
                    ax[i_c, i_d].set_title(dim+' C' + str(i_c+1) + ' v. expl.: ' +
                                           str(np.round(var_exp[dim][i_c], 2)))
            ax[i_c, i_d].legend()
            # name = ''.join([i[0]+str(i[1]) for i in conditioning.items()])
            # f.savefig(sv_folder+name+'.png')

    elif exp_nets == 'exps':
        # experiments
        main_folder = '/home/'+home+'/fof_data/'
        if home == 'manuel':
            sv_folder = main_folder+'/psths/'
        elif home == 'molano':
            sv_folder = '/home/molano/Dropbox/project_Barna/FOF_project/psths/'
        std_conv = 50
        margin_psth = 1000
        inv = np.load('/home/'+home+'/fof_data/sess_inv_extended.npz',
                      allow_pickle=1)
        sel_rats = []  # ['LE113']  # 'LE101'
        sel_sess = []  # ['LE104_2021-06-02_13-14-24']
        # ['LE77_2020-12-04_08-27-33']  # ['LE113_2021-06-05_12-38-09']
        cond = {'ch': -1, 'prev_ch': 2, 'outc': -1, 'prev_outc': 2, 'prev_tr': -1,
                'ctxt': -1}
        lbls_cps = 'pct'
        compute_dPCA_exps(inv=inv, main_folder=main_folder, std_conv=std_conv,
                          margin_psth=margin_psth, sel_sess=sel_sess,
                          sel_rats=sel_rats, conditioning=cond, lbls_cps=lbls_cps,
                          sv_folder=sv_folder)
