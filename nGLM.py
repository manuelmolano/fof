#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:55:59 2021

@author: manuel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import glob
import plot_pshts as pp
import utils_fof as ut
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as sstats
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

model_cols = ['evidence',
              'L+1', 'L-1', 'L+2', 'L-2', 'L+3', 'L-3', 'L+4', 'L-4',
              'L+5', 'L-5', 'L+6-10', 'L-6-10',
              'T++1', 'T+-1', 'T-+1', 'T--1', 'T++2', 'T+-2', 'T-+2',
              'T--2', 'T++3', 'T+-3', 'T-+3', 'T--3', 'T++4', 'T+-4',
              'T-+4', 'T--4', 'T++5', 'T+-5', 'T-+5', 'T--5',
              'T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10',
              'intercept', 'trans_bias-', 'trans_bias+']
afterc_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                  'T+-1', 'T--1']]
aftere_cols = [x for x in model_cols if x not in ['L+1', 'T++1',
                                                  'T-+1', 'L+2',
                                                  'L-2']]


def get_fig(ncols=2, nrows=2, figsize=(8, 6)):
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharey=True)
    if ncols == 1 and nrows == 1:
        ax = [ax]
    else:
        ax = ax.flatten()
    for a in ax:
        a.invert_xaxis()
        a.axhline(y=0, linestyle='--', c='k', lw=0.5)
    return f, ax


def plot_kernels(weights_ac, weights_ae, std_ac=None, std_ae=None, ac_cols=None,
                 ae_cols=None, ax=None, ax_inset=None, inset_xs=0.5,
                 regressors=['T++', 'T-+', 'T+-', 'T--'], **kwargs):
    def get_krnl(name, cols, weights):
        indx = np.array([np.where(np.array([x.startswith(name)
                                            for x in cols]))[0]])
        indx = np.array([x for x in indx if len(x) > 0])
        xtcks = np.array(cols)[indx][0]
        kernel = np.nanmean(weights[indx], axis=0).flatten()
        try:
            xs = [int(x[len(name):len(name)+1]) for x in xtcks]
        except ValueError:
            xs = [1]
        return kernel, xs

    def xtcks_krnls(xs, ax):
        xtcks = np.arange(1, max(xs)+1)
        ax.set_xticks(xtcks)
        ax.set_xlim([xtcks[0]-0.5, xtcks[-1]+0.5])
        xtcks_lbls = [str(x) for x in xtcks]
        xtcks_lbls[-1] = '6-10'
        ax.set_xticklabels(xtcks_lbls)

    def get_opts_krnls(plot_opts, tag):
        opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
        opts['color'] = plot_opts['color'+tag]
        opts['linestyle'] = plot_opts['lstyle'+tag]
        return opts

    plot_opts = {'lw': 1,  'label': '', 'alpha': 1, 'color_ac': naranja,
                 'fntsz': 7, 'color_ae': (0, 0, 0), 'lstyle_ac': '-',
                 'lstyle_ae': '-', 'marker': '.'}
    plot_opts.update(kwargs)
    fntsz = plot_opts['fntsz']
    del plot_opts['fntsz']
    ac_cols = afterc_cols if ac_cols is None else ac_cols
    ae_cols = aftere_cols if ae_cols is None else ae_cols
    if ax is None:
        n_regr = len(regressors)
        if n_regr > 2:
            ncols = int(np.sqrt(n_regr))
            nrows = int(np.sqrt(n_regr))
            figsize = (8, 5)
        else:
            ncols = n_regr
            nrows = 1
            figsize = (8, 3)
        f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize,
                             sharey=True)
        if n_regr == 1:
            ax = np.array([ax])
        ax = ax.flatten()
        for a in ax:
            a.invert_xaxis()
            a.axhline(y=0, linestyle='--', c='k', lw=0.5)
    else:
        f = None
    for j, name in enumerate(regressors):
        ax[j].set_ylabel('Weight (a.u.)', fontsize=fntsz)
        ax[j].set_xlabel('Trials back from decision', fontsize=fntsz)
        # after correct
        kernel_ac, xs_ac = get_krnl(name=name, cols=ac_cols, weights=weights_ac)
        if std_ac is not None:
            s_ac, _ = get_krnl(name=name, cols=ac_cols, weights=std_ac)
        else:
            s_ac = np.zeros_like(kernel_ac)
        opts = get_opts_krnls(plot_opts=plot_opts, tag='_ac')
        ax[j].errorbar(xs_ac, kernel_ac, s_ac, **opts)

        # after error
        kernel_ae, xs_ae = get_krnl(name=name, cols=ae_cols, weights=weights_ae)
        if std_ae is not None:
            s_ae, _ = get_krnl(name=name, cols=ae_cols, weights=std_ae)
        else:
            s_ae = np.zeros_like(kernel_ae)
        opts = get_opts_krnls(plot_opts=plot_opts, tag='_ae')
        ax[j].errorbar(xs_ae, kernel_ae, s_ae, **opts)

        # tune fig
        xtcks_krnls(xs=xs_ac, ax=ax[j])

    return f, kernel_ac, kernel_ae, xs_ac, xs_ae


def get_vars(data, fix_tms):
    ch = data['choice'][fix_tms-1].astype(float)
    perf = np.roll(data['perf'][fix_tms-1].astype(float), shift=-1)
    prev_perf = data['perf'][fix_tms-1].astype(float)
    ev = np.roll(np.array(data['info_vals'].item()['coh'])[fix_tms], shift=-1)
    return ch, perf, prev_perf, ev


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


def get_transition_mat(repeat, conv_w=5):
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
    limit = -conv_w+1 if conv_w > 1 else len(repeat)
    kernel = np.ones((conv_w,))
    kernel = np.exp(-np.arange(conv_w)/2)
    transition = np.convolve(repeat, np.flip(kernel),
                             mode='full')[0:limit]
    transition_ev = np.concatenate((np.array([0]), transition[:-1]))
    return transition_ev


def get_GLM_regressors(data, exp_nets, mask=None, chck_corr=False, tau=2,
                       krnl_len=10):
    """
    Compute regressors.

    Parameters
    ----------
    data : dict
        dictionary containing behavioral data.
    chck_corr : bool, optional
        whether to check correlations (False)

    Returns
    -------
    df: dataframe
        dataframe containg evidence, lateral and transition regressors.

    """
    if exp_nets == 'exps':
        ev = data['coh'].values
        perf = data['hithistory'].values
        ch = data['R_response'].values
        # discard (make nan) non-standard-2afc task periods
        nan_indx = np.logical_and.reduce((ch != 1., ch != 2., ~mask))
        ev[nan_indx] = np.nan
        perf[nan_indx] = np.nan
        ch[nan_indx] = np.nan
        prev_perf = ~ (np.concatenate((np.array([True]),
                                       data['hithistory'][:-1])) == 1)
        prev_perf = prev_perf.astype('int')
        ev /= np.nanmax(ev)
        rep_ch_ = get_repetitions(ch)
    elif exp_nets == 'nets':
        fix_sgnl = data['stimulus'][:, 0]
        fix_tms = np.where(fix_sgnl == 1)[0][1:]  # drop first trial
        # data['signed_evidence']
        # ev = np.array(info['coh'])[fix_tms]
        ch, perf, prev_perf, ev = get_vars(data=data, fix_tms=fix_tms)
        # ev = np.roll(np.array(info['coh'])[fix_tms], shift=-1)
        # data['performance'].astype(float)
        # perf = np.array(info['performance']).astype(float)
        # prev_perf = ~ (np.concatenate((np.array([True]), perf[:-1])) == 1)
        # ch = data['choice'][fix_tms].astype(float)
        # ch = np.roll(data['choice'][fix_tms-1], shift=-1).astype(float)
        # discard (make nan) non-standard-2afc task periods
        if 'std_2afc' in data.keys():
            std_2afc = data['std_2afc']
        else:
            std_2afc = np.ones_like(ch)
        inv_choice = np.logical_and(ch != 1., ch != 2.)
        nan_indx = np.logical_or.reduce((std_2afc == 0, inv_choice))
        ev[nan_indx] = np.nan
        perf[nan_indx] = np.nan
        ch[nan_indx] = np.nan
        ch = -(ch-2)  # choices should belong to {0, 1}
        prev_perf = prev_perf.astype('int')
        ev /= np.nanmax(ev)
        rep_ch_ = get_repetitions(ch)

    # variables:
    # 'origidx': trial index within session
    # 'rewside': ground truth
    # 'hithistory': performance
    # 'R_response': choice (right == 1, left == 0, invalid == nan)
    # 'subjid': subject
    # 'sessid': session
    # 'res_sound': stimulus (left - right) [frame_i, .., frame_i+n]
    # 'sound_len': stim duration
    # 'frames_listened'
    # 'aftererror': not(performance) shifted
    # 'rep_response'
    df = {'origidx': np.arange(ch.shape[0]), 'R_response': ch, 'hit': perf,
          'evidence': ev, 'aftererror': prev_perf, 'rep_response': rep_ch_}
    df = pd.DataFrame(df)

    # Lateral module
    # L+
    df['L+1'] = np.nan  # np.nan considering invalids as errors
    df.loc[(df.R_response == 1) & (df.hit == 1), 'L+1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 1), 'L+1'] = -1
    df.loc[df.hit == 0, 'L+1'] = 0
    df['L+1'] = df['L+1'].shift(1)
    df.loc[df.origidx == 1, 'L+1'] = np.nan
    # L-
    df['L-1'] = np.nan
    df.loc[(df.R_response == 1) & (df.hit == 0), 'L-1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 0), 'L-1'] = -1
    df.loc[df.hit == 1, 'L-1'] = 0
    df['L-1'] = df['L-1'].shift(1)
    df.loc[df.origidx == 1, 'L-1'] = np.nan
    # shifts
    for i, item in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
        df['L+'+str(item)] = df['L+'+str(item-1)].shift(1)
        df['L-'+str(item)] = df['L-'+str(item-1)].shift(1)
        df.loc[df.origidx == 1, 'L+'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'L-'+str(item)] = np.nan

    # add from 6 to 10, assign them and drop prev cols cols
    cols_lp = ['L+'+str(x) for x in range(6, 11)]
    cols_ln = ['L-'+str(x) for x in range(6, 11)]

    df['L+6-10'] = np.nansum(df[cols_lp].values, axis=1)
    df['L-6-10'] = np.nansum(df[cols_ln].values, axis=1)
    df.drop(cols_lp+cols_ln, axis=1, inplace=True)
    df.loc[df.origidx <= 6, 'L+6-10'] = np.nan
    df.loc[df.origidx <= 6, 'L-6-10'] = np.nan

    # pre transition module
    df.loc[df.origidx == 1, 'rep_response'] = np.nan
    df['rep_response_11'] = df.rep_response
    df.loc[df.rep_response == 0, 'rep_response_11'] = -1
    df.rep_response_11.fillna(value=0, inplace=True)
    df.loc[df.origidx == 1, 'aftererror'] = np.nan

    # transition module
    df['T++1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 1), 'T++1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 0), 'T++1'] = 0
    df['T++1'] = df['T++1'].shift(1)
    zt_comps = df['T++1'].shift(1)
    df['T+-1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 0), 'T+-1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 1), 'T+-1'] = 0
    df['T+-1'] = df['T+-1'].shift(1)

    df['T-+1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 1), 'T-+1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 0), 'T-+1'] = 0
    df['T-+1'] = df['T-+1'].shift(1)

    df['T--1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 0), 'T--1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 1), 'T--1'] = 0
    df['T--1'] = df['T--1'].shift(1)

    # shifts now
    for i, item in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
        df['T++'+str(item)] = df['T++'+str(item-1)].shift(1)
        df['T+-'+str(item)] = df['T+-'+str(item-1)].shift(1)
        df['T-+'+str(item)] = df['T-+'+str(item-1)].shift(1)
        df['T--'+str(item)] = df['T--'+str(item-1)].shift(1)
        df.loc[df.origidx == 1, 'T++'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'T+-'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'T-+'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'T--'+str(item)] = np.nan

    cols_tpp = ['T++'+str(x) for x in range(6, 11)]
    # cols_tpp = [x for x in df.columns if x.startswith('T++')]
    cols_tpn = ['T+-'+str(x) for x in range(6, 11)]
    # cols_tpn = [x for x in df.columns if x.startswith('T+-')]
    cols_tnp = ['T-+'+str(x) for x in range(6, 11)]
    # cols_tnp = [x for x in df.columns if x.startswith('T-+')]
    cols_tnn = ['T--'+str(x) for x in range(6, 11)]
    # cols_tnn = [x for x in df.columns if x.startswith('T--')]

    df['T++6-10'] = np.nansum(df[cols_tpp].values, axis=1)
    df['T+-6-10'] = np.nansum(df[cols_tpn].values, axis=1)
    df['T-+6-10'] = np.nansum(df[cols_tnp].values, axis=1)
    df['T--6-10'] = np.nansum(df[cols_tnn].values, axis=1)
    df.drop(cols_tpp+cols_tpn+cols_tnp+cols_tnn, axis=1, inplace=True)
    df.loc[df.origidx < 6, ['T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10']] =\
        np.nan
    # transforming transitions to left/right space
    for col in [x for x in df.columns if x.startswith('T')]:
        df[col] = df[col] * (df.R_response.shift(1)*2-1)
        # {0 = Left; 1 = Right, nan=invalid}

    df['intercept'] = 1
    # zT
    limit = -krnl_len+1
    kernel = np.exp(-np.arange(krnl_len)/tau)
    zt = np.convolve(zt_comps, kernel, mode='full')[0:limit]
    df['trans_bias+'] = zt*df['L+1']
    df['trans_bias-'] = zt*df['L-1']
    # tr_bias_aux = np.convolve(zt_comps.shift(1), kernel,
    #                           mode='full')[0:limit]*(df['L+1']+df['L-1'])
    # plt.figure()
    # plt.plot(2*(ch-0.5), '-+', label='choice')
    # plt.plot(df['L+1'], '-+', label='L+')
    # plt.plot(df['L-1'], '-+', label='L-')
    # plt.plot(perf, '-+', label='perf')
    # plt.plot(2*(rep_ch_-0.5), '-+', label='reps')
    # plt.plot(zt_comps, '-+', label='t++')
    # plt.plot(df['T++1'], '-+', label='T++')  # T++1 is in the Left/rigth space
    # plt.plot(df['trans_bias'], '-+', label='bias')
    # plt.plot(tr_bias_aux, '-+', label='bias II')
    # plt.legend()
    # asdasd
    df.loc[:, model_cols].fillna(value=0, inplace=True)
    # check correlation between regressors
    if chck_corr:
        for j, (t, cols) in enumerate(zip(['after correct', 'after error'],
                                          [afterc_cols, aftere_cols])):
            fig, ax = plt.subplots(figsize=(16, 16))
            sns.heatmap(df.loc[df.aftererror == j,
                               cols].fillna(value=0).corr(),
                        vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
            ax.set_title(t)
    return df  # resulting df with lateralized T+


def get_cond_trials(b_data, e_data, exp_nets='nets', pvalue=0.0001, **exp_data):
    if exp_nets == 'exps':
        assert 'cl' in exp_data.keys(), 'Please provide cluster to analyze'
        exp_d = {'ev': 'stim_ttl_strt', 'evs_mrgn': 1e-2, 'plot': False,
                 'fixtn_time': .3, 'margin_psth': 100, 'std_conv': 20}
        exp_d.update(exp_data)
        # select trials
        evs, indx_good_evs = pp.preprocess_events(b_data=b_data, e_data=e_data,
                                                  evs_mrgn=exp_d['evs_mrgn'],
                                                  fixtn_time=exp_d['fixtn_time'],
                                                  ev=exp_d['ev'])
    else:
        indx_good_evs = None

    # get behavior
    df = get_GLM_regressors(b_data, exp_nets=exp_nets, mask=indx_good_evs,
                            chck_corr=False)

    if exp_nets == 'exps':
        # get spikes
        # XXX: events could be filter here, but I do all the filtering below
        # filt_evs = evs[indx_good_evs]
        spk_tms = e_data['spks'][e_data['clsts'] == exp_d['cl']][:, None]
        resps = ut.scatter(spk_tms=spk_tms, evs=evs, margin_psth=margin_psth,
                           plot=False)
        # XXX: responses measured in the form of spike-counts
        resps = np.array([len(r) for r in resps['aligned_spks']])
    else:
        fix_sgnl = data['stimulus'][:, 0]
        fix_tms = np.where(fix_sgnl == 1)[0][1:]
        resps = data['states']
        resps = resps[fix_tms, int(resps.shape[1]/2):]
        resps = sstats.zscore(resps, axis=0)
        # f, ax = plt.subplots(nrows=2)
        # ax[0].hist(resps.flatten(), 200)
        # ax[0].imshow(resps[:100, :].T, aspect='auto', vmin=-5, vmax=5)
        # asasd
        resps -= np.min(resps, axis=0)

    if exp_nets == 'exps':
        print('PSTHs not implemented for experimental data yet')
    else:
        states = data['states']
        states = states[:, int(states.shape[1]/2):]
        states = sstats.zscore(states, axis=0)
        states -= np.min(states, axis=0)
        lims = [6, 7]
        xs = np.arange(np.sum(lims))-lims[0]
        # fix_tms += 1
        fix_tms = fix_tms[fix_tms > lims[0]]
        fix_tms = fix_tms[fix_tms < states.shape[0]-lims[1]]
        states = np.array([states[fxt-lims[0]:fxt+lims[1], :] for fxt in fix_tms])
        # ch = data['choice'][fix_tms].astype(float)
        ch = np.roll(data['choice'][fix_tms-1], shift=-1)
        sts_1 = states[ch == 1]
        sts_2 = states[ch == 2]
        sign = []
        for tmstp in range(states.shape[1]):
            s_tsp_1 = sts_1[:, tmstp, :]
            s_tsp_2 = sts_2[:, tmstp, :]
            sign_n = [sstats.ranksums(s_tsp_1[:, i], s_tsp_2[:, i]).pvalue < pvalue
                      for i in range(states.shape[2])]
            sign.append(np.sum(sign_n)/states.shape[2])
        f, ax = plt.subplots(ncols=3)
        ax[0].plot(xs, sign, '+-', label='choice')
        ax[0].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        ch, perf, prev_perf, ev = get_vars(data=data, fix_tms=fix_tms)
        sts_1 = states[ch == 1]
        sts_2 = states[ch == 2]
        sign = []
        for tmstp in range(states.shape[1]):
            s_tsp_1 = sts_1[:, tmstp, :]
            s_tsp_2 = sts_2[:, tmstp, :]
            sign_n = [sstats.ranksums(s_tsp_1[:, i], s_tsp_2[:, i]).pvalue < pvalue
                      for i in range(states.shape[2])]
            sign.append(np.sum(sign_n)/states.shape[2])
        ax[0].plot(xs, sign, '+-', label='prev. choice')
        ax[0].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        ax[0].legend()
        sts_1 = states[perf == 0]
        sts_2 = states[perf == 1]
        sign = []
        for tmstp in range(states.shape[1]):
            s_tsp_1 = sts_1[:, tmstp, :]
            s_tsp_2 = sts_2[:, tmstp, :]
            sign_n = [sstats.ranksums(s_tsp_1[:, i], s_tsp_2[:, i]).pvalue < pvalue
                      for i in range(states.shape[2])]
            sign.append(np.sum(sign_n)/states.shape[2])
        ax[1].plot(xs, sign, '+-', label='perf')
        ax[1].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        sts_1 = states[prev_perf == 0]
        sts_2 = states[prev_perf == 1]
        sign = []
        for tmstp in range(states.shape[1]):
            s_tsp_1 = sts_1[:, tmstp, :]
            s_tsp_2 = sts_2[:, tmstp, :]
            sign_n = [sstats.ranksums(s_tsp_1[:, i], s_tsp_2[:, i]).pvalue < pvalue
                      for i in range(states.shape[2])]
            sign.append(np.sum(sign_n)/states.shape[2])
        ax[1].plot(xs, sign, '+-', label='prev. perf')
        ax[1].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        ax[1].legend()
        sts_1 = states[np.abs(ev) < 0.1]
        sts_2 = states[np.abs(ev-31.) < 1]
        sign = []
        for tmstp in range(states.shape[1]):
            s_tsp_1 = sts_1[:, tmstp, :]
            s_tsp_2 = sts_2[:, tmstp, :]
            sign_n = [sstats.ranksums(s_tsp_1[:, i], s_tsp_2[:, i]).pvalue < pvalue
                      for i in range(states.shape[2])]
            sign.append(np.sum(sign_n)/states.shape[2])
        ax[2].plot(xs, sign, '+-', label='evidence')
        ax[2].axvline(x=0, color=(.7, .7, .7), linestyle='--')

        # asd
        # clrs = [azul, rojo]
        # for i_c, c in enumerate([1, 2]):
        #     sts = states[ch == c]
        #     ax[0].plot(xs, np.std(np.mean(sts, axis=0), axis=1), color=clrs[i_c])
        # ax[0].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        # ev = np.array(info['coh'])[fix_tms]
        # clrs = grad_colors
        # for i_e, e in enumerate(np.unique(ev)):
        #     sts = states[ev == e]
        #     ax[1].plot(xs, np.std(np.mean(sts, axis=0), axis=1), color=clrs[i_e])

        # ax[1].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        # perf = data['perf'][fix_tms].astype(float)
        # clrs = ['k', naranja]
        # for i_p, p in enumerate([0, 1]):
        #     sts = states[perf == p]
        #     ax[2].plot(xs, np.std(np.mean(sts, axis=0), axis=1), color=clrs[i_p])
        # # ax[2].plot(xs, np.mean(np.std(states, axis=0), axis=1))
        # ax[2].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        # # ax[0].plot(np.mean(sts_2, axis=(0)), color=azul)
        # sfasasd

    # build data set
    not_nan_indx = df['R_response'].notna()

    # after correct/error regressors
    X_df_ac = df.loc[(df.aftererror == 0) & not_nan_indx,
                     afterc_cols].fillna(value=0)
    X_df_ae = df.loc[(df.aftererror == 1) & not_nan_indx,
                     aftere_cols].fillna(value=0)
    if exp_nets == 'exps':
        resps = resps[:, None]  # check if this works
        num_neurons = 1
    elif exp_nets == 'nets':
        num_neurons = 100

    # f_tr, ax_tr = get_fig(ncols=2, nrows=2, figsize=(8, 6))
    # f_l, ax_l = get_fig(ncols=2, nrows=1, figsize=(6, 6))
    # f_ev, ax_ev = get_fig(ncols=1, nrows=1, figsize=(8, 6))
    # f_tr_b, ax_tr_b = get_fig(ncols=1, nrows=1, figsize=(8, 6))
    f, ax = get_fig(ncols=5, nrows=2, figsize=(12, 6))
    ax[9].invert_xaxis()
    for i_n in range(num_neurons):
        # after correct
        resps_ac = resps[np.logical_and((df.aftererror == 0),
                                        not_nan_indx).values, i_n]
        exog, endog = sm.add_constant(X_df_ac), resps_ac
        mod = sm.GLM(endog, exog,
                     family=sm.families.Poisson(link=sm.families.links.log))
        res = mod.fit()
        weights_ac = res.params
        # print('After correct weights')
        # print(weights_ac)

        # after error
        resps_ae = resps[np.logical_and((df.aftererror == 1),
                                        not_nan_indx).values, i_n]
        exog, endog = sm.add_constant(X_df_ae), resps_ae
        mod = sm.GLM(endog, exog,
                     family=sm.families.Poisson(link=sm.families.links.log))
        res = mod.fit()
        weights_ae = res.params
        # print('After error weights')
        # print(weights_ae)
        # Poisson regression code
        regrss = ['T++', 'T-+', 'T+-', 'T--']
        plot_kernels(weights_ac=weights_ac.values, weights_ae=weights_ae.values,
                     regressors=regrss, ax=ax[:4])
        for i in range(4):
            ax[i].set_ylabel('Weight '+regrss[i])
        _, kernel_ac, _, _, _ = plot_kernels(weights_ac=weights_ac.values,
                                             weights_ae=weights_ae.values,
                                             regressors=['L+'], ax=ax[4:5])
        _, _, kernel_ae, _, _ = plot_kernels(weights_ac=weights_ac.values,
                                             weights_ae=weights_ae.values,
                                             regressors=['L-'], ax=ax[5:6])
        regrss = ['L+', 'L-']
        for i in range(2):
            ax[i+4].set_ylabel('Weight '+regrss[i])
        plot_kernels(weights_ac=weights_ac.values, weights_ae=weights_ae.values,
                     regressors=['evidence'], ax=ax[6:7])
        ax[6].set_ylabel('Weight evidence')
        ax[6].set_xlabel('')
        ax[6].set_xticks([])
        plot_kernels(weights_ac=weights_ac.values, weights_ae=weights_ae.values,
                     regressors=['trans_bias+', 'trans_bias-'], ax=ax[7:9])
        regrss = ['trans-bias+', 'trans-bias-']
        for i in range(2):
            ax[i+7].set_ylabel('Weight '+regrss[i])
            ax[i+7].set_xlabel('')
            ax[i+7].set_xticks([])
        ax[9].plot(np.abs(kernel_ac[0]), np.abs(kernel_ae[0]), '+')
        # asdasd
    print(1)


def compute_nGLM(main_folder, sel_sess, sel_rats, inv, std_conv=20,
                 margin_psth=1000, sel_qlts=['mua', 'good'], conditioning={},
                 sv_folder=''):
    ev_keys = ['fix_strt', 'stim_ttl_strt', 'outc_strt']
    rats = glob.glob(main_folder+'LE*')
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
                            get_cond_trials(b_data=b_data, e_data=e_data,
                                            ev=ev, cl=cl, std_conv=std_conv,
                                            margin_psth=margin_psth)
                            plt.title(session)
        # name = ''.join([i[0]+str(i[1]) for i in conditioning.items()])
        # f.savefig(sv_folder+name+'.png')


if __name__ == '__main__':
    plt.close('all')
    exps_nets = 'nets'
    analysis_type = 'dpca'
    std_conv = 50
    margin_psth = 1000
    if exps_nets == 'exps':
        home = 'manuel'
        main_folder = '/home/'+home+'/fof_data/'
        if home == 'manuel':
            sv_folder = main_folder+'/psths/'
        elif home == 'molano':
            sv_folder = '/home/molano/Dropbox/project_Barna/FOF_project/psths/'
        inv = np.load('/home/'+home+'/fof_data/sess_inv_extended.npz',
                      allow_pickle=1)
        sel_rats = []  # ['LE113']  # 'LE101'
        sel_sess = []  # ['LE104_2021-06-02_13-14-24']
        # ['LE104_2021-05-17_12-02-40']
        # ['LE77_2020-12-04_08-27-33']  # ['LE113_2021-06-05_12-38-09']
        compute_nGLM(inv=inv, main_folder=main_folder, std_conv=std_conv,
                     margin_psth=margin_psth, sel_sess=sel_sess,
                     sel_rats=sel_rats, sv_folder=sv_folder)
    if exps_nets == 'nets':
        main_folder = '/home/molano/Dropbox/project_Barna/FOF_project/' +\
            'networks/pretrained_RNNs_N2_fina_models/test_2AFC_activity/'
        data = np.load(main_folder+'data.npz', allow_pickle=1)
        e_data = data['states']
        b_data = data
        # del b_data['states']
        get_cond_trials(b_data=b_data, e_data=e_data)
