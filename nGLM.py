#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:55:59 2021

@author: manuel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
# import itertools
import pandas as pd
import glob
import plot_pshts as pp
import utils_fof as ut
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as sstats
from scipy.ndimage.interpolation import shift
from sklearn.linear_model import LogisticRegression
import plotting_functions as pf
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


GLM_VER = {'neural': 'lateral', 'behav': 'full'}
FIGS_VER = 'link_guassian'  # _minimal (<=29/10/21)
for k in GLM_VER.keys():
    FIGS_VER += '_'+k[0]+GLM_VER[k]


def save_fig(f, name):
    f.tight_layout()
    f.savefig(name, dpi=400, bbox_inches='tight')


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


def plot_all_weights(ax, weights_ac, weights_ae, behav_neural='neural'):
    # TRANSITION WEIGHTS
    if GLM_VER[behav_neural] == 'full':
        regrss = ['T++', 'T-+', 'T+-', 'T--']
        ax_tmp = np.array([ax[0:2], ax[4:6]]).flatten()
        plot_kernels(weights_ac=weights_ac, weights_ae=weights_ae,
                     regressors=regrss, ax=ax_tmp, behav_neural=behav_neural)
        for i in range(4):
            ax_tmp[i].set_ylabel('Weight '+regrss[i])
    elif GLM_VER[behav_neural] in ['lateral', 'minimal']:
        # zT
        plot_kernels(weights_ac=weights_ac, weights_ae=weights_ae,
                     regressors=['zT'], ax=ax[0:1],
                     behav_neural=behav_neural)
        ax[0].set_ylabel('Weight zT')

    # LATERAL WEIGHTS
    _, k_ac, _, _, _ = plot_kernels(weights_ac=weights_ac, weights_ae=weights_ae,
                                    regressors=['L+'], ax=ax[2:3],
                                    behav_neural=behav_neural)
    _, _, k_ae, _, _ = plot_kernels(weights_ac=weights_ac, weights_ae=weights_ae,
                                    regressors=['L-'], ax=ax[6:7],
                                    behav_neural=behav_neural)
    regrss = ['L+', 'L-']
    ax[2].set_ylabel('Weight L+')
    ax[6].set_ylabel('Weight L-')
    # EVIDENCE
    plot_kernels(weights_ac=weights_ac, weights_ae=weights_ae,
                 regressors=['evidence'], ax=ax[3:4],
                 behav_neural=behav_neural)
    ax[3].set_ylabel('Weight evidence')
    # TRANSITION-BIAS
    if behav_neural == 'neural':
        plot_kernels(weights_ac=weights_ac, weights_ae=weights_ae,
                     regressors=['trans_bias'], ax=ax[7:8],
                     behav_neural=behav_neural)
        ax[7].set_ylabel('Weight trans-bias')
    for a in [ax[3], ax[7]]:
        a.set_xlabel('')
        a.set_xticks([])


def plot_kernels(weights_ac, weights_ae, std_ac=None, std_ae=None, ax=None,
                 ax_inset=None, inset_xs=0.5, behav_neural='neural',
                 regressors=['T++', 'T-+', 'T+-', 'T--'], **kwargs):
    def get_krnl(name, cols, weights):
        indx = np.array([np.where(np.array([x.startswith(name)
                                            for x in cols]))[0]])
        indx = np.array([x for x in indx if len(x) > 0])
        try:
            xtcks = np.array(cols)[indx][0]
        except IndexError:
            return None, None
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
    ac_cols, ae_cols, _ = get_regressors(behav_neural=behav_neural)
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
        if kernel_ac is not None:
            if std_ac is not None:
                s_ac, _ = get_krnl(name=name, cols=ac_cols, weights=std_ac)
            else:
                s_ac = np.zeros_like(kernel_ac)
            opts = get_opts_krnls(plot_opts=plot_opts, tag='_ac')
            ax[j].errorbar(xs_ac, kernel_ac, s_ac, **opts)

        # after error
        kernel_ae, xs_ae = get_krnl(name=name, cols=ae_cols, weights=weights_ae)
        if kernel_ae is not None:
            if std_ae is not None:
                s_ae, _ = get_krnl(name=name, cols=ae_cols, weights=std_ae)
            else:
                s_ae = np.zeros_like(kernel_ae)
            opts = get_opts_krnls(plot_opts=plot_opts, tag='_ae')
            ax[j].errorbar(xs_ae, kernel_ae, s_ae, **opts)

        # tune fig
        xs_tune = xs_ac or xs_ae
        xtcks_krnls(xs=xs_tune, ax=ax[j])

    return f, kernel_ac, kernel_ae, xs_ac, xs_ae


def plt_p_VS_n(folder, lag, ax=None):
    data = np.load(folder+'/pvalues_'+str(lag)+'_'+FIGS_VER+'.npz')
    idx_mat = data['idx_mat']
    weights = data['weights']
    if ax is None:
        f, ax = plt.subplots(ncols=2)
        sv_fig = True
    else:
        sv_fig = False
    w_lp = weights[idx_mat == 'L+1_ac']
    w_lp = np.abs(w_lp)
    w_ln = weights[idx_mat == 'L-1_ae']
    w_ln = np.abs(w_ln)
    ax[0].plot(w_lp, w_ln, '.')
    ax[0].plot([np.min(w_lp), np.max(w_lp)], [np.min(w_lp), np.max(w_lp)],
               '--k', lw=0.5)
    ax[0].set_xlabel('L+1')
    ax[0].set_ylabel('L-1')

    w_trbp = weights[idx_mat == 'trans_bias_ac']
    w_trbp = np.abs(w_trbp)
    w_trbn = weights[idx_mat == 'trans_bias_ae']
    w_trbn = np.abs(w_trbn)
    ax[1].plot(w_trbp, w_trbn, '.')
    ax[1].plot([np.min(w_trbp), np.max(w_trbp)], [np.min(w_trbp), np.max(w_trbp)],
               '--k', lw=0.5)
    ax[1].set_xlabel('Transition-bias +')
    ax[1].set_ylabel('Transition-bias -')

    if sv_fig:
        save_fig(f=f, name=folder+'/p_VS_n_'+str(lag)+FIGS_VER+'.png')


def plot_weights_distr(folder, lag, plot=True):
    """
    Plot percentage of significant neurons to each regressor.

    Parameters
    ----------
    folder : str
        where the results stored.
    lag : int
        the lag with which the results were obtained.

    Returns
    -------
    None.

    """
    data = np.load(folder+'/pvalues_'+str(lag)+'_'+FIGS_VER+'.npz')
    idx_mat = data['idx_mat']
    pvalues = data['pvalues']
    weights = data['weights']
    unq_regrs = np.unique(idx_mat)
    unq_regrs = filter_regressors(regrs=unq_regrs)
    if plot:
        fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(18, 12))
        ax = ax.flatten()
    mean_ws = []
    std_ws = []
    for i_r, rgrs in enumerate(unq_regrs):
        rgrs_ac = rgrs+'_ac'
        w_ac = np.abs(weights[np.logical_and(idx_mat == rgrs_ac, pvalues < 0.01)])
        rgrs_ae = rgrs+'_ae'
        w_ae = np.abs(weights[np.logical_and(idx_mat == rgrs_ae, pvalues < 0.01)])
        if plot:
            ax[i_r].hist([w_ac, w_ae], 20)
            ax[i_r].set_title(rgrs)
        mean_ws.append([np.mean(w_ac), np.mean(w_ae)])
        std_ws.append([np.std(w_ac), np.std(w_ae)])

    # ax[0].legend()
    if plot:
        save_fig(f=fig, name=folder+'/weight_hists_'+str(lag)+FIGS_VER+'.png')
        plt.close(fig)
    return mean_ws, std_ws


def plot_perc_sign(folder, lag, plot=True):
    """
    Plot percentage of significant neurons to each regressor.

    Parameters
    ----------
    folder : str
        where the results stored.
    lag : int
        the lag with which the results were obtained.

    Returns
    -------
    None.

    """
    data = np.load(folder+'/pvalues_'+str(lag)+'_'+FIGS_VER+'.npz')
    idx_mat = data['idx_mat']
    pvalues = data['pvalues']
    perc_ac = []
    perc_ae = []
    unq_regrs = np.unique(idx_mat)
    unq_regrs = filter_regressors(regrs=unq_regrs)
    for rgrs in unq_regrs:
        rgrs_ac = rgrs+'_ac'
        if (idx_mat == rgrs_ac).any():
            num_smpls = np.sum(idx_mat == rgrs_ac)
            num_sign = np.sum(pvalues[idx_mat == rgrs_ac] < 0.01)
            perc_ac.append(100*num_sign/num_smpls)
        else:
            perc_ac.append(0)
        rgrs_ae = rgrs+'_ae'
        if (idx_mat == rgrs_ae).any():
            num_smpls = np.sum(idx_mat == rgrs_ae)
            num_sign = np.sum(pvalues[idx_mat == rgrs_ae] < 0.01)
            perc_ae.append(100*num_sign/num_smpls)
        else:
            perc_ae.append(0)
    if plot:
        x = np.arange(len(unq_regrs))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.bar(x - width/2, perc_ac, width, label='after correct')
        ax.bar(x + width/2, perc_ae, width, label='after error')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Percentage of significant neurons')
        ax.set_ylim([0, 100])
        # ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(unq_regrs)
        ax.legend()
        save_fig(f=fig, name=folder+'/perc_sign_neurons_'+str(lag)+FIGS_VER+'.png')
        plt.close(fig)
    return perc_ac, perc_ae, unq_regrs


def plot_corrs(mat, reset_mat, labels):
    f, ax = plt.subplots(ncols=mat.shape[2], figsize=(12, 3))
    corr_ac_mat = []
    corr_ae_mat = []
    for i_p in range(mat.shape[2]):
        ax[i_p].plot(mat[:, 0, i_p], reset_mat, '.')
        ax[i_p].plot(mat[:, 1, i_p], reset_mat, '.')
        corr_ac = np.corrcoef(mat[:, 0, i_p], reset_mat)[0, 1]
        corr_ae = np.corrcoef(mat[:, 1, i_p], reset_mat)[0, 1]
        ax[i_p].set_title(' corr. AC: '+str(np.round(corr_ac, 3))+' / AE: ' +
                          str(np.round(corr_ae, 3)))
        ax[i_p].set_xlabel(labels[i_p]+' mean weight')
        ax[i_p].set_ylabel('Reset Index')
        corr_ac_mat.append(corr_ac)
        corr_ae_mat.append(corr_ae)
    return corr_ac_mat, corr_ae_mat, f


def filter_regressors(regrs):
    return np.unique([x[:-3] for x in regrs if ('6-10' not in x) and
                      ('5' not in x) and ('4' not in x) and ('3' not in x) and
                      (not x.startswith('intercept'))])


def get_regressors(behav_neural):
    if GLM_VER[behav_neural] == 'full':  # all regressors (L, Tr, ev, trans-bias)
        cols = ['evidence',
                'L+1', 'L-1', 'L+3', 'L-3', 'L+4', 'L-4',
                'L+5', 'L-5', 'L+6-10', 'L-6-10',
                'T++1', 'T+-1', 'T-+1', 'T--1', 'T++2', 'T+-2', 'T-+2',
                'T--2', 'T++3', 'T+-3', 'T-+3', 'T--3', 'T++4', 'T+-4',
                'T-+4', 'T--4', 'T++5', 'T+-5', 'T-+5', 'T--5',
                'T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10',
                'intercept']
        if behav_neural == 'neural':
            cols.append('trans_bias')  # , 'curr_ch']
        afterc_cols = [x for x in cols if x not in ['L-1', 'T+-1', 'T--1']]
        aftere_cols = [x for x in cols if x not in ['L+1', 'T++1', 'T-+1']]
    elif GLM_VER[behav_neural] == 'lateral':  # L, zT, ev, trans-bias
        cols = ['evidence',
                'L+1', 'L-1', 'L+2', 'L-2', 'L+3', 'L-3', 'L+4', 'L-4',
                'L+5', 'L-5', 'L+6-10', 'L-6-10', 'zT', 'intercept']
        if behav_neural == 'neural':
            cols.append('trans_bias')  # , 'curr_ch']
        afterc_cols = [x for x in cols if x not in ['L-1']]
        aftere_cols = [x for x in cols if x not in ['L+1']]

    elif GLM_VER[behav_neural] == 'minimal':
        cols = ['evidence', 'L+1', 'L-1', 'zT', 'intercept']
        if behav_neural == 'neural':
            cols.append('trans_bias')  # , 'curr_ch']
        afterc_cols = [x for x in cols if x not in ['L-1']]
        aftere_cols = [x for x in cols if x not in ['L+1']]
    return afterc_cols, aftere_cols, cols


def get_vars(data, fix_tms):
    # XXX: when doing fix_tms-1 for each trial, we are actually taking the
    # choice/performance in the previous trial. Therefore, we have to shift
    # backwards to realign.
    ch = shift(data['choice'][fix_tms-1], shift=-1, cval=0).astype(float)
    perf = shift(data['perf'][fix_tms-1], shift=-1, cval=0).astype(float)
    prev_perf = data['perf'][fix_tms-1].astype(float)
    gt = shift(data['gt'][fix_tms-1], shift=-1, cval=0).astype(float)
    ev = np.array(data['info_vals'].item()['coh'])[fix_tms]
    putative_ev = ev*(-1)**(gt == 2)
    return ch, perf, prev_perf, putative_ev


def get_fixation_times(data, lags):
    fix_sgnl = data['stimulus'][:, 0]
    fix_tms = np.where(fix_sgnl == 1)[0][1:]  # drop first trial
    fix_tms = fix_tms[fix_tms > lags[0]]
    fix_tms = fix_tms[fix_tms < data['perf'].shape[0]-lags[1]]
    return fix_tms


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


def behavioral_glm(df):
    """
    Compute GLM weights for data in df conditioned on previous outcome.

    Parameters
    ----------
    df : dataframe
        dataframe containing regressors and response.

    Returns
    -------
    Lreg_ac : LogisticRegression model
        logistic model fit to after correct trials.
    Lreg_ae : LogisticRegression model
        logistic model fit to after error trials.

    """
    ac_cols, ae_cols, _ = get_regressors(behav_neural='behav')
    not_nan_indx = df['R_response'].notna()
    X_df_ac, y_df_ac =\
        df.loc[(df.aftererror == 0) & not_nan_indx,
               ac_cols].fillna(value=0),\
        df.loc[(df.aftererror == 0) & not_nan_indx, 'R_response']
    X_df_ae, y_df_ae =\
        df.loc[(df.aftererror == 1) & not_nan_indx,
               ae_cols].fillna(value=0),\
        df.loc[(df.aftererror == 1) & not_nan_indx, 'R_response']

    if len(np.unique(y_df_ac.values)) == 2 and len(np.unique(y_df_ae.values)) == 2:
        Lreg_ac = LogisticRegression(C=1, fit_intercept=False, penalty='l2',
                                     solver='saga', random_state=123,
                                     max_iter=10000000, n_jobs=-1)
        Lreg_ac.fit(X_df_ac.values, y_df_ac.values)
        Lreg_ae = LogisticRegression(C=1, fit_intercept=False, penalty='l2',
                                     solver='saga', random_state=123,
                                     max_iter=10000000, n_jobs=-1)
        Lreg_ae.fit(X_df_ae.values, y_df_ae.values)
    else:
        Lreg_ac = None
        Lreg_ae = None

    return Lreg_ac, Lreg_ae


def compute_GLM_regressors(data, exp_nets, mask=None, chck_corr=False, tau=2,
                           krnl_len=10, lags=[0, 0], behav_neural='neural'):
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

    afterc_cols, aftere_cols, model_cols =\
        get_regressors(behav_neural=behav_neural)
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
        ch = data['choice']
        perf = data['performance']
        prev_perf = data['prev_perf']
        ev = data['signed_evidence']
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
          'evidence': ev, 'aftererror': 1*(prev_perf == 0),
          'rep_response': rep_ch_}
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
    if GLM_VER[behav_neural] in ['lateral', 'full']:
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
    if GLM_VER[behav_neural] == 'full':
        # T+-
        df['T+-1'] = np.nan  # np.nan
        df.loc[(df.aftererror == 0) & (df.hit == 0), 'T+-1'] =\
            df.loc[(df.aftererror == 0) & (df.hit == 0), 'rep_response_11']
        df.loc[(df.aftererror == 1) | (df.hit == 1), 'T+-1'] = 0
        df['T+-1'] = df['T+-1'].shift(1)
        # T-+
        df['T-+1'] = np.nan  # np.nan
        df.loc[(df.aftererror == 1) & (df.hit == 1), 'T-+1'] =\
            df.loc[(df.aftererror == 1) & (df.hit == 1), 'rep_response_11']
        df.loc[(df.aftererror == 0) | (df.hit == 0), 'T-+1'] = 0
        df['T-+1'] = df['T-+1'].shift(1)
        # T--
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
        # sum trans. from 6 to 10
        cols_tpp = ['T++'+str(x) for x in range(6, 11)]
        cols_tpn = ['T+-'+str(x) for x in range(6, 11)]
        cols_tnp = ['T-+'+str(x) for x in range(6, 11)]
        cols_tnn = ['T--'+str(x) for x in range(6, 11)]
        df['T++6-10'] = np.nansum(df[cols_tpp].values, axis=1)
        df['T+-6-10'] = np.nansum(df[cols_tpn].values, axis=1)
        df['T-+6-10'] = np.nansum(df[cols_tnp].values, axis=1)
        df['T--6-10'] = np.nansum(df[cols_tnn].values, axis=1)
        df.drop(cols_tpp+cols_tpn+cols_tnp+cols_tnn, axis=1, inplace=True)
        df.loc[df.origidx < 6, ['T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10']] =\
            np.nan
    # intercept
    df['intercept'] = 1

    # zT
    if behav_neural == 'neural':
        zt_comps = df['T++1'].shift(1)
        limit = -krnl_len+1
        kernel = np.exp(-np.arange(krnl_len)/tau)
        zt = np.convolve(zt_comps, kernel, mode='full')[0:limit]
        df['trans_bias'] = zt*(df['L+1']+df['L-1'])
        # df['curr_ch'] = df['L+1']+df['L-1']
        # df['curr_ch'] = df['curr_ch'].shift(-1)
        if GLM_VER[behav_neural] in ['lateral', 'minimal']:
            df['zT'] = zt
    elif behav_neural == 'behav':
        # transforming transitions to left/right space
        for col in [x for x in df.columns if x.startswith('T')]:
            df[col] = df[col] * (df.R_response.shift(1)*2-1)
        if GLM_VER[behav_neural] in ['lateral', 'minimal']:
            zt_comps = df['T++1'].shift(1)
            limit = -krnl_len+1
            kernel = np.exp(-np.arange(krnl_len)/tau)
            df['zT'] = np.convolve(zt_comps, kernel, mode='full')[0:limit]

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


def cond_psths(data, exp_nets='nets', pvalue=0.0001, lags=[3, 4]):
    lags_mat = np.arange(np.sum(lags))-lags[0]
    if exp_nets == 'exps':
        print('PSTHs not implemented for experimental data yet')
    elif False:
        states = data['states']
        states = states[:, int(states.shape[1]/2):]
        states = sstats.zscore(states, axis=0)
        states -= np.min(states, axis=0)
        fix_tms = get_fixation_times(data, lags=lags)
        states = np.array([states[fxt-lags[0]:fxt+lags[1], :] for fxt in fix_tms])
        # ch = data['choice'][fix_tms].astype(float)
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
        f, ax = plt.subplots(ncols=3)
        ax[0].plot(lags_mat, sign, '+-', label='choice')
        ax[0].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        prev_ch = shift(ch, shift=1, cval=0)
        sts_1 = states[prev_ch == 1]
        sts_2 = states[prev_ch == 2]
        sign = []
        for tmstp in range(states.shape[1]):
            s_tsp_1 = sts_1[:, tmstp, :]
            s_tsp_2 = sts_2[:, tmstp, :]
            sign_n = [sstats.ranksums(s_tsp_1[:, i], s_tsp_2[:, i]).pvalue < pvalue
                      for i in range(states.shape[2])]
            sign.append(np.sum(sign_n)/states.shape[2])
        ax[0].plot(lags_mat, sign, '+-', label='prev. choice')
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
        ax[1].plot(lags_mat, sign, '+-', label='perf')
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
        ax[1].plot(lags_mat, sign, '+-', label='prev. perf')
        ax[1].axvline(x=0, color=(.7, .7, .7), linestyle='--')
        ax[1].legend()
        sts_1 = states[ev < 5]
        sts_2 = states[ev > 15]
        sign = []
        for tmstp in range(states.shape[1]):
            s_tsp_1 = sts_1[:, tmstp, :]
            s_tsp_2 = sts_2[:, tmstp, :]
            sign_n = [sstats.ranksums(s_tsp_1[:, i], s_tsp_2[:, i]).pvalue < pvalue
                      for i in range(states.shape[2])]
            sign.append(np.sum(sign_n)/states.shape[2])
        ax[2].plot(lags_mat, sign, '+-', label='evidence')
        ax[2].axvline(x=0, color=(.7, .7, .7), linestyle='--')


def GLMs(folder='', exp_nets='nets', lag=0, num_units=1024, plot=True,
         redo=False, **exp_data):
    """
    Compute both behavioral and neuro GLM.

    Parameters
    ----------
    folder : str
        where to find behav/neural data from networks.
    exp_nets : str, optional
        whether to analyze experiments or networks ('nets')
    lag : int, optional
        in networks lag from fixation to collect activity data (0)
    num_units : int, optional
        in networks number of units (1024)
    plot : boolean, optional
        whether to plot (True)
    **exp_data : dict
        experimental data info.
        exp_d = {'ev': 'stim_ttl_strt',  Event to align to
                 'evs_mrgn': 1e-2,  Margin error to align ttl and csv events
                 'plot': False, Whether to plot rasters/psths (?)
                 'fixtn_time': .3, Fixation time (s)
                 'margin_psth': 100, Margin for to plot raster/psth (?)
                 'std_conv': 20  Standard deviation of gaussian used for smoothing
                 }

    Returns
    -------
    idx_mat : list
        list with the regressor corresponding to the p-values in pvalues.
    pvalues : list
        list with the p-values associated to the regressors in idx_max.
    weights_mat: list
        list with the weights associated to the regressors in idx_max.
    """
    n_file_name = folder+'/pvalues_'+str(lag)+'_'+FIGS_VER+'.npz'
    b_file_name = folder+'/behav_'+FIGS_VER+'.npz'
    if not os.path.exists(b_file_name) or not os.path.exists(n_file_name) or redo:
        lags = [lag, lag+1]
        if exp_nets == 'exps':
            raise Exception('Code for experimental data is not up to date')
            assert 'cl' in exp_data.keys(), 'Please provide cluster to analyze'
            exp_d = {'ev': 'stim_ttl_strt', 'evs_mrgn': 1e-2, 'plot': False,
                     'fixtn_time': .3, 'margin_psth': 100, 'std_conv': 20}
            exp_d.update(exp_data)
            # select trials
            e_data = exp_d['e_data']
            evs, indx_good_evs =\
                pp.preprocess_events(b_data=exp_d['b_data'], e_data=e_data,
                                     evs_mrgn=exp_d['evs_mrgn'], ev=exp_d['ev'],
                                     fixtn_time=exp_d['fixtn_time'])
            # get spikes
            # XXX: events could be filter here, but I do all the filtering below
            # filt_evs = evs[indx_good_evs]
            spk_tms = e_data['spks'][e_data['clsts'] == exp_d['cl']][:, None]
            states = ut.scatter(spk_tms=spk_tms, evs=evs, margin_psth=margin_psth,
                                plot=False)
            # XXX: responses measured in the form of spike-counts
            states = np.array([len(r) for r in states['aligned_spks']])
        else:
            indx_good_evs = None
            # get and put together files
            datasets = glob.glob(folder+'data_*')
            data = {'choice': [], 'performance': [], 'prev_perf': [],
                    'states': np.empty((0, num_units)), 'signed_evidence': []}
            print('Experiment '+folder)
            print('Loading '+str(len(datasets))+' datasets')
            for f in datasets:
                data_tmp = np.load(f, allow_pickle=1)
                fix_tms = get_fixation_times(data=data_tmp, lags=lags)
                ch, perf, prev_perf, ev = get_vars(data=data_tmp, fix_tms=fix_tms)
                data['choice'] += ch.tolist()
                data['performance'] += perf.tolist()
                data['prev_perf'] += prev_perf.tolist()
                data['signed_evidence'] += ev.tolist()
                states = data_tmp['states']
                states = states[:, int(states.shape[1]/2):]
                states = sstats.zscore(states, axis=0)
                states = states[fix_tms+lag, :]
                data['states'] = np.concatenate((data['states'],
                                                 states[:, :num_units]), axis=0)

            for k in data.keys():
                data[k] = np.array(data[k])
    if not os.path.exists(b_file_name) or redo:
        # BEHAVIORAL GLM
        df = compute_GLM_regressors(data=data, exp_nets=exp_nets,
                                    mask=indx_good_evs, chck_corr=False,
                                    lags=lags, behav_neural='behav')
        Lreg_ac, Lreg_ae = behavioral_glm(df)
        weights_ac = Lreg_ac.coef_
        weights_ae = Lreg_ae.coef_
        xtcks = ['T++'+x for x in ['2', '3', '4', '5', '6-10']]
        reset, krnl_ac, krnl_ae = pf.compute_reset_index(weights_ac[None, :],
                                                         weights_ae[None, :],
                                                         xtcks=xtcks,
                                                         full_reset_index=False)
        data_bhv = {'weights_ac': weights_ac, 'weights_ae': weights_ae,
                    'reset': reset, 'krnl_ac': krnl_ac, 'krnl_ae': krnl_ae}
        np.savez(b_file_name, **data_bhv)

        f, ax = get_fig(ncols=4, nrows=2, figsize=(12, 6))
        plot_all_weights(ax=ax, weights_ac=weights_ac[0], weights_ae=weights_ae[0],
                         behav_neural='behav')
        ax[0].set_title(str(np.round(reset, 3)))
        save_fig(f=f, name=folder+'/behav_GLM'+FIGS_VER+'.png')
        plt.close(f)
    if not os.path.exists(n_file_name) or redo:
        # NEURO-GLM
        df = compute_GLM_regressors(data=data, exp_nets=exp_nets,
                                    mask=indx_good_evs, chck_corr=False,
                                    lags=lags, behav_neural='neural')
        # build data set
        not_nan_indx = df['R_response'].notna()

        # after correct/error regressors
        ac_cols, ae_cols, _ = get_regressors(behav_neural='neural')
        X_df_ac = df.loc[(df.aftererror == 0) & not_nan_indx,
                         ac_cols].fillna(value=0)
        X_df_ae = df.loc[(df.aftererror == 1) & not_nan_indx,
                         ae_cols].fillna(value=0)
        num_neurons = num_units  # XXX: for exps this should be 1

        if plot:
            f, ax = get_fig(ncols=4, nrows=2, figsize=(12, 6))
            f.suptitle(str(lag))
        idx_mat = []
        weights_mat = []
        pvalues = []
        for i_n in range(num_neurons):
            # print('Neuron ', i_n)
            # AFTER CORRECT
            resps_ac = data['states'][np.logical_and((df.aftererror == 0),
                                                     not_nan_indx).values, i_n]
            exog, endog = sm.add_constant(X_df_ac), resps_ac
            mod = sm.GLM(endog, exog)  # default is gaussian
            # family=sm.families.Poisson(link=sm.families.links.log))
            res = mod.fit()
            weights_ac = res.params
            idx_mat += [x+'_ac' for x in res.pvalues.index]
            pvalues += list(res.pvalues)
            weights_mat += list(weights_ac)
            # AFTER ERROR
            resps_ae = data['states'][np.logical_and((df.aftererror == 1),
                                                     not_nan_indx).values, i_n]
            exog, endog = sm.add_constant(X_df_ae), resps_ae
            mod = sm.GLM(endog, exog)  # default is gaussian
            # family=sm.families.Poisson(link=sm.families.links.log))
            res = mod.fit()
            weights_ae = res.params
            idx_mat += [x+'_ae' for x in res.pvalues.index]
            pvalues += list(res.pvalues)
            weights_mat += list(weights_ae)
            if plot:
                plot_all_weights(ax=ax, weights_ac=weights_ac.values,
                                 weights_ae=weights_ae.values)
            # ax[9].plot(np.abs(kernel_ac[0]), np.abs(kernel_ae[0]), '+')
            # asdasd
        data = {'idx_mat': idx_mat, 'pvalues': pvalues,
                'weights': weights_mat}
        np.savez(n_file_name, **data)


def batch_neuroGLM(main_folder, lag=0, redo=False, n_ch=16, plot=True):
    neural_folder = main_folder+'/neural_analysys/'
    if not os.path.exists(neural_folder):
        os.mkdir(neural_folder)
    # seeds = [[0, 2], np.arange(12, 16)]  # seeds 1, 3 for nch=2 don't do the task
    f_lp_ln, ax_lp_ln = plt.subplots(ncols=2)
    mean_percs = []
    mean_weights = []
    reset_mat = []
    for seed in np.arange(16):
        folder = main_folder+'/alg_ACER_seed_'+str(seed)+'_n_ch_'+str(n_ch) +\
            '/test_2AFC_activity/'
        GLMs(folder=folder, exp_nets='nets', plt=False, lag=lag,
             num_units=1024, redo=redo)
        perc_ac, perc_ae, labels = plot_perc_sign(folder=folder, lag=lag,
                                                  plot=plot)
        mean_percs.append([perc_ac, perc_ae])
        mean_ws, std_ws = plot_weights_distr(folder=folder, lag=lag, plot=plot)
        mean_weights.append(mean_ws)
        plt_p_VS_n(folder=folder, lag=lag, ax=ax_lp_ln)
        beahv_data = np.load(folder+'/behav.npz')
        reset_mat.append(beahv_data['reset'])
    xs = np.arange(len(labels))
    # plot mean percentages
    mean_percs = np.array(mean_percs)
    m_p = np.mean(mean_percs, axis=0)
    std_p = np.std(mean_percs, axis=0)
    f_perc, ax_perc = plt.subplots()
    ax_perc.set_title('Percentages '+str(n_ch))
    ax_perc.errorbar(xs, m_p[0], std_p[0], marker='+',
                     linestyle='none', label='After correct')
    ax_perc.errorbar(xs, m_p[1], std_p[1], marker='+',
                     linestyle='none', label='After error')
    ax_perc.set_xticks(xs)
    ax_perc.set_xticklabels(labels)
    ax_perc.legend()
    ax_perc.set_ylim([0, 100])
    # plot mean percentages VS reset
    corr_perc_ac, corr_perc_ae, f_perc_vs_res =\
        plot_corrs(mat=mean_percs, reset_mat=reset_mat, labels=labels)
    # plot mean weights
    mean_weights = np.array(mean_weights)
    m_w = np.mean(mean_weights, axis=0).T
    std_w = np.std(mean_weights, axis=0).T
    f_ws, ax_ws = plt.subplots()
    ax_ws.set_title('Weights '+str(n_ch))
    ax_ws.errorbar(xs, m_w[0], std_w[0], marker='+',
                   linestyle='none', label='After correct')
    ax_ws.errorbar(xs, m_w[1], std_w[1], marker='+',
                   linestyle='none', label='After error')
    ax_ws.set_xticks(xs)
    ax_ws.set_xticklabels(labels)
    ax_ws.legend()
    # plot mean percentages VS reset
    corr_ws_ac, corr_ws_ae, f_ws_vs_res =\
        plot_corrs(mat=mean_weights, reset_mat=reset_mat, labels=labels)
    # plot mean weights
    save_fig(f=f_lp_ln, name=neural_folder+'/p_vs_n_'+str(n_ch)+'_' +
             str(lag)+FIGS_VER+'.png')
    save_fig(f=f_perc, name=neural_folder+'/perc_sign_'+str(n_ch)+'_' +
             str(lag)+FIGS_VER+'.png')
    save_fig(f=f_ws, name=neural_folder+'/weights_'+str(n_ch)+'_' +
             str(lag)+FIGS_VER+'.png')
    save_fig(f=f_ws_vs_res, name=neural_folder+'/weights_vs_res_' +
             str(n_ch)+'_'+str(lag)+FIGS_VER+'.png')
    save_fig(f=f_perc_vs_res, name=neural_folder+'/perc_vs_res_' +
             str(n_ch)+'_'+str(lag)+FIGS_VER+'.png')
    return m_p, m_w, std_p, std_w, labels, corr_perc_ac, corr_perc_ae


def compute_nGLM_exps(main_folder, sel_sess, sel_rats, inv, std_conv=20,
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
                            GLMs(b_data=b_data, e_data=e_data, ev=ev, cl=cl,
                                 std_conv=std_conv, margin_psth=margin_psth)
                            plt.title(session)
        # name = ''.join([i[0]+str(i[1]) for i in conditioning.items()])
        # f.savefig(sv_folder+name+'.png')


### MAIN


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
        compute_nGLM_exps(inv=inv, main_folder=main_folder, std_conv=std_conv,
                          margin_psth=margin_psth, sel_sess=sel_sess,
                          sel_rats=sel_rats, sv_folder=sv_folder)
    elif exps_nets == 'nets':
        import sys
        if len(sys.argv) == 1:
            lag = 0
        else:
            lag = int(sys.argv[1])
        redo = False
        plot = True
        mean_perc_mat = []
        std_perc_mat = []
        mean_ws_mat = []
        std_ws_mat = []
        corr_ac_mat = []
        corr_ae_mat = []
        lags = [0, -1, 1, 2]
        for lag in lags:
            print('Using lag: ', lag)
            main_folder = '/home/molano/priors/AnnaKarenina_experiments/sims_21/'
            m_p, m_w, std_p, std_w, labels, corr_perc_ac, corr_perc_ae =\
                batch_neuroGLM(main_folder=main_folder, lag=lag, redo=redo,
                               plot=plot)
            mean_perc_mat.append(m_p)
            std_perc_mat.append(std_p)
            mean_ws_mat.append(m_w)
            std_ws_mat.append(std_w)
            corr_ac_mat.append(corr_perc_ac)
            corr_ae_mat.append(corr_perc_ae)
        plt.close('all')
        mean_perc_mat = np.array(mean_perc_mat)
        std_perc_mat = np.array(std_perc_mat)
        mean_ws_mat = np.array(mean_ws_mat)
        std_ws_mat = np.array(std_ws_mat)
        corr_ac_mat = np.array(corr_ac_mat)
        corr_ae_mat = np.array(corr_ae_mat)

        f, ax = plt.subplots(nrows=3)
        clrs = sns.color_palette()
        for i_r in range(mean_perc_mat.shape[2]):
            if np.sum(std_perc_mat[:, 0, i_r]) != 0:
                ax[0].errorbar(lags, mean_perc_mat[:, 0, i_r],
                               std_perc_mat[:, 0, i_r],
                               color=clrs[i_r], label=labels[i_r])
                ax[1].errorbar(lags, mean_ws_mat[:, 0, i_r],
                               std_ws_mat[:, 0, i_r],
                               color=clrs[i_r], label=labels[i_r])
                ax[2].plot(lags, corr_ac_mat[:, i_r], color=clrs[i_r],
                           label=labels[i_r])
            if np.sum(std_perc_mat[:, 1, i_r]) != 0:
                ax[0].errorbar(lags, mean_perc_mat[:, 1, i_r],
                               std_perc_mat[:, 1, i_r],
                               color=clrs[i_r], linestyle='--')
                ax[1].errorbar(lags, mean_ws_mat[:, 1, i_r],
                               std_ws_mat[:, 1, i_r],
                               color=clrs[i_r], linestyle='--')
                ax[2].plot(lags, corr_ae_mat[:, i_r], color=clrs[i_r],
                           label=labels[i_r],  linestyle='--')

        ax[1].set_ylabel('Weights')
        ax[1].set_xlabel('Lag')
        ax[0].set_ylabel('Percentage sign. neurons')
        ax[2].set_ylabel('Correlation with Reset Index')
        for a in ax:
            a.legend()
            a.set_xticks(lags)
