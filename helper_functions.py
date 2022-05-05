#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*
 * @Author: jorgedelpozolerida
 * @Date: 2020-04-07 18:39:43
 * @Last Modified by: jorgedelpozolerida
 * @Last Modified time: 2020-06-01 11:54:04
 */

"""
import numpy as np
from numpy import logical_and as and_
from copy import deepcopy as deepc
import seaborn as sns
from numpy import concatenate as conc
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.stats import entropy, mstats
import scipy.io as sio
import plotting_functions as pf
import GLM_nalt as nGLM
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import re
import glob
import sys
sys.path.append(os.path.expanduser("~/neurogym"))
import neurogym.utils.plotting as pl
rojo = np.array((228, 26, 28))/255
azul = np.array((55, 126, 184))/255
verde = np.array((77, 175, 74))/255
morado = np.array((152, 78, 163))/255
naranja = np.array((255, 127, 0))/255
marron = np.array((166, 86, 40))/255
amarillo = np.array((155, 155, 51))/255
rosa = np.array((252, 187, 161))/255
cyan = np.array((0, 1, 1))
gris = np.array((.5, .5, 0.5))
azul_2 = np.array([56, 108, 176])/255
rojo_2 = np.array([165, 15, 21])/255

COLORES = conc((azul.reshape((1, 3)), rojo.reshape((1, 3)),
                verde.reshape((1, 3)), morado.reshape((1, 3)),
                naranja.reshape((1, 3)), marron.reshape((1, 3)),
                amarillo.reshape((1, 3)), rosa.reshape((1, 3))),
               axis=0)
COLORES = np.concatenate((COLORES, 0.5*COLORES))
model_cols = ['evidence',
              'L+1', 'L-1', 'L+2', 'L-2', 'L+3', 'L-3', 'L+4', 'L-4',
              'L+5', 'L-5', 'L+6-10', 'L-6-10',
              'T++1', 'T+-1', 'T-+1', 'T--1', 'T++2', 'T+-2', 'T-+2',
              'T--2', 'T++3', 'T+-3', 'T-+3', 'T--3', 'T++4', 'T+-4',
              'T-+4', 'T--4', 'T++5', 'T+-5', 'T-+5', 'T--5',
              'T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10', 'intercept']
afterc_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                  'T+-1', 'T--1']]
aftere_cols = [x for x in model_cols if x not in ['L+1', 'T++1',
                                                  'T-+1', 'L+2',
                                                  'L-2']]

aftercc_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                   'T+-1', 'T--1', 'T-+1',
                                                   'T+-2', 'T--2']]

afteree_cols = [x for x in model_cols if x not in ['L+1', 'L+2', 'L-2',
                                                   'T+-1', 'T++1', 'T-+1',
                                                   'T++2', 'T-+2']]

afterec_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                   'T+-1', 'T++1', 'T--1',
                                                   'T++2', 'T-+2']]

afterce_cols = [x for x in model_cols if x not in ['L+1', 'L+2', 'L-2',
                                                   'T++1', 'T--1', 'T-+1',
                                                   'T+-2', 'T--2']]

### XXX: SECONDARY FUNCTIONS


def get_tag(tag, file):
    """
    Return value associated to a given tag in a file.

    Parameters
    ----------
    tag : str
        tag to look for in the basename of file.
    file : str
        file with tag and value separated by _.

    Returns
    -------
    val : str
        value associated to tag.

    """
    # process name
    file_name = os.path.basename(file)
    start_val = file_name.find(tag)
    assert start_val != -1, 'Tag ' + tag + ' not found in ' + file_name
    val = file_name[start_val + len(tag) + 1:]
    val = val[:val.find('_')] if '_' in val else val
    return val


def evidence_mask(ev, percentage=10):
    """
    Select trials where stimulus is lower than percetile 'percentage'.

    Parameters
    ----------
    ev : array
        matrix with the cummulative evidence for each trial.
    percentage : int, optional
        percentage of the evidence over which the mask will be 0 (10)

    Returns
    -------
    np.array
        vector of booleans indicating all trials for which ev < percentage.

    """
    ev_abs = np.abs(ev)
    return ev_abs <= np.percentile(ev_abs, percentage)


def get_times(num, per, step):
    """
    Create list of steps or times.

    Parameters
    ----------
    num : int
        number of samples.
    per : int
        period.
    step : int
        spacing between each step.

    Returns
    -------
    times : array
        array of indexes starting at 0 and ending at num-per with as spacing of
        step.

    """
    if per >= num:
        times = np.array([0])
    else:
        times = np.linspace(0, num - per, (num - per)//step + 1, dtype=int)
    return times


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
    repeat_choice = conc((rand_ch, mat))
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
    transition_ev -= conv_w/2
    return transition_ev


def probit(x, beta, alpha):
    """
    Return probit function with parameters alpha and beta.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha.

    """
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


def probit_lapse_rates(x, beta, alpha, piL, piR):
    """
    Return probit with lapse rates.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.
    piL : float
        lapse rate for left side.
    piR : TYPE
        lapse rate for right side.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha and lapse rates.

    """
    piL = 0
    piR = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


def remove_borders(mask):
    """
    Remove the change steps (borders) of a mask.

    Refines mask by removing blocks' borders, which are detected by
    a change of 1 or -1 (from True to False or viceversa).

    Parameters
    ----------
    mask : array
        array with 0s and 1s indicating a certain period (that should be of
                                                          several steps).

    Returns
    -------
    mask : array
        same array the 1s in the border of the period made 0.

    """
    mask = 1*mask
    if np.sum(mask) < len(mask):
        inside_blk_indx_on = np.diff(mask) != 1
        inside_blk_indx_on = np.append(False, inside_blk_indx_on)
        inside_blk_indx_off = np.diff(mask) != -1
        inside_blk_indx_off = np.append(inside_blk_indx_off, False)
        mask = and_.reduce((inside_blk_indx_on, inside_blk_indx_off, mask))
    return mask


def template_match(mat, templ, plot=False):
    """
    Find the time points in an array where a given template occurs.

    Parameters
    ----------
    mat : array
        array in which to find the template (the array is expected to have at least
                                             one occurrence of the template).
    templ : array
        template.

    Returns
    -------
    mask : array
        array with 1s at the time points right after the template has occurred.

    """
    mat = mat - np.mean(mat)
    temp_match = np.convolve(mat, np.flip(templ), mode='same')
    times = (np.where(temp_match == np.max(temp_match))[0] +
             np.ceil(len(templ)/2)-1).astype('int')
    mask = np.zeros_like(mat)
    times = times[times < mask.shape[0]]
    mask[times] = 1
    if plot:
        plt.figure()
        plt.plot(mat[max(0, times[0]-100):times[0]+10000], '-+', markersize=6)
        plt.plot(mask[max(0, times[0]-100):times[0]+10000])
    return mask


def get_average(mat):
    """
    Return average of arrays contained in mat.

    Averages across instances data contained in mat. If instances have different
    lenghts they are equalized by padding with nans.

    Parameters
    ----------
    mat : array/list
        array or list containing containing the arrays to average.

    Returns
    -------
    average_matrix : array
        average of arrays.

    """
    # Remove empty instances from data
    a_mat = [x for x in mat if len(x) > 0]
    max_ = np.max([len(x) for x in a_mat])
    a_mat_ =\
        [conc((x, np.nan*np.ones((((int(max_-len(x)),)+np.array(x).shape[1:])))))
         for x in a_mat]  # add nan to have same shape mats
    a_mat_ = np.array(a_mat_)
    average_matrix = np.nanmean(a_mat_, axis=0)
    return average_matrix


def get_std(mat):
    """
    Return std of arrays contained in mat.

    Averages across instances data contained in mat. If instances have different
    lenghts they are equalized by padding with nans.

    Parameters
    ----------
    mat : array/list
        array or list containing containing the arrays to average.

    Returns
    -------
    average_matrix : array
        std of arrays.

    """
    # Remove empty instances from data
    a_mat = [x for x in mat if len(x) > 0]
    max_ = np.max([len(x) for x in a_mat])
    a_mat_ =\
        [conc((x, np.nan*np.ones((((int(max_-len(x)),)+np.array(x).shape[1:])))))
         for x in a_mat]  # add nan to have same shape mats
    a_mat_ = np.array(a_mat_)
    average_matrix = np.nanstd(a_mat_, axis=0)
    return average_matrix


def get_median(mat):
    """
    Return median of arrays contained in mat.

    Averages across instances data contained in mat. If instances have different
    lenghts they are equalized by padding with nans.

    Parameters
    ----------
    mat : array/list
        array or list containing containing the arrays to average.

    Returns
    -------
    average_matrix : array
        median of arrays.

    """
    # Remove empty instances from data
    a_mat = [x for x in mat if len(x) > 0]
    max_ = np.max([len(x) for x in a_mat])
    a_mat_ =\
        [conc((x, np.nan*np.ones((((int(max_-len(x)),)+np.array(x).shape[1:])))))
         for x in a_mat]  # add nan to have same shape mats
    a_mat_ = np.array(a_mat_)
    average_matrix = np.nanmedian(a_mat_, axis=0)
    return average_matrix


### XXX: PRIMARY FUNCTIONS

def compute_transition_probs_mat(data_mat, choices, block_n_ch,
                                 block_tr_hist, num_blocks=3,
                                 extra_condition=None):
    """
    Compute transition probs mat.

    Parameters
    ----------
    data_mat : array
        array of choices from which the transition probs will be inferred.
    choices : array
        array of choices to consider.
    block_n_ch : array
        array with the number of choices 'active' for each trial.
    block_tr_hist : array
        array indicating the block at each trial.
    num_blocks : int, optional
        total number of blocks (3)
    extra_condition : array, optional
        mask that allows further filtering the trials (None)

    Returns
    -------
    trans_mat : array
        array with the transition probs between choices (rows: current choice,
                                                         columns: next choice)
    counts_mat : array
        aray with the number of each type of transition (same as trans_mat but
                                                         without normalizing).

    """
    if extra_condition is None:
        extra_condition = np.full(data_mat.shape, True, dtype=None)

    # get transition blocks
    blck_tr_hist_id = np.unique(block_tr_hist)
    blck_tr_hist_id = blck_tr_hist_id[:num_blocks]
    n_blcks_trh = blck_tr_hist_id.shape[0]
    # get number of choices blocks
    blck_n_ch_id = np.unique(block_n_ch)
    n_blcks_nch = blck_n_ch_id.shape[0]

    # get choices
    ch_bins = np.append(choices-0.5, choices[-1]+0.5)
    trans_mat = np.empty((n_blcks_trh, n_blcks_nch, choices.shape[0],
                          choices.shape[0]))
    trans_mat[:] = np.nan
    counts_mat = np.empty((n_blcks_trh, n_blcks_nch, choices.shape[0],
                           choices.shape[0]))
    counts_mat[:] = np.nan
    for ind_nch, bl_nch in enumerate(blck_n_ch_id):
        for ind_trh, bl_trh in enumerate(blck_tr_hist_id):
            for ind_ch, ch in enumerate(choices):
                # avoid blocks borders
                blk_nch_mask = block_n_ch == bl_nch
                blk_nch_mask = remove_borders(blk_nch_mask)
                blk_trh_mask = block_tr_hist == bl_trh
                blk_trh_mask = remove_borders(blk_trh_mask)
                condition = and_.reduce((data_mat == ch, blk_nch_mask,
                                         blk_trh_mask, extra_condition))
                indx = np.where(condition)[0]
                indx = indx[indx < len(data_mat)-1]
                next_ch = data_mat[indx + 1]
                counts = np.histogram(next_ch, ch_bins)[0]
                trans_mat[ind_trh, ind_nch, ind_ch, :] = counts/np.sum(counts)
                counts_mat[ind_trh, ind_nch, ind_ch, :] = counts.astype(int)
    return trans_mat, counts_mat


def process_data(folder, acr_tr_per=300000, step=30000, name='', spacing=10000,
                 cond_blcks=True, binning=300000, **analysis_opts):
    """
    Perform analysis of experiment in folder.

    Parameters
    ----------
    folder : str
        folder containing the behavioral data.
    step : int, optional
        step to use for the period-wise analysis of the data (30000)
    acr_tr_per : int, optional
        period to use for the period-wise analysis of the data (300000)
    name : str, optional
        name use to save figs ('')
    binning : int, optional
        binning to use when computing perf_cond
    **analysis_opts : dict
        dictionary indicating which analysis to perform. Example:
            {'glm': True, 'trans_mats': True, 'bias': True,
             'n-GLM': False, 'rebound': False, 'plot_figs': True,
             'step': 250000, 'per': 500000, 'reload': False}

    Returns
    -------
    all_data: dict
        dictionary with all analyses performed on behavioral data in folder.

    """
    opts = {'glm': False, 'bias': True, 'rebound': False, 'performances': True,
            'plot_figs': True, 'num_trials': True, 'reload': False,
            'save_matlab_file': False, 'glm_rbnd': False, 'trans_mats': False}
    opts.update(analysis_opts)
    if os.path.exists(folder+'/bhvr_data_all.npz') and not opts['reload']:
        data = load_behavioral_data(folder+'/bhvr_data_all.npz')
    else:
        data = pl.put_together_files(folder)
        if len(data) == 0:
            return {'glm_mats_ac': [], 'glm_mats_ae': [],
                    'glm_mats_acc': [], 'glm_mats_aee': [],
                    'glm_mats_ace': [], 'glm_mats_aec': [],
                    'bias_mats_psych': [], 'bias_mats_entr': [],
                    'bias_mats_wl': [], 'perf_mats': [],
                    'num_2afc_trs': [], 'perf_pi': None, 'aha_mmt': None,
                    'perfs_cond': None, 'trans_mats': {}}

        else:
            data = load_behavioral_data(folder+'/bhvr_data_all.npz')
    if opts['save_matlab_file']:
        # CoherenceVec = [0; 0.25; 0.375; 0.5; 0.625; 0.75; 1]
        # RewardSideList = 1/2
        # HitHistoryList = 0/1
        # CoherenceList = 1:1:7
        # SilenceList = 0
        # DelayList = 0
        # StimulusList = 1:1:n
        # EnviroProbRepeat = 0.2/0.8 or similar
        # StimulusR = 20 x n matrix evidence R
        # StimulusL = 20 x n matrix evidence L
        # StimulusBlock = StimulusR - StimulusL (I think) (edited)
        EnviroProbRepeat = np.convolve(data['gt'], np.ones((20,))/20, mode='same')
        EnviroProbRepeat = EnviroProbRepeat > 1.5
        cohs = np.unique(data['putative_ev'])
        cohs_list = [np.where(cohs == c)[0][0]+1 for c in data['putative_ev']]
        data_mat = {'CoherenceVec': cohs,
                    'RewardSideList': data['gt'],
                    'HitHistoryList': data['performance'],
                    'CoherenceList': cohs_list,
                    'SilenceList': [],
                    'DelayList': [],
                    'EnviroProbRepeat': EnviroProbRepeat}
        sio.savemat(folder+'/bhvr_data_all.mat', data_mat)
    if opts['num_trials'] and 'std_2afc' in data.keys():
        print('Computing number of 2AFC trials')
        num_2afc_trs = get_num_tr_single_exp(mat=data['std_2afc'].copy(),
                                             per=acr_tr_per, step=step)
    else:
        num_2afc_trs = []
    if opts['performances']:
        print('Computing performance')
        data_tmp = deepc(data)
        # plot performance
        perf, perf_pi, aha_mmt, perfs_cond =\
            get_perf_single_exp(data=data_tmp, spacing=spacing,
                                cond_blcks=cond_blcks, binning=binning)
        if opts['plot_figs']:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
            ax.plot([0, perf.shape[0]*spacing], [0, 0], '--k', lw=0.5)
            ax.plot(np.arange(perf.shape[0])*spacing, perf)
            ax.set_ylabel('Network - Perfect Integrator performance')
            ax.set_xlabel('Trials')
            f.savefig(folder+'/performance.png', dpi=400, bbox_inches='tight')
            plt.close(f)
            n_ch = np.max(data['gt'])
            ncols = 3 if cond_blcks else 1
            f, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(9, 3))
            colors = sns.color_palette("mako", n_colors=n_ch-1)
            for n in range(2, n_ch+1):
                if cond_blcks:
                    starts = ['1-', '2-', str(n)+'-'] if n > 2 else ['1-', '2-']
                    ax_tmp = ax
                    titles = ['Repeating', 'ClockWise', 'Counter-ClockWise']
                else:
                    starts = ['']
                    ax_tmp = [ax]
                    titles = ['']
                for i_s, strt in enumerate(starts):
                    key = str(n)+'-'+strt[:-1]
                    ax_tmp[i_s].errorbar(x=perfs_cond[key]['indx'],
                                         y=perfs_cond[key]['m_perf'],
                                         yerr=perfs_cond[key]['std_perf'],
                                         color=colors[n-2], label=str(n))
            for i_a, a in enumerate(ax_tmp):
                # a.legend()
                # a.set_ylim([-0.1, 1])
                a.set_title(titles[i_a])
                a.axhline(linestyle='--', color='k', lw=0.5)
            f.savefig(folder+'/performance_cond.png', dpi=400, bbox_inches='tight')
            plt.close(f)
    else:
        perf, perf_pi, aha_mmt, perfs_cond = [], None, None, None

    if opts['glm']:
        print('Computing GLM weights')
        data_tmp = deepc(data)
        flag, weights_ac, weights_ae =\
            get_glm_ws_single_exp(data=data_tmp, acr_tr_per=acr_tr_per,
                                  step=step)
        if opts['plot_figs'] and flag:
            # plot kernels
            regrss = ['T++', 'T-+', 'T+-', 'T--']
            f_tr, _, _, _, _ = pf.plot_kernels(weights_ac, weights_ae, ax=None,
                                               n_stps_ws=2, regressors=regrss)
            f_tr.savefig(folder+'/'+name+'_kernels_'+''.join(regrss)+'.png',
                         dpi=400, bbox_inches='tight')
            plt.close(f_tr)
            regrss = ['L+', 'L-']
            f_lat, _, _, _, _ = pf.plot_kernels(weights_ac, weights_ae, ax=None,
                                                n_stps_ws=2, regressors=regrss)
            f_lat.savefig(folder+'/'+name+'_kernels_'+''.join(regrss)+'.png',
                          dpi=400, bbox_inches='tight')
            plt.close(f_lat)
            # plot weights evolution
            tags_mat = [['evidence', 'intercept'], ['L+', 'L-'],
                        ['T++', 'T-+', 'T+-', 'T--']]
            figs = pf.plot_glm_weights(weights_ac, weights_ae,
                                       tags_mat, step, acr_tr_per)
            for ind_f, f in enumerate(figs):
                name_2 = ''.join(tags_mat[ind_f])
                f.savefig(folder+'/'+name+'_'+name_2+'.png', dpi=400,
                          bbox_inches='tight')
                plt.close(f)
    else:
        weights_ac, weights_ae = [], []

    if opts['bias']:
        print('Computing biases')
        data_tmp = deepc(data)
        # BIAS PSYCHO
        flag, bias_psych, _ =\
            get_bias_single_exp(data=data_tmp, acr_tr_per=acr_tr_per, step=step,
                                bias_type='psycho')
        if opts['plot_figs'] and flag:
            fig = pf.plot_bias(bias=bias_psych, step=step, per=acr_tr_per,
                               bias_type='psych')
            fig.savefig(folder+'/acr_training_bias_psycho_trans_cond_'+name+'.png',
                        dpi=400, bbox_inches='tight')
            plt.close(fig)
        # BIAS ENTROPY
        flag, bias_entr, _ =\
            get_bias_single_exp(data=data_tmp, acr_tr_per=acr_tr_per, step=step,
                                bias_type='entropy', ref_distr=True)
        if flag:
            bias_entr_plt = np.nanmean(bias_entr, axis=2)
            bias_entr_plt = np.expand_dims(bias_entr_plt, axis=2)
        if opts['plot_figs'] and flag:
            fig = pf.plot_bias(bias=bias_entr_plt, step=step,  per=acr_tr_per,
                               bias_type='entropy')
            fig.savefig(folder+'/acr_training_bias_entropy_'+name+'.png',
                        dpi=400, bbox_inches='tight')
            plt.close(fig)
        # AFTER CORRECT/ERROR BIAS
        flag, bias_wl, _ =\
            get_bias_single_exp(data=data_tmp, acr_tr_per=acr_tr_per, step=step,
                                bias_type='WL_rep_prob', ref_distr=True)
        if flag:
            bias_wl_plt = np.expand_dims(bias_wl, axis=2)
        if opts['plot_figs'] and flag:
            fig = pf.plot_bias(bias=bias_wl_plt, step=step,  per=acr_tr_per,
                               bias_type='entropy')
            fig.savefig(folder+'/acr_training_bias_WL_rep_prob_'+name+'.png',
                        dpi=400, bbox_inches='tight')
            plt.close(fig)
    else:
        bias_entr, bias_psych, bias_wl = [], [], []

    if opts['glm_rbnd']:
        print('Computing GLM rebound')
        data_tmp = deepc(data)
        weights_acc, weights_aee, weights_ace, weights_aec =\
            get_glm_double_cond_single_exp(data)
        if opts['plot_figs'] and len(weights_acc) > 0 and len(weights_aec) > 0:
            # plot kernels
            regrss = ['T++']
            f_tr, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4), sharey=True)
            kwargs = {'color_ac': naranja, 'color_ae': 'm'}
            pf.plot_kernels(weights_ac=weights_acc[None],
                            weights_ae=weights_aec[None],
                            ac_cols=aftercc_cols,
                            ae_cols=afterec_cols,
                            ax=np.array([ax]), n_stps_ws=2,
                            regressors=regrss, **kwargs)
            if len(weights_ace) > 0 and len(weights_aee) > 0:
                kwargs = {'color_ac': (.7, .7, .7), 'color_ae': 'k'}
                pf.plot_kernels(weights_ac=weights_ace[None],
                                weights_ae=weights_aee[None],
                                ac_cols=afterce_cols,
                                ae_cols=afteree_cols,
                                ax=np.array([ax]), n_stps_ws=2,
                                regressors=regrss, **kwargs)

            f_tr.savefig(folder+'/'+name+'_kernels_rebnd_'+''.join(regrss)+'.png',
                         dpi=400, bbox_inches='tight')
            # plt.close(f_tr)
    else:
        weights_acc, weights_aee, weights_ace, weights_aec = [], [], [], []

    if opts['trans_mats']:
        data_tmp = deepc(data)
        n_ch = min(6, np.max(data_tmp['gt']))
        trans_mats = get_trans_mats_after_seq(data=data_tmp, sv_folder=folder,
                                              n_ch=n_ch)
    else:
        trans_mats = {}

    all_data = {'glm_mats_ac': weights_ac, 'glm_mats_ae': weights_ae,
                'glm_mats_acc': weights_acc, 'glm_mats_aee': weights_aee,
                'glm_mats_ace': weights_ace, 'glm_mats_aec': weights_aec,
                'bias_mats_psych': bias_psych, 'bias_mats_entr': bias_entr,
                'bias_mats_wl': bias_wl, 'perf_mats': perf,
                'num_2afc_trs': num_2afc_trs, 'perf_pi': perf_pi,
                'perfs_cond': perfs_cond, 'aha_mmt': aha_mmt,
                'trans_mats': trans_mats}
    return all_data


def process_cond(main_folder, cond, val, step=30000, acr_tr_per=300000,
                 binning=300000, spacing=10000, name='', test_retrain='',
                 extra_folder={}, **opts):
    """
    Generate a file with the analyses of experiments in main_folder.

    Saves data for each given condition in single .npz file, which contains
    a dict of concatenated data. It does so for training data, different types of
    test data and retrain data.

    Parameters
    ----------
    main_folder : str
        folder containing the experiments to analyze.
    cond : str
        tag to look for in experiments folders.
    val : str
        tag to separate experiments into categories.
    step : int, optional
        step to use for the period-wise analysis of the data (30000)
    acr_tr_per : int, optional
        period to use for the period-wise analysis of the data (300000)
    name : str, optional
        name to use to save figs and data ('')
    test_retrain : str, optional
        specifies the type of data (training, test_2AFC, test, retrain) to analyze
    **opts : dict
        dictionary indicating which analysis to perform. Example:
            {'glm': True, 'trans_mats': True, 'bias': True,
             'n-GLM': False, 'rebound': False, 'plot_figs': True,
             'step': 250000, 'per': 500000, 'reload': False}

    Returns
    -------
    data : dict
        dictionary with all analyses performed on all experiments in main_folder.

    """
    def process_cond_core(folder, cond, val, data=None, val_stord=None):
        files = glob.glob(folder+'/*'+cond+'*')
        files = [f for f in files if os.path.basename(f).find(val) != -1]
        for ind_f, f in enumerate(files):
            # Get values from each file.
            curr_val = get_tag(val, f) if val_stord is None else val_stord
            work_f = f + '/' + test_retrain + '/'
            print('-------------------------')
            print(work_f)
            all_data = process_data(work_f, name=name, acr_tr_per=acr_tr_per,
                                    step=step, spacing=spacing, binning=binning,
                                    **opts)
            if data is None:
                data = {k: [] for k in all_data.keys()}
                data['val_mat'] = []
                data['shapes'] = []
            data['val_mat'].append(curr_val)
            for k in all_data.keys():
                data[k].append(all_data[k])
        return data, files

    data, files = process_cond_core(folder=main_folder, cond=cond, val=val)
    extra_files = []
    if len(extra_folder) > 0:
        data, extra_files = process_cond_core(folder=extra_folder['main_folder'],
                                              cond=extra_folder['cond'],
                                              val=extra_folder['val'],
                                              val_stord=extra_folder['val_stord'],
                                              data=data)
    files = files+extra_files
    if len(files) > 0:
        data['step'] = step
        data['files'] = files
        data['acr_tr_per'] = acr_tr_per
        data['spacing'] = spacing
        np.savez(main_folder+'/data_'+cond+'_'+test_retrain.replace('/', '') +
                 '_'+name+'.npz', **data)
        return data


def batch_analysis(main_folder, conds=['A2C', 'ACER', 'ACKTR', 'PPO2'],
                   val='n_ch',  name='', extra_folder={},
                   **analysis_opts):
    """
    Run analysis for different conditions.

    Parameters
    ----------
    main_folder : str
        folder containing all instance from the experiment.
    conds : list, optional
        list containing the tags corresponding to each condition
        (['A2C', 'ACER', 'ACKTR', 'PPO2'])
    val : str, optional
        tag to look for and store in each instance ('n_ch')
    name : str, optional
        extra tag to filter the instances ('')
    **analysis_opts : dict
        dictionary containing the parameters for the analysis of different
        simulations. Example:
            opts = {'train': {'glm': False, 'trans_mats': False, 'bias': True,
                              'rebound': False, 'plot_figs': True,
                              'step': 50000, 'per': 90000, 'reload': True},
                    'test_2AFC': {'glm': True, 'trans_mats': False, 'bias': True,
                                  'n-GLM': False, 'rebound': False,
                                  'plot_figs': True, 'step': 100000, 'per': 200000,
                                  'reload': True}}
            analysis_opts = {'analyze': ['train', 'test_2AFC'],
                             'opts': opts}

    Returns
    -------
    None.

    """
    for cond in conds:
        for exp in analysis_opts['analyze']:
            step = analysis_opts['opts'][exp]['step']
            per = analysis_opts['opts'][exp]['per']
            binning = analysis_opts['opts'][exp]['binning']
            opts = analysis_opts['opts'][exp].copy()
            del opts['step'], opts['per'], opts['binning']
            if '_all' not in exp:
                exp_name = '' if exp == 'train' else exp
                process_cond(main_folder, cond, val, step=step, acr_tr_per=per,
                             name=name, test_retrain=exp_name, binning=binning,
                             extra_folder=extra_folder, **opts)
            else:
                folders = np.unique([os.path.basename(x[0])
                                     for x in os.walk(main_folder)
                                     if '_model_' in x[0]])
                folders = [f for f in folders
                           if int(get_tag('model', f)) % 4000000 == 0]
                for f in folders:
                    process_cond(main_folder, cond, val, step=step, acr_tr_per=per,
                                 name=name,  binning=binning,
                                 test_retrain=exp+'/'+f, **opts)


def bias_entropy(choice, mask, num_chs=None, gt=None, plot=False, comp_bsln=False):
    """
    Compute entropy bias frm a choice array.

    The code can compute the absolute entropy (KLD from the uniform) of the choices
    probabilities conditioned on each choice or the KLD from the distribution of
    ground truth sides.
    Parameters
    ----------
    choice : array
        array containing the choice from which to compute the entropy.
    mask : array
        array of boolean values indicating which elements to use for the
        computation.
    num_chs : int, optional
        total number of possible choices (None)
    gt : array, optional
        array containing the ground truth sides, if not None, the entropy will be
        computed as the KLD between the distributions of choices and ground truth
        sides conditioned on the previous choice (None)
    plot : bool, optional
        bolean indicating whether to plot the transition matrices (False)
    comp_bsln : bool, optional
        bolean indicating whether to compute a baseline value by shuffling 100
        times the transition matrix obtained (False)

    Returns
    -------
    popt : list
        list containing (in the 2nd element) the bias.
    pcov : int
        0 value.

    the format of the returned parameters is inherited from the psych-bias fn

    """
    # Associate invalid trials (network fixates) with incorrect choice.
    invalid = choice == 0
    num_invalids = np.sum(invalid)

    if num_chs is None and gt is None:
        num_chs = np.unique(choice).shape[0] - 1*(num_invalids > 0)
    elif gt is not None:
        num_chs = np.unique(gt[mask]).shape[0]
    # assign random choices to the invalid trials
    aux = np.random.choice(num_chs, (num_invalids,)) + 1
    choice[invalid] = aux
    # one entropy value calculated for trial t + 1 after choice ch
    trans_mat = np.empty((num_chs, num_chs))
    trans_mat[:] = np.nan
    ref_trans_mat = np.empty((num_chs, num_chs))
    ref_trans_mat[:] = np.nan
    for ch in np.arange(1, num_chs+1):
        trans_mat[ch-1, :] = get_probs(mat=choice, val=ch, mask=mask,
                                       num_chs=num_chs)
        if gt is None:
            ref_trans_mat[ch-1, :] = np.ones((num_chs,))/num_chs
        else:
            ref_trans_mat[ch-1, :] = get_probs(mat=gt, val=ch, mask=mask,
                                               num_chs=num_chs, prev_mat=choice)
    if plot:
        plot_matrices(mat=trans_mat, ref_mat=ref_trans_mat)
    trans_mat = trans_mat.flatten()
    trans_mat = trans_mat/np.sum(trans_mat)
    ref_trans_mat = ref_trans_mat.flatten()
    ref_trans_mat = ref_trans_mat/np.sum(ref_trans_mat)
    bias = 1-entropy(pk=trans_mat.flatten(), qk=ref_trans_mat.flatten())
    if comp_bsln:
        bias_sh = []
        for i_sh in range(100):
            np.random.shuffle(trans_mat)
            bias_sh.append(1-entropy(pk=trans_mat.flatten(),
                                     qk=ref_trans_mat.flatten()))
        print(np.mean(bias_sh))
        print(bias)
        print(np.sum(bias_sh > bias))
        print('--------------------')
    popt = [np.nan, bias]
    pcov = 0
    return popt, pcov


def get_probs(mat, val, mask, num_chs, prev_mat=None):
    """
    Compute counts of each choice at t+1 conditioned on ch at t being val.

    Parameters
    ----------
    mat : array
        array with choices.
    val : int
        choice to condition on.
    mask : array
        array of booleans to filter trials.
    num_chs : int
        total number of possible choices.
    prev_mat : array, optional
        if not None, the previous choice will be conditioned on prev_mat
        instead of mat (this is used for the case in which mat is the ground truth
                        but the conditioning is made on the actual choices (None)

    Returns
    -------
    counts : array
        array with the counts of each possible choice at time t+1 after the choice
        indicated by val has been selected at time t.

    """
    if prev_mat is None:
        inds_ch = mat[:-1] == val
    else:
        inds_ch = prev_mat[:-1] == val
    # t + 1 selected where choice at t == ch
    inds_ch = conc((np.array([0]), inds_ch))
    inds_ch = np.where(and_(inds_ch, mask))[0]
    counts, _ = np.histogram(mat[inds_ch], bins=np.arange(num_chs+1)+0.5)
    counts = counts[:num_chs].astype('float')
    counts[counts == 0.] = 1
    return counts


def plot_matrices(mat, ref_mat):
    """
    Plot a matrix and a reference matrix (gt) side by side.

    Parameters
    ----------
    mat : 2D array
        matrix.
    ref_mat : 2D array
        reference matrix.

    Returns
    -------
    None.

    """
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(mat)
    plt.title('Network')
    plt.subplot(1, 2, 2)
    plt.imshow(ref_mat)


def bias_psychometric(choice, ev, mask=None, maxfev=10000):
    """
    Compute repeating bias by fitting probit function.

    Parameters
    ----------
    choice : array
        array of choices made bythe network.
    ev : array
        array with (signed) stimulus evidence.
    mask : array, optional
        array of booleans indicating the trials on which the bias
    # should be computed (None)

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of probit(xdata) - ydata is minimized
    pcov : 2d array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix.

    """
    choice = choice.astype(float)
    choice[and_(choice != 1, choice != 2)] = np.nan
    repeat = get_repetitions(choice).astype(float)
    repeat[np.isnan(choice)] = np.nan
    # choice_repeating is just the original right_choice mat
    # but shifted one element to the left.
    choice_repeating = conc(
        (np.array(np.random.choice([1, 2])).reshape(1, ),
         choice[:-1]))
    # the rep. evidence is the original evidence with a negative sign
    # if the repeating side is the left one
    rep_ev = ev*(-1)**(choice_repeating == 2)
    if mask is None:
        mask = ~np.isnan(repeat)
    else:
        mask = and_(~np.isnan(repeat), mask)
    rep_ev_mask = rep_ev[mask]  # xdata
    repeat_mask = repeat[mask]  # ydata
    try:
        # Use non-linear least squares to fit probit to xdata, ydata
        popt, pcov = curve_fit(probit_lapse_rates, rep_ev_mask,
                               repeat_mask, maxfev=maxfev)
    except RuntimeError as err:
        print(err)
        popt = [np.nan, np.nan, np.nan, np.nan]
        pcov = 0
    return popt, pcov, rep_ev_mask, repeat_mask


def distance(len_my_list, idx1, idx2, sign=1):
    d1 = abs(idx1 - idx2)
    d2 = len_my_list - d1
    sign = sign if ((d1 < d2) and (idx1 < idx2)) or ((d1 > d2) and (idx1 > idx2))\
        else -sign
    return sign*min(d1, d2)


def get_trans_prob_mat(choice, mask, n_ch):
    """
    Compute entropy bias frm a choice array.

    The code can compute the absolute entropy (KLD from the uniform) of the choices
    probabilities conditioned on each choice or the KLD from the distribution of
    ground truth sides.

    Parameters
    ----------
    choice : array
        array containing the choice from which to compute the entropy.
    mask : array
        array of boolean values indicating which elements to use for the

    Returns
    -------
    trans_mat : array
        array containing the transition probabilities

    """
    # Associate invalid trials (network fixates) with incorrect choice.
    invalid = and_(choice == 0, choice > n_ch)
    num_invalids = np.sum(invalid)
    # assign random choices to the invalid trials
    aux = np.random.choice(n_ch, (num_invalids,)) + 1
    choice[invalid] = aux
    # one entropy value calculated for trial t + 1 after choice ch
    trans_mat = np.empty((n_ch, n_ch))
    trans_mat[:] = np.nan
    aligned_mat = np.empty((n_ch, n_ch))
    aligned_mat[:] = np.nan
    for ch in np.arange(1, n_ch+1):
        probs = get_probs(mat=choice, val=ch, mask=mask, num_chs=n_ch)
        trans_mat[:, ch-1] = probs
        aligned_mat[:, ch-1] = np.roll(probs, -int(ch-n_ch/2))

    trans_mat = trans_mat/np.sum(trans_mat, axis=0)
    aligned_mat = np.sum(aligned_mat, axis=1)
    aligned_mat = aligned_mat/np.sum(aligned_mat)
    return trans_mat, aligned_mat


def get_trans_prob_mats_after_seq(data, n_ch=6, num_samples=None, ch_gt='choice',
                                  seq_prps={'templ_perf': [1, 1, 1, -1, 1],
                                            'templ_trans': np.array([1, 1, 1,
                                                                     np.nan,
                                                                     np.nan]),
                                            'start': 0}):
    """
    Compute entropy bias for passed conditions.

    Computes KLD from a uniform distr. of the choices probabilities conditioned
    on each choice or the KLD from the distribution of ground truth sides.
    It does so for each block type in tr_block, after_error and after_correct
    (Shape = (2, num_blocks))
    Parameters
    ----------
    ch : array
        array with choices.
    ev : array
        array with stimulus evidence.
    perf : array
        array with choices outcomes.
    c_tr : array
        array indicating catch trials (correct choice but reward not given).
    tr_block : TYPE
        array with transitions block.
    gt : array, optional
        array with ground truth sides (None)
    nch_mask : array, optional
        array to further filter trials (None)
    plot_tr_mat : bool, optional
        whether to plot transition matrices (False)
    ev_perc : int, optional
        percetile of evidence used to filter trials bias calculation (5)
    plt_mask : bool, optional (for debugging)
        boolean indicating whether to plot mask with choices, perf.. (False)

    Returns
    -------
    biases : array
        array with bias values.

    """
    if num_samples is None:
        num_samples = 0
    ch = data[ch_gt][-num_samples:]
    sig_ev = data['putative_ev'][-num_samples:]
    perf = data['performance'][-num_samples:]
    print('Number of samples: ', perf.shape[0])
    # filter by number of choices
    indx = data['nch'] == n_ch
    ch = ch[indx].astype(float)
    sig_ev = sig_ev[indx].astype(float)
    perf = perf[indx].astype(float)
    # make nan all choices larger than nch
    indx = np.logical_or(ch > n_ch, ch == 0)
    ch[indx] = np.nan
    n_max_ch = np.nanmax(ch)
    sign = 1 if n_ch != 2 else -1  # this is probably not necessary
    trans = [distance(n_max_ch, ch[x], ch[x+1], sign=sign)
             for x in range(ch.shape[0]-1)]
    trans = conc((trans, np.array([np.nan])))

    # context
    templ_perf = seq_prps['templ_perf']
    templ_trans = seq_prps['templ_trans']
    start = seq_prps['start']

    # transition blocks
    # select evidence below ev_perc quantile
    ev_mask = np.abs(sig_ev) < 0.00001
    # get rid of first value in evidence
    # ev_mask = conc((ev_mask[1:], np.array([0])))
    # After error and after correct bias for each block.
    trans_mats = []
    al_trans_mats = []
    num_samples_mat = []
    perf_mat = []
    ind_ctx = 0
    for ind_ctx in range(0, len(templ_perf)):
        templ_perf_temp = templ_perf[:1+ind_ctx]
        print(templ_perf_temp)
        perf_mask = template_match(perf, templ_perf_temp)
        perf_mask = conc((np.array([False]), perf_mask[:-1]))

        if ind_ctx >= start:
            templ_trans_temp = templ_trans[:1+ind_ctx-start]
            print(templ_trans_temp)
            t_len = len(templ_trans_temp)
            t_comp = templ_trans_temp[~np.isnan(templ_trans_temp)]
            indx = np.array([i+t_len for i in range(len(trans)-t_len-1)
                             if (trans[i:i+len(t_comp)] == t_comp).all()])
            trans_mask = np.zeros_like(perf_mask)
            if len(indx) > 0:
                trans_mask[indx+1] = 1
        else:
            trans_mask = np.ones_like(perf_mask)

        mask = and_.reduce((ev_mask, perf_mask, trans_mask))
        if False:
            plot_masks_cond(ch=ch, perf=perf, mask=mask, p_hist=perf_mask,
                            repeat=sig_ev, trans=trans_mask, num=100,
                            start=np.where(mask == 1)[0][0]-10)
            plt.title(templ_perf_temp+[1001]+templ_trans_temp.tolist())
        num_samples = np.sum(mask)
        print('Number of samples: ', num_samples)
        tr_mat, al_tr_mat = get_trans_prob_mat(mask=mask.copy(), choice=ch,
                                               n_ch=n_ch)
        trans_mats.append(tr_mat)
        al_trans_mats.append(al_tr_mat)
        num_samples_mat.append(num_samples)
        perf_mat.append(np.mean(perf[mask]))
    return trans_mats, al_trans_mats, num_samples_mat, perf_mat


def compute_AE_rebound(ch, ev, perf, nch_mask=None, tr_hist_mask=None,
                       bias_type='psycho', ev_perc=10, nch=None,
                       template_perf=None, template_rep=None,
                       templ_specs=None, whole_seq=True):
    """
    Compute bias for specific sequence of transitions and performances.

    Parameters
    ----------
    ch : array
        array with choices made by agent.
    ev : array
        array with stimulus evidence.
    perf : array
        array with performance.
    nch_mask : array, optional
        array of booleans to filter trials depending on number of choices (None)
    tr_hist_mask : array, optional
        array of booleans to filter trials depending on history block (None)
    bias_type : str, optional
        type of bias to compute ('entropy' or 'psycho') ('psycho')
    ev_perc : int, optional
        evidence percentile to filter trials for the entropy bias computation (10)
    nch : int, optional
        number of choices (for the entropy bias computation) (None)
    template_perf : array, optional
        array indicating the sequence of outcomes (None). If None the template
        will be:
            np.array([-1, 1, 1, 1, 1, -1, 1])
        where -1 and 1 correspond to error and correct, respectively.
    template_rep : array, optional
         array indicating the sequence of transitions (None). If None the template
        will be:
            np.array([0, 0, 1, 1, 1, 0, 0])
        where 0 corresponds to both alternations or repetition and 1 corresponds
        to repetition (-1 corresponds to alternation)
    templ_specs : dict, optional
        allows to easily update the specifications of the templates (None)
    whole_seq : bool, optional
        whether to compute the biases for the whole sequence of just for the last
        three steps (True)

    Returns
    -------
    biases : array
        array containing the biases corresponding to each element of the sequence
        (i.e. biases conditioning on the sub-sequence ending at each element).
    num_samples : array
        array containing the number of samples for each sub-sequence.

    """
    templ_specs_def = {'num_trials_seq': 3, 'after_corr': -1, 'rep_alt': 1}
    repetitions = get_repetitions(ch)
    mean_perf = np.mean(perf)
    mean_rep = np.mean(repetitions)
    perf_zero_mean = perf - mean_perf
    rep_zero_mean = repetitions - mean_rep
    if bias_type == 'entropy':
        # select evidence below ev_perc quantile
        ev_mask = evidence_mask(ev, percentage=ev_perc)
        # get rid of first value in evidence
        ev_mask = conc((ev_mask[1:], np.array([0])))
        bias_calculation = bias_entropy
        b_args = {'choice': ch, 'num_chs': nch}
    else:
        ev_mask = np.ones_like(perf) == 1
        bias_calculation = bias_psychometric
        b_args = {'choice': ch, 'ev': ev}
    if nch_mask is None:
        nch_mask = np.ones_like(perf) == 1
    if tr_hist_mask is None:
        tr_hist_mask = np.ones_like(perf) == 1
    nch_mask = remove_borders(nch_mask)
    tr_hist_mask = remove_borders(tr_hist_mask)
    if template_perf is None:
        templ_specs_def.update(templ_specs)
        nt_seq = templ_specs_def['num_trials_seq']
        ac = templ_specs_def['after_corr']
        # after error
        template_perf = np.array([-1, 1] + [1]*nt_seq + [ac] + [1])

    if template_rep is None:
        templ_specs_def.update(templ_specs)
        nt_seq = templ_specs_def['num_trials_seq']
        rep_alt = templ_specs_def['rep_alt']
        template_rep = np.array([0]*2+[rep_alt]*nt_seq+[0, 0])
    assert len(template_perf) == len(template_rep)
    if whole_seq:
        num_steps = len(template_perf)
    else:
        num_steps = 3
    biases = np.empty((num_steps, ))
    num_samples = np.empty((num_steps, ))
    biases[:] = np.nan
    num_samples[:] = np.nan
    for count, ind_seq in enumerate(range(len(template_perf)-num_steps,
                                          len(template_perf))):
        ind_perf = ind_seq + 1
        ind_rep = ind_seq + 1
        templ_perf = template_perf[:max(0, ind_perf)]
        templ_rep = template_rep[:max(0, ind_rep)]
        perf_mask = template_match(perf_zero_mean, templ_perf)
        if len(templ_rep) > 0:
            rep_mask = template_match(rep_zero_mean, templ_rep)
        else:
            rep_mask = np.ones_like(perf_mask)
        templ_mask = and_(rep_mask, perf_mask)
        mask = and_.reduce((ev_mask, nch_mask, tr_hist_mask, templ_mask))
        mask = conc((np.array([False]), mask[:-1]))
        if np.sum(mask) > 10:
            popt, pcov, _, _ = bias_calculation(mask=mask.copy(), **b_args)
        else:
            popt = [np.nan, np.nan]
        # we just care about second element of popt, which is the bias.
        biases[count] = popt[1]
        num_samples[count] = np.sum(mask)
    return biases, num_samples


def compute_winLose_rep_prob(ch, ev, perf, mask=None, figs=False, new_fig=False,
                             plt_mask=False, prev_perfs=[0, 1], per_point_err=.1,
                             bins=np.arange(20)-10, plot_pts=True, **plt_kwargs):
    """
    Compute repeating bias conditioned on context (rep/alt) and performance.

    Parameters
    ----------
    ch : array
        array with choices.
    ev : array
        array with stimulus evidence.
    perf : array
        array with choices outcomes.
    mask : array
        array indicating trials to use.
    conv_w : int
        number of transitions back taken into account to define the context.
    figs : bool, optional
        boolean indicating whether to plot the psychometric curves (False)
    new_fig : bool, optional
        boolean indicating whether to create a new figure for the plots (False)
    lw : float, optional
        line width for plots (2)
    plt_mask : bool, optional (for debugging)
        boolean indicating whether to plot mask with choices, perf.. (False)

    Returns
    -------
    biases : array
        array containing the biases for the different contexts:
     prev. perf x [alt cont + alt, rep cont + rep, alt cont + rep, rep cont + alt]
     err/corr  x [alt congr.,       rep. congr.,   alt. incongr., rep. incongr.]
    num_samples : array
        number of samples for each context.

    """
    if mask is None:
        mask = np.ones_like(perf)
    biases = np.empty((2, 1))
    plt_opts = {'lw': 1.5}  # only used if figs, but need to define it anyway
    if figs:
        if new_fig:
            plt.subplots(figsize=(2, 2))
        labels = ['After error', 'After correct']
        count = 0
    for ind_perf in prev_perfs:
        m = and_.reduce((perf == ind_perf,
                         mask == 1))
        m = np.concatenate((np.array([False]), m[:-1]))
        if ind_perf == 0 and plt_mask:
            plot_masks_cond(ch=ch, perf=perf,
                            mask=m, general_mask=mask)

        if np.sum(m) > 100:
            popt, pcov, ev_mask, rep_mask = bias_psychometric(choice=ch.copy(),
                                                              ev=ev.copy(),
                                                              mask=m.copy())
            # compare fit's prediction with repeating proportions
            # plot if required
            if figs:
                plt_opts['color'] = 'k'
                plt_opts['label'] = labels[count]+str(round(popt[1], 3))
                plt_opts['alpha'] = 0.6+0.4*ind_perf
                plt_opts['linestyle'] = '-'
                plt_opts.update(plt_kwargs)
            pred = psych_curves_fit(ev=ev, popt=popt, plot=figs,
                                    bins=bins, **plt_opts)
            plot_figs = figs and plot_pts
            if plot_figs:
                plt_opts['label'] = ''
                plt_opts['linestyle'] = ''
            means = psych_curves_props(x_values=ev_mask, y_values=rep_mask,
                                       bins=bins, plot=plot_figs, **plt_opts)
            error = np.nansum(np.abs(means-pred))
            if figs and ind_perf == 0:
                plt.title(str(per_point_err*len(bins))+' ' +
                          str(error))
            if error > per_point_err*len(bins):
                popt = [np.nan, np.nan, np.nan, np.nan]
        else:
            popt = [np.nan, np.nan, np.nan, np.nan]
        if figs:
            count += 1
        biases[ind_perf] = popt[1]
    return biases


def compute_bias_psycho(ch, ev, perf, conv_w, mask=None, figs=False, new_fig=False,
                        plt_mask=False, prev_perfs=[0, 1], per_point_err=.1,
                        min_num_tr=100, bins=np.arange(20)-10, plot_pts=True,
                        **plt_kwargs):
    """
    Compute repeating bias conditioned on context (rep/alt) and performance.

    Parameters
    ----------
    ch : array
        array with choices.
    ev : array
        array with stimulus evidence.
    perf : array
        array with choices outcomes.
    mask : array
        array indicating trials to use.
    conv_w : int
        number of transitions back taken into account to define the context.
    figs : bool, optional
        boolean indicating whether to plot the psychometric curves (False)
    new_fig : bool, optional
        boolean indicating whether to create a new figure for the plots (False)
    lw : float, optional
        line width for plots (2)
    plt_mask : bool, optional (for debugging)
        boolean indicating whether to plot mask with choices, perf.. (False)

    Returns
    -------
    biases : array
        array containing the biases for the different contexts:
     prev. perf x [alt cont + alt, rep cont + rep, alt cont + rep, rep cont + alt]
     err/corr  x [alt congr.,       rep. congr.,   alt. incongr., rep. incongr.]
    num_samples : array
        number of samples for each context.

    """
    if mask is None:
        mask = np.ones_like(perf)
    num_conts = 4
    values = [-conv_w/2, conv_w/2]
    repeat = get_repetitions(ch)
    rep_values = [0, 1, 1, 0]
    biases = np.empty((2, num_conts))
    num_samples = np.empty((2, num_conts))
    p_hist = np.convolve(perf, np.ones((conv_w,)), mode='full')[0:-conv_w+1]
    p_hist = np.concatenate((np.array([0]), p_hist[:-1]))
    trans = get_transition_mat(ch, conv_w=conv_w)
    plt_opts = {'lw': 1.5}  # only used if figs, but need to define it anyway
    if figs:
        if new_fig:
            plt.subplots(figsize=(2, 2))
        colrs = [rojo, azul, rojo_2, cyan]
        labels = ['alt. context alt- b:', 'rep. context rep- b:',
                  'alt. context rep- b:', 'rep. context alt- b:',
                  'alt. context alt+ b:', 'rep. context rep+ b:',
                  'alt. context rep+ b:', 'rep. context alt+ b:']
        count = 0
    for ind_perf in prev_perfs:
        # ind_tr = 0 --> alt cont + alt
        # ind_tr = 1 --> rep cont + rep
        # ind_tr = 2 --> alt cont + rep
        # ind_tr = 3 --> rep cont + alt
        for ind_tr in range(num_conts):
            m = and_.reduce((trans == values[ind_tr % 2], perf == ind_perf,
                             p_hist == conv_w, repeat == rep_values[ind_tr],
                             mask == 1))
            m = np.concatenate((np.array([False]), m[:-1]))
            if ind_perf == 0 and ind_tr == 3 and plt_mask:
                plot_masks_cond(ch=ch, repeat=repeat, trans=trans, perf=perf,
                                mask=m, p_hist=p_hist, general_mask=mask)

            if np.sum(m) > min_num_tr:
                popt, pcov, ev_mask, rep_mask = bias_psychometric(choice=ch.copy(),
                                                                  ev=ev.copy(),
                                                                  mask=m.copy())
                # TODO: this part needs to be cleaned
                # compare fit's prediction with repeating proportions
                # plot if required
                plot_figs = figs and ind_tr < 2  # plot for congruent case
                if plot_figs:
                    plt_opts['color'] = colrs[ind_tr]
                    plt_opts['label'] = labels[count]+str(round(popt[1], 3))
                    plt_opts['alpha'] = 0.6+0.4*ind_perf
                    plt_opts['linestyle'] = '-'
                    plt_opts.update(plt_kwargs)
                min_ = min(ev_mask)
                max_ = max(ev_mask)
                extrem = max(abs(min_), max_)
                bins = np.linspace(-extrem, extrem, 7)
                pred = psych_curves_fit(ev=ev, popt=popt, plot=plot_figs,
                                        bins=bins, **plt_opts)
                plot_figs = plot_figs and plot_pts
                if plot_figs:
                    plt_opts['label'] = ''
                    plt_opts['linestyle'] = ''

                psych_curves_props(x_values=ev_mask, y_values=rep_mask,
                                   bins=bins, plot=plot_figs, **plt_opts)
                error = 0  # np.nansum(np.abs(means-pred))
                if figs and ind_tr >= 2 and ind_perf == 0:
                    plt.title(str(per_point_err*len(bins))+' ' +
                              str(error))
                if error > per_point_err*len(bins):
                    popt = [np.nan, np.nan, np.nan, np.nan]
            else:
                popt = [np.nan, np.nan, np.nan, np.nan]
            if figs:
                count += 1
            biases[ind_perf, ind_tr] = popt[1]
            num_samples[ind_perf, ind_tr] = np.sum(mask)
    return biases  # , num_samples


def plot_masks_cond(ch, perf, mask, c_tr=None, general_mask=None, repeat=None,
                    trans=None, p_hist=None, num=500, start=9200):
    """
    Plot mask with choices, performance and other variables.

    Parameters
    ----------
    ch : array
        array with choices.
    perf : array
        array with choices outcomes.
    c_tr : array
        array indicating catch trials (correct choice but reward not given).
    mask : array
        array indicating which trials will be used.
    repeat : array, optional
        array with repeatitions (1 if repeating, 0 if alternating) (None)
    trans : TYPE, optional
        array indicating the number of past repetitions (None)
    p_hist : TYPE, optional
        array indicating the number of past correct (None)

    Returns
    -------
    None.

    """
    plt.subplots(figsize=(8, 8))
    plt.plot(ch[start:start+num], '-+',
             label='choice', lw=1)
    plt.plot(perf[start:start+num]-3, '--+', label='perf',
             lw=1)
    plt.plot(mask[start:start+num]-3, '-+', label='mask',
             lw=1)
    if general_mask is not None:
        plt.plot(general_mask[start:start+num] - 4, '-+',
                 label='general mask', lw=1)
    if c_tr is not None:
        plt.plot(c_tr[start:start+num] - 4, '-+',
                 label='catch trial', lw=1)
    if repeat is not None:
        plt.plot(repeat[start:start+num], '-+',
                 label='repeat', lw=1)
        plt.plot(trans[start:start+num], '-+',
                 label='transitions', lw=1)
    if p_hist is not None:
        plt.plot(p_hist[start:start+num], '-+',
                 label='perf_hist', lw=1)
    for ind in range(num):
        plt.plot([ind, ind], [-3, 3], '--',
                 color=(.7, .7, .7))
    plt.legend()


def psych_curves_fit(ev, popt, bins=None, plot=False, **plot_opts):
    """
    Plot psychometric curves.

    Parameters
    ----------
    ev : array
        array with evidences that will be used to compute the xs.
    popt : list
        list with fitting parameters.
    **plot_opts : dict
        plotting options.

    Returns
    -------
    None.

    """
    num_values = 10
    # conf = 0.95
    if bins is None:
        bin_edges = mstats.mquantiles(ev,
                                      (np.arange(num_values)+1)/num_values)
    else:
        bin_edges = bins
    x = bin_edges[:-1]+np.diff(bin_edges)/2
    y = probit_lapse_rates(x, popt[0], popt[1], popt[2], popt[3])
    if plot:
        plt.plot(x, y, **plot_opts)
        plt.legend(loc="lower right")
        ax = plt.gca()
        pf.plot_dashed_lines(ax=ax, minimo=-np.max(x), maximo=np.max(x))
        plt.xlabel('Repeating evidence')
        plt.ylabel('Repeating probability')
        ax.set_xlabel('Repeating evidence')
        ax.set_ylabel('Probability of repeat')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-.4, .4])
    return y


def psych_curves_props(x_values, y_values, bins=None, plot=False, num_values=7,
                       **plt_opts):
    """
    Plot average values of y_values in x_values.

    Parameters
    ----------
    x_values : array
        x values.
    y_values : array
        y values.
    color : tuple, optional
        color of the trace ((0, 0, 0))

    Returns
    -------
    None.

    """
    # conf = 0.95
    if bins is None:
        bin_edges = mstats.mquantiles(x_values,
                                      (np.arange(num_values)+1)/num_values)
    else:
        bin_edges = bins
    xss_bin = np.searchsorted(bin_edges, x_values)
    xss_unq = np.unique(xss_bin)
    mean_perf =\
        np.array([np.nanmean(y_values[xss_bin == x]) for x in xss_unq])
    std_perf = [np.nanstd(y_values[xss_bin == x]) /
                np.sqrt(np.sum(xss_bin == x)) for x in xss_unq]
    mean_indx = [np.nanmean(x_values[xss_bin == x]) for x in xss_unq]
    hist_1 = np.histogram(x_values[y_values == 1], bins=bin_edges)[0]
    hist_all = np.histogram(x_values, bins=bin_edges)[0]
    # plot
    props = hist_1/hist_all
    if plot:
        # plt_opts['alpha'] = 0.2
        # plt.plot(bin_edges[:-1]+np.diff(bin_edges)/2, props, marker='.',
        #          **plt_opts)
        plt_opts['alpha'] = 1
        plt.errorbar(mean_indx, mean_perf, std_perf, marker='.', **plt_opts)
        # asdasd
    return props


def plot_psycho_curve(ev, choice, popt, ax, plot_errbars=False, **plt_opts):
    """
    Plot psycho-curves (fits and props) using directly the fit parameters.

    THIS FUNCTION ASSUMES PUTATIVE EVIDENCE (it will compute response proportions
                                             for all values of ev)

    Parameters
    ----------
    ev : array
        array with **putative** evidence for each trial.
    choice : array
        array with choices made by agent.
    popt : list
        list containing fitted parameters (beta, alpha, piL, piR).
    ax : axis
        where to plot.
    **plt_opts : dict
        plotting options.

    Returns
    -------
    means : list
        response means for each evidence value.
    sems : list
        sem for the responses.
    x : array
        evidences values for which the means/sems are computed.
    y_fit : array
        y values for the fit.
    x_fit : array
        x values for the fit.

    """
    x_fit = np.linspace(np.min(ev), np.max(ev), 20)
    y_fit = probit_lapse_rates(x_fit, popt[0], popt[1], popt[2], popt[3])
    ax.plot(x_fit, y_fit, '-', **plt_opts)
    means = []
    sems = []
    for e in np.unique(ev):
        means.append(np.mean(choice[ev == e]))
        sems.append(np.std(choice[ev == e])/np.sqrt(np.sum(ev == e)))
    x = np.unique(ev)
    plt_opts['linestyle'] = ''
    if 'label' in plt_opts.keys():
        del plt_opts['label']
    if plot_errbars:
        ax.errorbar(x, means, sems, **plt_opts)
    ax.plot([0, 0], [0, 1], '--', lw=0.2, color=(.5, .5, .5))
    return means, sems, x, y_fit, x_fit


def compute_bias_entropy(ch, ev, perf, tr_block, c_tr, gt=None,
                         nch_mask=None, plot_tr_mat=False, ev_perc=5,
                         plt_masks=False):
    """
    Compute entropy bias for passed conditions.

    Computes KLD from a uniform distr. of the choices probabilities conditioned
    on each choice or the KLD from the distribution of ground truth sides.
    It does so for each block type in tr_block, after_error and after_correct
    (Shape = (2, num_blocks))
    Parameters
    ----------
    ch : array
        array with choices.
    ev : array
        array with stimulus evidence.
    perf : array
        array with choices outcomes.
    c_tr : array
        array indicating catch trials (correct choice but reward not given).
    tr_block : TYPE
        array with transitions block.
    gt : array, optional
        array with ground truth sides (None)
    nch_mask : array, optional
        array to further filter trials (None)
    plot_tr_mat : bool, optional
        whether to plot transition matrices (False)
    ev_perc : int, optional
        percetile of evidence used to filter trials bias calculation (5)
    plt_mask : bool, optional (for debugging)
        boolean indicating whether to plot mask with choices, perf.. (False)

    Returns
    -------
    biases : array
        array with bias values.

    """
    min_num_samps = 100
    blocks, counts = np.unique(tr_block[nch_mask], return_counts=True)
    blocks = blocks[counts > min_num_samps]
    b_args = {'choice': ch, 'plot': plot_tr_mat}
    # select evidence below ev_perc quantile
    ev_mask = evidence_mask(ev, percentage=ev_perc)
    # get rid of first value in evidence
    ev_mask = conc((ev_mask[1:], np.array([0])))
    if gt is not None:
        b_args.update({'gt': gt})
    num_blocks = blocks.shape[0]
    biases = np.empty((2, num_blocks))
    biases[:] = np.nan
    if nch_mask is None:
        nch_mask = np.ones_like(perf) == 1
    nch_mask = remove_borders(nch_mask)
    # After error and after correct bias for each block.
    for ind_perf in range(2):
        counter = 0
        for ind_blk, blk in enumerate(blocks):
            tr_block_mask = tr_block == blk
            tr_block_mask = remove_borders(tr_block_mask)
            mask = and_.reduce((perf == ind_perf, c_tr == 0, ev_mask, nch_mask,
                                tr_block_mask))
            mask = conc((np.array([False]), mask[:-1]))
            if ind_perf == 1 and ind_blk == 1 and plt_masks:
                plot_masks_cond(ch=ch, perf=perf, mask=mask, c_tr=c_tr)
            if np.sum(mask) > min_num_samps:
                counter += 1
                popt, pcov = bias_entropy(mask=mask.copy(), **b_args)
                if b_args['plot']:
                    plt.title('After '+str(ind_perf)+' blck: '+str(blk))
            else:
                popt = [np.nan, np.nan]
            # we just care about second element of popt, which is the bias.
            biases[ind_perf, ind_blk] = popt[1]
    return biases


def load_behavioral_data(file):
    """
    Load and organize behavioral data.

    Parameters
    ----------
    file : str
        path to file.

    Returns
    -------
    data : dict
        keys: 'choice', 'gt', 'performance',
        'signed_evidence' (for 2AFC, difference between the 2 stimuli),
        'evidence' (ground truth signal - mean of signals of other stims),
        'catch_trial', 'reward', 'stimulus', 'nch' (number of choices),
        'tr_block' (transition block), 'sel_chs' (effective choices)

    """
    data = np.load(file, allow_pickle=1)
    choice = data['choice'].flatten()
    stimulus = data['stimulus']
    gt = data['gt'].flatten()
    signed_evidence = stimulus[:, 1] - stimulus[:, 2]
    non_rew_stims = stimulus.copy()
    # make rewarded stim = nan (gt goes from 1 to 4, but 1st dim is fixation cue)
    non_rew_stims[np.arange(non_rew_stims.shape[0]), gt] = np.nan
    evidence = np.abs(stimulus[np.arange(non_rew_stims.shape[0]), gt] -
                      np.nanmean(non_rew_stims[:, np.unique(gt)], axis=1))
    if 'coh' in data.keys():
        putative_ev = data['coh']*(-1)**(gt == 2)
        # plt.figure()
        # plt.plot(putative_ev, signed_evidence, '.')
        # putative_ev = data['coh']*(-1)**(choice == 1)
        # plt.figure()
        # plt.plot(putative_ev, signed_evidence, '.')
        # asdsda
    else:
        putative_ev = signed_evidence
        print('xxxxxxxxxxxxxxxxxx')
        print('Could not load putative evidence')
        print('xxxxxxxxxxxxxxxxxx')
    if 'catch_trial' in data.keys():
        catch_trial = data['catch_trial']
        if len(catch_trial) == 0:
            catch_trial = np.zeros((choice.shape[0],))
    else:
        catch_trial = np.zeros((choice.shape[0],))
    performance = choice == gt
    if 'reward' in data.keys():
        reward = data['reward']
    else:
        reward = performance
    if 'nch' in data.keys():
        nch = data['nch']
    else:
        nch = np.ones_like(choice)*np.max(gt)

    if 'sel_chs' in data.keys():
        sel_chs = data['sel_chs']
    else:
        sel_chs = np.zeros_like(choice)
    if 'curr_block' in data.keys():
        tr_block = data['curr_block']
        if isinstance(tr_block[0], str) and tr_block[0].find('-') != -1:
            p = re.compile('[1-9]')
            std_2afc = 1*np.array([(x.startswith('2-1') or x.startswith('1-2'))
                                   and len(re.findall(p, x[3:])) == 0
                                   for x in tr_block])
    else:
        tr_block = np.zeros_like(choice)
    data = {'choice': choice, 'gt': gt,
            'performance': performance, 'signed_evidence': signed_evidence,
            'evidence': evidence, 'catch_trial': catch_trial, 'reward': reward,
            'stimulus': stimulus, 'nch': nch, 'tr_block': tr_block,
            'sel_chs': sel_chs, 'putative_ev': putative_ev}
    if 'std_2afc' in locals():
        data['std_2afc'] = std_2afc
    return data


def bias_across_training(data, per=100000, step=None, ref_distr=False,
                         bias_type='entropy', conv_w=3, plt_psych_curves=False):
    """
    Return bias and performance matrix for nch block passed.

    Parameters
    ----------
    data : dict
        dictionary containing behavioral data.
    per : int, optional
        window use to compute bias (100000)
    step : int, optional
        step to 'slide' the window. If None it will be set equal to per (None)
    ref_distr : bool, optional
        whether to use the ground truth transition distr. as reference (False)
    bias_type : str, optional
        type of bias to compute ('entropy')
    conv_w : int, optional
        window to use to compute psych bias (3)
    plt_psych_curves : bool, optional
        whether to plot psych curves (False)

    Returns
    -------
    bias_mat : array
        steps x outcome x number of blocks if bias_type == 'entropy'
        steps x outcome x 4 if bias_type == 'psycho'
        (4 --> [alt cont + alt, rep cont + rep, alt cont + rep, rep cont + alt])
    perf_mat : array
        array containing the performance of the network in each period.

    """
    choice = data['choice']
    if ref_distr:
        ground_truth = data['gt']
    evidence = data['evidence']
    signed_evidence = data['signed_evidence']
    performance = data['performance']
    catch_trials = data['catch_trial']
    tr_block = data['tr_block']
    if 'std_2afc' in data.keys() and bias_type == 'psycho':
        nch_blck_mask = data['std_2afc'] == 1
        num_blocks = 4
    elif bias_type == 'psycho':
        nch_blck_mask = np.ones_like(data['choice']) == 1
        num_blocks = 4
    else:
        # no mask
        nch_blck_mask = np.ones_like(data['choice']) == 1
        # get only the tr-blocks that overlap with the nch-blck
        num_blocks = np.unique(tr_block[nch_blck_mask]).shape[0]
    if per is None:
        per = choice.shape[0]
    if step is None:
        step = per
    # Bias only computed for selected steps
    steps = get_times(choice.shape[0], per, step)
    bias_mat = np.empty((len(steps), 2, num_blocks))
    perf_mat = np.empty((len(steps)))
    bias_mat[:] = np.nan
    perf_mat[:] = np.nan
    for ind, ind_per in enumerate(steps):
        ev = evidence[ind_per:ind_per+per+1]
        sig_ev = signed_evidence[ind_per:ind_per+per+1]
        prf = performance[ind_per:ind_per+per+1]
        ch = choice[ind_per:ind_per+per+1]
        c_tr = catch_trials[ind_per:ind_per+per+1]
        nch_mask = nch_blck_mask[ind_per:ind_per+per+1]
        tr_blk = tr_block[ind_per:ind_per+per+1]
        gt = ground_truth[ind_per:ind_per+per+1] if ref_distr else None
        if bias_type == 'entropy':
            biases = compute_bias_entropy(ch=ch, gt=gt, ev=ev, perf=prf, c_tr=c_tr,
                                          tr_block=tr_blk, nch_mask=nch_mask,
                                          plot_tr_mat=False)  # ind_per==steps[-1]
        elif bias_type == 'psycho':
            figs_flag = ind_per == steps[-1] and plt_psych_curves
            biases = compute_bias_psycho(ch=ch, ev=sig_ev, perf=prf, mask=nch_mask,
                                         conv_w=conv_w, figs=figs_flag,
                                         new_fig=figs_flag)
        elif bias_type == 'WL_rep_prob':
            figs_flag = ind_per == steps[-1] and plt_psych_curves
            biases = compute_winLose_rep_prob(ch=ch, ev=sig_ev, perf=prf,
                                              mask=nch_mask, figs=figs_flag,
                                              new_fig=figs_flag)

        perf_mat[ind] = np.mean(prf)
        # count the number of new trials (bias is computed with a sliding
        # window and different periods overlap)
        for ind_perf in range(2):
            for ind_blk in range(biases.shape[1]):
                bias_mat[ind, ind_perf, ind_blk] = biases[ind_perf, ind_blk]

    return bias_mat, perf_mat


def get_GLM_regressors(data, chck_corr=False):
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
    ev = data['signed_evidence']
    perf = data['performance'].astype(float)
    ch = data['choice'].astype(float)
    # discard (make nan) non-standard-2afc task periods
    if 'std_2afc' in data.keys():
        std_2afc = data['std_2afc']
    else:
        std_2afc = np.ones_like(ch)
    inv_choice = and_(ch != 1., ch != 2.)
    nan_indx = np.logical_or.reduce((std_2afc == 0, inv_choice))
    ev[nan_indx] = np.nan
    perf[nan_indx] = np.nan
    ch[nan_indx] = np.nan
    ch = -(ch-2)  # choices should belong to {0, 1}
    prev_perf = ~ (conc((np.array([True]), data['performance'][:-1])) == 1)
    prev_perf = prev_perf.astype('int')
    prevprev_perf = (conc((np.array([False]), prev_perf[:-1])) == 1)
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
    df = {'origidx': np.arange(ch.shape[0]),
          'R_response': ch,
          'hit': perf,
          'evidence': ev,
          'aftererror': prev_perf,
          'rep_response': rep_ch_,
          'prevprev_perf': prevprev_perf}
    df = pd.DataFrame(df)

    # Lateral module
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


def weights_across_training(data, per, step):
    """
    Compute GLM weights across training.

    Parameters
    ----------
    data : dict
        dictionary containing behavioral data.
    per : int, optional
         window use to compute bias
    step : int, optional
        step to 'slide' the window. If None it will be set equal to per

    Returns
    -------
    weights_ac : array
        array containing weights across training for GLM fit to after correct trial
    weights_ae : array
        array containing weights across training for GLM fit to after error trial

    """
    df = get_GLM_regressors(data)
    # condition on num-ch block
    if per is None:
        per = df.shape[0]
    steps = get_times(df.shape[0], per, step)
    weights_ac = []
    weights_ae = []
    for ind, ind_per in enumerate(steps):
        indx = and_(df.origidx.values >= ind_per,
                    df.origidx.values < ind_per+per)
        df_tmp = df[indx]
        Lreg_ac, Lreg_ae = glm(df_tmp)
        if Lreg_ac is not None:
            weights_ac.append(Lreg_ac.coef_)
            weights_ae.append(Lreg_ae.coef_)
        else:
            weights_ac.append(np.ones((1, len(afterc_cols)))*np.nan)
            weights_ae.append(np.ones((1, len(aftere_cols)))*np.nan)
    weights_ac = np.asarray(weights_ac)
    weights_ae = np.asarray(weights_ae)

    return weights_ac, weights_ae


def glm(df):
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
    not_nan_indx = df['R_response'].notna()
    X_df_ac, y_df_ac =\
        df.loc[(df.aftererror == 0) & not_nan_indx,
               afterc_cols].fillna(value=0),\
        df.loc[(df.aftererror == 0) & not_nan_indx, 'R_response']
    X_df_ae, y_df_ae =\
        df.loc[(df.aftererror == 1) & not_nan_indx,
               aftere_cols].fillna(value=0),\
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


def glm_dc(df, prv_outs, after_cols):
    """
    Compute GLM weights for data in df conditioned on 2 previous outcome.

    Parameters
    ----------
    df : dataframe
        dataframe containing regressors and response.
    prv_outs : list
        previous outcomes.
    after_cols : list
        list specifying regressors to use.

    Returns
    -------
    Lreg : LogisticRegression model
        logistic model fit to trials conditioned on 2 previous ouctomes.

    """
    not_nan_indx = df['R_response'].notna()
    X_df, y_df =\
        df.loc[(df.aftererror == prv_outs[1]) &
               (df.prevprev_perf == prv_outs[0]) &
               not_nan_indx, after_cols].fillna(value=0),\
        df.loc[(df.aftererror == prv_outs[1]) &
               (df.prevprev_perf == prv_outs[0]) &
               not_nan_indx, 'R_response']

    if len(np.unique(y_df.values)) == 2:
        Lreg = LogisticRegression(C=1, fit_intercept=False, penalty='l2',
                                  solver='saga', random_state=123,
                                  max_iter=10000000, n_jobs=-1)
        Lreg.fit(X_df.values, y_df.values)
    else:
        Lreg = None

    return Lreg


def glm_double_cond(df):
    """
    Compute GLM weights for data in df conditioned on 2 previous outcome.

    Parameters
    ----------
    df : dataframe
        dataframe containing regressors and response.

    Returns
    -------
    Lregacc/ee/ec/ce : LogisticRegression model
        logistic model fit to trials conditioned on 2 previous ouctomes.

    """
    Lreg_acc = glm_dc(df=df, prv_outs=[0, 0], after_cols=aftercc_cols)
    Lreg_aee = glm_dc(df=df, prv_outs=[1, 1], after_cols=afteree_cols)
    Lreg_ace = glm_dc(df=df, prv_outs=[0, 1], after_cols=afterce_cols)
    Lreg_aec = glm_dc(df=df, prv_outs=[1, 0], after_cols=afterec_cols)
    return Lreg_acc, Lreg_aee, Lreg_ace, Lreg_aec


def get_bias_single_exp(data, acr_tr_per=200000, step=40000, ref_distr=False,
                        bias_type='entropy'):
    """
    Compute bias for behavioral data provided in data.

    Parameters
    ----------
    data : dict
        dictionary containing the behavioral data.
    acr_tr_per : int, optional
         window use to compute bias (200000)
    step : int, optional
        step to 'slide' the window. If None it will be set equal to per (None)
    ref_distr : bool, optional
        whether to use the ground truth transition distr. as reference (False)
    bias_type : str, optional
        type of bias to compute ('entropy')

    Returns
    -------
    bool
        True if there is enough data to compute bias.
    bias_mat : array
        steps x outcome x number of blocks if bias_type == 'entropy'
        steps x outcome x 4 if bias_type == 'psycho'
        (4 --> [alt cont + alt, rep cont + rep, alt cont + rep, rep cont + alt])
    perf_mat : array
        array containing the performance of the network in each period.

    """
    if acr_tr_per is None or data['choice'].shape[0] >= acr_tr_per:
        bias_mat, perf_mat =\
            bias_across_training(data, per=acr_tr_per, step=step,
                                 bias_type=bias_type, ref_distr=ref_distr)
        return True, bias_mat, perf_mat
    else:
        print('Not enough data')
        return False, [], []


def get_glm_ws_single_exp(data, acr_tr_per=200000, step=40000):
    """
    Compute GLM weights for behavioral data in data.

    Parameters
    ----------
    data : dict
        dictionary containing the behavioral data.
    acr_tr_per : int, optional
         window use to apply GLM (200000)
    step : int, optional
        step to 'slide' the window. If None it will be set equal to per (None)

    Returns
    -------
    bool
        True if there is enough data to compute bias.
    weights_ac : array
        array containing weights across training for GLM fit to after correct trial
    weights_ae : array
        array containing weights across training for GLM fit to after error trial

    """
    if acr_tr_per is None or data['choice'].shape[0] >= acr_tr_per:
        weights_ac, weights_ae =\
            weights_across_training(data, per=acr_tr_per, step=step)
        return True, weights_ac, weights_ae
    else:
        print('Not enough data')
        return False, [], []


def get_rebound_single_exp(data, min_num_tr=200000, period=-2000000, rep_alt=1,
                           cond_nch_blck=2, num_trials_seq=6, after_corr=-1):
    """
    Compute bias for rep/alt sequence.

    Parameters
    ----------
    data : dict
        dictionary containing the behavioral data.
    min_num_tr : int, optional
        minimum number of trials to do the analysis (200000)
    period : int, optional
        period from the last trial to use for the analysis (-2000000)
    rep_alt : int, optional
        whether to compute the bias for a rep (1) or alt (-1) sequence (1)
    cond_nch_blck : int, optional
        condition the trials to have cond_nch_blck choices (2)
    num_trials_seq : int, optional
        number of trials in the sequence (6)
    after_corr : int, optional
        sequence with correct (1) or error (-1) in the second to last trials (-1)

    Returns
    -------
    flag: bool
        indicates whether the analysis was succesful.
    biases: list
        biases for all sequences lengths until num_trials_seq.
    bss: array
        biases for a sequence of length of num_trials_seq.
    num_samples: array
        number of samples for all sequences lengths until num_trials_seq.

    """
    if min_num_tr is None or data['choice'].shape[0] >= min_num_tr:
        choice = data['choice'][period:]
        evidence = data['signed_evidence'][period:]
        performance = data['performance'][period:]
        nch_blck_mask = data['nch'][period:] == cond_nch_blck
        tr_hist_mask = None  # data['tr_block'][period:] == 1
        biases = []
        num_samples = []
        for ind_n_tr_seq in range(1, num_trials_seq+1):
            whole_seq = True  # ind_n_tr_seq == num_trials_seq
            template_rep = np.array([0]*2+[rep_alt]*ind_n_tr_seq+[0, 0])
            # after error
            template_perf =\
                np.array([-1, 1] + [1]*ind_n_tr_seq + [after_corr] + [1])
            bss, ns = compute_AE_rebound(ch=choice, ev=evidence,
                                         perf=performance,
                                         nch_mask=nch_blck_mask,
                                         tr_hist_mask=tr_hist_mask,
                                         bias_type='psycho',
                                         template_perf=template_perf,
                                         template_rep=template_rep,
                                         whole_seq=whole_seq)
            biases.append(np.array(bss))
            num_samples.append(np.array(ns))
        return True, biases, bss, num_samples
    else:
        print('Not enough data')
        return False, [], [], []


def get_perf_single_exp(data, window=10000, spacing=10000, margin=.05,
                        coh0_w=10000, binning=300000, cond_blcks=False):
    """
    Compute smoothed performance across training.

    Parameters
    ----------
    data : dict
        dictionary containing the behavioral data.
    window : int, optional
        window to use for smoothing the performance (10000)
    spacing : int, optional
        spacing to subsample the performance vector (10000)

    Returns
    -------
    perf : array
        smoothed and subsampled performance.

    """
    # performance ideal observer (perfect integrator)
    n_ch = np.max(data['gt'])
    cohs = data['putative_ev']
    stim = data['stimulus'][:, 1:n_ch+1]
    ch_io = np.argmax(stim, axis=1)+1
    perf_io = np.mean(ch_io == data['gt'])
    print('Performance perfect integrator: ', perf_io)
    perf = data['performance']
    perf_conv = np.convolve(perf, np.ones((window,))/window, mode='valid')
    indx = np.linspace(0, perf_conv.shape[0]-1, (perf_conv.shape[0]-1)//spacing)
    perf_conv = perf_conv[indx.astype('int')]
    perf_coh = data['performance'].copy().astype('float')
    coh0_indx = np.where(np.abs(cohs) < 0.00001)[0]
    perf_coh = perf_coh[coh0_indx]
    perf_coh = np.convolve(perf_coh, np.ones((coh0_w,))/coh0_w, mode='same')
    aha_moment = np.where(perf_coh > (1/n_ch+margin))[0]
    aha_moment = coh0_indx[aha_moment[0]] if len(aha_moment) > 0 else None

    # performances conditioned on block and num. choices
    tr_block = data['tr_block']
    n_ch_mat = data['nch']
    perfs_cond = {}
    for n in range(2, n_ch+1):
        indx_n = np.where(n_ch_mat == n)[0]
        if cond_blcks:
            starts = ['1-', '2-', str(n)+'-'] if n > 2 else ['1-', '2-']
        else:
            starts = ['']
        for i_s, strt in enumerate(starts):
            indx_blk = np.array([i for i, x in zip(range(len(tr_block)), tr_block)
                                 if x.startswith(strt)])
            indx = np.intersect1d(np.intersect1d(indx_n, indx_blk), coh0_indx)
            bins = np.linspace(0, cohs.shape[0], (cohs.shape[0])//binning+1)
            xss_bin = np.searchsorted(bins, indx)
            xss_unq = np.unique(xss_bin)
            mean_perf =\
                np.array([np.nanmean(perf[indx[xss_bin == x]]) for x in xss_unq])
            std_perf = [np.nanstd(perf[indx[xss_bin == x]]) /
                        np.sqrt(np.sum(xss_bin == x)) for x in xss_unq]
            mean_indx = [np.nanmean(indx[xss_bin == x]) for x in xss_unq]
            perfs_cond[str(n)+'-'+strt[:-1]] = {}
            perfs_cond[str(n)+'-'+strt[:-1]]['m_perf'] = mean_perf-1/n
            perfs_cond[str(n)+'-'+strt[:-1]]['std_perf'] = std_perf
            perfs_cond[str(n)+'-'+strt[:-1]]['indx'] = mean_indx
    # plt.figure()
    # plt.plot(np.arange(len(perf))*spacing, perf)
    # plt.plot(coh0_indx, perf_coh)
    # plt.plot([aha_moment, aha_moment], [0, 1], '--')
    # plt.figure()
    # plt.plot(np.arange(len(perf))*spacing-aha_moment, perf)
    return perf_conv, perf_io, aha_moment, perfs_cond


def get_num_tr_single_exp(mat, per, step):
    """
    Compute number of trials fulfilling a certain condition as indicated by mat.

    Parameters
    ----------
    mat : array
        mask indicating which trials fulfil a given condition.
    per : int, optional
         window use to compute number of samples
    step : int, optional
        step to 'slide' the window.

    Returns
    -------
    num_trs : array
        array indicating the number of trials fulfilling the condition indicated
        by mat for each period step.

    """
    print('Proportion of 2AFC trials: ', np.sum(mat)/len(mat))
    steps = get_times(mat.shape[0], per, step)
    num_trs = np.empty((len(steps)))
    for ind, ind_per in enumerate(steps):
        num_trs[ind] = np.sum(mat[ind_per:ind_per+per+1])
    return num_trs


def get_glm_double_cond_single_exp(data, per=500000):
    """
    Compute number of trials fulfilling a certain condition as indicated by mat.

    Parameters
    ----------
    data : dict
        dictionary containing the behavioral data.
    per : int, optional
         window use to apply GLM.

    Returns
    -------
    weights_acc/ee/ac/ce : array
        array containing weights across training for GLM fit conditioned on 2
        previous outcomes

    """
    df = get_GLM_regressors(data, chck_corr=False)
    if per is None:
        per = df.shape[0]
    indx = df.origidx.values >= df.shape[0]-per
    df_tmp = df[indx]
    Lreg_acc, Lreg_aee, Lreg_ace, Lreg_aec = glm_double_cond(df_tmp)
    weights_acc = Lreg_acc.coef_ if Lreg_acc is not None else []
    weights_aee = Lreg_aee.coef_ if Lreg_aee is not None else []
    weights_ace = Lreg_ace.coef_ if Lreg_ace is not None else []
    weights_aec = Lreg_aec.coef_ if Lreg_aec is not None else []

    return weights_acc, weights_aee, weights_ace, weights_aec


def get_trans_mats_after_seq(data, n_ch=6, ch_gt='choice', perf_ctxt=[1, 1, 1],
                             sv_folder=''):
    all_contexts = {}
    trans_ctxt_mat = [[np.nan, 1, 1], [np.nan, -1, -1], [np.nan, 0, 0]]
    t_cxt_nm = ['cw', 'ccw', 'rep']
    for tc, tc_nm in zip(trans_ctxt_mat, t_cxt_nm):
        contexts = {tc_nm+'_rbnd': {'perf_seq': perf_ctxt+[-1, 1],
                                    'trans_seq': tc+[np.nan, np.nan]},
                    tc_nm+'_to_cw': {'perf_seq': perf_ctxt+[1, 1],
                                     'trans_seq': tc+[1, 1]},
                    tc_nm+'_to_ccw': {'perf_seq': perf_ctxt+[1, 1],
                                      'trans_seq': tc+[-1, -1]},
                    tc_nm+'_to_rep': {'perf_seq': perf_ctxt+[1, 1],
                                      'trans_seq': tc+[0, 0]},
                    tc_nm+'_neutr_trans': {'perf_seq': perf_ctxt+[1, 1],
                                           'trans_seq': tc+[2, tc[-1]]}}
        for k in contexts.keys():
            seq_props = {'templ_perf': contexts[k]['perf_seq'],
                         'templ_trans': np.array(contexts[k]['trans_seq']),
                         'start': 0}
            tr_mats, al_tr_mats, num_s_mat, perf_mat = \
                get_trans_prob_mats_after_seq(data=data, n_ch=n_ch, ch_gt=ch_gt,
                                              seq_prps=seq_props)
            trans_name = ''.join(['_'+str(x) for x in seq_props['templ_trans']])
            name = ch_gt+'_trans_mat_'+str(n_ch)+trans_name
            pf.plot_trans_prob_mats_after_error(trans_mats=tr_mats,
                                                al_trans_mats=al_tr_mats,
                                                num_samples_mat=num_s_mat,
                                                perf_mat=perf_mat, n_ch=n_ch,
                                                sv_folder=sv_folder, name=name)
            contexts[k]['tr_mats'] = tr_mats
            contexts[k]['al_tr_mats'] = al_tr_mats
            contexts[k]['num_s_mat'] = num_s_mat
            contexts[k]['perf_mat'] = perf_mat
        all_contexts.update(contexts)
    return all_contexts


if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) == 1:
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'var_nch_predef_mats_larger_nets/'  # 'bernstein_shorter_rollout/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'variable_nch_predef_tr_mats/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'pre_train_tr_history_agent/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'bernstein_persistence/'
        main_folder = '/home/molano/priors/AnnaKarenina_experiments/sims_21/'
        main_folder = '/home/manuel/priors_analysis/annaK/sims_21/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'optimal_observer/'
    elif len(sys.argv) == 2:
        main_folder = sys.argv[1]
    opts = {'train': {'glm': False, 'bias': False,
                      'glm_rbnd': False, 'plot_figs': True,
                      'performances': True, 'trans_mats': False,
                      'step': 100000, 'per': 200000, 'binning': 20000,
                      'reload': True},
            'test_2AFC': {'glm': True, 'bias': False,
                          'glm_rbnd': False, 'plot_figs': True,
                          'performances': True, 'trans_mats': False,
                          'step': 100000, 'per': 200000, 'binning': 300000,
                          'reload': True},
            'test_nch_6': {'glm': False, 'trans_mats': True, 'bias': False,
                           'n-GLM': False, 'glm_rbnd': False, 'plot_figs': True,
                           'step': 100000, 'per': 200000, 'binning': 300000,
                           'reload': True},
            'test_2AFC_all': {'glm': True, 'trans_mats': False, 'bias': False,
                              'n-GLM': False, 'glm_rbnd': True, 'plot_figs': True,
                              'step': 200000, 'per': 300000, 'binning': 300000,
                              'reload': True}}
    analysis_opts = {'analyze': ['train'], 'opts': opts}
    name = 'bin_20K'  # 'sims_21_n_ch_16'
    # extra_folder = {'main_folder': '/gpfs/projects/hcli64/molano/anna_k/sims_21/',
    #                'cond': 'ACER*n_ch_16*', 'val': 'n_ch', 'val_stord': 0.01}
    extra_folder = {}
    batch_analysis(main_folder, conds=['ACER*n_ch_2*'], val='n_ch',
                   extra_folder=extra_folder, name=name, **analysis_opts)


# import helper_functions as hf
# import matplotlib.pyplot as plt
# from neurogym.utils import plotting as pl
# import numpy as np

# plt.close('all')
# data = pl.put_together_files('/home/molano/priors/AnnaKarenina_experiments/' +
#                              'pre_train_tr_history_agent/' +
#                              'alg_A2C_seed_38_n_ch_8/')
# f, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
# ax[0].plot(0.5*np.ones_like(data['reward']), '--k')
# ax[0].plot(np.convolve(data['performance'], np.ones((100,))/100, mode='valid'),
#            label='perf. (w=100)')
# ax[0].plot(np.convolve(data['performance'], np.ones((5000,))/5000, mode='valid'),
#            label='perf. (w=5000)')
# ax[0].plot(np.convolve(data['reward'], np.ones((5000,))/5000, mode='valid'),
#            label='reward (w=5000)')
# for k in data.keys():
#     if k.find('mean_perf_') != -1:
#         ax[0].plot(data[k], label=k)
# tr_blck_procssd = [x.replace('-0', '') for x in data['curr_block']]
# tr_blck_procssd = np.array([len(x.replace('-', ''))-x.startswith('0')
#                             for x in tr_blck_procssd])
# num_choices = [len(x) for x in data['curr_block']]
# ax[0].plot(1/tr_blck_procssd, '--c', label='chance level')
# ax[0].legend(loc='upper left')

# ax[1].plot(data['gt'], '-+', label='gt')
# ax[1].plot(data['choice'], '-+', label='choice')
# ax[1].plot(data['phase'], '-+', label='phase')
# ax[1].legend(loc='upper left')
# print('\nCorrect choices')
# print(np.unique(data['gt'], return_counts=1))

# print('\nSelected choices')
# print(np.unique(data['choice'], return_counts=1))
# # blck_indx = [int(blk.replace('-', '')) for blk in data['tr_block']]
# print('\nTransition blocks')
# blks, counts = np.unique(data['curr_block'], return_counts=1)
# tr_blk = [float(x.replace('-', '')) for x in data['curr_block']]
# ax[2].plot(np.log10(tr_blk))
# print(blks)
# print(counts/np.sum(counts))

# # chs_indx = [int(blk.replace('-', '')) for blk in data['sel_chs']]
# print('\nSelected choices')
# blks, counts = np.unique(data['sel_chs'], return_counts=1)
# print(blks)
# print(counts/np.sum(counts))
# print(np.sum(data['curr_block'] == '2-1-0-0-0-0-0-0'))
# print(np.sum(data['curr_block'] == '1-2-0-0-0-0-0-0'))
# f, ax = plt.subplots(nrows=8, ncols=1, sharex=True)
# prev_choice = np.concatenate((np.array([0]), data['choice'][:-1]))
# for ch in range(8):
#     stim_cond = data['stimulus'][prev_choice == ch+1, :]
#     ax[ch].imshow(stim_cond[-10:, 1:9].T, aspect='auto')


# data = np.load('/home/molano/priors/AnnaKarenina_experiments/' +
#                 'var_nch_predef_mats_long_stim/data_ACER__.npz',
#                 allow_pickle=1)
# indx = np.array([x.find('n_ch_16') != -1 for x in data['files']])
# biases = data['bias_mats_psych'][indx]
# files = data['files'][indx]
# perfs = data['perf_mats'][indx]
# indx_empty = [len(x) != 0 for x in biases]
# biases = biases[indx_empty]
# files = files[indx_empty]
# perfs = perfs[indx_empty]
# print(files)
# print([np.mean(x[-10:]) for x in perfs])
# [np.sum(np.abs(x[-1, 0, :2]))/np.sum(np.abs(x[-1, 1, :2])) for x in biases]
