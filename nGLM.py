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


def plot_masks_cond(ch, prev_rep=None, trans=None, num=50, start=0):
    """
    Plot mask with choices, performance and other variables.

    Parameters
    ----------
    ch : array
        array with choices.
    prev_rep : array, optional
        array with previous repeatitions (1 if repeating, 0 if alternating) (None)
    zt: TYPE, optional
        array indicating the exponential sum of previous transitions (None)

    Returns
    -------
    None.

    """
    plt.subplots(figsize=(8, 8))
    plt.plot(ch[start:start+num], '-+', label='choice', lw=1)
    if prev_rep is not None:
        plt.plot(prev_rep[start:start+num], '-+', label='prev. repeat', lw=1)
        plt.plot(trans[start:start+num], '-+', label='transitions', lw=1)
    for ind in range(num):
        plt.plot([ind, ind], [-3, 3], '--', color=(.7, .7, .7))
    plt.legend()


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
    ev = data['coh'].values
    perf = data['hithistory'].values
    ch = data['R_response'].values
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
    prev_perf = ~ (np.concatenate((np.array([True]),
                                   data['performance'][:-1])) == 1)
    prev_perf = prev_perf.astype('int')
    prevprev_perf = (np.concatenate((np.array([False]), prev_perf[:-1])) == 1)
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

def get_cond_trials(b_data, e_data, ev, cl, evs_mrgn=1e-2, plot=False,
                    fixtn_time=.3, margin_psth=100, std_conv=20):
    df = get_GLM_regressors(b_data, chck_corr=False)
    # get spikes
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    # select trials
    evs, indx_good_evs = pp.preprocess_events(b_data=b_data, e_data=e_data,
                                              ev=ev, evs_mrgn=evs_mrgn,
                                              fixtn_time=fixtn_time)
    filt_evs = evs[indx_good_evs]
    ch_mat = b_data['R_response'].values[indx_good_evs]
    prev_ch_mat = b_data['R_response'].shift(periods=1).values[indx_good_evs]
    outc_mat = b_data['hithistory'].values[indx_good_evs]
    prev_outc_mat = b_data['hithistory'].shift(periods=1).values[indx_good_evs]
    prev_tr_mat = b_data['rep_response'].shift(periods=1).values[indx_good_evs]
    zt = get_transition_mat(1*b_data['rep_response'], conv_w=5)[indx_good_evs]
    if plot:
        plot_masks_cond(ch=ch_mat, prev_repeat=prev_tr_mat, zt=zt, num=200,
                        start=0)
        import sys
        sys.exit()
    resps = ut.scatter(spk_tms=spk_tms, evs=filt_evs, margin_psth=margin_psth,
                       plot=False)
    resps = [len(r) for r in resps['aligned_spks']]


def compute_nGLM(main_folder, sel_sess, sel_rats, inv, std_conv=20,
                 margin_psth=1000, sel_qlts=['mua', 'good'], conditioning={},
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
                            trR, min_n_tr =\
                                get_cond_trials(b_data=b_data, e_data=e_data,
                                                ev=ev, cl=cl, std_conv=std_conv,
                                                margin_psth=margin_psth)
                            if min_n_tr > 10:
                                all_trR.append(trR)
                                min_num_tr = min(min_num_tr, min_n_tr)
        name = ''.join([i[0]+str(i[1]) for i in conditioning.items()])
        # f.savefig(sv_folder+name+'.png')


if __name__ == '__main__':
    plt.close('all')
    analysis_type = 'dpca'
    std_conv = 50
    margin_psth = 1000
    home = 'manuel'
    main_folder = '/home/'+home+'/fof_data/'
    if home == 'manuel':
        sv_folder = main_folder+'/psths/'
    elif home == 'molano':
        sv_folder = '/home/molano/Dropbox/project_Barna/FOF_project/psths/'
    inv = np.load('/home/'+home+'/fof_data/sess_inv_extended.npz', allow_pickle=1)
    sel_rats = []  # ['LE113']  # 'LE101'
    sel_sess = []  # ['LE104_2021-06-02_13-14-24']  # ['LE104_2021-05-17_12-02-40']
    # ['LE77_2020-12-04_08-27-33']  # ['LE113_2021-06-05_12-38-09']

    compute_nGLM(inv=inv, main_folder=main_folder, std_conv=std_conv,
                 margin_psth=margin_psth, sel_sess=sel_sess, sel_rats=sel_rats,
                 sv_folder=sv_folder)
