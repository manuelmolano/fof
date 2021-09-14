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
from copy import deepcopy as dpcp
import plot_pshts as pp
import utils as ut
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')


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



def get_cond_trials(b_data, e_data, ev, cl, evs_mrgn=1e-2, plot=False,
                    fixtn_time=.3, margin_psth=100, std_conv=20):
    # get spikes
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    # select trials
    filt_evs, indx_good_evs = pp.preprocess_events(b_data=b_data, e_data=e_data,
                                                   ev=ev, evs_mrgn=evs_mrgn,
                                                   fixtn_time=fixtn_time)
    ch_mat = b_data['R_response'].values
    prev_ch_mat = b_data['R_response'].shift(periods=1).values
    outc_mat = b_data['hithistory'].values
    prev_outc_mat = b_data['hithistory'].shift(periods=1).values
    prev_tr_mat = b_data['rep_response'].shift(periods=1).values
    zt = get_transition_mat(1*b_data['rep_response'], conv_w=5)
    if plot:
        plot_masks_cond(ch=ch_mat, prev_repeat=prev_tr_mat, zt=zt, num=200,
                        start=0)
        import sys
        sys.exit()
    for i_tr in range (len(ch_mat)):
            psth_cnv, peri_ev = ut.convolve_psth(spk_times=spk_tms, events=evs,
                                         std=std_conv, margin=margin_psth)

        peri_ev = feats['peri_ev']


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
