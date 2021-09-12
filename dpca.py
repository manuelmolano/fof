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
# import utils as ut
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')


def get_cond_trials(b_data, e_data, ev, cl, conditions, evs_mrgn=1e-2,
                    fixtn_time=.3, margin_psth=1000, std_conv=20):
    # get spikes
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    # select trials
    filt_evs, indx_good_evs = pp.preprocess_events(b_data=b_data, e_data=e_data,
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
        feats = pp.plt_psths(spk_tms=spk_tms, evs=evs, std_conv=std_conv,
                             margin_psth=margin_psth, plot=False)
        peri_ev = feats['peri_ev']
        idx = [np.arange(len(peri_ev))]+[case[i] for i in active_idx]
        if len(peri_ev) > 0:
            trialR[idx] = peri_ev
        min_num_tr = min(min_num_tr, len(peri_ev))
    return trialR, min_num_tr


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
    analysis_type = 'dpca'
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
        cond = {'ch': True, 'prev_ch': True, 'outc': False, 'prev_outc': True,
                'prev_tr': False}
        lbls_cps = 'cprt'
        compute_dPCA(inv=inv, main_folder=main_folder, std_conv=std_conv,
                     margin_psth=margin_psth, sel_sess=sel_sess, sel_rats=sel_rats,
                     conditioning=cond, lbls_cps=lbls_cps, sv_folder=sv_folder)
    elif analysis_type == 'stats':
        units_stats(inv=inv, main_folder=main_folder, sv_folder=sv_folder)
