#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:37:33 2022

@author: molano
"""

import glob
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import utils_fof as ut
rojo = np.array((228, 26, 28))/255
azul = np.array((55, 126, 184))/255
verde = np.array((77, 175, 74))/255
morado = np.array((152, 78, 163))/255


def psth(file, ev_algmt='S', pre_post=[-2, 2], w=0.1):
    """

    Parameters
    ----------
    file : str
        where to get the data.

    Returns
    -------
    None.
    bhv_ss:
     HitHistoryList: [348×1 double]
     CoherenceList: [348×1 double]
      CoherenceVec: [-1 -0.4816 -0.2282 0 0.2282 0.4816 1]
    RewardSideList: [348×1 double]
      StimulusList: [348×1 double]
       EnviroOrder: {348×1 cell}
      ParsedEvents: {348×1 cell}
     Cube_FrameCoh: [7×200×40 double]

     EventsTimesLabel
     {'CenterLedOn','PreStimDelay','Stimulus','MovementTime','OutcomeBegin' ,
      'OutcomeEnd','TimeOutBegin','EarlyWithdrawalBegin','EarlyWithdrawalEnd'};
     EventsTimesLabel =  {'LOn','P_S','S','MT','O_B' ,'O_E', 'TO','E_B','E_E'};

    """
    mat = loadmat(file)
    bhv_ss = mat['bhv_ss'][0][0]
    # perf = bhv_ss[0]
    # stim_vals = bhv_ss[2][0, :]
    # coh = stim_vals[bhv_ss[1]-1]
    # gt = bhv_ss[3]
    # stim = bhv_ss[4]
    blk = bhv_ss[5][:, 0]
    ctx = np.array(['']*len(blk))
    ctx[blk == 'Switching'] = '2'
    ctx[blk == 'Repetitive'] = '1'
    ev_times_lbl = mat['EventsTimesLabel'][0, :]
    all_evs_times = mat['ev_times_ss']
    spk_times = mat['spk_times_ss']
    ev_times = all_evs_times[:, ev_times_lbl == ev_algmt]
    units = np.unique(spk_times[:, 1])
    bins = np.linspace(pre_post[0], pre_post[1], int(np.diff(pre_post)/w)+1)
    _, ax = plt.subplots(nrows=5, ncols=6)
    ax = ax.flatten()
    for i_un, un in enumerate(units):
        spk_un = spk_times[spk_times[:, 1] == un, 0]
        algn_spks = np.array([spk_un - et for et in ev_times])
        algn_spks = algn_spks[np.logical_and(algn_spks > pre_post[0],
                                             algn_spks < pre_post[1])]
        # indx = np.searchsorted(bins, algn_spks)
        psth = np.histogram(algn_spks, bins)
        ax[i_un].plot(bins[:-1]+w/2, psth[0])
        # print(1)
    # evs = bhv_ss[7]
    # print(bhv_ss)


def insert_nans(mat, odd, filling=np.nan):
    if len(mat.shape) == 1:
        new_mat = np.array(2*len(mat)*[filling])
        if odd:
            new_mat[np.arange(0, 2*len(mat), 2)] = mat
        else:
            new_mat[np.arange(1, 2*len(mat), 2)] = mat
    else:
        new_mat = np.array([mat.shape[1]*[filling]]*2*mat.shape[0])
        if odd:
            new_mat[np.arange(0, 2*len(mat), 2), :] = mat
        else:
            new_mat[np.arange(1, 2*len(mat), 2), :] = mat
    return new_mat


def batch_dms_data(main_folder):
    rats = ['Patxi', 'Rat15', 'Rat31', 'Rat32', 'Rat7']
    for r in rats:
        print('xxxxxxxxxxxxxxxx')
        print(r)
        files = glob.glob(main_folder+'/'+r+'*mat')
        num_unts = []
        for f in files:
            # print('--------------------------')
            # print(f)
            data, units = get_dms_data(file=f)
            num_unts.append(len(units))
        num_unts = np.array(num_unts)
        print('-------------')
        print('Number of sessions')
        print(len(num_unts))
        print('Proportion of sessions with units')
        print(np.sum(num_unts != 0)/len(num_unts))
        print('Median number of units in sessions with units')
        print(np.median(num_unts[num_unts != 0]))
        print('-------------')


def get_dms_data(file, ev_algmt='S', pre_post=[-1, 0], w=0.1):
    """

    Parameters
    ----------
    file : str
        where to get the data.

    Returns
    -------
    None.
    bhv_ss:
     HitHistoryList: [348×1 double]
     CoherenceList: [348×1 double]
      CoherenceVec: [-1 -0.4816 -0.2282 0 0.2282 0.4816 1]
    RewardSideList: [348×1 double]
      StimulusList: [348×1 double]
       EnviroOrder: {348×1 cell}
      ParsedEvents: {348×1 cell}
     Cube_FrameCoh: [7×200×40 double]

     EventsTimesLabel
     {'CenterLedOn','PreStimDelay','Stimulus','MovementTime','OutcomeBegin' ,
      'OutcomeEnd','TimeOutBegin','EarlyWithdrawalBegin','EarlyWithdrawalEnd'};
     EventsTimesLabel =  {'LOn','P_S','S','MT','O_B' ,'O_E', 'TO','E_B','E_E'};

     Output:
             ctx = data['contexts']
             gt  = data['gt']
             choice=data['choice']
             eff_choice=data['prev_choice']
             rw  = data['reward']
             obsc = data['obscategory']
             dyns =data['states']
        stim_trials[idx] = {'stim_coh': obsc[ngt_tot[idx]+1:ngt_tot[idx]+2],
                            'ctx': ctxseq[ngt_tot[idx]+1],
                            'gt': gt[ngt_tot[idx+1]],
                            'resp': dyns[ngt_tot[idx]+1:ngt_tot[idx]+2, :],
                            'choice': eff_choice[ngt_tot[idx+1]+1],
                            'rw': rw[ngt_tot[idx+1]],
                            'start_end': np.array([igt+1, ngt_tot[idx+1]]),
                            }


    """
    mat = loadmat(file)
    bhv_ss = mat['bhv_ss'][0][0]
    # context
    blk = bhv_ss[5][:, 0]
    contexts = np.array(['']*len(blk))
    contexts[blk == 'Switching'] = '2'
    contexts[blk == 'Repetitive'] = '1'
    # ground truth
    gt = bhv_ss[3].astype(float)
    gt = gt.flatten()
    inv_gt = np.logical_and(gt != 1., gt != 2.)
    # print('gt values:')
    # print(np.unique(gt, return_counts=1))
    # reward
    reward = bhv_ss[0].astype(float)
    reward = reward.flatten()
    # print('reward values:')
    # print(np.unique(reward, return_counts=1))
    inv_rw = np.logical_and(reward != 1., reward != 0.)
    # choice
    choice = gt.copy().astype(float)
    choice[reward == 0] = np.abs(gt[reward == 0]-3)
    # print('choice values:')
    # print(np.unique(choice, return_counts=1))
    # print('-------------------')
    inv_ch = np.logical_and(choice != 1, choice != 2)
    prev_choice = np.insert(choice[:-1], 0, 0)
    # stim strength
    stim_vals = bhv_ss[2][0, :]
    coh_list = bhv_ss[1].flatten()
    obscategory = np.abs(stim_vals[coh_list-1])
    ev_times_lbl = mat['EventsTimesLabel'][0, :]
    all_evs_times = mat['ev_times_ss']
    spk_times = mat['spk_times_ss']
    ev_times = all_evs_times[:, ev_times_lbl == ev_algmt]
    data = {}
    if len(spk_times) > 0:
        units = np.unique(spk_times[:, 1])
        states = []
        for i_un, un in enumerate(units):
            spk_un = spk_times[spk_times[:, 1] == un, 0]
            algn_spks = np.array([spk_un - et for et in ev_times])
            algn_spks[np.logical_or(algn_spks < pre_post[0],
                                    algn_spks > pre_post[1])] = np.nan
            resp = np.sum(~np.isnan(algn_spks), axis=1)
            states.append(resp)
        states = np.array(states).T
        # evs = bhv_ss[7]
        data['choice'] = insert_nans(mat=choice, odd=False)
        data['stimulus'] = None
        indxs = np.floor(np.arange(2*len(contexts))/2).astype(int)
        contexts = contexts[indxs]
        # this is because of the way this info is retrieved in
        # transform_stim_trials_ctxtgt
        contexts = [[c] for c in contexts]
        data['contexts'] = contexts
        gt[np.logical_or.reduce((inv_gt, inv_rw, inv_ch))] = np.nan
        data['gt'] = insert_nans(mat=gt, odd=False, filling=-1)
        data['prev_choice'] = insert_nans(mat=prev_choice, odd=True)
        data['reward'] = insert_nans(mat=reward, odd=False)
        data['obscategory'] = insert_nans(mat=obscategory, odd=True)
        data['states'] = insert_nans(mat=states, odd=True)
        np.savez(file[:file.find('data_for_python.mat')-1], **data)
        # for k in data.keys():
        #     if data[k] is not None and k != 'states':
        #         print(k)
        #         print(data[k][:10])
        #         print(data[k][-10:])
    else:
        units = []
    return data, units


def batch_fof_data(inv, main_folder, pre_post=[-1000, 1000], sel_sess=[],
                   sel_rats=[], name='ch', sel_qlts=['good', 'mua'], evs_mrgn=1e-2,
                   fixtn_time=.3, ev='stim_ttl_strt', plot=True):
    rats = glob.glob(main_folder+'LE*')
    features = {'sign_mat': []}
    for r in rats:
        rat = os.path.basename(r)
        sessions = glob.glob(r+'/LE*')
        for sess in sessions:
            session = os.path.basename(sess)
            name = SV_FOLDER+session
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
                get_fof_data(sess=sess, e_data=e_data, sel_clstrs=sel_clstrs,
                             pre_post=pre_post, name=name, sel_qlts=sel_qlts,
                             evs_mrgn=evs_mrgn, fixtn_time=fixtn_time, ev=ev,
                             plot=plot)
                print('----')

                # asdasd
                # ut.append_features(features=features, new_data=feats)

    # np.savez(sv_folder+'/feats.npz', **features)
    return features


def get_fof_data(sess, e_data, sel_clstrs, pre_post=[-1000, 1000], name='',
                 sel_qlts=['good', 'mua'], evs_mrgn=1e-2, fixtn_time=.3,
                 ev='stim_ttl_strt', plot=True):
    """

    Parameters
    ----------
    file : str
        where to get the data.

    Returns
    -------
    None.
    bhv_ss:
     HitHistoryList: [348×1 double]
     CoherenceList: [348×1 double]
      CoherenceVec: [-1 -0.4816 -0.2282 0 0.2282 0.4816 1]
    RewardSideList: [348×1 double]
      StimulusList: [348×1 double]
       EnviroOrder: {348×1 cell}
      ParsedEvents: {348×1 cell}
     Cube_FrameCoh: [7×200×40 double]

     EventsTimesLabel
     {'CenterLedOn','PreStimDelay','Stimulus','MovementTime','OutcomeBegin' ,
      'OutcomeEnd','TimeOutBegin','EarlyWithdrawalBegin','EarlyWithdrawalEnd'};
     EventsTimesLabel =  {'LOn','P_S','S','MT','O_B' ,'O_E', 'TO','E_B','E_E'};

     Output:
             ctx = data['contexts']
             gt  = data['gt']
             choice=data['choice']
             eff_choice=data['prev_choice']
             rw  = data['reward']
             obsc = data['obscategory']
             dyns =data['states']
        stim_trials[idx] = {'stim_coh': obsc[ngt_tot[idx]+1:ngt_tot[idx]+2],
                            'ctx': ctxseq[ngt_tot[idx]+1],
                            'gt': gt[ngt_tot[idx+1]],
                            'resp': dyns[ngt_tot[idx]+1:ngt_tot[idx]+2, :],
                            'choice': eff_choice[ngt_tot[idx+1]+1],
                            'rw': rw[ngt_tot[idx+1]],
                            'start_end': np.array([igt+1, ngt_tot[idx+1]]),
                            }


    """

    b_file = sess+'/df_trials'
    b_data = pd.read_pickle(b_file)
    # context
    blk = b_data['prob_repeat'].values
    contexts = np.array(['']*len(blk))
    contexts[blk < 0.5] = '2'
    contexts[blk > 0.5] = '1'
    # ground truth
    gt = b_data['rewside'].values+1
    inv_gt = np.logical_and(gt != 1., gt != 2.)
    # reward
    reward = b_data['hithistory'].values
    inv_rw = np.logical_and(reward != 1., reward != 0.)
    # choice
    choice = b_data['R_response'].values+1
    inv_ch = np.logical_and(choice != 1, choice != 2)
    prev_choice = np.insert(choice[:-1], 0, 0)
    # stim strength
    obscategory = np.abs(b_data['coh'].values-0.5)
    # select trials
    filt_evs, indx_good_evs =\
        ut.preprocess_events(b_data=b_data, e_data=e_data,
                             ev=ev, evs_mrgn=evs_mrgn,
                             fixtn_time=fixtn_time)
    print('Number of valid trials:', np.sum(indx_good_evs))
    print('Number of trials:', len(indx_good_evs))
    # indx_valid =\
    #   np.logical_and.reduce((indx_good_evs, filt_evs > pre_post[0],
    #                          filt_evs < np.max(spk_tms)-pre_post[1]))
    # filt_evs = filt_evs[indx_good_evs]
    filt_evs = 1000*filt_evs
    indx_inv = np.logical_or.reduce((inv_gt, inv_rw, inv_ch, ~indx_good_evs))
    states = []
    for i_cl, cl in enumerate(sel_clstrs):
        cl_qlt = e_data['clstrs_qlt'][i_cl]
        print('Cluster ', cl)
        if cl_qlt in sel_qlts:
            f, ax, resp =\
                get_responses(filt_evs=filt_evs, e_data=e_data,
                              b_data=b_data,  # indx_valid=indx_good_evs,
                              cl=cl, pre_post=pre_post, plot=plot)
            states.append(resp)
            if plot:
                mean_fr = np.round(np.mean(resp), 2)
                print('Mean fr:'+str(mean_fr))
                print('-----')
                title = name+'_'+str(cl)+' FR:'+str(mean_fr)
                title = title[title.rfind('/')+1:]
                ax[0].set_title(title)
                f.savefig(name+'_'+str(cl))
                plt.close(f)
    states = np.array(states).T
    data = {}
    data['choice'] = insert_nans(mat=choice, odd=False)
    data['stimulus'] = None
    indxs = np.floor(np.arange(2*len(contexts))/2).astype(int)
    contexts = contexts[indxs]
    # this is because of the way this info is retrieved in
    # transform_stim_trials_ctxtgt
    contexts = [[c] for c in contexts]
    data['contexts'] = contexts
    gt[indx_inv] = np.nan
    data['gt'] = insert_nans(mat=gt, odd=False, filling=-1)
    data['prev_choice'] = insert_nans(mat=prev_choice, odd=True)
    data['reward'] = insert_nans(mat=reward, odd=False)
    data['obscategory'] = insert_nans(mat=obscategory, odd=True)
    data['states'] = insert_nans(mat=states, odd=True)
    np.savez(SV_FOLDER+os.path.basename(sess)+'.npz', **data)


def get_responses(filt_evs, e_data, b_data, cl, pre_post=[-1000, 0],
                  plot=False):
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    spk_tms = 1000*spk_tms.flatten()
    spk_tms = spk_tms.astype(int)

    algn_spks = np.array([spk_tms-x for x in filt_evs])
    if plot:
        f, ax = plot_psth(algn_spks=algn_spks.copy(), pre_post=pre_post,
                          behav_data=b_data)  # , indx_valid=indx_valid)
    else:
        f, ax = None, None
    algn_spks[np.logical_or(algn_spks < pre_post[0],
                            algn_spks > pre_post[1])] = np.nan
    resp = np.sum(~np.isnan(algn_spks), axis=1)
    # print(1)
    return f, ax, resp


def plot_psth(algn_spks, pre_post, behav_data, w=5):
    rew_side = behav_data['rewside'].values
    # rew_side = rew_side[indx_valid]
    bins = np.linspace(pre_post[0], pre_post[1], int(np.diff(pre_post)/w)+1)
    f, ax = plt.subplots(nrows=2)
    colors = [verde, morado]
    offset = 0
    for i_r, r in enumerate(np.unique(rew_side)):
        spks_side = algn_spks[rew_side == r]
        spks_side[np.logical_or(spks_side < pre_post[0],
                                spks_side > pre_post[1])] = np.nan
        for i_tr in range(spks_side.shape[0]):
            spks = spks_side[i_tr, :]
            spks = spks[~np.isnan(spks)]
            ax[0].scatter(spks, np.ones((len(spks)))*(i_tr+offset),
                          color=colors[i_r], s=2)
        offset = spks_side.shape[0]
        psth = np.histogram(spks_side, bins)
        ax[1].plot(bins[:-1]+w/2, psth[0], color=colors[i_r])
        ax[1].axvline(x=0, color='k', linestyle='--')
    return f, ax


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    area = 'fof'  # 'fof'  # 'dms'
    if area == 'dms':
        # main_folder = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys/'
        main_folder = '/home/molano/DMS_electro/DataEphys/pre_processed/'
        batch_dms_data(main_folder=main_folder)
    elif area == 'fof':
        home = 'molano'
        main_folder = '/home/'+home+'/fof_data/2022/'
        # SV_FOLDER =\
        #     '/home/molano/Dropbox/project_Barna/FOF_project/psths/2022/tests2/'
        SV_FOLDER = '/home/molano/Dropbox/project_Barna/FOF_project/2022/' +\
            'files_pop_analysis/'
        if not os.path.exists(SV_FOLDER):
            os.mkdir(SV_FOLDER)
        inv = np.load(main_folder+'/sess_inv_extended.npz',
                      allow_pickle=1)
        batch_fof_data(inv=inv, main_folder=main_folder, plot=True,
                       pre_post=[-1000, 0], sel_sess=['LE113_2021-06-05_12-38-09'])
    # sel_sess=['LE113_2021-06-21_13-54-42',
    #           'LE81_2020-12-21_10-00-52',
    #           'LE81_2020-11-30_10-41-44'])
    # get_data_file(file=MAIN_FOLDER+'/Rat32_ss_26_data_for_python.mat')
    # files = glob.glob('/home/molano/DMS_electro/DataEphys/pre_processed/' +
    #                   '*data_for_py*')

    # rats = np.unique([os.path.basename(x[:x.find('_ss_')]) for x in files])

    # for rt in rats:
    #     r_files = [x for x in files if x.find(rt) != -1]
