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
MAIN_FOLDER = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys/'# '/home/molano/DMS_electro/DataEphys/pre_processed/'


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
    perf = bhv_ss[0]
    stim_vals = bhv_ss[2][0, :]
    coh = stim_vals[bhv_ss[1]-1]
    gt = bhv_ss[3]
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


def get_data_file(file, ev_algmt='S', pre_post=[-1, 0], w=0.1):
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
    # reward
    reward = bhv_ss[0].astype(float)
    reward = reward.flatten()
    inv_rw = np.logical_and(reward != 1., reward != 0.)
    # choice
    choice = gt.copy().astype(float)
    choice[reward == 0] = np.abs(gt[reward == 0]-3)
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


# --- MAIN
if __name__ == '__main__':
    rats = ['Patxi', 'Rat15', 'Rat31', 'Rat32', 'Rat7']
    for r in rats:
        print('xxxxxxxxxxxxxxxx')
        print(r)
        files = glob.glob(MAIN_FOLDER+'/'+r+'*mat')
        num_unts = []
        for f in files:
            # print('--------------------------')
            # print(f)
            data, units = get_data_file(file=f)
            num_unts.append(len(units))
        num_unts = np.array(num_unts)
        print('-------------')
        print('Number of sessions')
        print(len(num_unts))
        print('Proportion of sessions with units')
        print(np.sum(num_unts != 0)/len(num_unts))
        print('Median number of units in sessions with units')
        print(np.median(num_unts[num_unts != 0]))
        print('-------------')        # get_data_file(file=MAIN_FOLDER+'/Rat32_ss_26_data_for_python.mat')
    # files = glob.glob('/home/molano/DMS_electro/DataEphys/pre_processed/' +
    #                   '*data_for_py*')

    # rats = np.unique([os.path.basename(x[:x.find('_ss_')]) for x in files])

    # for rt in rats:
    #     r_files = [x for x in files if x.find(rt) != -1]
