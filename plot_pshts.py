#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:26:32 2021

@author: molano
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import glob
# import utils as ut
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')
import utils as ut

def rm_top_right_lines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def histogram_psth(spk_times, events, bins, bin_size):
    spks_mat = np.tile(spk_times, (1, len(events)))-events.T
    hists = np.array([np.histogram(spks_mat[:, i], bins)[0]
                      for i in range(spks_mat.shape[1])])
    hists = hists/bin_size
    psth = np.mean(hists, axis=0)
    return psth


def convolve_psth(spk_times, events, std=20, margin_psth=1000):
    if len(events) > 0:
        krnl_len = 5*std
        # pass spikes to ms
        spk_times = 1000*spk_times.flatten()
        spk_times = spk_times.astype(int)
        spk_dist = spk_times[-1] - spk_times[0]
        # build spike vector
        spk_vct = np.zeros((spk_dist+2*margin_psth, ))
        spk_vct[spk_times-spk_times[0]+margin_psth] = 1
        # convolve
        x = np.linspace(norm.ppf(1e-5, scale=std),
                        norm.ppf(1-1e-5, scale=std), krnl_len)
        kernel = norm.pdf(x, scale=std)
        kernel = kernel/np.sum(kernel)  # XXX: why isn't kernel already normalized?
        spk_conv = np.convolve(spk_vct, kernel)
        # spk_conv = gaussian_filter1d(spk_vct, std)
        # pass events to ms
        events = 1000*events
        # offset events
        events = events.astype(int)-spk_times[0]+margin_psth
        peri_evs = np.array([spk_conv[x-margin_psth:x+margin_psth]
                             for x in events])
        try:
            psth = np.mean(peri_evs, axis=0)*1000
        except:
            print(1)
    else:
        psth = []
    return psth


def plt_psths(cl, spk_tms, evs, ax, margin_psth=1000,
              std_conv=20, lbl=''):
    ax.axvline(x=0, linestyle='--', color=(.7, .7, .7))
    psth_cnv = convolve_psth(spk_times=spk_tms, events=evs, std=std_conv,
                             margin_psth=margin_psth)
    xs = np.arange(2*margin_psth)-margin_psth
    xs = xs/1000
    ax.plot(xs, psth_cnv, label=lbl)


def psth_choice_cond(cl, e_data, b_data, session, ax, ev='stim_ttl_strt', std_conv=20,
                     margin_psth=1000, sv_folder=''):
    import time
    offset = 1 if ev != 'fix_strt' else 0
    trial_times = ut.date_2_secs(b_data.fix_onset_dt)
    events = e_data[ev]
    start = time.time()
    ev_indx = np.searchsorted(trial_times, events)-offset
    aux, unq_idx, counts = np.unique(ev_indx, return_counts=1, return_index=1)
    filt_evs_b = events[unq_idx]
    print(time.time()-start)
    start = time.time()
    filt_evs = [(np.min(np.abs(events-tr)), events[np.argmin(np.abs(events-tr))])
                for tr in trial_times]
    print(time.time()-start)
    # indx of regular trials
    indx = np.logical_and(b_data['special_trial'] == 0, b_data['soundrfail'] == 0)
    if np.max(counts) != 1:
        print(np.unique(counts, return_counts=1))
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    choice = b_data['R_response'].values
    indx_ch = np.logical_and(choice[ev_indx] > .5, indx[ev_indx])
    plt_psths(cl=cl, spk_tms=spk_tms, evs=events[indx_ch], ax=ax,
              std_conv=std_conv, margin_psth=margin_psth, lbl='Right')
    indx_ch = np.logical_and(choice[ev_indx] < .5, indx[ev_indx])
    plt_psths(cl=cl, spk_tms=spk_tms, evs=events[indx_ch], ax=ax,
              std_conv=std_conv, margin_psth=margin_psth, lbl='Left')


if __name__ == '__main__':
    plt.close('all')
    std_conv = 20
    margin_psth = 1000
    sv_folder = '/home/molano/fof_data/pshts/'
    home = 'molano'
    main_folder = '/home/'+home+'/fof_data/'
    inv = np.load('/home/molano/fof_data/sess_inv.npz', allow_pickle=1)
    sel_rats = []  # ['LE113']  # 'LE101'
    sel_sess = ['LE113_2021-06-02_14-28-00']  # ['LE113_2021-06-05_12-38-09']
    # file = main_folder+'/'+rat+'/sessions/'+session+'/extended_df'
    home = 'molano'
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
            else:
                print(str(inv['num_stms_csv'][idx[0]])+' / ' +
                      str(inv['sil_per'][idx[0]])+' / ' +
                      str(inv['num_stim_ttl'][idx[0]])+' / ' +
                      str(inv['stim_ttl_dists_med'][idx[0]])+' / ' +
                      str(inv['stim_ttl_dists_max'][idx[0]])+' / ' +
                      str(inv['num_stim_analogue'][idx[0]])+' / ' +
                      str(inv['stim_analogue_dists_med'][idx[0]])+' / ' +
                      str(inv['stim_analogue_dists_max'][idx[0]]))
            e_file = sess+'/e_data.npz'
            e_data = np.load(e_file, allow_pickle=1)
            sel_clstrs = e_data['sel_clstrs']
            if inv['sil_per'][idx[0]] < 1e-2 and len(sel_clstrs) > 0:
                b_file = sess+'/df_trials'
                b_data = pd.read_pickle(b_file)
                for i_cl, cl in enumerate(sel_clstrs):
                    cl_qlt = e_data['clstrs_qlt'][i_cl]
                    f, ax = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
                    if inv['stim_ttl_dists_max'][idx[0]] < 0.1:
                        ev = 'fix_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[0], margin_psth=margin_psth,
                                         sv_folder=sv_folder)
                        ev = 'stim_ttl_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[1], margin_psth=margin_psth,
                                         sv_folder=sv_folder)
                        ev = 'outc_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[2], margin_psth=margin_psth,
                                         sv_folder=sv_folder)
                    if inv['stim_analogue_dists_max'][idx[0]] < 0.1:
                        ev = 'stim_anlg_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[1], margin_psth=margin_psth,
                                         sv_folder=sv_folder)
                    ax[0].legend()
                    ax[0].set_ylabel('Firing rate (Hz)')
                    ax[0].set_xlabel('Peri-fixation time (s)')
                    ax[1].set_xlabel('Peri-stim time (s)')
                    ax[2].set_xlabel('Peri-outcome time (s)')
                    for a in ax:
                        rm_top_right_lines(a)
                    num_spks = np.sum(e_data['clsts'] == cl)
                    f.suptitle(str(cl)+' / #spks: '+str(num_spks) +
                               ' / qlt: '+cl_qlt)
                    f.savefig(sv_folder+str(cl)+'_'+session+'.png')
                    plt.close(f)
