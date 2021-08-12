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


def histogram_psth(spk_times, events, bins, bin_size):
    spks_mat = np.tile(spk_times, (1, len(events)))-events.T
    hists = np.array([np.histogram(spks_mat[:, i], bins)[0]
                      for i in range(spks_mat.shape[1])])
    hists = hists/bin_size
    psth = np.mean(hists, axis=0)
    return psth


def convolve_psth(spk_times, events, std=20, margin_psth=1000):
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
    peri_evs = np.array([spk_conv[x-margin_psth:x+margin_psth] for x in events])
    psth = np.mean(peri_evs, axis=0)*1000
    return psth


def plt_psths(spks, clstrs, sel_clstrs, clstrs_qlt, evs, ax, ncols, nrows,
              margin_psth=1000, std_conv=20, lbl=''):
    for i_cl, cl in enumerate(sel_clstrs):
        spk_tms = spks[clstrs == cl][:, None]
        ax[i_cl].axvline(x=0, linestyle='--', color=(.7, .7, .7))
        psth_cnv = convolve_psth(spk_times=spk_tms, events=evs, std=std_conv,
                                 margin_psth=margin_psth)
        xs = np.arange(2*margin_psth)-margin_psth
        xs = xs/1000
        ax[i_cl].plot(xs, psth_cnv, label=lbl)
        ax[i_cl].set_title(str(cl)+' / #spks: '+str(len(spk_tms)) +
                           ' / qlt: '+clstrs_qlt[i_cl])
        ax[i_cl].legend()
        if i_cl < ncols*(nrows-1):
            ax[i_cl].set_xticks([])
        if i_cl == ncols*(nrows-1):
            ax[i_cl].set_xlabel('Peri-stimulus time (s)')
            ax[i_cl].set_ylabel('Firing rate (Hz)')


if __name__ == '__main__':
    plt.close('all')
    std_conv = 20
    margin_psth = 1000
    sv_folder = '/home/molano/fof_data/pshts/'
    home = 'molano'
    main_folder = '/home/'+home+'/fof_data/'
    inv = np.load('/home/molano/fof_data/sess_inv.npz', allow_pickle=1)
    # rat = 'LE113'  # 'LE101'
    # session = 'LE113_p4_noenv_20210605-123818'  # 'LE101_2021-06-11_12-10-11'
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
            idx = [i for i, x in enumerate(inv['session']) if x.endswith(session)]
            if len(idx) != 1:
                print(str(idx))
            e_file = sess+'/e_data.npz'
            b_file = sess+'/df_trials'
            e_data = np.load(e_file, allow_pickle=1)
            b_data = pd.read_pickle(b_file)
            trial_times = ut.date_2_secs(b_data.fix_onset_dt)
            ev = 'stim_ttl_strt'
            events = e_data[ev]
            ev_indx = np.searchsorted(trial_times, events)-1
            _, counts = np.unique(ev_indx, return_counts=1)
            assert np.max(counts) == 1
            choice = b_data['R_response'].values
            sel_clstrs = e_data['sel_clstrs']
            ncols = int(np.ceil(np.sqrt(len(sel_clstrs))))
            nrows = (ncols-1) if ncols*(ncols-1) >= len(sel_clstrs) else ncols
            f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 8))
            ax = ax.flatten()
            plt_psths(spks=e_data['spks'], clstrs=e_data['clsts'],
                      sel_clstrs=e_data['sel_clstrs'],
                      clstrs_qlt=e_data['clstrs_qlt'],
                      evs=events[choice[ev_indx] > .5],
                      ax=ax, ncols=ncols, nrows=nrows,
                      std_conv=std_conv, lbl='Right')
            plt_psths(spks=e_data['spks'], clstrs=e_data['clsts'],
                      sel_clstrs=e_data['sel_clstrs'],
                      clstrs_qlt=e_data['clstrs_qlt'],
                      evs=events[choice[ev_indx] < .5],
                      ax=ax, ncols=ncols, nrows=nrows,
                      std_conv=std_conv, lbl='Left')
            f.savefig(sv_folder+session+'.png')
