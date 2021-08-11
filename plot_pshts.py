#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:26:32 2021

@author: molano
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')
# import utils as ut


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


def plt_psths(spks, clstrs, sel_clstrs, clstrs_qlt, evs,
              std_conv=20):
    for i_cl, cl in enumerate(sel_clstrs):
        spk_tms = spks[clstrs == cl][:, None]
        ax[i_cl].axvline(x=0, linestyle='--', color=(.7, .7, .7))
        psth_cnv = convolve_psth(spk_times=spk_tms, events=evs, std=std_conv,
                                 margin_psth=margin_psth)
        xs = np.arange(2*margin_psth)-margin_psth
        xs = xs/1000
        ax[i_cl].plot(xs, psth_cnv, label='clstr '+str(cl))
        ax[i_cl].set_title('#spks: '+str(len(spk_tms))+' / qlt: '+clstrs_qlt[i_cl])
        ax[i_cl].legend()
        if i_cl < ncols*(nrows-1):
            ax[i_cl].set_xticks([])
        if i_cl == ncols*(nrows-1):
            ax[i_cl].set_xlabel('Peri-stimulus time (s)')
            ax[i_cl].set_ylabel('Firing rate (Hz)')


if __name__ == '__main__':
    plt.close('all')
    main_folder = '/home/molano/fof_data'
    # rat = 'LE113'  # 'LE101'
    # session = 'LE113_p4_noenv_20210605-123818'  # 'LE101_2021-06-11_12-10-11'
    # file = main_folder+'/'+rat+'/sessions/'+session+'/extended_df'
    e_file = '/home/manuel/fof_data/LE113/LE113_2021-06-05_12-38-09/e_data.npz'
    b_file = '/home/manuel/fof_data/LE113/LE113_2021-06-05_12-38-09/df'
    std_conv = 20
    margin_psth = 1000
    e_data = np.load(e_file, allow_pickle=1)
    b_data = pd.read_pickle(b_file)
    tmp = b_data.MSG == 'REWARD_SIDE'
    b_data.loc[tmp]
    ev = 'stim_ttl_strt'
    sel_clstrs = e_data['sel_clstrs']
    ncols = int(np.ceil(np.sqrt(len(sel_clstrs))))
    nrows = (ncols-1) if ncols*(ncols-1) >= len(sel_clstrs) else ncols
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 8))
    ax = ax.flatten()
    plt_psths(spks=e_data['spks'], clstrs=e_data['clsts'],
              sel_clstrs=e_data['sel_clstrs'], clstrs_qlt=e_data['clstrs_qlt'],
              evs=e_data[ev], std_conv=std_conv)
