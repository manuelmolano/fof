#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:26:32 2021

@author: molano
"""
import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')
import utils as ut
# plt.close('all')
main_folder = '/home/molano/fof_data'
# rat = 'LE113'  # 'LE101'
# session = 'LE113_p4_noenv_20210605-123818'  # 'LE101_2021-06-11_12-10-11'
# file = main_folder+'/'+rat+'/sessions/'+session+'/extended_df'
file = '/home/molano/fof_data/behavioral_data/LE113/sessions/' +\
    'LE113_p4_noenv_20210605-123818/extended_df'
margin_spks_plot = 1
bin_size = .1
ext_df =\
    np.load(file, allow_pickle=1)
events = ut.date_2_secs(ext_df.loc[(ext_df['stim_ttl_strt'] == 1.),
                                   'PC-TIME'])[:, None]
clstrs = [x for x in list(ext_df) if x.startswith('cl_')]
bins = np.linspace(-margin_spks_plot, margin_spks_plot-bin_size,
                   int(2*margin_spks_plot/bin_size))
f, ax = plt.subplots(ncols=5, nrows=3, figsize=(10, 6))
ax = ax.flatten()
for i_cl, cl in enumerate(clstrs):
    spk_times = ut.date_2_secs(ext_df.loc[(ext_df[cl] == 1.), 'PC-TIME'])[:, None]
    print('----------')
    print(len(spk_times))
    print(len(spk_times)/(spk_times[-1]-spk_times[0]))
    spks_mat = np.tile(spk_times, (1, len(events)))-events.T
    hists = np.array([np.histogram(spks_mat[:, i], bins)[0]
                      for i in range(spks_mat.shape[1])])
    hists = hists/bin_size
    psth = np.mean(hists, axis=0)
    ax[i_cl].plot(bins[:-1]+bin_size/2, psth)
    ax[i_cl].set_title(str(cl)+' / #spks: '+str(len(spk_times)))
    # plt.figure()
    # plt.imshow(spks_mat.T)
