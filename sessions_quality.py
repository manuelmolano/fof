#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:28:28 2021

@author: molano
"""
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import glob
import shutil
# import utils as ut
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')
import utils as ut
colors = sns.color_palette()


def set_title(ax, inv, inv_sbsmpld):
    ax.set_title(str(np.round(inv['num_stms_csv'][idx[0]], 3))+' / ' +
                 str(np.round(inv['sil_per'][idx[0]], 3))+' /// ' +
                 str(np.round(inv['num_stim_ttl'][idx[0]], 3))+' / ' +
                 str(np.round(inv['stim_ttl_dists_med'][idx[0]], 3))+' / ' +
                 str(np.round(inv['stim_ttl_dists_max'][idx[0]], 3))+' /// ' +
                 str(np.round(inv_sbsmpld['num_stim_analogue'][idx[0]], 3))+' / ' +
                 str(np.round(inv['stim_analogue_dists_med'][idx[0]], 3))+' / ' +
                 str(np.round(inv['stim_analogue_dists_max'][idx[0]], 3)))
    for k in inv.keys():
        if not np.isnan(inv[k]) and not np.isnan(inv_sbsmpld[k]) and\
           k not in ['num_stim_analogue', 'sil_per']:
            assert inv[k] == inv_sbsmpld[k], str(inv[k] - inv_sbsmpld[k])


if __name__ == '__main__':
    plt.close('all')
    std_conv = 20
    margin_psth = 2000
    xs = np.arange(2*margin_psth)-margin_psth
    xs = xs/1000
    sv_folder = '/home/molano/fof_data/ttl_psths/'
    home = 'molano'
    main_folder = '/home/'+home+'/fof_data/'
    inv = np.load('/home/molano/fof_data/sess_inv.npz', allow_pickle=1)
    inv_sbsmpld = np.load('/home/molano/fof_data/sess_inv_sbsTrue.npz',
                          allow_pickle=1)
    sess_classification = ['bad']*len(inv['session'])
    issue = ['']*len(inv['session'])
    observations = ['']*len(inv['session'])
    sel_rats = []  # ['LE113']  # 'LE101'
    sel_sess = []  # ['LE113_2021-06-02_14-28-00']  # ['LE113_2021-06-05_12-38-09']
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
                print(idx)
                continue

            offset = inv['offset'][idx[0]]
            e_file = sess+'/e_data.npz'
            e_data = np.load(e_file, allow_pickle=1)
            samples = np.load(sess+'/ttls_sbsmpl.npz', allow_pickle=1)
            samples = samples['samples']
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
            ax.remove()
            ax_traces = plt.axes([.05, 0.55, 0.9, .4])
            set_title(ax=ax_traces, inv=inv, inv_sbsmpld=inv_sbsmpld)
            ax_size = 0.17
            margin = .06
            idx_max = np.where(samples[:, 0] == np.max(samples[:, 0]))[0][0]
            idx_midd = int(samples.shape[0]/2)
            num_ps = int(1e5)
            for ttl in range(4):
                aux1 = ttl % 2 != 0
                aux2 = ttl > 1
                ax_loc = [.05+(ax_size+margin)*aux1, .05+(ax_size+margin)*aux2,
                          ax_size, ax_size]
                ax_hist = plt.axes(ax_loc)
                sample = samples[:, ttl]
                ax_hist.hist(sample, 100)
                ax_hist.set_title('TTL: '+str(ttl+36))
                ax_hist.set_yscale('log')
                sample = sample/np.max(sample)
                # plot around a max event
                ax_traces.plot(np.arange(num_ps)+idx_max,
                               sample[idx_max:idx_max+num_ps]+ttl,
                               label=str(ttl+36), color=colors[ttl])
                # plot around the middle point of the recording
                ax_traces.plot(np.arange(num_ps)+idx_max+num_ps+1e3,
                               sample[idx_midd:idx_midd+num_ps]+ttl,
                               color=colors[ttl])
            ax_traces.legend()
            # PSTHs
            evs_lbls = ['stim_ttl_strt', 'fix_strt', 'outc_strt', 'stim_anlg_strt']
            for i_e, ev in enumerate(evs_lbls):
                aux1 = i_e % 2 != 0
                aux2 = i_e > 1
                ax_loc = [.55+(ax_size+margin)*aux1, .05+(ax_size+margin)*aux2,
                          ax_size, ax_size]
                ax_psth = plt.axes(ax_loc)
                ax_psth.set_title(ev)
                chnls = [0, 1] if i_e < 3 else [2, 3]
                lbls = [' (36)', ' (37)'] if i_e < 3 else [' (38)', ' (39)']
                evs = e_data[ev]
                print(ev)
                if len(evs) > 0:
                    evs = e_data['s_rate_eff']*(evs+offset)
                    evs = evs.astype(int)
                    peri_evs_1 = np.array([samples[x-margin_psth:x+margin_psth,
                                                   chnls[0]] for x in evs])
                    peri_evs_2 = np.array([samples[x-margin_psth:x+margin_psth,
                                                   chnls[1]] for x in evs])
                    try:
                        for i_ex in range(10):
                            ax_psth.plot(xs, peri_evs_1[i_ex], color=colors[0],
                                         lw=.5, alpha=.2)
                            ax_psth.plot(xs, peri_evs_2[i_ex], color=colors[1],
                                         lw=.5, alpha=.2)
                        psth_1 = np.mean(peri_evs_1, axis=0)
                        psth_2 = np.mean(peri_evs_2, axis=0)
                        ax_psth.plot(xs, psth_1, color=colors[0], lw=1,
                                     label='ch '+lbls[0])
                        ax_psth.plot(xs, psth_2, color=colors[1], lw=1,
                                     label='ch '+lbls[1])
                    except ValueError:
                        print('to-do')
                    ax_psth.legend()
            f.savefig(sv_folder+'/'+session+'.png')
            good = input("Is this session good?")
            if good == 'y':
                fldr = 'good'
            elif good == 'n':
                fldr = 'bad'
            elif good == ' ':
                fldr = 'revisit'
            else:
                raise ValueError('Specify the quality of the session with y/n')
            sess_classification[idx[0]] = fldr
            prob = input("issue:")
            issue[idx[0]] = prob
            obs = input("Observations:")
            observations[idx[0]] = obs
            extended_inv = {}
            for it in inv.items():
                extended_inv[it[0]] = it[1]
            extended_inv['sess_class'] = sess_classification
            extended_inv['issue'] = issue
            extended_inv['observations'] = observations
            f.savefig(sv_folder+fldr+'/'+session+'.png')
            plt.close(f)
            np.savez('/home/molano/fof_data/sess_inv_extended.npz', **inv)
    #
    #
    #
    # fltr_k = None
    # trace1 = samples[:, 0]
    # trace1 = trace1/np.max(trace1)
    # trace1_filt = ss.medfilt(trace1, fltr_k) if fltr_k is not None else trace1
    # trace2 = samples[:, 1]
    # trace2 = trace2/np.max(trace2)
    # trace2_filt = ss.medfilt(trace2, fltr_k) if fltr_k is not None else trace2
    # signal = 1*((trace1_filt-trace2_filt) > 0.5)
   
    # # stim starts/ends
    # stim_starts = np.where(np.diff(signal) > 0.9)[0]
    # ttl_stim_strt = stim_starts/e_data['s_rate_eff']
    # ttl_stim_strt = e_data['s_rate_eff']*ttl_stim_strt
    # ttl_stim_strt = ttl_stim_strt.astype(int)
    # print(ttl_stim_strt[:10])
    # print(ttl_stim_strt[-10:])
    # for i in range(100):
    #     plt.plot(samples[ttl_stim_strt[i]-20:ttl_stim_strt[i]+100, 0],
    #              color=colors[0])
    #     plt.plot(samples[ttl_stim_strt[i]-20:ttl_stim_strt[i]+100, 1],
    #              color=colors[1])
        
    # signal = 1*((trace2_filt-trace1_filt) > 0.5)
    # stim_starts = np.where(np.diff(signal) > 0.9)[0]
    # ttl_stim_strt = stim_starts/e_data['s_rate_eff']
    # ttl_stim_strt = e_data['s_rate_eff']*ttl_stim_strt
    # ttl_stim_strt = ttl_stim_strt.astype(int)
    # print('------------------------')
    # print(ttl_stim_strt[:10])
    # print(ttl_stim_strt[-10:])
    # for i in range(100):
    #     plt.plot(samples[ttl_stim_strt[i]-20:ttl_stim_strt[i]+100, 0],
    #              color=colors[0])
    #     plt.plot(samples[ttl_stim_strt[i]-20:ttl_stim_strt[i]+100, 1],
    #              color=colors[1])

    #
    #
    #