#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 18:01:57 2021

@author: molano
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
main_folder = '/home/molano/Dropbox/project_Barna/FOF_project'
folder = main_folder+'/psths/'
rats = ['LE113', 'LE81', 'LE104']
corrs = []
repeated = []
for i_r, rat1 in enumerate(rats):
    print(rat1)
    trs_1 = np.load(folder+'/'+rat1+'_traces.npz', allow_pickle=True)
    # if i_r > 0:
    #     corrs.append(np.ones_like(corrs_tmp))
    corrs_within_rat = []
    corrs_between_rats = []
    for k1 in trs_1.items():
        psth1 = k1[1][0, :]
        corrs_tmp = []
        for rat2 in rats:
            trs_2 = np.load(folder+'/'+rat2+'_traces.npz', allow_pickle=True)
            for k2 in trs_2.items():
                psth2 = k2[1][0, :]
                corr = np.corrcoef(psth1, psth2)[0, 1]
                corrs_tmp.append(corr)
                if rat1 == rat2:
                    corrs_within_rat.append(corr)
                else:
                    corrs_between_rats.append(corr)
                if corr > 0.9 and corr < 1-1e-6:
                    f, ax = plt.subplots(ncols=2)
                    ax[0].plot(psth1, label=k1[0][:-9])
                    ax[0].plot(psth2, label=k2[0][:-9])
                    ax[1].plot(psth1/np.max(psth1))
                    ax[1].plot(psth2/np.max(psth2))
                    ax[0].set_title('Correlation: '+str(np.round(corr, 4)))
                    ax[0].legend()
                    f.savefig(main_folder+'/psth_corrs/'+k1[0][:-9]+'_' +
                              k2[0][:-9]+'.png')
                    repeated.append(k1[0][:-9])
                    plt.close(f)
                    if rat1 != rat2:
                        print(k1[0][:-9]+'  '+k2[0][:-9])
        corrs.append(corrs_tmp)
    f = plt.figure()
    plt.hist([corrs_within_rat, corrs_between_rats], 20)
    f.savefig(main_folder+'/psth_corrs/'+rat1+'_corrs_hist.png')

corrs = np.array(corrs)
plt.figure()
plt.imshow(corrs, aspect='auto')
f.savefig(main_folder+'/psth_corrs/corrs_mat.png')
