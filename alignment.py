#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:55:52 2021

@author: manuel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilsJ.Behavior import ComPipe

plt.close('all')
main_folder = '/home/manuel/fof_data/'
s_rate = 3e4
# p = ComPipe.chom('LE113',  # sujeto (nombre de la carpeta under parentpath)
#                  parentpath=main_folder,
#                  analyze_trajectories=False)  # precarga sesiones disponibles
# p.load_available()  # just in case, refresh
# print(p.available[2])  # exmpl target sess / filename string is the actual arg
# p.load(p.available[2])
# p.process()
# p.trial_sess.head()  # preprocessed df stored in attr. trial_sess

df = pd.read_csv(main_folder+'/LE113/sessions/LE113_p4_noenv_20210605-123818.csv',
                 sep=';', skiprows=6)
df['PC-TIME'] = pd.to_datetime(df['PC-TIME'])
strt_snd_times = df.loc[(df['MSG'] == 'StartSound') &
                        (df.TYPE == 'TRANSITION'), 'PC-TIME']

bnc1_times = df.loc[(df['+INFO'] == 'BNC1High') &
                    (df.TYPE == 'EVENT'), 'PC-TIME']
bnc2_high_times = df.loc[(df['+INFO'] == 'BNC2High') &
                         (df.TYPE == 'EVENT'), 'PC-TIME']
bnc2_low_times = df.loc[(df['+INFO'] == 'BNC2Low') &
                        (df.TYPE == 'EVENT'), 'PC-TIME']

sst_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
                    for x in strt_snd_times])
bnc2H_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
                      for x in bnc2_high_times])
bnc2L_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
                      for x in bnc2_low_times])

path = main_folder+'/LE113/electro/LE113_2021-06-05_12-38-09/'
events = np.load(path+'/events.npz', allow_pickle=1)
plt.figure()
offset = 15000000
num_secs = 100000
samples = events['samples'].T/np.max(events['samples'], axis=0)[:, None]
samples = samples[:, :int(num_secs*s_rate)]
plt.plot((offset+np.arange(samples.shape[1]))/s_rate, samples[0, :], label='ttl1')
plt.plot((offset+np.arange(samples.shape[1]))/s_rate, samples[1, :], label='ttl2')
# plt.plot((offset+np.arange(samples.shape[1]))/s_rate, samples[2, :], label='ttl3')
# plt.plot((offset+np.arange(samples.shape[1]))/s_rate, samples[3, :], label='ttl4')

trace1 = samples[0, :]
trace2 = samples[1, :]
stim = 1*((trace2-trace1) > 0.9)
starts_bis = np.where(np.diff(stim) > 0.9)[0]
# plt.plot((offset+np.arange(samples.shape[1]))/s_rate, abv_th1, label='abv_th1')
# plt.plot((offset+np.arange(samples.shape[1]))/s_rate, abv_th2, label='abv_th2')
# plt.plot((offset+np.arange(samples.shape[1]))/s_rate, stim, label='stim')


num_secs = num_secs+offset/s_rate
ev_strt = events['ev_strt'][2]/s_rate
ev_strt = ev_strt[ev_strt < num_secs]
for i in ev_strt:
    label = 'ttl-stim' if i == ev_strt[0] else ''
    plt.plot(np.array([i, i]), [0, 0.5], 'c', label=label)


sst_sec -= sst_sec[0]
sst_sec += ev_strt[0]
sst_sec = sst_sec[sst_sec < num_secs]
for i in sst_sec:
    label = 'start-sound' if i == sst_sec[0] else ''
    plt.plot(np.array([i, i]), [0.5, 1], 'b', label=label)


# bnc2H_sec -= bnc2H_sec[0]
# bnc2H_sec += ev_strt[0]
# bnc2H_sec = bnc2H_sec[bnc2H_sec < 200]
# for i in bnc2H_sec:
#     label = 'bnc2 high' if i == bnc2H_sec[0] else ''
#     plt.plot(np.array([i, i]), [-0.5, 0], 'g', label=label)

# plt.legend()

# bnc2L_sec -= bnc2L_sec[0]
# bnc2L_sec += ev_strt[0]
# bnc2L_sec = bnc2L_sec[bnc2L_sec < 200]
# for i in bnc2L_sec:
#     label = 'bnc2 low' if i == bnc2L_sec[0] else ''
#     plt.plot(np.array([i, i]), [-0.5, 0], 'b', label=label)

plt.legend()

