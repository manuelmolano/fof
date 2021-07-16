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
p = ComPipe.chom('LE113',  # sujeto (nombre de la carpeta under parentpath)
                  parentpath=main_folder,
                  analyze_trajectories=False)  # precarga sesiones disponibles
p.load_available()  # just in case, refresh
print(p.available[2])  # exmpl target sess / filename string is the actual arg
p.load(p.available[2])
p.process()
p.trial_sess.head()  # preprocessed df stored in attr. trial_sess
df = p.sess
# df = pd.read_csv(main_folder+
#                  '/LE113/sessions/LE113_p4_noenv_20210605-123818.csv',
#                   sep=';', skiprows=6)
# df['PC-TIME'] = pd.to_datetime(df['PC-TIME'])
csv_strt_snd_times = df.loc[(df['MSG'] == 'StartSound') &
                            (df.TYPE == 'TRANSITION'), 'PC-TIME']
csv_strt_outc_times = df.loc[((df['MSG'] == 'Reward') |
                              (df['MSG'] == 'Punish')) &
                             (df.TYPE == 'TRANSITION'), 'PC-TIME']
# translate date to seconds
csv_ss_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
                       for x in csv_strt_snd_times])
csv_ss_sec = csv_ss_sec-csv_ss_sec[0]
csv_so_sec = np.array([60*60*x.hour+60*x.minute+x.second+x.microsecond/1e6
                       for x in csv_strt_outc_times])
csv_so_sec = csv_so_sec-csv_ss_sec[0]
path = main_folder+'/LE113/electro/LE113_2021-06-05_12-38-09/'
events = np.load(path+'/events.npz', allow_pickle=1)
print(len(events['outc_starts']))
print(len(csv_strt_outc_times))
plt.figure()
offset = 29550000
num_secs = 100000
samples = events['samples'].T/np.max(events['samples'], axis=0)[:, None]
samples = samples[:, :int(num_secs*s_rate)]
plt.plot((offset+np.arange(samples.shape[1]))/s_rate,
         samples[0, :], label='ttl1 (ch35)')
plt.plot((offset+np.arange(samples.shape[1]))/s_rate,
         samples[1, :], linestyle='--', label='ttl2 (ch36)')

# num_secs = num_secs+offset/s_rate
# ev_strt = events['stim_starts']/s_rate
# ev_strt = ev_strt[ev_strt < num_secs]
# for i in ev_strt:
#     label = 'ttl-stim' if i == ev_strt[0] else ''
#     plt.plot(np.array([i, i]), [0, 0.5], 'c', label=label)

# csv_ss_sec -= csv_ss_sec[0]
# csv_ss_sec += ev_strt[0]
# csv_ss_sec = csv_ss_sec[csv_ss_sec < num_secs]
# for i in csv_ss_sec:
#     label = 'start-sound' if i == csv_ss_sec[0] else ''
#     plt.plot(np.array([i, i]), [0.5, 1], 'b', label=label)


num_secs = num_secs+offset/s_rate
ev_strt = events['outc_starts']/s_rate
ev_strt = ev_strt[ev_strt < num_secs]
for i in ev_strt:
    label = 'ttl-outcome' if i == ev_strt[0] else ''
    plt.plot(np.array([i, i]), [0, 0.5], '--c', label=label, lw=2)

csv_so_sec -= csv_so_sec[0]
csv_so_sec += ev_strt[0]
csv_so_sec = csv_so_sec[csv_so_sec < num_secs]
for i in csv_so_sec:
    label = 'CSV outcome' if i == csv_so_sec[0] else ''
    plt.plot(np.array([i, i]), [0.5, 1], '--b', label=label, lw=2)


plt.legend()
