
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from statannot import add_stat_annotation

### ----------- history axis ------------------- 
dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
IDX_RAT = 'Rat31_'
timep = '/-01-00s/'
NITERATIONS = 50
### compute the angles between history encoding axises
dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']

dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']

dataname  = dir+timep+IDX_RAT+'data_beh_ae.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_behae, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

dataname  = dir+timep+IDX_RAT+'data_beh_ac.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_behac, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

wiac_hist,wiae_hist = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
wiac_hist_beh,wiae_hist_beh= np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
for i in range(NITERATIONS):
    wiac_hist[:,i],wiae_hist[:,i]= wi_ac[:, i*5+3],wi_ae[:, i*5+3]
    wiac_hist_beh[:,i],wiae_hist_beh[:,i] = wi_behac[:, i*5+4],wi_behae[:, i*5+4]

### Predicting upcoming stimulus 
### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hist_ac_iter, hist_ae_iter = np.zeros((NITERATIONS,np.shape(wiac_hist)[0])), np.zeros((NITERATIONS,np.shape(wiac_hist)[0]))
for i in range(NITERATIONS):
    hist_ac_iter[i,:] = wiac_hist[:,i]/np.linalg.norm(wiac_hist[:,i]) 
    hist_ae_iter[i,:] = wiae_hist[:,i]/np.linalg.norm(wiae_hist[:,i])

histbeh_ac_iter, histbeh_ae_iter = np.zeros((NITERATIONS,np.shape(wiac_hist)[0])), np.zeros((NITERATIONS,np.shape(wiac_hist)[0]))
for i in range(NITERATIONS):
    histbeh_ac_iter[i,:] = wiac_hist_beh[:,i]/np.linalg.norm(wiac_hist_beh[:,i]) 
    histbeh_ae_iter[i,:] = wiae_hist_beh[:,i]/np.linalg.norm(wiae_hist_beh[:,i])

### Get distributions
hist_ag_ce_iter, beh_ag_ce_iter, beh_hist_cpc_iter, beh_hist_cpe_iter = np.zeros(NITERATIONS),np.zeros(NITERATIONS),np.zeros(NITERATIONS),np.zeros(NITERATIONS)
for i in range(NITERATIONS):
    hist_ag_ce_iter[i]   = np.arccos(np.dot(hist_ac_iter[i,:],hist_ae_iter[i,:]))
    beh_ag_ce_iter[i]    = np.arccos(np.dot(histbeh_ac_iter[i,:],histbeh_ae_iter[i,:]))
    beh_hist_cpc_iter[i] = np.arccos(np.dot(histbeh_ac_iter[i,:],hist_ac_iter[i,:]))
    beh_hist_cpe_iter[i] = np.arccos(np.dot(histbeh_ae_iter[i,:],hist_ae_iter[i,:]))

# figure ploting -- distribution of bootstrap results
fig_hist_beh,ax_hist_beh = plt.subplots(figsize=(6,3))
BOX_WDTH = 0.25

df = {'trbias (ac/ae) angle': hist_ag_ce_iter*180/np.pi, 'beh-trbias(ac)': beh_hist_cpc_iter*180/np.pi,
      'beh-trbias(ae)': beh_hist_cpe_iter*180/np.pi}
order = ['trbias (ac/ae) angle', 'beh-trbias(ac)', 'beh-trbias(ae)']
df = pd.DataFrame(df)
sns.set(style='whitegrid')

ax_hist_beh = sns.boxplot(data=df, order=order)
box_pairs = [('beh-trbias(ac)', 'beh-trbias(ae)')]
add_stat_annotation(ax_hist_beh, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
ax_hist_beh.set_ylim([60,120])
ax_hist_beh.set_yticks([60,90,120])


timep = '/00-01s/'
### compute the angles between history encoding axises    
dataname  = dir+timep+IDX_RAT+'data_beh.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_comp_dc = data_dec['coefs']

wicomp_stim_dc = np.zeros((np.shape(wi_comp_dc)[0],NITERATIONS))
wicomp_beh_dc= np.zeros((np.shape(wi_comp_dc)[0],NITERATIONS))
for i in range(NITERATIONS):
    wicomp_stim_dc[:,i] = wi_comp_dc[:, i*2+0]
    wicomp_beh_dc[:,i] = wi_comp_dc[:, i*2+1]

### Encoding received stimulus and determining behaviour
### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
stim_comp_iter,beh_comp_iter = np.zeros((NITERATIONS,np.shape(wi_comp_dc)[0])),np.zeros((NITERATIONS,np.shape(wi_comp_dc)[0]))
for i in range(NITERATIONS):
    stim_comp_iter[i,:] = wicomp_stim_dc[:,i]/np.linalg.norm(wicomp_stim_dc[:,i])
    beh_comp_iter[i,:] = wicomp_beh_dc[:,i]/np.linalg.norm(wicomp_beh_dc[:,i])
    
    
stim_beh_ag_comp_iter = np.zeros(NITERATIONS)
for i in range(NITERATIONS): 
    stim_beh_ag_comp_iter[i] = np.arccos(np.dot(stim_comp_iter[i,:],beh_comp_iter[i,:]))
fig_stim_beh, ax_stim_beh=plt.subplots(figsize=(1,2)) 
BOX_WDTH = 0.25
    
df = {'stim and behaviour encoding angle': stim_beh_ag_comp_iter*180/np.pi}
df = pd.DataFrame(df)
sns.set(style='whitegrid')
ax_stim_beh = sns.boxplot(data=df)

ax_stim_beh.set_ylim([0,90])
ax_stim_beh.set_yticks([0,45,90])

# ### Get distributions
# stim_hist_ag_ac_iter, stim_hist_ag_ae_iter, histstim_beh_ag_ac_iter, histstim_beh_ag_ae_iter = np.zeros(NITERATIONS),np.zeros(NITERATIONS),np.zeros(NITERATIONS),np.zeros(NITERATIONS)
# for i in range(NITERATIONS):
#     stim_hist_ag_ac_iter[i] = np.arccos(np.dot(stim_comp_iter[i,:],hist_ac_iter[i,:]))
#     stim_hist_ag_ae_iter[i] = np.arccos(np.dot(stim_comp_iter[i,:],hist_ae_iter[i,:]))
#     histstim_beh_ag_ac_iter[i] = np.arccos(np.dot(beh_comp_iter[i,:],histbeh_ac_iter[i,:]))
#     histstim_beh_ag_ae_iter[i] = np.arccos(np.dot(beh_comp_iter[i,:],histbeh_ae_iter[i,:]))


# fig_stim_beh, ax_stim_beh=plt.subplots(figsize=(8,4))
# BOX_WDTH = 0.25

# df = {'stim and prediction(ac) angle': stim_hist_ag_ac_iter*180/np.pi, 'stim and prediction(ae) angle': stim_hist_ag_ae_iter*180/np.pi,'two beh axes (ac)': histstim_beh_ag_ac_iter*180/np.pi,
#       'two beh axes (ae)': histstim_beh_ag_ae_iter*180/np.pi}
# order = ['stim and prediction(ac) angle', 'stim and prediction(ae) angle','two beh axes (ac)',
#       'two beh axes (ae)']
# df = pd.DataFrame(df)
# sns.set(style='whitegrid')

# ax_stim_beh = sns.boxplot(data=df, order=order)
# box_pairs = [('stim and prediction(ac) angle', 'stim and prediction(ae) angle'),('two beh axes (ac)',
#       'two beh axes (ae)')]
# add_stat_annotation(ax_stim_beh, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)