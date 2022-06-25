"""
Computing all relevant AXES
@YX 18 Jun
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from statannot import add_stat_annotation

def ang_AXES(IDX_RAT,NITERATIONS):
    ### /Users/yuxiushao/Public/DataML/Auditory/DataEphys/00-01s
    dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # timep = '/00-01s/'
    timep = '/01-02s/'
    ### ----------- stimulus/behaviour axis ------------------- 
    # IDX_RAT='Rat15_'
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    dataname  = dir+timep+IDX_RAT+'data_dec.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_comp, bi_comp = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    wiac_stim,wiae_stim,wicomp_stim = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
    wiac_beh,wiae_beh,wicomp_beh = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
    for i in range(NITERATIONS):
    	wiac_stim[:,i],wiae_stim[:,i],wicomp_stim[:,i] = wi_ac[:, i*5+3],wi_ae[:, i*5+3],wi_comp[:, i*5+3]
    	wiac_beh[:,i],wiae_beh[:,i],wicomp_beh[:,i] = wi_ac[:, i*5+4],wi_ae[:, i*5+4],wi_comp[:, i*5+4]
    
    ### Encoding received stimulus 
    ### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    stim_ac_iter, stim_ae_iter, stim_comp_iter = np.zeros((NITERATIONS,np.shape(wi_ac)[0])), np.zeros((NITERATIONS,np.shape(wi_ac)[0])), np.zeros((NITERATIONS,np.shape(wi_ac)[0]))
    for i in range(NITERATIONS):
    	stim_ac_iter[i,:] = wiac_stim[:,i]/np.linalg.norm(wiac_stim[:,i]) 
    	stim_ae_iter[i,:] = wiae_stim[:,i]/np.linalg.norm(wiae_stim[:,i])
    	stim_comp_iter[i,:] = wicomp_stim[:,i]/np.linalg.norm(wicomp_stim[:,i])
    ### ~~~~~~~~~~~ mean ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	
    wiac_stim, wiae_stim, wicomp_stim= np.mean(wiac_stim,axis=1),np.mean(wiae_stim,axis=1),np.mean(wicomp_stim,axis=1)
    # ag_ce, ag_cpc, ag_cpe = 0,0,0  
    stim_ac=wiac_stim/np.linalg.norm(wiac_stim)
    stim_ae=wiae_stim/np.linalg.norm(wiae_stim)
    stim_comp=wicomp_stim/np.linalg.norm(wicomp_stim)
    
    
    ### (Dominated effect) making decisions according to received stimulus 
    ### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    beh_ac_iter, beh_ae_iter, beh_comp_iter = np.zeros((NITERATIONS,np.shape(wiac_beh)[0])), np.zeros((NITERATIONS,np.shape(wiac_beh)[0])), np.zeros((NITERATIONS,np.shape(wiac_beh)[0]))
    for i in range(NITERATIONS):
    	beh_ac_iter[i,:] = wiac_beh[:,i]/np.linalg.norm(wiac_beh[:,i]) 
    	beh_ae_iter[i,:] = wiae_beh[:,i]/np.linalg.norm(wiae_beh[:,i])
    	beh_comp_iter[i,:] = wicomp_beh[:,i]/np.linalg.norm(wicomp_beh[:,i])
    wiac_beh, wiae_beh, wicomp_beh= np.mean(wiac_beh,axis=1),np.mean(wiae_beh,axis=1),np.mean(wicomp_beh,axis=1)
    # ag_ce, ag_cpc, ag_cpe = 0,0,0  
    beh_ac=wiac_beh/np.linalg.norm(wiac_beh)
    beh_ae=wiae_beh/np.linalg.norm(wiae_beh)
    beh_comp=wicomp_beh/np.linalg.norm(wicomp_beh)
    
    ce_dotprod   = np.dot(stim_ac,stim_ae)
    stim_ag_ce   = np.arccos(ce_dotprod)
    
    cpc_dotprod  = np.dot(beh_comp,stim_ac)
    beh_stim_cpc = np.arccos(cpc_dotprod)
    
    cpe_dotprod  = np.dot(stim_ae,beh_comp)
    beh_stim_cpe = np.arccos(cpe_dotprod)

    print('stimulus ac/ae axis: ',stim_ag_ce*180/np.pi)
    print('behaviour ac-stimulus content: ',beh_stim_cpc*180/np.pi,"; ae-stimulus content ",beh_stim_cpe*180/np.pi)
    
    ### Get distributions
    stim_ag_ce_iter, beh_stim_cpc_iter, beh_stim_cpe_iter = np.zeros(NITERATIONS),np.zeros(NITERATIONS),np.zeros(NITERATIONS)
    for i in range(NITERATIONS):
    	stim_ag_ce_iter[i] = np.arccos(np.dot(stim_ac_iter[i,:],stim_ae_iter[i,:]))
    	beh_stim_cpc_iter[i] = np.arccos(np.dot(beh_comp_iter[i,:],stim_ac_iter[i,:]))
    	beh_stim_cpe_iter[i] = np.arccos(np.dot(beh_comp_iter[i,:],stim_ae_iter[i,:]))

    fig_stim_beh, ax_stim_beh=plt.subplots(figsize=(4,4))
    BOX_WDTH = 0.25
    
    df = {'stim(ac/ae) angle': stim_ag_ce_iter*180/np.pi, 'beh-stim(ac)': beh_stim_cpc_iter*180/np.pi,
          'beh-stim(ae)': beh_stim_cpe_iter*180/np.pi}
    order = ['stim(ac/ae) angle', 'beh-stim(ac)', 'beh-stim(ae)']
    df = pd.DataFrame(df)
    sns.set(style='whitegrid')
    
    ax_stim_beh = sns.boxplot(data=df, order=order)
    box_pairs = [('beh-stim(ac)', 'beh-stim(ae)')]
    add_stat_annotation(ax_stim_beh, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    ### ----------- history axis ------------------- 
    timep = '/-01-00s/'
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    wiac_hist,wiae_hist = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
    wiac_hist_beh,wiae_hist_beh= np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
    for i in range(NITERATIONS):
    	wiac_hist[:,i],wiae_hist[:,i]= wi_ac[:, i*5+3],wi_ae[:, i*5+3]
    	wiac_hist_beh[:,i],wiae_hist_beh[:,i] = wi_ac[:, i*5+4],wi_ae[:, i*5+4]
    
    ### Predicting upcoming stimulus 
    ### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    hist_ac_iter, hist_ae_iter = np.zeros((NITERATIONS,np.shape(wiac_hist)[0])), np.zeros((NITERATIONS,np.shape(wiac_hist)[0]))
    for i in range(NITERATIONS):
    	hist_ac_iter[i,:] = wiac_hist[:,i]/np.linalg.norm(wiac_hist[:,i]) 
    	hist_ae_iter[i,:] = wiae_hist[:,i]/np.linalg.norm(wiae_hist[:,i])

    
    wiac_hist, wiae_hist= np.mean(wiac_hist,axis=1),np.mean(wiae_hist,axis=1)
    	# ag_ce, ag_cpc, ag_cpe = 0,0,0  
    hist_ac=wiac_hist/np.linalg.norm(wiac_hist)
    hist_ae=wiae_hist/np.linalg.norm(wiae_hist)
    
    wiac_hist_beh, wiae_hist_beh= np.mean(wiac_hist_beh,axis=1),np.mean(wiae_hist_beh,axis=1)
    # ag_ce, ag_cpc, ag_cpe = 0,0,0  
    hist_beh_ac=wiac_hist_beh/np.linalg.norm(wiac_hist_beh)
    hist_beh_ae=wiae_hist_beh/np.linalg.norm(wiae_hist_beh)
    
    ce_dotprod   = np.dot(hist_ac,hist_ae)
    hist_ag_ce   = np.arccos(ce_dotprod)
    
    cpc_dotprod  = np.dot(hist_ac,beh_comp)
    beh_hist_cpc = np.arccos(cpc_dotprod)
    
    cpe_dotprod  = np.dot(hist_ae,beh_comp)
    beh_hist_cpe = np.arccos(cpe_dotprod)
    
    print('history ac/ae axis: ',hist_ag_ce*180/np.pi)
    print('behaviour ac-hist content: ',beh_hist_cpc*180/np.pi,"; ae-hist content ",beh_hist_cpe*180/np.pi)
    
    ### Get distributions
    hist_ag_ce_iter, beh_hist_cpc_iter, beh_hist_cpe_iter = np.zeros(NITERATIONS),np.zeros(NITERATIONS),np.zeros(NITERATIONS)
    for i in range(NITERATIONS):
    	hist_ag_ce_iter[i]   = np.arccos(np.dot(hist_ac_iter[i,:],hist_ae_iter[i,:]))
    	beh_hist_cpc_iter[i] = np.arccos(np.dot(beh_comp_iter[i,:],hist_ac_iter[i,:]))
    	beh_hist_cpe_iter[i] = np.arccos(np.dot(beh_comp_iter[i,:],hist_ae_iter[i,:]))
    
    # figure ploting -- distribution of bootstrap results
    fig_hist_beh,ax_hist_beh = plt.subplots(figsize=(4, 4))
    BOX_WDTH = 0.25
    
    df = {'hist(ac/ae) angle': hist_ag_ce_iter*180/np.pi, 'beh-hist(ac)': beh_hist_cpc_iter*180/np.pi,
          'beh-hist(ae)': beh_hist_cpe_iter*180/np.pi}
    order = ['hist(ac/ae) angle', 'beh-hist(ac)', 'beh-hist(ae)']
    df = pd.DataFrame(df)
    sns.set(style='whitegrid')
    
    ax_hist_beh = sns.boxplot(data=df, order=order)
    box_pairs = [('beh-hist(ac)', 'beh-hist(ae)')]
    add_stat_annotation(ax_hist_beh, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)


        ### Get distributions
    hist_stim_c_iter, hist_stim_e_iter = np.zeros(NITERATIONS),np.zeros(NITERATIONS)
    for i in range(NITERATIONS):
    	hist_stim_c_iter[i]  = np.arccos(np.dot(hist_ac_iter[i,:],stim_ac_iter[i,:]))
    	hist_stim_e_iter[i]  = np.arccos(np.dot(hist_ae_iter[i,:],stim_ae_iter[i,:]))
    
    # figure ploting -- distribution of bootstrap results
    fig_hist_stim,ax_hist_stim = plt.subplots(figsize=(4, 4))
    BOX_WDTH = 0.25
    
    df = {'hist-stim angle(ac)': hist_stim_c_iter*180/np.pi, 'hist-stim angle(ae)': hist_stim_e_iter*180/np.pi}
    order = ['hist-stim angle(ac)', 'hist-stim angle(ae)']
    df = pd.DataFrame(df)
    sns.set(style='whitegrid')
    
    ax_hist_stim = sns.boxplot(data=df, order=order)
    box_pairs = [('hist-stim angle(ac)', 'hist-stim angle(ae)')]
    add_stat_annotation(ax_hist_stim, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)