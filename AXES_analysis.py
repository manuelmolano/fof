"""
Computing all relevant AXES
@YX 18 Jun
"""

import numpy as np
def ang_AXES(IDX_RAT,)
	### /Users/yuxiushao/Public/DataML/Auditory/DataEphys/00-01s
	dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
	timep = '/00-01s/'
	### ----------- stimulus/behaviour axis ------------------- 
	# IDX_RAT='Rat15_'
	### compute the angles between history encoding axises
	dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
	data_dec  = np.load(dataname, allow_pickle=True)
	wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']

	dataname  = dir+IDX_RAT+'data_dec_ac.npz'
	data_dec  = np.load(dataname, allow_pickle=True)
	wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']

	dataname  = dir+IDX_RAT+'data_dec.npz'
	data_dec  = np.load(dataname, allow_pickle=True)
	wi_comp, bi_comp = data_dec['coefs_correct'], data_dec['intercepts_correct']

	wiac_stim,wiae_stim,wicomp_stim = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
	wiac_beh,wiae_beh,wicomp_beh = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
	for i in range(NITERATIONS):
		wiac_stim[:,i],wiae_stim[:,i],wicomp_stim[:,i] = wi_ac[:, i*5+3],wi_ae[:, i*5+3],wi_comp[:, i*5+3]
		wiac_beh[:,i],wiae_beh[:,i],wicomp_beh[:,i] = wi_ac[:, i*5+4],wi_ae[:, i*5+4],wi_comp[:, i*5+4]

	wiac_stim, wiae_stim, wicomp_stim= np.mean(wiac_stim,axis=1),np.mean(wiae_stim,axis=1),np.mean(wicomp_stim,axis=1)
	# ag_ce, ag_cpc, ag_cpe = 0,0,0  
	stim_ac=wiac_stim/np.linalg.norm(wiac_stim)
	stim_ae=wiae_stim/np.linalg.norm(wiae_stim)
	stim_comp=wicomp_stim/np.linalg.norm(wicomp_stim)

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

	### ----------- history axis ------------------- 
	timep = '/-05s/'
	### compute the angles between history encoding axises
	dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
	data_dec  = np.load(dataname, allow_pickle=True)
	wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']

	dataname  = dir+IDX_RAT+'data_dec_ac.npz'
	data_dec  = np.load(dataname, allow_pickle=True)
	wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']

	wiac_hist,wiae_hist = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
	wiac_hist_beh,wiae_hist_beh= np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
	for i in range(NITERATIONS):
		wiac_hist[:,i],wiae_hist[:,i]= wi_ac[:, i*5+3],wi_ae[:, i*5+3]
		wiac_hist_beh[:,i],wiae_hist_beh[:,i] = wi_ac[:, i*5+4],wi_ae[:, i*5+4]

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