import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.decomposition import PCA

from numpy import *
from numpy.random import rand, randn, randint
import itertools
import scipy.stats as sstats

def valid_hist_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files):
    unique_choices = [0,1]
    
    error_false   = []
    correct_false = []
    for state in unique_states:
        if state>=4:
            break
        for idxf in range(len(files)):
            data_temp  = Xdata_set[idxf,'error'].copy()
            label_temp = ylabels_set[idxf,'error'].copy()
            try:
                totaltrials=np.shape(data_temp[state])[0]
            except:
                error_false = np.append(error_false,idxf)
                continue
            if totaltrials<1:
                error_false = np.append(error_false,idxf)
                continue

    for state in unique_states:
        if state<4:
            continue
        for coh in unique_cohs:
            for choice in unique_choices:
                for idxf in range(len(files)):
                    data_temp  = Xdata_set[idxf,'correct'].copy()
                    # print(data_temp.keys())
                    label_temp = ylabels_set[idxf,'correct'].copy()
                    try:
                        totaltrials=np.shape(data_temp[state])[0]
                    except:
                        correct_false = np.append(correct_false,idxf)
                        continue
                    if totaltrials<1:
                        correct_false = np.append(correct_false,idxf)
                        continue
    return np.unique(correct_false), np.unique(error_false)

##### validate behaviour state
def valid_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files):
    unique_choices = [0,1]
    error_false  = []
    correct_false = []

    for idxf in range(len(files)):
        data_temp  = Xdata_set[idxf,'error'].copy()
        for coh in unique_cohs:
            for state in unique_states:
                if state>=4:
                    break
                totaltrials=0
                for choice in unique_choices:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                    except:
                        error_false=np.append(error_false,idxf)
                        continue
                if (totaltrials<1):
                    # print('error',state,'--',coh,'--',choice,'--',idxf) 
                    error_false=np.append(error_false,idxf)
            

    for idxf in range(len(files)):
        data_temp  = Xdata_set[idxf,'correct'].copy()
        for coh in unique_cohs:
            for state in unique_states:
                if state<4:
                    continue
                totaltrials=0
                for choice in unique_choices:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                    except:
                        correct_false=np.append(correct_false,idxf)
                        continue
                if (totaltrials<1):
                    # print('correct',state,'--',coh,'--',choice,'--',idxf) 
                    correct_false=np.append(correct_false,idxf)
    return np.unique(correct_false), np.unique(error_false)



def merge_pseudo_hist_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files,idx_delete,EACHSTATES=20,RECORD_TRIALS=0, RECORDED_TRIALS=[]):
    unique_choices = [0,1]
    Xmerge_trials_correct,ymerge_labels_correct = {},{}
    Xmerge_trials_error,ymerge_labels_error = {},{}

    merge_trials_hist = {}
    
    for state in unique_states:
        if state>=4:
            break
        for idxf in range(len(files)):
            if(idxf in idx_delete):
                continue
            data_temp  = Xdata_set[idxf,'error'].copy()
            label_temp = ylabels_set[idxf,'error'].copy()
            try:
                totaltrials=np.shape(data_temp[state])[0]
            except:
                print('exist for initial finding')
                continue
            if totaltrials<1:
                continue                  
            idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
            if(RECORD_TRIALS):
                merge_trials_hist[state,idxf] = idxsample
            else:
                idxsample = RECORDED_TRIALS[state,idxf]
            if (idxf == 0):
                ymerge_labels_error[state] = label_temp[state][idxsample,:]
                Xmerge_trials_error[state] = data_temp[state][idxsample,:]
            else:
                try:    
                    Xmerge_trials_error[state] = np.hstack((Xmerge_trials_error[state],data_temp[state][idxsample,:]))
                    ymerge_labels_error[state] = np.hstack((ymerge_labels_error[state],label_temp[state][idxsample,:]))
                except:
                    ymerge_labels_error[state] = label_temp[state][idxsample,:]
                    Xmerge_trials_error[state] = data_temp[state][idxsample,:]
    
    for state in unique_states:
        if state<4:
            continue
        for idxf in range(len(files)):
            if(idxf in idx_delete):
                continue
            data_temp  = Xdata_set[idxf,'correct'].copy()
            # print(data_temp.keys())
            label_temp = ylabels_set[idxf,'correct'].copy()
            try:
                totaltrials=np.shape(data_temp[state])[0]
            except:
                print('exist for initial finding')
                continue
            if totaltrials<1:
                continue                  
            idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
            if(RECORD_TRIALS):
                merge_trials_hist[state,idxf] = idxsample
            else:
                idxsample = RECORDED_TRIALS[state,idxf]
            if (idxf == 0):
                ymerge_labels_correct[state] = label_temp[state][idxsample,:]
                Xmerge_trials_correct[state] = data_temp[state][idxsample,:]
            else:
                try:    
                    Xmerge_trials_correct[state] = np.hstack((Xmerge_trials_correct[state],data_temp[state][idxsample,:]))
                    ymerge_labels_correct[state] = np.hstack((ymerge_labels_correct[state],label_temp[state][idxsample,:]))
                except:
                    ymerge_labels_correct[state] = label_temp[state][idxsample,:]
                    Xmerge_trials_correct[state] = data_temp[state][idxsample,:]
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials_hist

def merge_pseudo_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,vfiles,falsefiles,metadata,EACHSTATES=60, RECORD_TRIALS=1, RECORDED_TRIALS=[]):
    unique_choices = [0,1]
    Xmerge_trials_correct,ymerge_labels_correct = {},{}
    yright_ratio_correct = {}
    Xmerge_trials_error,ymerge_labels_error = {},{}
    yright_ratio_error = {}
    merge_trials = {}
    for state in unique_states:
        if state<4:
            continue
        for coh in unique_cohs:
            for idxf in range(len(vfiles)):
                if(idxf in falsefiles):
                    continue
                data_temp  = Xdata_set[idxf,'correct'].copy()
                temp_trials = []
                temp_beh    = []
                for choice in unique_choices: 
                    if np.shape(temp_trials)[0]==0:
                        temp_trials=data_temp[state,coh,choice]
                        temp_beh   = choice*np.ones(np.shape(temp_trials)[0])
                    else:
                        temp_trials=np.vstack((temp_trials,data_temp[state,coh,choice]))
                        temp_beh   = np.hstack((temp_beh,choice*np.ones(np.shape(data_temp[state,coh,choice])[0])))
                totaltrials = np.shape(temp_trials)[0]              
                idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
                if(RECORD_TRIALS):
                    merge_trials[state,coh,idxf] = idxsample
                else:
                    idxsample = RECORDED_TRIALS[state,coh,idxf]
                try:
                    ymerge_labels_correct[state,coh] = np.vstack((ymerge_labels_correct[state,coh],temp_beh[idxsample])) 
                    Xmerge_trials_correct[state,coh] = np.hstack((Xmerge_trials_correct[state,coh],temp_trials[idxsample,:]))
                except:
                    ymerge_labels_correct[state,coh] = temp_beh[idxsample]#[np.sum(temp_beh[idxsample])/len(idxsample)]
                    Xmerge_trials_correct[state,coh] = temp_trials[idxsample,:]
    for state in unique_states:
        if state>=4:
            break
        for coh in unique_cohs:
            for idxf in range(len(vfiles)):
                if(idxf in falsefiles):
                    continue
                data_temp  = Xdata_set[idxf,'error'].copy()
                temp_trials = []
                temp_beh    = []
                for choice in unique_choices: 
                    if np.shape(temp_trials)[0]==0:
                        temp_trials=data_temp[state,coh,choice]
                        temp_beh   = choice*np.ones(np.shape(temp_trials)[0])
                    else:
                        temp_trials=np.vstack((temp_trials,data_temp[state,coh,choice]))
                        temp_beh   = np.hstack((temp_beh,choice*np.ones(np.shape(data_temp[state,coh,choice])[0])))
                totaltrials = np.shape(temp_trials)[0]              
                idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
                if(RECORD_TRIALS):
                    merge_trials[state,coh,idxf] = idxsample
                else:
                    idxsample = RECORDED_TRIALS[state,coh,idxf]
                try:
                    ymerge_labels_error[state,coh] = np.vstack((ymerge_labels_error[state,coh],temp_beh[idxsample])) 
                    Xmerge_trials_error[state,coh] = np.hstack((Xmerge_trials_error[state,coh],temp_trials[idxsample,:]))
                except:
                    ymerge_labels_error[state,coh] = temp_beh[idxsample]#[np.sum(temp_beh[idxsample])/len(idxsample)]
                    Xmerge_trials_error[state,coh] = temp_trials[idxsample,:]
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials


def behaviour_trbias_proj(coeffs_pool, intercepts_pool, Xmerge_trials,
                                     ymerge_labels, unique_states,unique_cohs,unique_choices, EACHSTATES=20):

    MAXV = 100
    NDEC = int(np.shape(coeffs_pool)[1]/5)
    NN   = np.shape(Xmerge_trials[unique_states[0],unique_cohs[0]])[1]
    NS, NC, NCH = len(unique_states),len(unique_cohs),len(unique_choices)
    nbins_trbias = 5
    psychometric_trbias = np.zeros((len(unique_cohs), nbins_trbias))
    trbias_range        = np.zeros((len(unique_cohs), nbins_trbias))
    
    # fig, ax =plt.subplots(figsize=(4,4))
    for idxcoh, coh in enumerate(unique_cohs):
        maxtrbias,mintrbias=-MAXV,MAXV
        evidences   = []#np.zeros(NS*NCH*EACHSTATES)
        rightchoice = []#np.zeros(NS*NCH*EACHSTATES)
        for idxs, state in enumerate(unique_states):
            for idx in range(EACHSTATES):
                Xdata_test = Xmerge_trials[state,coh][idx,:]
                idxdecoder = np.random.choice(np.arange(0, NDEC, 1),
                                size=1, replace=True)
                linw_bias, linb_bias = coeffs_pool[:, idxdecoder*5+3], intercepts_pool[0, 5*idxdecoder+3]
                evidences   = np.append(evidences,np.squeeze(
                Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias))
                temp_perc   = np.sum(ymerge_labels[state,coh][:,idx])/np.shape(ymerge_labels[state,coh])[0]
                rightchoice = np.append(rightchoice,temp_perc)
        
        ### 
        maxtrbias ,mintrbias = max(evidences),min(evidences)
        binss = np.linspace(mintrbias,maxtrbias,nbins_trbias+1)
        perc_right = np.zeros(nbins_trbias)
        ax_trbias  = (binss[1:]+binss[:-1])/2.0
        for i in range(1,nbins_trbias+1):
            idxbinh = np.where(evidences<binss[i])[0]
            idxbinl = np.where(evidences>binss[i-1])[0]
            idxbin  = np.intersect1d(idxbinh,idxbinl)
            perc_right[i-1] = np.sum(rightchoice[idxbin])/len(idxbin)
        # ax.plot(ax_trbias,perc_right)
        psychometric_trbias[idxcoh,:] = perc_right.copy()
        trbias_range[idxcoh,:] = trbias_range.copy()
    return psychometric_trbias,trbias_range