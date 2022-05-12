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



def merge_pseudo_hist_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,vfiles,idx_delete,EACHSTATES=20):
    unique_choices = [0,1]
    Xmerge_trials_correct,ymerge_labels_correct = {},{}
    Xmerge_trials_error,ymerge_labels_error = {},{}
    
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
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error

def merge_pseudo_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,vfiles,falsefiles,metadata,EACHSTATES=60):
    unique_choices = [0,1]
    Xmerge_trials_correct,ymerge_labels_correct = {},{}
    yright_ratio_correct = {}
    Xmerge_trials_error,ymerge_labels_error = {},{}
    yright_ratio_error = {}
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
                try:
                    ymerge_labels_error[state,coh] = np.vstack((ymerge_labels_error[state,coh],temp_beh[idxsample])) 
                    Xmerge_trials_error[state,coh] = np.hstack((Xmerge_trials_error[state,coh],temp_trials[idxsample,:]))
                except:
                    ymerge_labels_error[state,coh] = temp_beh[idxsample]#[np.sum(temp_beh[idxsample])/len(idxsample)]
                    Xmerge_trials_error[state,coh] = temp_trials[idxsample,:]
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error