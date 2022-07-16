import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.decomposition import PCA

from numpy import *
from numpy.random import rand, randn, randint
import itertools
import scipy.stats as sstats

def valid_hist_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files,THRESH_TRIAL):
    unique_choices = [0,1]
    
    error_false     = []
    correct_false   = []
    num_hist_trials = np.zeros((8,len(files))) 
    min_hist_trials = 1e5
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
            if totaltrials<THRESH_TRIAL:
                error_false = np.append(error_false,idxf)
                continue
            if totaltrials<min_hist_trials:
                min_hist_trials = totaltrials
            num_hist_trials[state,idxf] = totaltrials

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
                    if totaltrials<THRESH_TRIAL:
                        correct_false = np.append(correct_false,idxf)
                        continue

                    if totaltrials<min_hist_trials:
                        min_hist_trials=totaltrials
                    num_hist_trials[state,idxf] = totaltrials

    return np.unique(correct_false), np.unique(error_false), min_hist_trials, num_hist_trials

##### validate behaviour state
def valid_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files, THRESH_TRIAL):
    unique_choices = [0,1]
    error_false  = []
    correct_false = []

    min_beh_trials = 1e5
    num_beh_trials = np.zeros((len(unique_cohs),8,len(files)))
    min_beh_trials = 1e5

    for idxf in range(len(files)):
        data_temp  = Xdata_set[idxf,'error'].copy()
        for idxc, coh in enumerate(unique_cohs):
            for state in unique_states:
                if state>=4:
                    break
                totaltrials=0
                for choice in unique_choices:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                        if coh>-2 and np.shape(data_temp[state,coh,choice])[0]==0:#>-1
                        #     # print('file-',idxf,'  choice-',choice,' trials-',np.shape(data_temp[state,coh,choice])[0])
                            error_false=np.append(error_false,idxf)
                        #     # print('error!!!!!!')
                            continue

                    except:
                        error_false=np.append(error_false,idxf)
                        continue
                if (totaltrials<THRESH_TRIAL):
                    # print('error',state,'--',coh,'--',choice,'--',idxf) 
                    error_false=np.append(error_false,idxf)
                    continue
                if totaltrials<min_beh_trials:
                    min_beh_trials=totaltrials 
                num_beh_trials[idxc,state,idxf] = totaltrials
            

    for idxf in range(len(files)):
        data_temp  = Xdata_set[idxf,'correct'].copy()
        for idxc, coh in enumerate(unique_cohs):
            for state in unique_states:
                if state<4:
                    continue
                totaltrials=0
                for choice in unique_choices:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                        if coh>-2 and np.shape(data_temp[state,coh,choice])[0]==0:#>-1
                        #     # print('file-',idxf,'  choice-',choice,' trials-',np.shape(data_temp[state,coh,choice])[0])
                            error_false=np.append(error_false,idxf)
                        #     # print('error!!!!!!')
                            continue
                    except:
                        correct_false=np.append(correct_false,idxf)
                        continue
                if (totaltrials<THRESH_TRIAL):
                    # print('correct',state,'--',coh,'--',choice,'--',idxf) 
                    correct_false=np.append(correct_false,idxf)
                    continue
                if totaltrials<min_beh_trials:
                    min_beh_trials = totaltrials 
                num_beh_trials[idxc,state,idxf]= totaltrials
    return np.unique(correct_false), np.unique(error_false), min_beh_trials, num_beh_trials



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

def merge_pseudo_hist_trials_individual(Xdata_set,ylabels_set,unique_states,unique_cohs,files,idx_delete,EACHSTATES=20,RECORD_TRIALS=0, RECORDED_TRIALS=[]):
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
            #### generate sampled true trials for individual neurons in each pseudo trial
            NN = np.shape(data_temp[state])[1]
            sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
            Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
            ymerge_trials_t     = np.zeros((EACHSTATES,np.shape(label_temp[state])[1]))
            for iii in range(EACHSTATES):
                sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                Xmerge_trials_t[iii,:]     = np.array([data_temp[state][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                ymerge_trials_t[iii,:]     = label_temp[state][iidrandom,:]

            if(RECORD_TRIALS):
                merge_trials_hist[state,idxf] = sampled_true_trials
            else:
                idxsample = RECORDED_TRIALS[state,idxf]
            if (idxf == 0):
                ymerge_labels_error[state] = ymerge_trials_t
                Xmerge_trials_error[state] = Xmerge_trials_t
            else:
                try:    
                    Xmerge_trials_error[state] = np.hstack((Xmerge_trials_error[state],Xmerge_trials_t))
                    ymerge_labels_error[state] = np.hstack((ymerge_labels_error[state],ymerge_trials_t))
                except:
                    ymerge_labels_error[state] = ymerge_trials_t
                    Xmerge_trials_error[state] = Xmerge_trials_t
    
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
            #### generate sampled true trials for individual neurons in each pseudo trial
            NN = np.shape(data_temp[state])[1]
            sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
            Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
            ymerge_trials_t     = np.zeros((EACHSTATES,np.shape(label_temp[state])[1]))
            for iii in range(EACHSTATES):
                sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                Xmerge_trials_t[iii,:]     = np.array([data_temp[state][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                ymerge_trials_t[iii,:]     = label_temp[state][iidrandom,:]

            if(RECORD_TRIALS):
                merge_trials_hist[state,idxf] = sampled_true_trials
            else:
                idxsample = RECORDED_TRIALS[state,idxf]
            if (idxf == 0):
                ymerge_labels_correct[state] = ymerge_trials_t
                Xmerge_trials_correct[state] = Xmerge_trials_t
            else:
                try:    
                    Xmerge_trials_correct[state] = np.hstack((Xmerge_trials_correct[state],Xmerge_trials_t))
                    ymerge_labels_correct[state] = np.hstack((ymerge_labels_correct[state],ymerge_trials_t))
                except:
                    ymerge_labels_correct[state] = ymerge_trials_t
                    Xmerge_trials_correct[state] = Xmerge_trials_t

    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials_hist

def shuffle_pseudo_hist_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files,idx_delete,EACHSTATES=20,RECORD_TRIALS=0, RECORDED_TRIALS=[]):
    unique_choices = [0,1]
    Xmerge_trials_correct,ymerge_labels_correct = {},{}
    Xmerge_trials_error,ymerge_labels_error = {},{}

    merge_trials_hist = {}
    NSTATES  = 4
    ctxt_iid = np.zeros((8,2))
    ctxt_iid[0,:] = np.array([0,1])
    ctxt_iid[1,:] = np.array([0,1])

    ctxt_iid[2,:] = np.array([2,3])
    ctxt_iid[3,:] = np.array([2,3])

    ctxt_iid[4,:] = np.array([4,5])
    ctxt_iid[5,:] = np.array([4,5])

    ctxt_iid[6,:] = np.array([6,7])
    ctxt_iid[7,:] = np.array([6,7])


    for state in unique_states:
        if state>=4:
            break
        for idxf in range(len(files)):
            if(idxf in idx_delete):
                continue
            data_temp  = Xdata_set[idxf,'error'].copy()
            label_temp = ylabels_set[idxf,'error'].copy()
            ### record shuffling trials 
            data_shuffle  = np.zeros((EACHSTATES,np.shape(data_temp[state])[1]))
            label_shuffle = np.zeros((EACHSTATES,np.shape(label_temp[state])[1]))

            shuffle_states = np.random.choice(np.squeeze(ctxt_iid[state,:]),size=EACHSTATES,replace=True)
            for idx_s, shuffle_state in enumerate(shuffle_states):
                try:
                    totaltrials=np.shape(data_temp[shuffle_state])[0]
                except:
                    print('exist for initial finding')
                    continue
                if totaltrials<1:
                    continue                  
                idxsample = np.random.choice(np.arange(totaltrials),size=1,replace=False)
                data_shuffle[idx_s,:] = data_temp[shuffle_state][idxsample,:]
                label_shuffle[idx_s,:]= label_temp[shuffle_state][idxsample,:]

            if (idxf == 0):
                ymerge_labels_error[state] = label_shuffle
                Xmerge_trials_error[state] = data_shuffle
            else:
                try:    
                    Xmerge_trials_error[state] = np.hstack((Xmerge_trials_error[state],data_shuffle))
                    ymerge_labels_error[state] = np.hstack((ymerge_labels_error[state],label_shuffle))
                except:
                    ymerge_labels_error[state] = label_shuffle
                    Xmerge_trials_error[state] = data_shuffle
    
    for state in unique_states:
        if state<4:
            continue
        for idxf in range(len(files)):
            if(idxf in idx_delete):
                continue
            data_temp  = Xdata_set[idxf,'correct'].copy()
            label_temp = ylabels_set[idxf,'correct'].copy()

            ### record shuffling trials 
            data_shuffle  = np.zeros((EACHSTATES,np.shape(data_temp[state])[1]))
            label_shuffle = np.zeros((EACHSTATES,np.shape(label_temp[state])[1]))

            shuffle_states = np.random.choice(np.squeeze(ctxt_iid[state,:]),size=EACHSTATES,replace=True)

            for idx_s, shuffle_state in enumerate(shuffle_states):
                try:
                    totaltrials=np.shape(data_temp[shuffle_state])[0]
                except:
                    print('exist for initial finding')
                    continue
                if totaltrials<1:
                    continue                  
                idxsample = np.random.choice(np.arange(totaltrials),size=1,replace=False)
                data_shuffle[idx_s,:] = data_temp[shuffle_state][idxsample,:]
                label_shuffle[idx_s,:]= label_temp[shuffle_state][idxsample,:]

            if (idxf == 0):
                ymerge_labels_correct[state] = label_shuffle
                Xmerge_trials_correct[state] = data_shuffle
            else:
                try:    
                    Xmerge_trials_correct[state] = np.hstack((Xmerge_trials_correct[state],data_shuffle))
                    ymerge_labels_correct[state] = np.hstack((ymerge_labels_correct[state],label_shuffle))
                except:
                    ymerge_labels_correct[state] = label_shuffle
                    Xmerge_trials_correct[state] = data_shuffle
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials_hist

def merge_pseudo_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,vfiles,falsefiles,metadata,EACHSTATES=60, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[],STIM_BEH=1):
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
                label_temp = ylabels_set[idxf,'correct'].copy()
                temp_trials = []
                temp_beh    = []
                for choice in unique_choices: 
                    if np.shape(temp_trials)[0]==0:
                        temp_trials= data_temp[state,coh,choice]
                        if(STIM_BEH==0):
                            temp_beh   = label_temp[state,coh,choice][:,3::6]-2
                            temp_beh   = temp_beh.flatten()
                        elif(STIM_BEH==1):
                            temp_beh   = choice*np.ones(np.shape(temp_trials)[0])
                        # print('~~~~~~~~~~~ shape labels:',np.shape(temp_beh))
                        # temp_beh   = np.reshape(temp_beh,(1,-1))
                    else:
                        temp_trials = np.vstack((temp_trials,data_temp[state,coh,choice]))
                        pend_ylabel = label_temp[state,coh,choice][:,3::6]-2
                        pend_ylabel = pend_ylabel.flatten()
                        if(STIM_BEH==0):
                            temp_beh   = np.hstack((temp_beh,pend_ylabel))
                        elif(STIM_BEH==1):
                            temp_beh   = np.hstack((temp_beh,choice*np.ones(np.shape(data_temp[state,coh,choice])[0])))
                totaltrials = np.shape(temp_trials)[0] 
                ### permute -- for the same state and coherence
                tpermute_idx = np.random.permutation(np.arange(totaltrials))
                temp_beh     = temp_beh[tpermute_idx]
                temp_trials  = temp_trials[tpermute_idx,:]  
                
                if(RECORD_TRIALS):
                    # if(coh==0):
                    #     print('file:',idxf,'~~~~state:',state,'~~~~~~~~~~ total ac trials coh 0:',totaltrials)
                    idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
                    merge_trials[state,coh,idxf] = idxsample
                else:
                    idxsample = RECORDED_TRIALS_SET[state,coh,idxf]
                try:
                    # if(STIM_BEH==1):
                    #     labels = np.reshape(temp_beh[idxsample],(-1,1))
                    # elif(STIM_BEH==0):
                    #     labels = temp_beh[idxsample].copy()
                    ymerge_labels_correct[state,coh] = np.vstack((ymerge_labels_correct[state,coh],temp_beh[idxsample])) #labels))
                    Xmerge_trials_correct[state,coh] = np.hstack((Xmerge_trials_correct[state,coh],temp_trials[idxsample,:]))
                except:
                    # if(STIM_BEH==1):
                    #     labels = np.reshape(temp_beh[idxsample],(-1,1))
                    # elif(STIM_BEH==0):
                    #     labels = temp_beh[idxsample].copy()
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
                label_temp = ylabels_set[idxf,'error'].copy()
                temp_trials = []
                temp_beh    = []
                for choice in unique_choices: 
                    if np.shape(temp_trials)[0]==0:
                        temp_trials= data_temp[state,coh,choice]
                        if(STIM_BEH==0):
                            temp_beh   = label_temp[state,coh,choice][:,3::6]#choice*np.ones(np.shape(temp_trials)[0])
                            temp_beh   = temp_beh.flatten()
                        elif(STIM_BEH==1):
                            temp_beh   = choice*np.ones(np.shape(temp_trials)[0])
                    else:
                        temp_trials = np.vstack((temp_trials,data_temp[state,coh,choice]))
                        pend_ylabel = label_temp[state,coh,choice][:,3::6]
                        pend_ylabel = pend_ylabel.flatten()
                        if(STIM_BEH==0):
                            temp_beh   = np.hstack((temp_beh,pend_ylabel))
                        elif(STIM_BEH==1):
                            temp_beh   = np.hstack((temp_beh,choice*np.ones(np.shape(data_temp[state,coh,choice])[0])))
                        # temp_beh   = np.hstack((temp_beh,choice*np.ones(np.shape(data_temp[state,coh,choice])[0])))
                totaltrials = np.shape(temp_trials)[0]  
                ### permute -- for the same state and coherence
                tpermute_idx = np.random.permutation(np.arange(totaltrials))
                temp_beh     = temp_beh[tpermute_idx]
                temp_trials  = temp_trials[tpermute_idx,:]  
                
                if(RECORD_TRIALS):
                    # if(coh==0):
                    #     print('file:',idxf,'~~~~state:',state,'~~~~~~~~~~ total ae trials coh 0:',totaltrials)
                    idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
                    merge_trials[state,coh,idxf] = idxsample
                else:
                    idxsample = RECORDED_TRIALS_SET[state,coh,idxf]
                try:                    
                    ymerge_labels_error[state,coh] = np.vstack((ymerge_labels_error[state,coh],temp_beh[idxsample]))# np.hstack((ymerge_labels_correct[state,coh],labels))# 
                    Xmerge_trials_error[state,coh] = np.hstack((Xmerge_trials_error[state,coh],temp_trials[idxsample,:]))
                except:
                    ymerge_labels_error[state,coh] = temp_beh[idxsample]#[np.sum(temp_beh[idxsample])/len(idxsample)]
                    Xmerge_trials_error[state,coh] = temp_trials[idxsample,:]
            # if coh==0:
            #     print("~~~~~~~ratio:",np.mean(ymerge_labels_error[state,0],axis=0))
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials


def merge_pseudo_beh_trials_individual(Xdata_set,ylabels_set,unique_states,unique_cohs,vfiles,falsefiles,metadata,EACHSTATES=60, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[],STIM_BEH=1):
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
            if coh >-2:
                for idxf in range(len(vfiles)):
                    if(idxf in falsefiles):
                        continue
                    data_temp  = Xdata_set[idxf,'correct'].copy()
                    label_temp = ylabels_set[idxf,'correct'].copy()
                    temp_trials = {}
                    temp_beh    = {}
                    totaltrials = {}
                    for choice in unique_choices: 
                        temp_trials[choice] = data_temp[state,coh,choice]
                        # print('file-',idxf,' state - coh-choice-',state, coh, choice,' shape:',np.shape(data_temp[state,coh,choice]))
                        if(STIM_BEH==0):
                            temp_beh[choice]   = label_temp[state,coh,choice][:,3::6]-2
                            temp_beh[choice]   = temp_beh.flatten()
                        elif(STIM_BEH==1):
                            temp_beh[choice]   = choice*np.ones(np.shape(temp_trials[choice])[0])
                        totaltrials[choice]  = np.shape(temp_trials[choice])[0]


                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN = np.shape(temp_trials[choice])[1]
                    sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
                    Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
                    ymerge_trials_t     = np.zeros((1,EACHSTATES))
                    if(totaltrials[unique_choices[0]]>0 and totaltrials[unique_choices[1]]>0 ):

                        # for iii in range(int(EACHSTATES/2)):
                        #     sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[0]]),size=NN, replace=True)
                        #     Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[0]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                        #     cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                        #     ymerge_trials_t[0,iii]     = 0#temp_beh[unique_choices[0]][cchoice]

                        # for iii in range(int(EACHSTATES/2),EACHSTATES):
                        #     sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[1]]),size=NN, replace=True)
                        #     Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[1]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                        #     cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                        #     ymerge_trials_t[0,iii]     = 1#temp_beh[unique_choices[1]][cchoice]

                        sampled_true = np.random.choice(np.arange(totaltrials[unique_choices[0]]),size=int(EACHSTATES/2), replace=True)
                        Xmerge_trials_t[:int(EACHSTATES/2),:] = temp_trials[unique_choices[0]][sampled_true,:]
                        ymerge_trials_t[0,:int(EACHSTATES/2)]     = 0#temp_beh[unique_choices[0]][cchoice]

                        sampled_true = np.random.choice(np.arange(totaltrials[unique_choices[1]]),size=int(EACHSTATES/2), replace=True)
                        Xmerge_trials_t[int(EACHSTATES/2):,:] = temp_trials[unique_choices[1]][sampled_true,:]
                        ymerge_trials_t[0,int(EACHSTATES/2):]     = 1#temp_beh[unique_choices[0]][cchoice]

                    elif(totaltrials[unique_choices[0]]>0 and totaltrials[unique_choices[1]]==0 ):
                        for iii in range(int(EACHSTATES)):
                            sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[0]]),size=NN, replace=True)
                            Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[0]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                            cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                            ymerge_trials_t[0,iii]     = temp_beh[unique_choices[0]][cchoice]
                    elif(totaltrials[unique_choices[0]]==0 and totaltrials[unique_choices[1]]>0 ):
                        for iii in range(int(EACHSTATES)):
                            sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[1]]),size=NN, replace=True)
                            Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[1]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                            cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                            ymerge_trials_t[0,iii]     = temp_beh[unique_choices[1]][cchoice]
                    
                    if(RECORD_TRIALS):
                        merge_trials[state,coh,idxf] = sampled_true_trials
                    else:
                        idxsample = RECORDED_TRIALS_SET[state,coh,idxf]
                    try:
                        ymerge_labels_correct[state,coh] = np.vstack((ymerge_labels_correct[state,coh],ymerge_trials_t))
                        Xmerge_trials_correct[state,coh] = np.hstack((Xmerge_trials_correct[state,coh],Xmerge_trials_t))
                    except:
                        ymerge_labels_correct[state,coh] = ymerge_trials_t
                        Xmerge_trials_correct[state,coh] = Xmerge_trials_t
                ### permute -- for the same state and coherence
                tpermute_idx = np.random.permutation(np.arange(np.shape(Xmerge_trials_correct[state,coh])[0]))
                Xmerge_trials_correct[state,coh]    = Xmerge_trials_correct[state,coh][tpermute_idx,:]
                # print('~~~~~~ correct shape data:',np.shape(Xmerge_trials_correct[state,coh]))
                ymerge_labels_correct[state,coh]  = ymerge_labels_correct[state,coh][:,tpermute_idx]  
            else:
                for idxf in range(len(vfiles)):
                    if(idxf in falsefiles):
                        continue
                    data_temp  = Xdata_set[idxf,'correct'].copy()
                    label_temp = ylabels_set[idxf,'correct'].copy()
                    
                    temp_trials = []
                    temp_beh    = []
                    for choice in unique_choices: 
                        if np.shape(temp_trials)[0]==0:
                            temp_trials= data_temp[state,coh,choice]
                            if(STIM_BEH==0):
                                temp_beh   = label_temp[state,coh,choice][:,3::6]
                                temp_beh   = temp_beh.flatten()
                            elif(STIM_BEH==1):
                                temp_beh   = choice*np.ones(np.shape(temp_trials)[0])
                        else:
                            temp_trials = np.vstack((temp_trials,data_temp[state,coh,choice]))
                            pend_ylabel = label_temp[state,coh,choice][:,3::6]
                            pend_ylabel = pend_ylabel.flatten()
                            if(STIM_BEH==0):
                                temp_beh   = np.hstack((temp_beh,pend_ylabel))
                            elif(STIM_BEH==1):
                                temp_beh   = np.hstack((temp_beh,choice*np.ones(np.shape(data_temp[state,coh,choice])[0])))


                    #### generate sampled true trials for individual neurons in each pseudo trial
                    totaltrials = np.shape(temp_trials)[0]  
                    NN = np.shape(temp_trials)[1]
                    sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
                    Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
                    ymerge_trials_t     = np.zeros((1,EACHSTATES))
                    for iii in range(int(EACHSTATES)):
                        sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                        Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                        cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                        ymerge_trials_t[0,iii]     = np.mean(temp_beh[sampled_true_trials[iii,:]])

                
                    if(RECORD_TRIALS):
                        merge_trials[state,coh,idxf] = sampled_true_trials
                    else:
                        idxsample = RECORDED_TRIALS_SET[state,coh,idxf]
                    try:
                        ymerge_labels_correct[state,coh] = np.vstack((ymerge_labels_correct[state,coh],ymerge_trials_t))
                        Xmerge_trials_correct[state,coh] = np.hstack((Xmerge_trials_correct[state,coh],Xmerge_trials_t))
                    except:
                        ymerge_labels_correct[state,coh] = ymerge_trials_t
                        Xmerge_trials_correct[state,coh] = Xmerge_trials_t

    for state in unique_states:
        if state>=4:
            break
        for coh in unique_cohs:
            if coh>-2:
                for idxf in range(len(vfiles)):
                    if(idxf in falsefiles):
                        continue
                    data_temp  = Xdata_set[idxf,'error'].copy()
                    label_temp = ylabels_set[idxf,'error'].copy()
                    
                    temp_trials = {}
                    temp_beh    = {}
                    totaltrials = {}
                    for choice in unique_choices: 
                        temp_trials[choice] = data_temp[state,coh,choice]
                        # print('file-',idxf,' state - coh-choice-',state, coh, choice,' shape:',np.shape(data_temp[state,coh,choice]))
                        if(STIM_BEH==0):
                            temp_beh[choice]   = label_temp[state,coh,choice][:,3::6]-2
                            temp_beh[choice]   = temp_beh.flatten()
                        elif(STIM_BEH==1):
                            temp_beh[choice]   = choice*np.ones(np.shape(temp_trials[choice])[0])
                        totaltrials[choice]  = np.shape(temp_trials[choice])[0]


                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN = np.shape(temp_trials[choice])[1]
                    sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
                    Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
                    ymerge_trials_t     = np.zeros((1,EACHSTATES))
                    if(totaltrials[unique_choices[0]]>0 and totaltrials[unique_choices[1]]>0 ):

                        # for iii in range(int(EACHSTATES/2)):
                        #     sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[0]]),size=NN, replace=True)
                        #     Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[0]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                        #     cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                        #     ymerge_trials_t[0,iii]     = 0#temp_beh[unique_choices[0]][cchoice]

                        # for iii in range(int(EACHSTATES/2),EACHSTATES):
                        #     sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[1]]),size=NN, replace=True)
                        #     Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[1]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                        #     cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                        #     ymerge_trials_t[0,iii]     = 1#temp_beh[unique_choices[1]][cchoice]

                        sampled_true = np.random.choice(np.arange(totaltrials[unique_choices[0]]),size=int(EACHSTATES/2), replace=True)
                        Xmerge_trials_t[:int(EACHSTATES/2),:] = temp_trials[unique_choices[0]][sampled_true,:]
                        ymerge_trials_t[0,:int(EACHSTATES/2)]     = 0#temp_beh[unique_choices[0]][cchoice]

                        sampled_true = np.random.choice(np.arange(totaltrials[unique_choices[1]]),size=int(EACHSTATES/2), replace=True)
                        Xmerge_trials_t[int(EACHSTATES/2):,:] = temp_trials[unique_choices[1]][sampled_true,:]
                        ymerge_trials_t[0,int(EACHSTATES/2):]     = 1#temp_beh[unique_choices[0]][cchoice]

                    elif(totaltrials[unique_choices[0]]>0 and totaltrials[unique_choices[1]]==0 ):
                        for iii in range(int(EACHSTATES)):
                            sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[0]]),size=NN, replace=True)
                            Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[0]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                            cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                            ymerge_trials_t[0,iii]     = temp_beh[unique_choices[0]][cchoice]
                    elif(totaltrials[unique_choices[0]]==0 and totaltrials[unique_choices[1]]>0 ):
                        for iii in range(int(EACHSTATES)):
                            sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials[unique_choices[1]]),size=NN, replace=True)
                            Xmerge_trials_t[iii,:]     = np.array([temp_trials[unique_choices[1]][item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                            cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                            ymerge_trials_t[0,iii]     = temp_beh[unique_choices[1]][cchoice]

                
                    if(RECORD_TRIALS):
                        merge_trials[state,coh,idxf] = sampled_true_trials
                    else:
                        idxsample = RECORDED_TRIALS_SET[state,coh,idxf]
                        
                    # print('~~~~shape.....t',np.shape(Xmerge_trials_t),np.shape(ymerge_trials_t))

                    try:
                        ymerge_labels_error[state,coh] = np.vstack((ymerge_labels_error[state,coh],ymerge_trials_t))
                        Xmerge_trials_error[state,coh] = np.hstack((Xmerge_trials_error[state,coh],Xmerge_trials_t))
                        # print('~~~~~~ merg error shape data:',np.shape(Xmerge_trials_error[state,coh]))
                    except:
                        # print('~~~~~error file:',idxf)
                        ymerge_labels_error[state,coh] = ymerge_trials_t
                        Xmerge_trials_error[state,coh] = Xmerge_trials_t
                        # print('~~~~shape.....t',np.shape(Xmerge_trials_error[state,coh]),np.shape(ymerge_labels_error[state,coh]))

                ### permute -- for the same state and coheren
                tpermute_idx = np.random.permutation(np.arange(np.shape(Xmerge_trials_error[state,coh])[0]))
                Xmerge_trials_error[state,coh]    = Xmerge_trials_error[state,coh][tpermute_idx,:]
                # print('~~~~~~ error shape data:',np.shape(Xmerge_trials_error[state,coh]))
                ymerge_labels_error[state,coh]  = ymerge_labels_error[state,coh][:,tpermute_idx]  
            else:
                for idxf in range(len(vfiles)):
                    if(idxf in falsefiles):
                        continue
                    data_temp  = Xdata_set[idxf,'error'].copy()
                    label_temp = ylabels_set[idxf,'error'].copy()
                    
                    temp_trials = []
                    temp_beh    = []
                    for choice in unique_choices: 
                        if np.shape(temp_trials)[0]==0:
                            temp_trials= data_temp[state,coh,choice]
                            if(STIM_BEH==0):
                                temp_beh   = label_temp[state,coh,choice][:,3::6]
                                temp_beh   = temp_beh.flatten()
                            elif(STIM_BEH==1):
                                temp_beh   = choice*np.ones(np.shape(temp_trials)[0])
                        else:
                            temp_trials = np.vstack((temp_trials,data_temp[state,coh,choice]))
                            pend_ylabel = label_temp[state,coh,choice][:,3::6]
                            pend_ylabel = pend_ylabel.flatten()
                            if(STIM_BEH==0):
                                temp_beh   = np.hstack((temp_beh,pend_ylabel))
                            elif(STIM_BEH==1):
                                temp_beh   = np.hstack((temp_beh,choice*np.ones(np.shape(data_temp[state,coh,choice])[0])))


                    #### generate sampled true trials for individual neurons in each pseudo trial
                    totaltrials = np.shape(temp_trials)[0]  
                    NN = np.shape(temp_trials)[1]
                    sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
                    Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
                    ymerge_trials_t     = np.zeros((1,EACHSTATES))
                    for iii in range(int(EACHSTATES)):
                        sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                        Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                        cchoice = np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                        ymerge_trials_t[0,iii]     = np.mean(temp_beh[sampled_true_trials[iii,:]])
                        # if coh>-2:
                        #     if ymerge_trials_t[0,iii]>0.5:
                        #         ymerge_trials_t[0,iii] = 1
                        #     else:
                        #         ymerge_trials_t[0,iii] = 0

                
                    if(RECORD_TRIALS):
                        merge_trials[state,coh,idxf] = sampled_true_trials
                    else:
                        idxsample = RECORDED_TRIALS_SET[state,coh,idxf]

                    try:
                        ymerge_labels_error[state,coh] = np.vstack((ymerge_labels_error[state,coh],ymerge_trials_t))
                        Xmerge_trials_error[state,coh] = np.hstack((Xmerge_trials_error[state,coh],Xmerge_trials_t))
                    except:
                        ymerge_labels_error[state,coh] = ymerge_trials_t
                        Xmerge_trials_error[state,coh] = Xmerge_trials_t


    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials

def merge_pseudo_beh_trials_stimperiod(Xdata_set,ylabels_set,unique_states,unique_cohs,vfiles,falsefiles,EACHSTATES=60, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[]):
    unique_choices = [0,1]
    cohs_true      =  [-1.00000000e+00, -7.55200000e-01,  -4.41500000e-01, -0.00000000e+00,  4.41500000e-01,  7.55200000e-01, 1.00000000e+0]#[-1., -0.4816, -0.2282,  0. ,  0.2282,  0.4816,  1.] #[-1., -0.4816, -0.2282,  0. ,  0.2282,  0.4816,  1.]    
    Xmerge_trials,ymerge_labels = {},{}
    merge_trials = {}

    for coh_t in cohs_true:
    	for idxf in range(len(vfiles)): 
            if(idxf in falsefiles):  
                continue 
            temp_trials = [] 
            temp_beh    = [] 
            for state in unique_states: 
                for coh in unique_cohs: 
                    for choice in unique_choices: 
                        if state<4: 
                            data_temp  = Xdata_set[idxf,'error'][state,coh,choice].copy() 
                            label_temp = ylabels_set[idxf,'error'][state,coh,choice].copy() 
                        elif(state>=4): 
                            data_temp  = Xdata_set[idxf,'correct'][state,coh,choice].copy() 
                            label_temp = ylabels_set[idxf,'correct'][state,coh,choice].copy()-2 
                            label_temp[:,5]+=2
                        possible_cohs = (label_temp[:,5].copy()).flatten() 
                        possible_trials = np.where(possible_cohs==coh_t) 
                        if(len(possible_trials)<1): 
                            continue  
                        if np.shape(temp_trials)[0]==0: 
                            temp_trials = data_temp[possible_trials]  
                            temp_beh    = label_temp[possible_trials] 
                        else: 
                            temp_trials = np.vstack((temp_trials,data_temp[possible_trials]))  
                            temp_beh    = np.vstack((temp_beh,label_temp[possible_trials])) 
            totaltrials = np.shape(temp_trials)[0]   
		                
            if(RECORD_TRIALS):
                # print('stim:',coh_t,' file:',idxf, ' shape:',np.shape(totaltrials))
                idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
                merge_trials[coh_t,idxf] = idxsample
            else:
                idxsample = RECORDED_TRIALS_SET[coh_t,idxf]
            try:
                ymerge_labels[coh_t] = np.hstack((ymerge_labels[coh_t],temp_beh[idxsample]))
                Xmerge_trials[coh_t] = np.hstack((Xmerge_trials[coh_t],temp_trials[idxsample,:]))
            except:
                ymerge_labels[coh_t] = temp_beh[idxsample]
                Xmerge_trials[coh_t] = temp_trials[idxsample,:]
    return Xmerge_trials,ymerge_labels,merge_trials


def behaviour_trbias_proj(coeffs_pool, intercepts_pool, Xmerge_trials,
                                     ymerge_labels, unique_states,unique_cohs,unique_choices, num_beh_trials, EACHSTATES=20,FIX_TRBIAS_BINS=[],NBINS=5,mmodel=[],PCA_n_components=0):

    MAXV = 100
    NDEC = int(np.shape(coeffs_pool)[1]/5)
    NN   = np.shape(Xmerge_trials[unique_states[0],unique_cohs[0]])[1]
    NS, NC, NCH = len(unique_states),len(unique_cohs),len(unique_choices)
    nbins_trbias = NBINS
    psychometric_trbias = np.zeros((len(unique_cohs), nbins_trbias))
    trbias_range        = np.zeros((len(unique_cohs), nbins_trbias))
    
    # fig, ax =plt.subplots(figsize=(4,4))
    # print('unique cohs:',unique_cohs)
    for idxcoh, coh in enumerate(unique_cohs):
        maxtrbias,mintrbias=-MAXV,MAXV
        evidences   = []#np.zeros(NS*NCH*EACHSTATES)
        rightchoice = []#np.zeros(NS*NCH*EACHSTATES)
        trbias_w    = []
        for idxs, state in enumerate(unique_states):
            ### cal weight for averaging
            num_true = np.mean(num_beh_trials[idxcoh,state,:]) 
            weight_per = num_true/EACHSTATES 
            if(PCA_n_components>0):
                Xmerge_trials[state,coh] = mmodel.transform(Xmerge_trials[state,coh])
            for idx in range(EACHSTATES):
                Xdata_test = Xmerge_trials[state,coh][idx,:]
                idxdecoder = idx+idxs*EACHSTATES+idxcoh*(len(unique_states)*EACHSTATES)#np.random.choice(np.arange(0, NDEC, 1),size=1, replace=True)
                idxdecoder = np.mod(idxdecoder, NDEC)
                linw_bias, linb_bias = coeffs_pool[:, idxdecoder*5+3], intercepts_pool[0, 5*idxdecoder+3]
                # print('~~~~merge:',np.shape(Xdata_test),np.shape(linw_bias))
                evidences   = np.append(evidences,np.squeeze(
                Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias))
                temp_perc   = np.sum(ymerge_labels[state,coh][:,idx])/np.shape(ymerge_labels[state,coh])[0]
                # if coh == 0:
                #     print('~~~~~right percentage:', temp_perc)
                # if temp_perc>0.5:
                #     temp_perc=1   
                # else: 
                #     temp_perc = 0
                # temp_perc   = np.sum(ymerge_labels[state,coh][idx,:])/np.shape(ymerge_labels[state,coh])[1]
                # print('~~~~~~expectation:',temp_perc)
                rightchoice = np.append(rightchoice,temp_perc)
                trbias_w    = np.append(trbias_w, weight_per)
        
        ### 
        maxtrbias ,mintrbias = 0.8*max(evidences),1.2*min(evidences)
        binss = np.linspace(mintrbias,maxtrbias,nbins_trbias+1)
        perc_right = np.zeros(nbins_trbias)
        ax_trbias  = (binss[1:]+binss[:-1])/2.0
        for i in range(1,nbins_trbias+1):
            idxbinh = np.where(evidences<binss[i])[0]
            idxbinl = np.where(evidences>binss[i-1])[0]
            idxbin  = np.intersect1d(idxbinh,idxbinl)
            
            ### cal normalization coefficient
            normalization   = trbias_w[idxbin]/np.sum(trbias_w[idxbin])
            perc_right[i-1] = np.sum(rightchoice[idxbin])/len(idxbin)# np.sum(rightchoice[idxbin]*normalization)#
        # ax.plot(ax_trbias,perc_right)
        psychometric_trbias[idxcoh,:] = perc_right.copy()
        trbias_range[idxcoh,:]        = ax_trbias.copy()
    return psychometric_trbias,trbias_range


def behaviour_stim_proj(yevi,ylabels, fit=False, name='',NBINS=5):
    # print('>>>>> 111shape:',np.shape(yevi),np.shape(ylabels))
    evidences = np.squeeze(yevi[:,0].copy())
    trbias_w   = ylabels[:,4::6].copy()
    rightchoice = np.mean(trbias_w,axis=1)
    nbins_trbias        = NBINS
    psychometric_trbias = np.zeros(nbins_trbias)
    trbias_range        = np.zeros(nbins_trbias)
     
    ### 
    maxtrbias ,mintrbias = max(evidences),min(evidences)
    binss = np.linspace(mintrbias,maxtrbias,nbins_trbias+1)
    perc_right = np.zeros(nbins_trbias)
    ax_trbias  = (binss[1:]+binss[:-1])/2.0
    for i in range(1,nbins_trbias+1):
        idxbinh = np.where(evidences<binss[i])[0]
        idxbinl = np.where(evidences>binss[i-1])[0]
        idxbin  = np.intersect1d(idxbinh,idxbinl)
        ### cal normalization coefficient
        perc_right[i-1] = np.sum(rightchoice[idxbin])/len(idxbin)#np.sum(rightchoice[idxbin])/len(idxbin)
    # ax.plot(ax_trbias,perc_right)
    psychometric_trbias[:] = perc_right.copy()
    trbias_range[:]        = ax_trbias.copy()
    return psychometric_trbias,trbias_range


def coh_ch_stateratio(Xdata_set,ylabels_set,unique_states,unique_cohs,files, idx_delete):
    unique_choices = [0,1]
    coh_ch_stateratio_correct, coh_ch_stateratio_error = np.zeros((3,2,4)),np.zeros((3,2,4))

    ### ~~~~~~~~~~~ after correct 
    for idxf in range(len(files)):
        if(idxf in idx_delete): 
            continue
        data_temp  = Xdata_set[idxf,'error'].copy()
        for idxc, coh in enumerate(unique_cohs):
            for state in unique_states:
                if state>=4:
                    break
                for choice in unique_choices:
                    coh_ch_stateratio_error[idxc,choice,state]+=np.shape(data_temp[state,coh,choice])[0]

            

    for idxf in range(len(files)):
        if(idxf in idx_delete):
            continue
        data_temp  = Xdata_set[idxf,'correct'].copy()
        for idxc, coh in enumerate(unique_cohs):
            for state in unique_states:
                if state<4:
                    continue
                totaltrials=0
                for choice in unique_choices:
                    coh_ch_stateratio_correct[idxc,choice,state-4]+=np.shape(data_temp[state,coh,choice])[0]
                    # if(choice==1 and idxc==2):
                    #     print('~~~~state',state,' num.',np.shape(data_temp[state,coh,choice])[0])

    ### calculate the ratio
    for idxc, coh in enumerate(unique_cohs):
        for idxch, choice in enumerate(unique_choices):
            coh_ch_stateratio_correct[idxc,idxch,:] /= np.sum(coh_ch_stateratio_correct[idxc,idxch,:])
            coh_ch_stateratio_error[idxc,idxch,:] /=np.sum(coh_ch_stateratio_error[idxc,idxch,:])
            coh_ch_stateratio_correct[idxc,idxch,0::3] = np.mean(coh_ch_stateratio_correct[idxc,idxch,0::3])
            coh_ch_stateratio_correct[idxc,idxch,1:3]  = np.mean(coh_ch_stateratio_correct[idxc,idxch,1:3])
            coh_ch_stateratio_error[idxc,idxch,0::3]   = np.mean(coh_ch_stateratio_error[idxc,idxch,0::3])
            coh_ch_stateratio_error[idxc,idxch,1:3]    = np.mean(coh_ch_stateratio_error[idxc,idxch,1:3])
    ### avoid sub-sampling 
    correct_chR = (coh_ch_stateratio_correct[1,1,:]+coh_ch_stateratio_correct[2,1,:])/2.0
    correct_chL = (coh_ch_stateratio_correct[0,0,:]+coh_ch_stateratio_correct[1,0,:])/2.0 
    error_chR   = (coh_ch_stateratio_error[1,1,:]+coh_ch_stateratio_error[2,1,:])/2.0
    error_chL   = (coh_ch_stateratio_error[0,0,:]+coh_ch_stateratio_error[1,0,:])/2.0 
    for idxc,coh in enumerate(unique_cohs):
        coh_ch_stateratio_correct[idxc,0,:] = correct_chL 
        coh_ch_stateratio_correct[idxc,1,:] = correct_chR   
        coh_ch_stateratio_error[idxc,0,:]   = error_chL 
        coh_ch_stateratio_error[idxc,1,:]   = error_chR 

    return coh_ch_stateratio_correct, coh_ch_stateratio_error


def merge_pseudo_beh_trials_balanced(Xdata_set,ylabels_set,unique_states,unique_cohs,vfiles,falsefiles,coh_ch_stateratio_correct,coh_ch_stateratio_error,EACHSTATES=60, RECORD_TRIALS=1,RECORDED_TRIALS_SET=[],STIM_BEH=1):
    unique_choices = [0,1]
    Xmerge_trials_correct,ymerge_labels_correct = {},{}
    Xmerge_trials_error,ymerge_labels_error     = {},{}
    Neach_state_correct, Neach_state_error      = {},{}
    merge_trials = {}
    for idxc, coh in enumerate(unique_cohs):
        for idxch, choice in enumerate(unique_choices):
            Neach_state_correct[coh,choice] = np.zeros(4)
            Neach_state_error[coh,choice]   = np.zeros(4)
            for i in range(4):
                Neach_state_correct[coh,choice][i] = max(int(EACHSTATES*coh_ch_stateratio_correct[idxc,idxch,i]),1)
                Neach_state_error[coh,choice][i]   = max(int(EACHSTATES*coh_ch_stateratio_error[idxc,idxch,i]),1)

    #### only zero coherence
    unique_cohs = [-1,0,1]#
    for idxc, coh in enumerate(unique_cohs):
        for idxf in range(len(vfiles)):
            if(idxf in falsefiles):
                continue
            ### after correct
            data_temp  = Xdata_set[idxf,'correct'].copy()
            label_temp = ylabels_set[idxf,'correct'].copy()
            Xmerge_trials_t, ymerge_trials_t = [],[]
            for idxch, choice in enumerate(unique_choices):#Neach_state_correct[coh,choice]
                for istate in range(4):
                    pot_trials = data_temp[istate+4,coh,choice].copy()
                    n_pottrials = np.shape(pot_trials)[0]
                    if(RECORD_TRIALS):
                        idxsample  = np.random.choice(np.arange(n_pottrials),size=int(Neach_state_correct[coh,choice][istate]), replace=True)
                        merge_trials[coh,choice,istate+4,idxf] = idxsample
                    else:
                        idxsample = RECORDED_TRIALS_SET[coh,choice,istate+4,idxf]
                    
                    if (len(Xmerge_trials_t)==0):
                        Xmerge_trials_t  = pot_trials[idxsample,:].copy()
                        ymerge_trials_t  = np.reshape(choice*np.ones(len(idxsample)),(1,-1))
                    else:
                        Xmerge_trials_t  = np.vstack((Xmerge_trials_t, pot_trials[idxsample,:].copy()))
                        ymerge_trials_t = np.hstack((ymerge_trials_t,np.reshape(choice*np.ones(len(idxsample)),(1,-1))))
            # print('balanced? ',np.sum(ymerge_trials_t),' ',np.shape(ymerge_trials_t)[1])
            try:
                ymerge_labels_correct[coh] = np.vstack((ymerge_labels_correct[coh],ymerge_trials_t))
                Xmerge_trials_correct[coh] = np.hstack((Xmerge_trials_correct[coh],Xmerge_trials_t))
            except:
                ymerge_labels_correct[coh] = ymerge_trials_t
                Xmerge_trials_correct[coh] = Xmerge_trials_t

            ### after error
            data_temp  = Xdata_set[idxf,'error'].copy()
            label_temp = ylabels_set[idxf,'error'].copy()
            Xmerge_trials_t, ymerge_trials_t = [],[]
            for idxch, choice in enumerate(unique_choices):#Neach_state_error[coh,choice]
                for istate in range(4):
                    pot_trials = data_temp[istate,coh,choice].copy()
                    n_pottrials = np.shape(pot_trials)[0]
                    # print('balanced? ',np.sum(ymerge_trials_t),' ',np.shape(ymerge_trials_t)[1])
                    if(RECORD_TRIALS):
                        idxsample  = np.random.choice(np.arange(n_pottrials),size=int(Neach_state_error[coh,choice][istate]), replace=True)
                        merge_trials[coh,choice,istate,idxf] = idxsample
                    else:
                        idxsample = RECORDED_TRIALS_SET[coh,choice,istate,idxf]
                    
                    if (len(Xmerge_trials_t)==0):
                        Xmerge_trials_t  = pot_trials[idxsample,:].copy()
                        ymerge_trials_t  = np.reshape(choice*np.ones(len(idxsample)),(1,-1))
                    else:
                        Xmerge_trials_t  = np.vstack((Xmerge_trials_t, pot_trials[idxsample,:].copy()))
                        ymerge_trials_t = np.hstack((ymerge_trials_t,np.reshape(choice*np.ones(len(idxsample)),(1,-1))))
            try:
                ymerge_labels_error[coh] = np.vstack((ymerge_labels_error[coh],ymerge_trials_t))
                Xmerge_trials_error[coh] = np.hstack((Xmerge_trials_error[coh],Xmerge_trials_t))
            except:
                ymerge_labels_error[coh] = ymerge_trials_t
                Xmerge_trials_error[coh] = Xmerge_trials_t

        ### permute -- for the same state and coheren
        tpermute_idx = np.random.permutation(np.arange(np.shape(Xmerge_trials_error[coh])[0]))
        Xmerge_trials_error[coh]    = Xmerge_trials_error[coh][tpermute_idx,:]
        # print('~~~~~~ error shape data:',np.shape(Xmerge_trials_error[state,coh]))
        ymerge_labels_error[coh]  = ymerge_labels_error[coh][:,tpermute_idx] 

        tpermute_idx = np.random.permutation(np.arange(np.shape(Xmerge_trials_correct[coh])[0]))
        Xmerge_trials_correct[coh]    = Xmerge_trials_correct[coh][tpermute_idx,:]
        ymerge_labels_correct[coh]  = ymerge_labels_correct[coh][:,tpermute_idx]  


    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials

def behaviour_trbias_proj_balanced(coeffs_pool, intercepts_pool, Xmerge_trials,
                                     ymerge_labels, unique_states,unique_cohs,unique_choices, EACHSTATES=20,FIX_TRBIAS_BINS=[],NBINS=5,mmodel=[],PCA_n_components=0):

    MAXV = 100
    NDEC = int(np.shape(coeffs_pool)[1]/5)
    NN   = np.shape(Xmerge_trials[unique_cohs[0]])[1]
    NS, NC, NCH = len(unique_states),len(unique_cohs),len(unique_choices)
    nbins_trbias = NBINS
    psychometric_trbias = np.zeros((len(unique_cohs), nbins_trbias))
    trbias_range        = np.zeros((len(unique_cohs), nbins_trbias))
    
    # fig, ax =plt.subplots(figsize=(4,4))
    # print('unique cohs:',unique_cohs)
    for idxcoh, coh in enumerate(unique_cohs):
        maxtrbias,mintrbias=-MAXV,MAXV
        evidences   = []#np.zeros(NS*NCH*EACHSTATES)
        rightchoice = []#np.zeros(NS*NCH*EACHSTATES)
        trbias_w    = []
        if(PCA_n_components>0):
            Xmerge_trials[coh] = mmodel.transform(Xmerge_trials[coh])
        print('trials:',EACHSTATES,' ',np.shape(Xmerge_trials[coh])[0])
        for idx in range(np.shape(Xmerge_trials[coh])[0]):#EACHSTATES):
            Xdata_test = Xmerge_trials[coh][idx,:]
            idxdecoder = idx+idxcoh*(EACHSTATES)#np.random.choice(np.arange(0, NDEC, 1),size=1, replace=True)
            idxdecoder = np.mod(idxdecoder, NDEC)
            linw_bias, linb_bias = coeffs_pool[:, idxdecoder*5+3], intercepts_pool[0, 5*idxdecoder+3]
            # print('~~~~merge:',np.shape(Xdata_test),np.shape(linw_bias))
            evidences   = np.append(evidences,np.squeeze(
            Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias))
            temp_perc   = np.sum(ymerge_labels[coh][:,idx])/np.shape(ymerge_labels[coh])[0]
            rightchoice = np.append(rightchoice,temp_perc)
        
        ### 
        maxtrbias ,mintrbias = 0.8*max(evidences),1.2*min(evidences)
        binss = np.linspace(mintrbias,maxtrbias,nbins_trbias+1)
        perc_right = np.zeros(nbins_trbias)
        ax_trbias  = (binss[1:]+binss[:-1])/2.0
        for i in range(1,nbins_trbias+1):
            idxbinh = np.where(evidences<binss[i])[0]
            idxbinl = np.where(evidences>binss[i-1])[0]
            idxbin  = np.intersect1d(idxbinh,idxbinl)
            
            ### cal normalization coefficient
            perc_right[i-1] = np.sum(rightchoice[idxbin])/len(idxbin)# np.sum(rightchoice[idxbin]*normalization)#
        # ax.plot(ax_trbias,perc_right)
        psychometric_trbias[idxcoh,:] = perc_right.copy()
        trbias_range[idxcoh,:]        = ax_trbias.copy()
    return psychometric_trbias,trbias_range