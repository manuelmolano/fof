import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.decomposition import PCA

from numpy import *
from numpy.random import rand, randn, randint
import itertools
import scipy.stats as sstats

##### validate behaviour state
def valid_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files, THRESH_TRIAL):
    unique_choices = [0,1]
    error_false    = []
    correct_false  = []

    min_beh_trials = 1e5
    num_beh_trials = np.zeros((len(unique_cohs),8,len(files)))
    min_beh_trials = 1e5

    # ### add context 
    # for idxf in range(len(files)):
    #     data_temp  = Xdata_set[idxf,'error'].copy()
    #     for state in unique_states:
    #         if state>=4:
    #             break
    #         totaltrials=0
    #         for idxc, coh in enumerate(unique_cohs):
    #             for choice in unique_choices:
    #                 try:
    #                     if(np.shape(data_temp[state,coh,choice])[0]==0):
    #                         print('file error 00000, count----------',idxf)
    #                     totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
    #                 except:
    #                     error_false=np.append(error_false,idxf)
    #                     continue
    #         if (totaltrials<THRESH_TRIAL):
    #             # print('error',state,'--',coh,'--',choice,'--',idxf) 
    #             error_false=np.append(error_false,idxf)
    #             continue
    #         if totaltrials<min_beh_trials:
    #             min_beh_trials=totaltrials 
    #         num_beh_trials[idxc,state,idxf] = totaltrials
            

    # for idxf in range(len(files)):
    #     data_temp  = Xdata_set[idxf,'correct'].copy()
    #     for state in unique_states:
    #         if state<4:
    #             continue
    #         totaltrials=0
    #         for idxc, coh in enumerate(unique_cohs):
    #             for choice in unique_choices:
    #                 try:
    #                     if(np.shape(data_temp[state,coh,choice])[0]==0):
    #                         print('correct 00000, count----------',idxf)
    #                     totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
    #                 except:
    #                     correct_false=np.append(correct_false,idxf)
    #                     continue
    #         if (totaltrials<THRESH_TRIAL):
    #             # print('correct',state,'--',coh,'--',choice,'--',idxf) 
    #             correct_false=np.append(correct_false,idxf)
    #             continue
    #         if totaltrials<min_beh_trials:
    #             min_beh_trials = totaltrials 
    #         num_beh_trials[idxc,state,idxf]= totaltrials

    ### add context 
    for idxf in range(len(files)):
        data_temp  = Xdata_set[idxf,'error'].copy()
        for idxc, coh in enumerate(unique_cohs):
            for state in unique_states:
                if state>=4:
                    break
                totaltrials=0
                for choice in unique_choices:
                    try:
                        # if(np.shape(data_temp[state,coh,choice])[0]==0):
                        #     print('gpt 81 file error 00000, count----------',idxf)
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
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
                        # if(np.shape(data_temp[state,coh,choice])[0]==0):
                        #     print('correct 00000, count----------',idxf)
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
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

    # ### behaviour 
    # for idxf in range(len(files)):
    #     data_temp  = Xdata_set[idxf,'error'].copy()
    #     for idxc, coh in enumerate(unique_cohs):
    #         for choice in unique_choices:
    #             totaltrials=0
    #             for state in unique_states:
    #                 if state>=4:
    #                     break
    #                 try:
    #                     totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
    #                 except:
    #                     error_false=np.append(error_false,idxf)
    #                     continue
    #             if (totaltrials<THRESH_TRIAL):
    #                 error_false=np.append(error_false,idxf)
    #                 continue
            

    # for idxf in range(len(files)):
    #     data_temp  = Xdata_set[idxf,'correct'].copy()
    #     for idxc, coh in enumerate(unique_cohs):
    #         for choice in unique_choices:
    #             totaltrials=0
    #             for state in unique_states:
    #                 if state<4:
    #                     continue
    #                 try:
    #                     totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
    #                 except:
    #                     correct_false=np.append(correct_false,idxf)
    #                     continue
    #             if (totaltrials<THRESH_TRIAL):
    #                 correct_false=np.append(correct_false,idxf)
    #                 continue
    ### behaviour 

    ### error 
    rep_states, alt_states = [0,1],[2,3]
    for idxf in range(len(files)):
        data_temp  = Xdata_set[idxf,'error'].copy()
        for idxc, coh in enumerate(unique_cohs):
            for choice in unique_choices:
                totaltrials=0
                for state in rep_states:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                    except:
                        error_false=np.append(error_false,idxf)
                        continue
                if (totaltrials<THRESH_TRIAL):
                    error_false=np.append(error_false,idxf)
                    continue
                totaltrials=0
                for state in alt_states:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                    except:
                        error_false=np.append(error_false,idxf)
                        continue
                if (totaltrials<THRESH_TRIAL):
                    error_false=np.append(error_false,idxf)
                    continue
            

    rep_states, alt_states = [4,5],[6,7]
    for idxf in range(len(files)):
        data_temp  = Xdata_set[idxf,'correct'].copy()
        for idxc, coh in enumerate(unique_cohs):
            for choice in unique_choices:
                totaltrials=0
                for state in rep_states:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                    except:
                        correct_false=np.append(correct_false,idxf)
                        continue
                if (totaltrials<THRESH_TRIAL):
                    correct_false=np.append(correct_false,idxf)
                    continue

                totaltrials=0
                for state in alt_states:
                    try:
                        totaltrials=totaltrials+np.shape(data_temp[state,coh,choice])[0]
                    except:
                        correct_false=np.append(correct_false,idxf)
                        continue
                if (totaltrials<THRESH_TRIAL):
                    correct_false=np.append(correct_false,idxf)
                    continue
    return np.unique(correct_false), np.unique(error_false), min_beh_trials, num_beh_trials


##### training and testing
def train_test_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files, falsefiles, THRESH_TRIAL):
    unique_choices = [0,1]

    Xdata_hist_trainset, ylabels_hist_trainset, Xdata_hist_testset, ylabels_hist_testset = {}, {}, {}, {}
    Xdata_beh_trainset, ylabels_beh_trainset, Xdata_beh_testset, ylabels_beh_testset = {}, {}, {}, {}
    Xdata_psy_trainset, ylabels_psy_trainset, Xdata_psy_testset, ylabels_psy_testset = {}, {}, {}, {}

    for idxf in range(len(files)):
        if(idxf in falsefiles):
            continue
        Xdata_hist_trainset[idxf,'error'], ylabels_hist_trainset[idxf,'error'], Xdata_hist_testset[idxf,'error'], ylabels_hist_testset[idxf,'error']= {},{},{},{}
        Xdata_hist_trainset[idxf,'correct'], ylabels_hist_trainset[idxf,'correct'], Xdata_hist_testset[idxf,'correct'], ylabels_hist_testset[idxf,'correct']= {},{},{},{}

        Xdata_beh_trainset[idxf,'error'], ylabels_beh_trainset[idxf,'error'], Xdata_beh_testset[idxf,'error'], ylabels_beh_testset[idxf,'error']= {},{},{},{}
        Xdata_beh_trainset[idxf,'correct'], ylabels_beh_trainset[idxf,'correct'], Xdata_beh_testset[idxf,'correct'], ylabels_beh_testset[idxf,'correct']= {},{},{},{}

        Xdata_psy_trainset[idxf,'error'], ylabels_psy_trainset[idxf,'error'], Xdata_psy_testset[idxf,'error'], ylabels_psy_testset[idxf,'error']= {},{},{},{}
        Xdata_psy_trainset[idxf,'correct'], ylabels_psy_trainset[idxf,'correct'], Xdata_psy_testset[idxf,'correct'], ylabels_psy_testset[idxf,'correct']= {},{},{},{}

        for state in unique_states:
            ### error 
            if state>=4:
                break
            data_temp  = Xdata_set[idxf,'error'].copy()
            label_temp = ylabels_set[idxf,'error'].copy()               
            for coh in unique_cohs:
                Xhist_t,yhist_t = [],[] 
                numtrials = 0
                for choice in unique_choices:
                    ### sample train and test for behaviour decoder 
                    numtrials_t = np.shape(data_temp[state,coh,choice])[0]
                    if numtrials_t>0:
                        numtrials+=numtrials_t
                        if(len(Xhist_t)==0):
                            Xhist_t = data_temp[state,coh,choice].copy() 
                            yhist_t = label_temp[state,coh,choice].copy()  
                        else:
                            Xhist_t = np.vstack((Xhist_t,data_temp[state,coh,choice].copy()))
                            yhist_t = np.vstack((yhist_t,label_temp[state,coh,choice].copy())) 
                    else:
                        continue 
                if numtrials == 0:
                    print('NO!!!!!! file:~~~',idxf,'~~~state~~~',state,'~~coh',coh)
                    continue 
                idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
                idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)  

                Xdata_psy_trainset[idxf,'error'][state,coh],ylabels_psy_trainset[idxf,'error'][state,coh]=Xhist_t[idxtrain,:],yhist_t[idxtrain,:] 
                Xdata_psy_testset[idxf,'error'][state,coh],ylabels_psy_testset[idxf,'error'][state,coh]=Xhist_t[idxtest,:],yhist_t[idxtest,:] 
                try:
                    Xdata_hist_trainset[idxf,'error'][state],ylabels_hist_trainset[idxf,'error'][state]=np.vstack((Xdata_hist_trainset[idxf,'error'][state],Xdata_psy_trainset[idxf,'error'][state,coh])),np.vstack((ylabels_hist_trainset[idxf,'error'][state],ylabels_psy_trainset[idxf,'error'][state,coh]))#Xhist_t[idxtrain,:],yhist_t[idxtrain,:] 
                    Xdata_hist_testset[idxf,'error'][state],ylabels_hist_testset[idxf,'error'][state]=np.vstack((Xdata_hist_testset[idxf,'error'][state],Xdata_psy_testset[idxf,'error'][state,coh])),np.vstack((ylabels_hist_testset[idxf,'error'][state],ylabels_psy_testset[idxf,'error'][state,coh]))#Xhist_t[idxtest,:],yhist_t[idxtest,:] 
                except:
                    Xdata_hist_trainset[idxf,'error'][state],ylabels_hist_trainset[idxf,'error'][state]=Xdata_psy_trainset[idxf,'error'][state,coh],ylabels_psy_trainset[idxf,'error'][state,coh]
                    Xdata_hist_testset[idxf,'error'][state],ylabels_hist_testset[idxf,'error'][state]=Xdata_psy_testset[idxf,'error'][state,coh],ylabels_psy_testset[idxf,'error'][state,coh]

        for state in unique_states:
            ### correct
            if state<4:
                continue
            data_temp  = Xdata_set[idxf,'correct'].copy()
            label_temp = ylabels_set[idxf,'correct'].copy()  
                        
            for coh in unique_cohs:
                Xhist_t,yhist_t = [],[]  
                numtrials = 0
                for choice in unique_choices:
                    ### sample train and test for behaviour decoder 
                    numtrials_t = np.shape(data_temp[state,coh,choice])[0]
                    if numtrials_t>0:
                        numtrials+=numtrials_t
                        if(len(Xhist_t)==0):
                            Xhist_t = data_temp[state,coh,choice].copy() 
                            yhist_t = label_temp[state,coh,choice].copy()  
                        else:
                            Xhist_t = np.vstack((Xhist_t,data_temp[state,coh,choice].copy()))
                            yhist_t = np.vstack((yhist_t,label_temp[state,coh,choice].copy())) 
                    else:
                        continue 
                if numtrials == 0:
                    print('NO!!!!!! file:~~~',idxf,'~~~state~~~',state,'~~coh',coh)
                    continue 
 
                idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
                idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)

                Xdata_psy_trainset[idxf,'correct'][state,coh],ylabels_psy_trainset[idxf,'correct'][state,coh]=Xhist_t[idxtrain,:],yhist_t[idxtrain,:] 
                Xdata_psy_testset[idxf,'correct'][state,coh],ylabels_psy_testset[idxf,'correct'][state,coh]=Xhist_t[idxtest,:],yhist_t[idxtest,:] 
                try:
                    Xdata_hist_trainset[idxf,'correct'][state],ylabels_hist_trainset[idxf,'correct'][state]=np.vstack((Xdata_hist_trainset[idxf,'correct'][state],Xdata_psy_trainset[idxf,'correct'][state,coh])),np.vstack((ylabels_hist_trainset[idxf,'correct'][state],ylabels_psy_trainset[idxf,'correct'][state,coh]))#Xhist_t[idxtrain,:],yhist_t[idxtrain,:] 
                    Xdata_hist_testset[idxf,'correct'][state],ylabels_hist_testset[idxf,'correct'][state]=np.vstack((Xdata_hist_testset[idxf,'correct'][state],Xdata_psy_testset[idxf,'correct'][state,coh])),np.vstack((ylabels_hist_testset[idxf,'correct'][state],ylabels_psy_testset[idxf,'correct'][state,coh]))#Xhist_t[idxtest,:],yhist_t[idxtest,:] 
                except:
                    Xdata_hist_trainset[idxf,'correct'][state],ylabels_hist_trainset[idxf,'correct'][state]=Xdata_psy_trainset[idxf,'correct'][state,coh],ylabels_psy_trainset[idxf,'correct'][state,coh]
                    Xdata_hist_testset[idxf,'correct'][state],ylabels_hist_testset[idxf,'correct'][state]=Xdata_psy_testset[idxf,'correct'][state,coh],ylabels_psy_testset[idxf,'correct'][state,coh]  
            # Xdata_hist_trainset[idxf,'correct'][state],ylabels_hist_trainset[idxf,'correct'][state]=Xhist_t[idxtrain,:],yhist_t[idxtrain,:] 
            # Xdata_hist_testset[idxf,'correct'][state],ylabels_hist_testset[idxf,'correct'][state]=Xhist_t[idxtest,:],yhist_t[idxtest,:] 

        # for coh in unique_cohs:
        #     for choice in unique_choices:
        #         data_temp  = Xdata_set[idxf,'error'].copy()
        #         label_temp = ylabels_set[idxf,'error'].copy()  
        #         Xbeh_t,ybeh_t = [],[]              
        #         numtrials = 0
        #         for state in unique_states:
        #             ### error 
        #             if state>=4:
        #                 break
        #             numtrials_t = np.shape(data_temp[state,coh,choice])[0]
        #             if numtrials_t>0:
        #                 numtrials+=numtrials_t
        #                 if(len(Xbeh_t)==0):
        #                     Xbeh_t = data_temp[state,coh,choice].copy() 
        #                     ybeh_t = label_temp[state,coh,choice].copy()  
        #                 else:
        #                     Xbeh_t = np.vstack((Xbeh_t,data_temp[state,coh,choice].copy()))
        #                     ybeh_t = np.vstack((ybeh_t,label_temp[state,coh,choice].copy())) 
        #             else:
        #                 continue 
        #         idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
        #         idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)  
        #         Xdata_beh_trainset[idxf,'error'][coh,choice],ylabels_beh_trainset[idxf,'error'][coh,choice]=Xbeh_t[idxtrain,:],ybeh_t[idxtrain,:] 
        #         Xdata_beh_testset[idxf,'error'][coh,choice],ylabels_beh_testset[idxf,'error'][coh,choice]=Xbeh_t[idxtest,:],ybeh_t[idxtest,:] 


        # for coh in unique_cohs:
        #     for choice in unique_choices:
        #         data_temp  = Xdata_set[idxf,'correct'].copy()
        #         label_temp = ylabels_set[idxf,'correct'].copy()  
        #         Xbeh_t,ybeh_t = [],[]              
        #         numtrials = 0
        #         for state in unique_states:
        #             ### correct
        #             if state<4:
        #                 continue
        #             numtrials_t = np.shape(data_temp[state,coh,choice])[0]
        #             if numtrials_t>0:
        #                 numtrials+=numtrials_t
        #                 if(len(Xbeh_t)==0):
        #                     Xbeh_t = data_temp[state,coh,choice].copy() 
        #                     ybeh_t = label_temp[state,coh,choice].copy()  
        #                 else:
        #                     Xbeh_t = np.vstack((Xbeh_t,data_temp[state,coh,choice].copy()))
        #                     ybeh_t = np.vstack((ybeh_t,label_temp[state,coh,choice].copy())) 
        #             else:
        #                 continue 
        #         idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
        #         idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)  
        #         Xdata_beh_trainset[idxf,'correct'][coh,choice],ylabels_beh_trainset[idxf,'correct'][coh,choice]=Xbeh_t[idxtrain,:],ybeh_t[idxtrain,:] 
        #         Xdata_beh_testset[idxf,'correct'][coh,choice],ylabels_beh_testset[idxf,'correct'][coh,choice]=Xbeh_t[idxtest,:],ybeh_t[idxtest,:] 

        ctxt_rep, ctxt_alt = [0,1],[2,3]
        for coh in unique_cohs:
            for choice in unique_choices:
                data_temp  = Xdata_set[idxf,'error'].copy()
                label_temp = ylabels_set[idxf,'error'].copy()  
                ### repeating context -- after error
                Xbeh_t,ybeh_t = [],[]              
                numtrials = 0
                for state in ctxt_rep:
                    ### error 
                    if state>=4:
                        break
                    numtrials_t = np.shape(data_temp[state,coh,choice])[0]
                    if numtrials_t>0:
                        numtrials+=numtrials_t
                        if(len(Xbeh_t)==0):
                            Xbeh_t = data_temp[state,coh,choice].copy() 
                            ybeh_t = label_temp[state,coh,choice].copy()  
                            # print('gpt 382rep error:',np.unique(ybeh_t[:,1]))
                        else:
                            Xbeh_t = np.vstack((Xbeh_t,data_temp[state,coh,choice].copy()))
                            ybeh_t = np.vstack((ybeh_t,label_temp[state,coh,choice].copy())) 
                            # print('rep error:',np.unique(ybeh_t[:,1]),np.shape(ybeh_t))
                    else:
                        continue 
                idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
                idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)  
                Xdata_beh_trainset[idxf,'error'][coh,choice],ylabels_beh_trainset[idxf,'error'][coh,choice]=Xbeh_t[idxtrain,:],ybeh_t[idxtrain,:] 
                Xdata_beh_testset[idxf,'error'][coh,choice],ylabels_beh_testset[idxf,'error'][coh,choice]=Xbeh_t[idxtest,:],ybeh_t[idxtest,:] 

                ### alternating context -- after error  
                Xbeh_t,ybeh_t = [],[]            
                numtrials = 0
                for state in ctxt_alt:
                    ### error 
                    if state>=4:
                        break
                    numtrials_t = np.shape(data_temp[state,coh,choice])[0]
                    if numtrials_t>0:
                        numtrials+=numtrials_t
                        if(len(Xbeh_t)==0):
                            Xbeh_t = data_temp[state,coh,choice].copy() 
                            ybeh_t = label_temp[state,coh,choice].copy() 
                            # print('alt error:',np.unique(ybeh_t[:,1])) 
                        else:
                            Xbeh_t = np.vstack((Xbeh_t,data_temp[state,coh,choice].copy()))
                            ybeh_t = np.vstack((ybeh_t,label_temp[state,coh,choice].copy())) 
                            # print('alt error:',np.unique(ybeh_t[:,1]))
                    else:
                        continue 
                idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
                idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)  
                Xdata_beh_trainset[idxf,'error'][coh,choice],ylabels_beh_trainset[idxf,'error'][coh,choice]=np.vstack((Xdata_beh_trainset[idxf,'error'][coh,choice],Xbeh_t[idxtrain,:])),np.vstack((ylabels_beh_trainset[idxf,'error'][coh,choice],ybeh_t[idxtrain,:]))
                Xdata_beh_testset[idxf,'error'][coh,choice],ylabels_beh_testset[idxf,'error'][coh,choice]=np.vstack((Xdata_beh_testset[idxf,'error'][coh,choice],Xbeh_t[idxtest,:])),np.vstack((ylabels_beh_testset[idxf,'error'][coh,choice],ybeh_t[idxtest,:])) 


        ctxt_rep, ctxt_alt = [4,5],[6,7]
        for coh in unique_cohs:
            for choice in unique_choices:
                data_temp  = Xdata_set[idxf,'correct'].copy()
                label_temp = ylabels_set[idxf,'correct'].copy()  
                ### repeating context -- after correct 
                Xbeh_t,ybeh_t = [],[]              
                numtrials = 0
                for state in ctxt_rep:
                    ### correct
                    if state<4:
                        continue
                    numtrials_t = np.shape(data_temp[state,coh,choice])[0]
                    if numtrials_t>0:
                        numtrials+=numtrials_t
                        if(len(Xbeh_t)==0):
                            Xbeh_t = data_temp[state,coh,choice].copy() 
                            ybeh_t = label_temp[state,coh,choice].copy()  
                            # print('rep correct:',np.unique(ybeh_t[:,1]))
                        else:
                            Xbeh_t = np.vstack((Xbeh_t,data_temp[state,coh,choice].copy()))
                            ybeh_t = np.vstack((ybeh_t,label_temp[state,coh,choice].copy())) 
                            # print('rep correct:',np.unique(ybeh_t[:,1]))
                    else:
                        continue 
                idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
                idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)  
                Xdata_beh_trainset[idxf,'correct'][coh,choice],ylabels_beh_trainset[idxf,'correct'][coh,choice]=Xbeh_t[idxtrain,:],ybeh_t[idxtrain,:] 
                Xdata_beh_testset[idxf,'correct'][coh,choice],ylabels_beh_testset[idxf,'correct'][coh,choice]=Xbeh_t[idxtest,:],ybeh_t[idxtest,:] 

                ### alternating context -- after correct 
                Xbeh_t,ybeh_t = [],[]              
                numtrials = 0
                for state in ctxt_alt:
                    ### correct
                    if state<4:
                        continue
                    numtrials_t = np.shape(data_temp[state,coh,choice])[0]
                    if numtrials_t>0:
                        numtrials+=numtrials_t
                        if(len(Xbeh_t)==0):
                            Xbeh_t = data_temp[state,coh,choice].copy() 
                            ybeh_t = label_temp[state,coh,choice].copy()  
                            # print('alt correct:',np.unique(ybeh_t[:,1]))
                        else:
                            Xbeh_t = np.vstack((Xbeh_t,data_temp[state,coh,choice].copy()))
                            ybeh_t = np.vstack((ybeh_t,label_temp[state,coh,choice].copy())) 
                            # print('alt correct:',np.unique(ybeh_t[:,1]))
                    else:
                        continue 
                idxtrain = np.random.choice(np.arange(numtrials),size=int(numtrials/2),replace=False) 
                idxtest  = np.setdiff1d(np.arange(numtrials),idxtrain)  
                Xdata_beh_trainset[idxf,'correct'][coh,choice],ylabels_beh_trainset[idxf,'correct'][coh,choice]=np.vstack((Xdata_beh_trainset[idxf,'correct'][coh,choice],Xbeh_t[idxtrain,:])),np.vstack((ylabels_beh_trainset[idxf,'correct'][coh,choice],ybeh_t[idxtrain,:])) 
                Xdata_beh_testset[idxf,'correct'][coh,choice],ylabels_beh_testset[idxf,'correct'][coh,choice]=np.vstack((Xdata_beh_testset[idxf,'correct'][coh,choice],Xbeh_t[idxtest,:])),np.vstack((ylabels_beh_testset[idxf,'correct'][coh,choice],ybeh_t[idxtest,:]))

                # print('shape gpt file 465:',np.shape(Xdata_beh_testset[idxf,'correct'][coh,choice]),np.shape(Xdata_beh_trainset[idxf,'correct'][coh,choice])) 
    return Xdata_hist_trainset, ylabels_hist_trainset, Xdata_hist_testset, ylabels_hist_testset,Xdata_psy_trainset, ylabels_psy_trainset, Xdata_psy_testset, ylabels_psy_testset,Xdata_beh_trainset, ylabels_beh_trainset, Xdata_beh_testset, ylabels_beh_testset 



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
                    # print('ymerge_:',ymerge_labels_correct[state][0,2::6])
                    Xmerge_trials_correct[state] = data_temp[state][idxsample,:]
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials_hist


def merge_pseudo_hist_trials_individual(Xdata_set,ylabels_set,unique_states,unique_cohs,nselect, files,idx_delete,EACHSTATES=20,RECORD_TRIALS=0, RECORDED_TRIALS=[]):
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
            temp_trials  = (Xdata_set[idxf,'error'][state].copy())
            temp_labels  = ylabels_set[idxf,'error'][state].copy()

            #### generate sampled true trials for individual neurons in each pseudo trial
            NN = np.shape(temp_trials)[1]
            totaltrials = np.shape(temp_trials)[0]
            sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
            Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
            ymerge_trials_t     = np.zeros((EACHSTATES,np.shape(temp_labels)[1]))
            for iii in range(EACHSTATES):
                sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                ymerge_trials_t[iii,:]     = temp_labels[iidrandom,:]
                ymerge_trials_t[iii,4]     = np.mean(temp_labels[sampled_true_trials[iii,:],4]) ### choice



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
                    # print('ymerge_:',ymerge_labels_error[state][0,2::6])
                except:
                    ymerge_labels_error[state] = ymerge_trials_t
                    Xmerge_trials_error[state] = Xmerge_trials_t
        Xmerge_trials_error[state] = Xmerge_trials_error[state][:,nselect]
    
    for state in unique_states:
        if state<4:
            continue
        for idxf in range(len(files)):
            if(idxf in idx_delete):
                continue
            temp_trials  = (Xdata_set[idxf,'correct'][state].copy())
            temp_labels  = ylabels_set[idxf,'correct'][state].copy()
            #### generate sampled true trials for individual neurons in each pseudo trial
            NN = np.shape(temp_trials)[1]
            totaltrials = np.shape(temp_trials)[0]
            sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
            Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
            ymerge_trials_t     = np.zeros((EACHSTATES,np.shape(temp_labels)[1]))
            for iii in range(EACHSTATES):
                sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                ymerge_trials_t[iii,:]     = temp_labels[iidrandom,:]
                ymerge_trials_t[iii,4]     = np.mean(temp_labels[sampled_true_trials[iii,:],4])

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
        Xmerge_trials_correct[state] = Xmerge_trials_correct[state][:,nselect]

    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials_hist

def shuffle_pseudo_hist_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files,idx_delete,EACHSTATES=20,RECORD_TRIALS=0, RECORDED_TRIALS=[]):
    ### using the test dataset
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

            # if (idxf == 0):
            #     ymerge_labels_correct[state] = label_shuffle
            #     Xmerge_trials_correct[state] = data_shuffle
            # else:
            try:    
                Xmerge_trials_correct[state] = np.hstack((Xmerge_trials_correct[state],data_shuffle))
                ymerge_labels_correct[state] = np.hstack((ymerge_labels_correct[state],label_shuffle))
            except:
                ymerge_labels_correct[state] = label_shuffle
                Xmerge_trials_correct[state] = data_shuffle
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials_hist

def merge_pseudo_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,nselect, vfiles,falsefiles,EACHSTATES=60, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[],STIM_BEH=1):
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
                if np.shape(temp_trials)[0]==0:
                    temp_trials= data_temp[state,coh]
                    temp_beh   = np.reshape(label_temp[state,coh][:,4],(1,-1)).flatten()
                    temp_beh   = temp_beh -2
                else:
                    temp_trials = np.vstack((temp_trials,data_temp[state,coh]))
                    temp_beh    = np.hstack((temp_beh,np.reshape(label_temp[state,coh][:,4],(1,-1))-2))
                    temp_beh    = temp_beh.flatten()
                    
                totaltrials = np.shape(temp_trials)[0] 
                
                if(RECORD_TRIALS):
                    idxsample = np.random.choice(np.arange(totaltrials),size=EACHSTATES,replace=True)
                    merge_trials[state,coh,idxf] = idxsample
                else:
                    idxsample = RECORDED_TRIALS_SET[state,coh,idxf]
                try:
                    ymerge_labels_correct[state,coh] = np.vstack((ymerge_labels_correct[state,coh],temp_beh[idxsample])) #labels))
                    Xmerge_trials_correct[state,coh] = np.hstack((Xmerge_trials_correct[state,coh],temp_trials[idxsample,:]))
                except:
                    ymerge_labels_correct[state,coh] = temp_beh[idxsample]#[np.sum(temp_beh[idxsample])/len(idxsample)]
                    Xmerge_trials_correct[state,coh] = temp_trials[idxsample,:]
            Xmerge_trials_correct[state,coh]=Xmerge_trials_correct[state,coh][:,nselect]

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
                if np.shape(temp_trials)[0]==0:
                    temp_trials= data_temp[state,coh]
                    temp_beh   = np.reshape(label_temp[state,coh][:,4],(1,-1)).flatten()
                else:
                    temp_trials = np.vstack((temp_trials,data_temp[state,coh]))
                    temp_beh    = np.hstack((temp_beh,np.reshape(label_temp[state,coh][:,4],(1,-1))))
                    temp_beh    = temp_beh.flatten()
                totaltrials = np.shape(temp_trials)[0]  
                
                if(RECORD_TRIALS):
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
            Xmerge_trials_error[state,coh] = Xmerge_trials_error[state,coh][:,nselect]
    return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials

# def merge_pseudo_beh_trials_individual(Xdata_set,ylabels_set,unique_states,unique_cohs,nselect,vfiles,falsefiles,EACHSTATES=60, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[],STIM_BEH=1):
#     unique_choices = [0,1]
#     Xmerge_trials_correct,ymerge_labels_correct = {},{}
#     yright_ratio_correct = {}
#     Xmerge_trials_error,ymerge_labels_error = {},{}
#     yright_ratio_error = {}
#     merge_trials = {}

#     for coh in unique_cohs:
#         for choice in unique_choices:
#             for idxf in range(len(vfiles)):
#                 if(idxf in falsefiles):
#                     continue
#                 temp_trials  = Xdata_set[idxf,'correct'][coh,choice].copy()
#                 temp_beh     = ylabels_set[idxf,'correct'][coh,choice].copy()
#                 totaltrials  = np.shape(temp_trials)[0]
#                 #### generate sampled true trials for individual neurons in each pseudo trial
#                 NN = np.shape(temp_trials)[1]
#                 sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
#                 Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
#                 ymerge_trials_t     = np.zeros((1,EACHSTATES))
#                 for iii in range(EACHSTATES):
#                     sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
#                     Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
#                     iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
#                     ymerge_trials_t[0,iii]     = choice
                    
#                 if(RECORD_TRIALS):
#                     merge_trials[coh,choice,idxf] = sampled_true_trials
#                 else:
#                     idxsample = RECORDED_TRIALS_SET[coh,choice,idxf]
#                 try:
#                     ymerge_labels_correct[coh,choice] = np.vstack((ymerge_labels_correct[coh,choice],ymerge_trials_t))
#                     Xmerge_trials_correct[coh,choice] = np.hstack((Xmerge_trials_correct[coh,choice],Xmerge_trials_t))
#                 except:
#                     ymerge_labels_correct[coh,choice]     = ymerge_trials_t
#                     Xmerge_trials_correct[coh,choice] = Xmerge_trials_t
#             Xmerge_trials_correct[coh,choice] = Xmerge_trials_correct[coh,choice][:,nselect]

#     for coh in unique_cohs:
#         for choice in unique_choices:
#             for idxf in range(len(vfiles)):
#                 if(idxf in falsefiles):
#                     continue
#                 temp_trials  = Xdata_set[idxf,'error'][coh,choice].copy()
#                 temp_beh     = ylabels_set[idxf,'error'][coh,choice].copy()
#                 totaltrials  = np.shape(temp_trials)[0]
#                 #### generate sampled true trials for individual neurons in each pseudo trial
#                 NN = np.shape(temp_trials)[1]
#                 sampled_true_trials = np.zeros((EACHSTATES,NN),dtype=int32)
#                 Xmerge_trials_t     = np.zeros((EACHSTATES,NN))
#                 ymerge_trials_t     = np.zeros((1,EACHSTATES))
#                 for iii in range(EACHSTATES):
#                     sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
#                     Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
#                     iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
#                     ymerge_trials_t[0,iii]     = choice
                    
#                 if(RECORD_TRIALS):
#                     merge_trials[coh,choice,idxf] = sampled_true_trials
#                 else:
#                     idxsample = RECORDED_TRIALS_SET[coh,choice,idxf]
#                 try:
#                     ymerge_labels_error[coh,choice] = np.vstack((ymerge_labels_error[coh,choice],ymerge_trials_t))
#                     Xmerge_trials_error[coh,choice] = np.hstack((Xmerge_trials_error[coh,choice],Xmerge_trials_t))
#                 except:
#                     ymerge_labels_error[coh,choice]     = ymerge_trials_t
#                     Xmerge_trials_error[coh,choice]     = Xmerge_trials_t
#             Xmerge_trials_error[coh,choice] = Xmerge_trials_error[coh,choice][:,nselect]


#     return Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error, merge_trials


'''
Separate context repeating from context alternating
'''
def merge_pseudo_beh_trials_individual(Xdata_set,ylabels_set,unique_states,unique_cohs,nselect,vfiles,falsefiles,EACHSTATES=60, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[],STIM_BEH=1):
    unique_choices = [0,1]
    Xmerge_trials_correct_rep,ymerge_labels_correct_rep = {},{}
    yright_ratio_correct_rep = {}
    Xmerge_trials_error_rep,ymerge_labels_error_rep = {},{}
    yright_ratio_error_rep = {}

    Xmerge_trials_correct_alt,ymerge_labels_correct_alt = {},{}
    yright_ratio_correct_alt = {}
    Xmerge_trials_error_alt,ymerge_labels_error_alt = {},{}
    yright_ratio_error_alt = {}

    merge_trials = {}

    for coh in unique_cohs:
        for choice in unique_choices:
            ymerge_labels_correct_rep[coh,choice]     = []
            Xmerge_trials_correct_rep[coh,choice]     = []
  
            ymerge_labels_correct_alt[coh,choice]     = []
            Xmerge_trials_correct_alt[coh,choice]     = []
            
            ymerge_labels_error_rep[coh,choice]     = []
            Xmerge_trials_error_rep[coh,choice]     = []
  
            ymerge_labels_error_alt[coh,choice]     = []
            Xmerge_trials_error_alt[coh,choice]     = []
            
            for idxf in range(len(vfiles)):
                if(idxf in falsefiles):
                    continue
                temp_trials  = Xdata_set[idxf,'correct'][coh,choice].copy()
                temp_beh     = ylabels_set[idxf,'correct'][coh,choice].copy()
                temp_ctxt    = temp_beh[:,1].copy()  ### context information 
                totaltrials  = np.shape(temp_trials)[0]
                reptrials    = np.where(temp_ctxt==0+2)[0]
                alttrials    = np.where(temp_ctxt==1+2)[0]
                #### generate sampled true trials for individual neurons in each pseudo trial
                NN = np.shape(temp_trials)[1]

                sampled_true_trials_rep = np.zeros((EACHSTATES,NN),dtype=int32)
                Xmerge_trials_t_rep     = np.zeros((EACHSTATES,NN))
                ymerge_trials_t_rep     = np.zeros((1,EACHSTATES))

                sampled_true_trials_alt = np.zeros((EACHSTATES,NN),dtype=int32)
                Xmerge_trials_t_alt     = np.zeros((EACHSTATES,NN))
                ymerge_trials_t_alt     = np.zeros((1,EACHSTATES))

                for iii in range(EACHSTATES):
                    # sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                    # Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                    # iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                    # ymerge_trials_t[0,iii]     = choice

                    ### repeating 
                    try:
                        sampled_true_trials_rep[iii,:] = np.random.choice(reptrials,size=NN, replace=True)
                        Xmerge_trials_t_rep[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials_rep[iii,:])])
                        iidrandom=np.random.choice(sampled_true_trials_rep[iii,:],size=1,replace=False)
                        ymerge_trials_t_rep[0,iii]     = choice
                    except:
                        print('No repeating trials (beh-c)')
                        sampled_true_trials_rep = []
                        Xmerge_trials_t_rep     = []
                        ymerge_trials_t_rep     = []
                        continue
                    ### alternating
                    try:
                        sampled_true_trials_alt[iii,:] = np.random.choice(alttrials,size=NN, replace=True)
                        Xmerge_trials_t_alt[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials_alt[iii,:])])
                        iidrandom=np.random.choice(sampled_true_trials_alt[iii,:],size=1,replace=False)
                        ymerge_trials_t_alt[0,iii]     = choice
                    except:
                        print('No alternating trials (beh-c')
                        sampled_true_trials_alt = []
                        Xmerge_trials_t_alt     = []
                        ymerge_trials_t_alt     = []
                        continue
                    
                if(RECORD_TRIALS):
                    try:
                        merge_trials[coh,choice,idxf] = np.vstack((sampled_true_trials_rep,sampled_true_trials_alt))
                    except:
                        merge_trials[coh,choice,idxf] = []
                else:
                    idxsample = RECORDED_TRIALS_SET[coh,choice,idxf]
                # try:
                #     ### repeating
                #     ymerge_labels_correct_rep[coh,choice] = np.vstack((ymerge_labels_correct_rep[coh,choice],ymerge_trials_t_rep))
                #     Xmerge_trials_correct_rep[coh,choice] = np.hstack((Xmerge_trials_correct_rep[coh,choice],Xmerge_trials_t_rep))
                #     ### alternating
                #     ymerge_labels_correct_alt[coh,choice] = np.vstack((ymerge_labels_correct_alt[coh,choice],ymerge_trials_t_alt))
                #     Xmerge_trials_correct_alt[coh,choice] = np.hstack((Xmerge_trials_correct_alt[coh,choice],Xmerge_trials_t_alt))
                # except:
                #     ### repeating
                #     ymerge_labels_correct_rep[coh,choice]     = ymerge_trials_t_rep
                #     Xmerge_trials_correct_rep[coh,choice]     = Xmerge_trials_t_rep
                #     ### alternationg
                #     ymerge_labels_correct_alt[coh,choice]     = ymerge_trials_t_alt
                #     Xmerge_trials_correct_alt[coh,choice]     = Xmerge_trials_t_alt
                
                if (len(ymerge_labels_correct_rep[coh,choice])==0):
                    ### repeating
                    ymerge_labels_correct_rep[coh,choice]     = ymerge_trials_t_rep
                    Xmerge_trials_correct_rep[coh,choice]     = Xmerge_trials_t_rep
                elif(len(ymerge_trials_t_rep)==0):
                    ### repeating
                    ymerge_labels_correct_rep[coh,choice]     = ymerge_labels_correct_rep[coh,choice]
                    Xmerge_trials_correct_rep[coh,choice]     = Xmerge_trials_correct_rep[coh,choice]
                else:
                    ### repeating
                    ymerge_labels_correct_rep[coh,choice] = np.vstack((ymerge_labels_correct_rep[coh,choice],ymerge_trials_t_rep))
                    Xmerge_trials_correct_rep[coh,choice] = np.hstack((Xmerge_trials_correct_rep[coh,choice],Xmerge_trials_t_rep))
                    
                if (len(ymerge_labels_correct_alt[coh,choice])==0):
                    ### repeating
                    ymerge_labels_correct_alt[coh,choice]     = ymerge_trials_t_alt
                    Xmerge_trials_correct_alt[coh,choice]     = Xmerge_trials_t_alt
                elif(len(ymerge_trials_t_alt)==0):
                    ### repeating
                    ymerge_labels_correct_alt[coh,choice]     = ymerge_labels_correct_alt[coh,choice]
                    Xmerge_trials_correct_alt[coh,choice]     = Xmerge_trials_correct_alt[coh,choice]
                else:
                    ### repeating
                    ymerge_labels_correct_alt[coh,choice] = np.vstack((ymerge_labels_correct_alt[coh,choice],ymerge_trials_t_alt))
                    Xmerge_trials_correct_alt[coh,choice] = np.hstack((Xmerge_trials_correct_alt[coh,choice],Xmerge_trials_t_alt))

            # print('gpt 1012 ...shape correct:',coh,choice,'...',np.shape(Xmerge_trials_correct_rep[coh,choice]))
            # print('shape correct:',coh,choice,'...',np.shape(Xmerge_trials_correct_alt[coh,choice]))
            Xmerge_trials_correct_rep[coh,choice] = Xmerge_trials_correct_rep[coh,choice][:,nselect]
            Xmerge_trials_correct_alt[coh,choice] = Xmerge_trials_correct_alt[coh,choice][:,nselect]

    for coh in unique_cohs:
        for choice in unique_choices:
            for idxf in range(len(vfiles)):
                if(idxf in falsefiles):
                    continue
                temp_trials  = Xdata_set[idxf,'error'][coh,choice].copy()
                temp_beh     = ylabels_set[idxf,'error'][coh,choice].copy()
                temp_ctxt    = temp_beh[:,1].copy()
                totaltrials  = np.shape(temp_trials)[0]
                reptrials    = np.where(temp_ctxt==0)[0]
                alttrials    = np.where(temp_ctxt==1)[0]
                # print('gpt 1028 ctxt......error',np.unique(temp_ctxt),idxf,coh,choice)
                #### generate sampled true trials for individual neurons in each pseudo trial
                NN = np.shape(temp_trials)[1]
                ### repeating
                sampled_true_trials_rep = np.zeros((EACHSTATES,NN),dtype=int32)
                Xmerge_trials_t_rep     = np.zeros((EACHSTATES,NN))
                ymerge_trials_t_rep     = np.zeros((1,EACHSTATES))
                ### alternating
                sampled_true_trials_alt = np.zeros((EACHSTATES,NN),dtype=int32)
                Xmerge_trials_t_alt     = np.zeros((EACHSTATES,NN))
                ymerge_trials_t_alt     = np.zeros((1,EACHSTATES))
                for iii in range(EACHSTATES):
                    # sampled_true_trials[iii,:] = np.random.choice(np.arange(totaltrials),size=NN, replace=True)
                    # Xmerge_trials_t[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials[iii,:])])
                    # iidrandom=np.random.choice(sampled_true_trials[iii,:],size=1,replace=False)
                    # ymerge_trials_t[0,iii]     = choice
                    try:
                        ### repeating 
                        sampled_true_trials_rep[iii,:] = np.random.choice(reptrials,size=NN, replace=True)
                        Xmerge_trials_t_rep[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials_rep[iii,:])])
                        iidrandom=np.random.choice(sampled_true_trials_rep[iii,:],size=1,replace=False)
                        ymerge_trials_t_rep[0,iii]     = choice
                    except:
                        # print('gpt 1051  No repeating trials (beh-e)',idxf, coh, choice)
                        sampled_true_trials_rep = []
                        Xmerge_trials_t_rep     = []
                        ymerge_trials_t_rep     = []
                        continue
                    try:    
                        ### alternating
                        sampled_true_trials_alt[iii,:] = np.random.choice(alttrials,size=NN, replace=True)
                        Xmerge_trials_t_alt[iii,:]     = np.array([temp_trials[item,i] for i, item in enumerate(sampled_true_trials_alt[iii,:])])
                        iidrandom=np.random.choice(sampled_true_trials_alt[iii,:],size=1,replace=False)
                        ymerge_trials_t_alt[0,iii]     = choice
                    except:
                        # print('gpt 1063 ...No alternating trials (beh-e)',idxf, coh, choice)
                        sampled_true_trials_alt = []
                        Xmerge_trials_t_alt     = []
                        ymerge_trials_t_alt     = []              
                    
                if(RECORD_TRIALS):
                    try:
                        merge_trials[coh,choice,idxf] = np.vstack((sampled_true_trials_rep,sampled_true_trials_alt))
                    except:
                        merge_trials[coh,choice,idxf] =[]
                else:
                    idxsample = RECORDED_TRIALS_SET[coh,choice,idxf]
                # try:
                #     ### repeating
                #     ymerge_labels_error_rep[coh,choice] = np.vstack((ymerge_labels_error_rep[coh,choice],ymerge_trials_t_rep))
                #     Xmerge_trials_error_rep[coh,choice] = np.hstack((Xmerge_trials_error_rep[coh,choice],Xmerge_trials_t_rep))
                #     ### alternating
                #     ymerge_labels_error_alt[coh,choice] = np.vstack((ymerge_labels_error_alt[coh,choice],ymerge_trials_t_alt))
                #     Xmerge_trials_error_alt[coh,choice] = np.hstack((Xmerge_trials_error_alt[coh,choice],Xmerge_trials_t_alt))
                # except:
                #     ### repeating
                #     ymerge_labels_error_rep[coh,choice]     = ymerge_trials_t_rep
                #     Xmerge_trials_error_rep[coh,choice]     = Xmerge_trials_t_rep
                #     ### alternating
                #     ymerge_labels_error_alt[coh,choice]     = ymerge_trials_t_alt
                #     Xmerge_trials_error_alt[coh,choice]     = Xmerge_trials_t_alt
                
                if (len(ymerge_labels_error_rep[coh,choice])==0):
                    ### repeating
                    ymerge_labels_error_rep[coh,choice]     = ymerge_trials_t_rep
                    Xmerge_trials_error_rep[coh,choice]     = Xmerge_trials_t_rep
                elif(len(ymerge_trials_t_rep)==0):
                    ### repeating
                    ymerge_labels_error_rep[coh,choice]     = ymerge_labels_error_rep[coh,choice]
                    Xmerge_trials_error_rep[coh,choice]     = Xmerge_trials_error_rep[coh,choice]
                else:
                    ### repeating
                    ymerge_labels_error_rep[coh,choice] = np.vstack((ymerge_labels_error_rep[coh,choice],ymerge_trials_t_rep))
                    Xmerge_trials_error_rep[coh,choice] = np.hstack((Xmerge_trials_error_rep[coh,choice],Xmerge_trials_t_rep))
                    
                if (len(ymerge_labels_error_alt[coh,choice])==0):
                    ### repeating
                    ymerge_labels_error_alt[coh,choice]     = ymerge_trials_t_alt
                    Xmerge_trials_error_alt[coh,choice]     = Xmerge_trials_t_alt
                elif(len(ymerge_trials_t_alt)==0):
                    ### repeating
                    ymerge_labels_error_alt[coh,choice]     = ymerge_labels_error_alt[coh,choice]
                    Xmerge_trials_error_alt[coh,choice]     = Xmerge_trials_error_alt[coh,choice]
                else:
                    ### repeating
                    ymerge_labels_error_alt[coh,choice] = np.vstack((ymerge_labels_error_alt[coh,choice],ymerge_trials_t_alt))
                    Xmerge_trials_error_alt[coh,choice] = np.hstack((Xmerge_trials_error_alt[coh,choice],Xmerge_trials_t_alt))
            
            Xmerge_trials_error_rep[coh,choice] = Xmerge_trials_error_rep[coh,choice][:,nselect]
            Xmerge_trials_error_alt[coh,choice] = Xmerge_trials_error_alt[coh,choice][:,nselect]


    return Xmerge_trials_correct_rep,ymerge_labels_correct_rep,Xmerge_trials_error_rep,ymerge_labels_error_rep, Xmerge_trials_correct_alt,ymerge_labels_correct_alt,Xmerge_trials_error_alt,ymerge_labels_error_alt, merge_trials

def behaviour_trbias_proj(coeffs_pool, intercepts_pool, Xmerge_trials,
                                     ymerge_labels, unique_states,unique_cohs,unique_choices, num_beh_trials, EACHSTATES=20,FIX_TRBIAS_BINS=[],NBINS=5,mmodel=[],PCA_n_components=0):

    #### using Xdata_hist_testset
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

# ##### only one decoders for the repeating trials
# def behaviour_trbias_proj_new(coeffs_pool, intercepts_pool, Xmerge_trials,
#                                      ymerge_labels, unique_states,unique_cohs,unique_choices, num_beh_trials, EACHSTATES=20,FIX_TRBIAS_BINS=[],NBINS=5,mmodel=[],PCA_n_components=0,CONDITION_CTXT=-1):

#     #### using Xdata_hist_testset
#     MAXV = 100
#     NDEC = int(np.shape(coeffs_pool)[1]/6)#5  13 Oct
#     NN   = np.shape(Xmerge_trials[unique_states[0],unique_cohs[0]])[1]
#     NS, NC, NCH = len(unique_states),len(unique_cohs),len(unique_choices)
#     nbins_trbias = NBINS
#     psychometric_trbias = np.zeros((len(unique_cohs), nbins_trbias))
#     trbias_range        = np.zeros((len(unique_cohs), nbins_trbias))
    
#     for idxcoh, coh in enumerate(unique_cohs):
#         maxtrbias,mintrbias=-MAXV,MAXV
#         evidences   = []#
#         rightchoice = []#
#         for idxs, state in enumerate(unique_states):
#             if(PCA_n_components>0):
#                 Xmerge_trials[state,coh] = mmodel.transform(Xmerge_trials[state,coh])
#             for idx in range(EACHSTATES):
#                 Xdata_test = Xmerge_trials[state,coh][idx,:]
#                 idxdecoder = idx+idxs*EACHSTATES+idxcoh*(len(unique_states)*EACHSTATES)#np.random.choice(np.arange(0, NDEC, 1),size=1, replace=True)
#                 idxdecoder = np.mod(idxdecoder, NDEC)
#                 linw_bias, linb_bias = coeffs_pool[:, idxdecoder*6+3], intercepts_pool[0, 6*idxdecoder+3]
#                 evidences   = np.append(evidences,np.squeeze(
#                 Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias))
#                 temp_perc   = np.sum(ymerge_labels[state,coh][:,idx])/np.shape(ymerge_labels[state,coh])[0]

#                 rightchoice = np.append(rightchoice,temp_perc)
        
#         ### 
#         maxtrbias ,mintrbias = 0.8*max(evidences),1.2*min(evidences)
#         binss = np.linspace(mintrbias,maxtrbias,nbins_trbias+1)
#         perc_right = np.zeros(nbins_trbias)
#         ax_trbias  = (binss[1:]+binss[:-1])/2.0
#         for i in range(1,nbins_trbias+1):
#             idxbinh = np.where(evidences<binss[i])[0]
#             idxbinl = np.where(evidences>binss[i-1])[0]
#             idxbin  = np.intersect1d(idxbinh,idxbinl)
            
#             perc_right[i-1] = np.sum(rightchoice[idxbin])/len(idxbin)# 
#         psychometric_trbias[idxcoh,:] = perc_right.copy()
#         trbias_range[idxcoh,:]        = ax_trbias.copy()
#     return psychometric_trbias,trbias_range


##### consider decoders for repeating 3 and alternating 4
def behaviour_trbias_proj_new(coeffs_pool, intercepts_pool, Xmerge_trials,
                                     ymerge_labels, unique_states,unique_cohs,unique_choices, num_beh_trials, EACHSTATES=20,FIX_TRBIAS_BINS=[],NBINS=5,mmodel=[],PCA_n_components=0,CONDITION_CTXT=-1):

    #### using Xdata_hist_testset
    MAXV = 100
    NDEC = int(np.shape(coeffs_pool)[1]/6)#5  13 Oct
    NN   = np.shape(Xmerge_trials[unique_states[0],unique_cohs[0]])[1]
    NS, NC, NCH = len(unique_states),len(unique_cohs),len(unique_choices)
    nbins_trbias = NBINS
    psychometric_trbias = np.zeros((len(unique_cohs), nbins_trbias))
    trbias_range        = np.zeros((len(unique_cohs), nbins_trbias))
    
    if (unique_states[0]<4):
        # print('Psychometric curves for After Error Trials...')
        rep_states = [0,1] # unique_states[:2]
        alt_states = [2,3] # unique_states[2:]
    for idxcoh, coh in enumerate(unique_cohs):
        maxtrbias,mintrbias=-MAXV,MAXV
        evidences   = []#
        rightchoice = []#

        ### repeating
        for idxs, state in enumerate(unique_states[:2]):#unique_states):
            for idx in range(EACHSTATES):
                Xdata_test = Xmerge_trials[state,coh][idx,:]
                idxdecoder = idx+idxs*EACHSTATES+idxcoh*(len(unique_states)*EACHSTATES)#np.random.choice(np.arange(0, NDEC, 1),size=1, replace=True)
                idxdecoder = np.mod(idxdecoder, NDEC)
                linw_bias, linb_bias = coeffs_pool[:, idxdecoder*6+3], intercepts_pool[0, 6*idxdecoder+3]
                evidences   = np.append(evidences,np.squeeze(
                Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias))
                temp_perc   = np.sum(ymerge_labels[state,coh][:,idx])/np.shape(ymerge_labels[state,coh])[0]

                rightchoice = np.append(rightchoice,temp_perc)
        ### alternating
        for idxs, state in enumerate(unique_states[2:]):#unique_states):
            for idx in range(EACHSTATES):
                Xdata_test = Xmerge_trials[state,coh][idx,:]
                idxdecoder = idx+(idxs+2)*EACHSTATES+idxcoh*(len(unique_states)*EACHSTATES)
                ### here adding the previous 2
                idxdecoder = np.mod(idxdecoder, NDEC)
                linw_bias, linb_bias = coeffs_pool[:, idxdecoder*6+4], intercepts_pool[0, 6*idxdecoder+4]
                evidences   = np.append(evidences,np.squeeze(
                Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias))
                temp_perc   = np.sum(ymerge_labels[state,coh][:,idx])/np.shape(ymerge_labels[state,coh])[0]

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
            
            perc_right[i-1] = np.sum(rightchoice[idxbin])/len(idxbin)# 
        psychometric_trbias[idxcoh,:] = perc_right.copy()
        trbias_range[idxcoh,:]        = ax_trbias.copy()
    return psychometric_trbias,trbias_range

def behaviour_trbias_proj_evi(coeffs_trbias, intercepts_trbias,coeffs_beh, intercepts_beh, Xmerge_trials,
                                     ymerge_labels, unique_states,unique_cohs,unique_choices, num_beh_trials, pop_sub, pop_zero,USE_POP=1,EACHSTATES=20,FIX_TRBIAS_BINS=[],NBINS=5,mmodel=[],PCA_n_components=0):

    MAXV = 100
    NDEC = int(np.shape(coeffs_trbias)[1]/5)
    NN   = np.shape(Xmerge_trials[unique_states[0],unique_cohs[0]])[1]
    NS, NC, NCH = len(unique_states),len(unique_cohs),len(unique_choices)
    nbins_trbias = NBINS
    trbias_encodings = {}
    behaviour_encodings ={}
    
    for idxcoh, coh in enumerate(unique_cohs):
        maxtrbias,mintrbias=-MAXV,MAXV
        evidences   = []
        rightchoice = []
        for idxs, state in enumerate(unique_states):
            if(PCA_n_components>0):
                Xmerge_trials[state,coh] = mmodel.transform(Xmerge_trials[state,coh])
            for idx in range(EACHSTATES):
                Xdata_test = Xmerge_trials[state,coh][idx,:]
                idxdecoder = idx+idxs*EACHSTATES+idxcoh*(len(unique_states)*EACHSTATES)#np.random.choice(np.arange(0, NDEC, 1),size=1, replace=True)
                idxdecoder = np.mod(idxdecoder, NDEC)
                linw_bias, linb_bias = coeffs_trbias[:, idxdecoder*5+3], intercepts_trbias[0, 5*idxdecoder+3]
                linw_beh, linb_beh   = coeffs_beh[:,idxdecoder*5+4], intercepts_beh[0,5*idxdecoder+4]
                evidences   = np.append(evidences,np.squeeze(
                Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias))
                temp_perc   = np.squeeze(
                Xdata_test @ linw_beh.reshape(-1, 1) + linb_beh)

                rightchoice = np.append(rightchoice,temp_perc)
        
        trbias_encodings[coh] = evidences.copy() 
        behaviour_encodings[coh] = rightchoice.copy()
    return behaviour_encodings, trbias_encodings


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
    coh_ch_stateratio_error = np.ones_like(coh_ch_stateratio_error)/4.0

    return coh_ch_stateratio_correct, coh_ch_stateratio_error


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