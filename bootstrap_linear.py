# Load packages;
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import generate_pseudo_trials as gpt
from collections import Counter
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as sstats
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import Comp_pipeline as cp 
PRINT_PER = 1000
THRESH_TRIAL = 2



def bootstrap_linsvm_step_PCAonly(Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[], PCA_n_components=4):

    ### ac/ae ratio 
    CRATIO = ACE_RATIO/(1+ACE_RATIO)
    ERATIO = 1-CRATIO
    # NN      = np.shape(Xdata_hist_set[unique_states[0],'correct'])[1] 
    nlabels = 6*(len(files)-len(false_files))
    ntrain  = int(train_percent*N_pseudo_dec) # *4
    ntest   = (N_pseudo_dec-ntrain)*4 # state
    itest   = N_pseudo_dec-ntrain

    Xtest_set_correct, ytest_set_correct, =\
        np.zeros((n_iterations, ntest, NN)),\
        np.zeros((n_iterations, ntest, nlabels))  

    Xtest_set_error, ytest_set_error, =\
        np.zeros((n_iterations, ntest, NN)),\
        np.zeros((n_iterations, ntest, nlabels))  

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)
    
    CONTROL = 0
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # print('...... iteration index:', i,'........')
        ### >>>>>> generate training and testing dataset for decoder and testing(onestep)
        # N_pseudo_dec, N_pseudo_beh = 100,25
        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error,merge_trials_hist=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,2*N_pseudo_dec,RECORD_TRIALS, RECORDED_TRIALS_SET[i])

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials_hist
        #### --------- state 0 -----------------------
        Xdata_trainc,Xdata_testc=Xmerge_hist_trials_correct[4][:ntrainc,:],Xmerge_hist_trials_correct[4][ntrainc:ntrainc+itest,:] 
        Xdata_traine,Xdata_teste=Xmerge_hist_trials_error[0][:ntraine,:],Xmerge_hist_trials_error[0][ntraine:ntraine+itest,:] 
        ylabels_trainc,ylabels_testc = ymerge_hist_labels_correct[4][:ntrainc,:],ymerge_hist_labels_correct[4][ntrainc:ntrainc+itest,:] 
        ylabels_traine,ylabels_teste = ymerge_hist_labels_error[0][:ntraine,:],ymerge_hist_labels_error[0][ntraine:ntraine+itest,:] 
        for state in range(1,4): 
            Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_hist_trials_correct[state+4][:ntrainc ,:])),np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrainc:ntrainc+itest,:]))  
            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ymerge_hist_labels_correct[state+4][:ntrainc,:])),np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrainc:ntrainc+itest,:]))  
            Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_hist_trials_error[state][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntraine:ntraine+itest,:])) 
            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ymerge_hist_labels_error[state][:ntraine,:])),np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntraine:ntraine+itest,:])) 

        if DOREVERSE:
            ylabels_traine[:, 3] = 1-ylabels_traine[:, 3]
        ylabels_trainc = ylabels_trainc - 2
        ylabels_trainc[:,5::6] +=2 ### coherence
        if(i==0):
            Xdata_train   = np.append(Xdata_trainc, Xdata_traine, axis=0)
            ylabels_train = np.append(ylabels_trainc, ylabels_traine, axis=0)
        else:
            Xdata_train_t = np.append(Xdata_trainc, Xdata_traine, axis=0)
            ylabels_train_t = np.append(ylabels_trainc, ylabels_traine, axis=0)

            Xdata_train   = np.append(Xdata_train, Xdata_train_t,axis=0) 
            ylabels_train = np.append(ylabels_train, ylabels_train_t,axis=0)

    mmodel = PCA(n_components=PCA_n_components,svd_solver='full')
    mmodel.fit_transform(Xdata_train)
    print('explain_variance',np.sum(mmodel.explained_variance_ratio_))
    print('----shape:',np.shape(mmodel.transform(Xdata_train)))

    return mmodel

def bootstrap_linsvm_step_gaincontrol(data_tr,NN, unique_states,unique_cohs,nselect, files,false_files, pop_correct, pop_zero, pop_error, single_pop, CONDITION_CTXT, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[], mmodel=[],PCA_n_components=0):

    ### ac/ae ratio 
    CRATIO = ACE_RATIO/(1+ACE_RATIO)# according to theratio#0.5#share the same #  
    ERATIO = 1-CRATIO
    # NN      = np.shape(Xdata_hist_set[unique_states[0],'correct'])[1] 
    nlabels = 6*(len(files)-len(false_files))
    ntrain  = int(train_percent*N_pseudo_dec) # *4
    ntest   = (N_pseudo_dec-ntrain)*4 # state
    itest   = N_pseudo_dec-ntrain

    
    SVMAXIS=3

    if(PCA_n_components>0):
        NN = PCA_n_components

    Xtest_set_correct, ytest_set_correct, =\
        np.zeros((n_iterations, ntest, NN)),\
        np.zeros((n_iterations, ntest, nlabels))  

    Xtest_set_error, ytest_set_error, =\
        np.zeros((n_iterations, ntest, NN)),\
        np.zeros((n_iterations, ntest, nlabels))  

    yevi_set_correct = np.zeros((n_iterations, ntest, 3+2))
    yevi_set_error   = np.zeros((n_iterations, ntest, 3+2))

    yevi_set_correct_supp = np.zeros((n_iterations, ntest, 2+2))
    yevi_set_error_supp   = np.zeros((n_iterations, ntest, 2+2))### 3 Sept 

    stats = list()
    lin_pact = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_ctxt = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_xor  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                      shrinking=False, tol=1e-6)

    #### ------ gain control ---------
    if (CONDITION_CTXT):
        lin_bias_r  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
        lin_bias_a  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
    else:
        lin_bias_l  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
        lin_bias_r  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
    #### -----------------------------
    lin_cc    = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                 shrinking=False, tol=1e-6)

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    # print("ntrainc ---",ntrainc,'ntraine ----',ntraine)

    stats_c,stats_e = list(), list()
    stats_c_pop, stats_e_pop = list(),list()
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        ### generate training and testing dataset independently for each fold
        data_traintest_tr = cp.dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
        Xdata_hist_trainset, Xdata_hist_testset = data_traintest_tr['Xdata_hist_trainset'],data_traintest_tr['Xdata_hist_testset']
        ylabels_hist_trainset,ylabels_hist_testset = data_traintest_tr['ylabels_hist_trainset'],data_traintest_tr['ylabels_hist_testset']
        ### training dataset 
        Xdata_train_correct,ylabels_train_correct,Xdata_train_error,ylabels_train_error,merge_trials_hist_train=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_trainset,ylabels_hist_trainset,unique_states,unique_cohs,nselect, files,false_files,ntrain,RECORD_TRIALS, RECORDED_TRIALS_SET[i]) #individual
        ### testing dataset
        Xdata_test_correct,ylabels_test_correct,Xdata_test_error,ylabels_test_error,merge_trials_hist_test=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_testset,ylabels_hist_testset,unique_states,unique_cohs,nselect, files,false_files,itest,RECORD_TRIALS, RECORDED_TRIALS_SET[i]) #individual

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials_hist_train

        #### --------- state 0 -----------------------
        Xdata_trainc,Xdata_testc = Xdata_train_correct[4],Xdata_test_correct[4]
        Xdata_traine,Xdata_teste = Xdata_train_error[0],Xdata_test_error[0]
        ylabels_trainc,ylabels_testc = ylabels_train_correct[4],ylabels_test_correct[4]
        ylabels_traine,ylabels_teste = ylabels_train_error[0],ylabels_test_error[0]
        for state in range(1,4): 
            Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xdata_train_correct[state+4])),np.vstack((Xdata_testc,Xdata_test_correct[state+4]))  
            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ylabels_train_correct[state+4])),np.vstack((ylabels_testc,ylabels_test_correct[state+4]))  
            Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xdata_train_error[state])),np.vstack((Xdata_teste,Xdata_test_error[state])) 
            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ylabels_train_error[state])),np.vstack((ylabels_teste,ylabels_test_error[state])) 


        if(CONTROL==1):
            if DOREVERSE:
                ylabels_traine[:, 3] = 1-ylabels_traine[:, 3]
            Xdata_train   = Xdata_traine.copy()
            ylabels_traine[:,2::6] = 1-ylabels_traine[:,2::6]
            ylabels_train = ylabels_traine.copy() 
        elif(CONTROL==2):
            ylabels_trainc = ylabels_trainc - 2
            ylabels_trainc[:,5::6] +=2 ### coherence
            Xdata_train   = Xdata_trainc.copy()
            ylabels_train = ylabels_trainc.copy() 
        elif(CONTROL==0):  
            if DOREVERSE:
                ylabels_traine[:, 3] = 1-ylabels_traine[:, 3]
            ylabels_trainc = ylabels_trainc - 2
            ylabels_trainc[:,5::6] +=2 ### coherence
            ylabels_traine[:,2::6] = 1-ylabels_traine[:,2::6] ####reverse traine
            Xdata_train   = np.append(Xdata_trainc, Xdata_traine, axis=0)
            ylabels_train = np.append(ylabels_trainc, ylabels_traine, axis=0)
        if(PCA_n_components>0):
            Xdata_train = mmodel.transform(Xdata_train)

        #### reverse ylabels_[2]
        ylabels_teste[:,2::6] = 1-ylabels_teste[:,2::6]


        # pop_correct = np.union1d(pop_correct,pop_zero)
        # pop_error   = np.union1d(pop_error,pop_zero)
        pop = np.union1d(pop_correct,pop_error)


        ### identical conditions v.s. opposite conditions 
        Xdata_testc_cross = Xdata_testc.copy() 
        '''
        ### in prev ch left, pop_correct has lower firing rate therefore doesn't gate; pop_error has higher firing rate therefore gates the interference
        '''

        # Xdata_testc[:,pop_correct]   = Xdata_teste[:,pop_correct]
        # Xdata_testc[:,pop_error]     = Xdata_teste[:,pop_error]

        # fit model
        # model.fit(X_train,y_train) i.e model.fit(train set, train label as it
        # is a classifier)
        lin_pact.fit(Xdata_train, np.squeeze(ylabels_train[:, 0]))
        lin_ctxt.fit(Xdata_train, np.squeeze(ylabels_train[:, 1]))
        lin_xor.fit(Xdata_train, np.squeeze(ylabels_train[:, 2]))
        
        ### ---- most common tr. bias -----------
        ytr_bias = np.zeros(np.shape(ylabels_train)[0]) 
        for iset in range(np.shape(ylabels_train)[0]): 
            bias_labels=Counter(ylabels_train[iset,3::6]) 
            ytr_bias[iset]=(bias_labels.most_common(1)[0][0]) 
            ytr_bias[iset]=ylabels_train[iset,2] ### congruent
        if(CONDITION_CTXT):
            lin_bias_r.fit(np.vstack((Xdata_train[:ntrain,:],Xdata_train[1*ntrain:2*ntrain,:])), np.hstack((ytr_bias[:ntrain],ytr_bias[ntrain*1:ntrain*2])))
            lin_bias_a.fit(np.vstack((Xdata_train[2*ntrain:3*ntrain,:],Xdata_train[3*ntrain:4*ntrain,:])), np.hstack((ytr_bias[ntrain*2:ntrain*3],ytr_bias[ntrain*3:ntrain*4])))
        else:

            lin_bias_l.fit(np.vstack((Xdata_train[:ntrain,:],Xdata_train[2*ntrain:3*ntrain,:])), np.hstack((ytr_bias[:ntrain],ytr_bias[ntrain*2:ntrain*3])))
            lin_bias_r.fit(np.vstack((Xdata_train[1*ntrain:2*ntrain,:],Xdata_train[3*ntrain:4*ntrain,:])), np.hstack((ytr_bias[ntrain*1:ntrain*2],ytr_bias[ntrain*3:ntrain*4])))

        ### --- percentage of right choices -----
        ycchoice = np.zeros(np.shape(ylabels_train)[0])
        for iset in range(np.shape(ylabels_train)[0]):
            cchoice_labels = np.mean(ylabels_train[iset,4::6])#Counter(ylabels_train[iset,4::6])
            # ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
            if cchoice_labels>0.5:
                ycchoice[iset] =1 
            else:
                ycchoice[iset]=0
        lin_cc.fit(Xdata_train, ycchoice)

        if i == 0:
            intercepts = np.zeros((1, 3 + 1 + 1+1)) ## conditional decoders
            intercepts[:, 0] = lin_pact.intercept_[:]
            intercepts[:, 1] = lin_ctxt.intercept_[:]
            intercepts[:, 2] = lin_xor.intercept_[:]#lin_bias_.intercept_[:]#
            if(CONDITION_CTXT):
                intercepts[:, 3] = (lin_bias_r.intercept_[:])#lin_bias_l.intercept_[:]#
                intercepts[:, 4] = (lin_bias_a.intercept_[:])#lin_bias_r.intercept_[:]#
            else:
                intercepts[:, 3] = lin_bias_l.intercept_[:]#
                intercepts[:, 4] = lin_bias_r.intercept_[:]#

            intercepts[:, 5] = lin_cc.intercept_[:]

            coeffs = np.zeros((NN, 3 + 1 + 1+1))
            coeffs[:, 0] = lin_pact.coef_[:]
            coeffs[:, 1] = lin_ctxt.coef_[:]
            coeffs[:, 2] = lin_xor.coef_[:]#lin_bias_.coef_[:]#
            if (CONDITION_CTXT):
                coeffs[:, 3] = (lin_bias_r.coef_[:])#lin_bias_l.coef_[:]#
                coeffs[:, 4] = (lin_bias_a.coef_[:])#lin_bias_r.coef_[:]#
            else:
                coeffs[:, 3] = lin_bias_l.coef_[:]#
                coeffs[:, 4] = lin_bias_r.coef_[:]#

            coeffs[:, 5] = lin_cc.coef_[:]

        else:
            tintercepts = np.zeros((1, 3 + 1 + 1+1))
            tintercepts[:, 0] = lin_pact.intercept_[:]
            tintercepts[:, 1] = lin_ctxt.intercept_[:]
            tintercepts[:, 2] = lin_xor.intercept_[:]#lin_bias_.intercept_[:]#
            if(CONDITION_CTXT):
                tintercepts[:, 3] = (lin_bias_r.intercept_[:])#lin_bias_l.intercept_[:]#
                tintercepts[:, 4] = (lin_bias_a.intercept_[:])#lin_bias_r.intercept_[:]#
            else:
                tintercepts[:, 3] = lin_bias_l.intercept_[:]#
                tintercepts[:, 4] = lin_bias_r.intercept_[:]#
            tintercepts[:, 5] = lin_cc.intercept_[:]
            intercepts = np.append(intercepts, tintercepts, axis=1)

            tcoeffs = np.zeros((NN, 3 + 1 + 1+1))
            tcoeffs[:, 0] = lin_pact.coef_[:]
            tcoeffs[:, 1] = lin_ctxt.coef_[:]
            tcoeffs[:, 2] = lin_xor.coef_[:]#lin_bias_.coef_[:]#
            if (CONDITION_CTXT):
                tcoeffs[:, 3] = (lin_bias_r.coef_[:])#lin_bias_l.coef_[:]#
                tcoeffs[:, 4] = (lin_bias_a.coef_[:])#lin_bias_r.coef_[:]#
            else:
                tcoeffs[:, 3] = lin_bias_l.coef_[:]#
                tcoeffs[:, 4] = lin_bias_r.coef_[:]#
            tcoeffs[:, 5] = lin_cc.coef_[:]
            coeffs = np.append(coeffs, tcoeffs, axis=1)

        #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>

        if(PCA_n_components>0):
            Xdata_testc = mmodel.transform(Xdata_testc)
            Xdata_teste = mmodel.transform(Xdata_teste)
        #### -------- AC testing trials 
        linw_pact, linb_pact = lin_pact.coef_[:], lin_pact.intercept_[:]
        linw_ctxt, linb_ctxt = lin_ctxt.coef_[:], lin_ctxt.intercept_[:]
        linw_xor, linb_xor   = lin_xor.coef_[:],  lin_xor.intercept_[:]
        if(CONDITION_CTXT):
            linw_bias_r, linb_bias_r = lin_bias_r.coef_[:], lin_bias_r.intercept_[:]
            linw_bias_a, linb_bias_a = lin_bias_a.coef_[:], lin_bias_a.intercept_[:]
        else:
            linw_bias_l, linb_bias_l =  lin_bias_l.coef_[:], lin_bias_l.intercept_[:]
            linw_bias_r, linb_bias_r =  lin_bias_r.coef_[:], lin_bias_r.intercept_[:]
        
        linw_cc, linb_cc     =  lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
        evidences_c = np.zeros((ntest, 3 + 2))
    
        evidences_c_supp = np.zeros((ntest, 2+2)) ### 3 Sept @YX 

        evidences_c[:, 0] = np.squeeze(
            Xdata_testc[:,:] @ linw_pact.reshape(-1, 1)[:] + linb_pact) #pop
        evidences_c[:, 1] = np.squeeze(
            Xdata_testc[:,:] @ linw_ctxt.reshape(-1, 1)[:] + linb_ctxt) # pop

        # evidences_c[:, 2] = np.squeeze(
        #     Xdata_testc[:,pop] @ linw_xor.reshape(-1, 1)[pop] + linb_xor)
        ##### instead -------------------Cross-projections---------------
        #### gain control 
        if(CONDITION_CTXT):
            evidences_c[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)
            evidences_c[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)
            evidences_c[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_c[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)

            ### supplementary -- single selectivity 
            evidences_c_supp[0*itest:1*itest, 0] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)
            evidences_c_supp[1*itest:2*itest, 0] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)
            evidences_c_supp[2*itest:3*itest, 0] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_c_supp[3*itest:4*itest, 0] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            ### supplementary -- all neurons
            evidences_c_supp[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_c_supp[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_c_supp[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c_supp[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
        else:
            evidences_c[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_c[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,:] @ (linw_bias_l.reshape(-1, 1))[:] + linb_bias_l)
            evidences_c[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_c[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,:] @ (linw_bias_l.reshape(-1, 1))[:] + linb_bias_l)

            ### supplementary -- single selectivity 
            evidences_c_supp[0*itest:1*itest, 0] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r) ### single_pop
            evidences_c_supp[1*itest:2*itest, 0] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l)
            evidences_c_supp[2*itest:3*itest, 0] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_c_supp[3*itest:4*itest, 0] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l)

            ### supplementary -- all neurons
            evidences_c_supp[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r) ### : alla
            evidences_c_supp[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_c_supp[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c_supp[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)

        if(CONDITION_CTXT):
            evidences_c[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_c[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_c[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)
            evidences_c[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)

            ### supplementary -- single selectivity 
            evidences_c_supp[0*itest:1*itest, 1] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_c_supp[1*itest:2*itest, 1] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_c_supp[2*itest:3*itest, 1] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)
            evidences_c_supp[3*itest:4*itest, 1] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)

            ### supplementary -- mixed
            evidences_c_supp[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c_supp[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c_supp[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_c_supp[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
        else:
            evidences_c[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,:] @ (linw_bias_l.reshape(-1, 1))[:] + linb_bias_l)
            evidences_c[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_c[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,:] @ (linw_bias_l.reshape(-1, 1))[:] + linb_bias_l)
            evidences_c[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)

            ### supplementary -- single selectivity 
            evidences_c_supp[0*itest:1*itest, 1] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l) # single_pop
            evidences_c_supp[1*itest:2*itest, 1] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_c_supp[2*itest:3*itest, 1] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l)
            evidences_c_supp[3*itest:4*itest, 1] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)

            ### supplementary -- mixed
            evidences_c_supp[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_testc[0*itest:1*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l) ### : ###overall
            evidences_c_supp[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_testc[1*itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c_supp[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_c_supp[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)

        evidences_c[:, 4] = np.squeeze(
            Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)

        # evaluate model
        predictions_c     = np.zeros((np.shape(evidences_c)[0], 3 + 1 + 1))
        predictions_c_pop = np.zeros((np.shape(evidences_c)[0], 1 + 1))
        predictions_c[:, 0] = lin_pact.predict(
            Xdata_testc)  # model.predict(X_test)
        predictions_c[:, 1] = lin_ctxt.predict(Xdata_testc)
        predictions_c[:, 2] = lin_xor.predict(Xdata_testc)

        # if(CONDITION_CTXT):
        #     predictions_c[0*itest:1*itest, 3] = lin_bias_r.predict(Xdata_testc[0*itest:1*itest,:])
        #     predictions_c[1*itest:2*itest, 3] = lin_bias_r.predict(Xdata_testc[1*itest:2*itest,:])
        #     predictions_c[2*itest:3*itest, 3] = lin_bias_a.predict(Xdata_testc[2*itest:3*itest,:])
        #     predictions_c[3*itest:4*itest, 3] = lin_bias_a.predict(Xdata_testc[3*itest:4*itest,:])
        # else:
        #     predictions_c[0*itest:1*itest, 3] = lin_bias_l.predict(Xdata_testc[0*itest:1*itest,:])
        #     predictions_c[1*itest:2*itest, 3] = lin_bias_r.predict(Xdata_testc[1*itest:2*itest,:])
        #     predictions_c[2*itest:3*itest, 3] = lin_bias_l.predict(Xdata_testc[2*itest:3*itest,:])
        #     predictions_c[3*itest:4*itest, 3] = lin_bias_r.predict(Xdata_testc[3*itest:4*itest,:])

        ### overall neurons
        evi_overall = evidences_c[:,3]-evidences_c[:,2]
        predictions_c[np.where(evi_overall>=0)[0], 3] = 1
        predictions_c[np.where(evi_overall<0)[0], 3]  = 0

        ##### 0 -- single selectivity; 1 -- mixed selectivity 
        evi_single = evidences_c_supp[:,1]-evidences_c_supp[:,0]
        predictions_c_pop[np.where(evi_single>=0)[0],0] = 1
        predictions_c_pop[np.where(evi_single<0)[0],0]  = 0 
        evi_mixed = evidences_c_supp[:,3]-evidences_c_supp[:,2]
        predictions_c_pop[np.where(evi_mixed>=0)[0],1] = 1
        predictions_c_pop[np.where(evi_mixed<0)[0],1]  = 0
        
        predictions_c[:, 4] = lin_cc.predict(Xdata_testc)

        ### modifying ylabels_testc[:,3], upcoming stimulus category 
        ytr_test_bias = np.zeros(np.shape(ylabels_testc)[0])
        for iset in range(np.shape(ylabels_testc)[0]):
            bias_labels=Counter(ylabels_testc[iset,3::6])
            ytr_test_bias[iset]=(bias_labels.most_common(1)[0][0])
        ylabels_testc[:,3] =  ytr_test_bias[:]#ylabels_testc[:,2]# congruent#
        # print('~~~~~~ correct match:',ylabels_testc[:,3])
        # print('~~~~~~~~~~~~~~~:',ylabels_testc[:,2])

        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
        # ylabels_testc[:,4::6]-=2
        for iset in range(np.shape(ylabels_testc)[0]):
            cchoice_labels = np.mean(ylabels_testc[iset,4::6])#Counter(ylabels_testc[iset,4::6])
            # ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
            if(cchoice_labels>2.5):
                ycchoice_test[iset] = 3
            else:
                ycchoice_test[iset] = 2
        ylabels_testc[:,4] = ycchoice_test[:]

        Xtest_set_correct[i, :, :], ytest_set_correct[i, :, :]=Xdata_testc[:, :].copy(), ylabels_testc[:, :].copy()
        
        if i == 0:
            yevi_set_correct[i, :, :]      = evidences_c.copy()
            yevi_set_correct_supp[i, :, :] = evidences_c_supp.copy()
        else:
            yevi_set_correct[i, :, :] = evidences_c.copy()
            yevi_set_correct_supp[i, :, :] = evidences_c_supp.copy()

        score_c, score_c_pop     = np.zeros((3 + 1 + 1, 1)), np.zeros((1 + 1, 1))
        for j in range(np.shape(score_c)[0]):
            score_c[j, 0] = accuracy_score(
                ylabels_testc[:, j]-2, predictions_c[:, j])

        for j in range(2):
            score_c_pop[j,0] = accuracy_score(ylabels_testc[:,3]-2,  predictions_c_pop[:,j])

        #### -------- AE testing trials --------------
        # evaluate evidence model
        evidences_e = np.zeros((ntest, 3 + 2))

        evidences_e_supp = np.zeros((ntest, 2+2)) ### 3 Sept @YX 

        evidences_e[:, 0] = np.squeeze(
            Xdata_teste[:,:] @ linw_pact.reshape(-1, 1)[:] + linb_pact) # pop
        evidences_e[:, 1] = np.squeeze(
            Xdata_teste[:,:] @ linw_ctxt.reshape(-1, 1)[:] + linb_ctxt) # pop
        
        # evidences_e[:, 2] = np.squeeze(
        #     Xdata_teste[:,pop] @ linw_xor.reshape(-1, 1)[pop] + linb_xor)
        #### gain control   ------ cross projections
        if(CONDITION_CTXT):
            evidences_e[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)
            evidences_e[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)
            evidences_e[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_e[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)

            ### supplementary -- single selectivity 
            evidences_e_supp[0*itest:1*itest, 0] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)
            evidences_e_supp[1*itest:2*itest, 0] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)
            evidences_e_supp[2*itest:3*itest, 0] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_e_supp[3*itest:4*itest, 0] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            ### supplementary -- all neurons
            evidences_e_supp[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_e_supp[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_e_supp[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e_supp[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)

        else:
            evidences_e[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_e[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,:] @ (linw_bias_l.reshape(-1, 1))[:] + linb_bias_l)
            evidences_e[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_e[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,:] @ (linw_bias_l.reshape(-1, 1))[:] + linb_bias_l)

            ### supplementary -- single selectivity 
            evidences_e_supp[0*itest:1*itest, 0] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_e_supp[1*itest:2*itest, 0] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l)
            evidences_e_supp[2*itest:3*itest, 0] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_e_supp[3*itest:4*itest, 0] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l)

            ### supplementary -- all neurons
            evidences_e_supp[0*itest:1*itest, 2] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e_supp[1*itest:2*itest, 2] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_e_supp[2*itest:3*itest, 2] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c_supp[3*itest:4*itest, 2] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)

        # evidences_e[:, 3] = np.squeeze(
        #         Xdata_teste[:,:] @ linw_bias.reshape(-1, 1) + linb_bias)
        #### gain control 
        if(CONDITION_CTXT):
            evidences_e[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_e[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,:] @ (linw_bias_r.reshape(-1, 1))[:] + linb_bias_r)
            evidences_e[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)
            evidences_e[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,:] @ (linw_bias_a.reshape(-1, 1))[:] + linb_bias_a)

            ### supplementary -- single selectivity 
            evidences_e_supp[0*itest:1*itest, 1] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_e_supp[1*itest:2*itest, 1] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_e_supp[2*itest:3*itest, 1] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)
            evidences_e_supp[3*itest:4*itest, 1] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,single_pop] @ (linw_bias_a.reshape(-1, 1))[single_pop] + linb_bias_a)
            ### supplementary -- all neurons
            evidences_e_supp[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e_supp[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e_supp[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_e_supp[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
        else:

            ### gain control 
            evidences_e[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_e[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_e[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)

            ### supplementary -- single selectivity 
            evidences_e_supp[0*itest:1*itest, 1] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l)
            evidences_e_supp[1*itest:2*itest, 1] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)
            evidences_e_supp[2*itest:3*itest, 1] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,single_pop] @ (linw_bias_l.reshape(-1, 1))[single_pop] + linb_bias_l)
            evidences_e_supp[3*itest:4*itest, 1] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,single_pop] @ (linw_bias_r.reshape(-1, 1))[single_pop] + linb_bias_r)

            ### supplementary -- all neurons
            evidences_e_supp[0*itest:1*itest, 3] = np.squeeze(
                    Xdata_teste[0*itest:1*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_e_supp[1*itest:2*itest, 3] = np.squeeze(
                    Xdata_teste[1*itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e_supp[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_e_supp[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)

            # evidences_e[0*itest:3*itest, 3] = np.squeeze(
            #         Xdata_teste[0*itest:3*itest,pop_error] @ (linw_bias_l.reshape(-1, 1))[pop_error] + linb_bias_l)
            # evidences_e[3*itest:4*itest, 3] = np.squeeze(
            #         Xdata_teste[3*itest:4*itest,pop_error] @ (linw_bias_r.reshape(-1, 1))[pop_error] + linb_bias_r)

        evidences_e[:, 4] = np.squeeze(
            Xdata_teste @ linw_cc.reshape(-1, 1) + linb_cc)

        # evaluate model
        predictions_e     = np.zeros((np.shape(evidences_e)[0], 3 + 1 + 1))
        predictions_e_pop = np.zeros((np.shape(evidences_e)[0], 1 + 1))
        predictions_e[:, 0] = lin_pact.predict(
            Xdata_teste)  # model.predict(X_test)
        predictions_e[:, 1] = lin_ctxt.predict(Xdata_teste)
        predictions_e[:, 2] = lin_xor.predict(Xdata_teste)
        
        # if(CONDITION_CTXT):
        #     predictions_e[0*itest:1*itest, 3] = lin_bias_r.predict(Xdata_teste[0*itest:1*itest,:])
        #     predictions_e[1*itest:2*itest, 3] = lin_bias_r.predict(Xdata_teste[1*itest:2*itest,:])
        #     predictions_e[2*itest:3*itest, 3] = lin_bias_a.predict(Xdata_teste[2*itest:3*itest,:])
        #     predictions_e[3*itest:4*itest, 3] = lin_bias_a.predict(Xdata_teste[3*itest:4*itest,:])
        # else:
        #     predictions_e[0*itest:1*itest, 3] = lin_bias_l.predict(Xdata_teste[0*itest:1*itest,:])
        #     predictions_e[1*itest:2*itest, 3] = lin_bias_r.predict(Xdata_teste[1*itest:2*itest,:])
        #     predictions_e[2*itest:3*itest, 3] = lin_bias_l.predict(Xdata_teste[2*itest:3*itest,:])
        #     predictions_e[3*itest:4*itest, 3] = lin_bias_r.predict(Xdata_teste[3*itest:4*itest,:])


        ### overall neurons
        evi_overall = evidences_e[:,3]-evidences_e[:,2]
        predictions_e[np.where(evi_overall>=0)[0], 3] = 1
        predictions_e[np.where(evi_overall<0)[0], 3]  = 0

        ##### 0 -- single selectivity; 1 -- mixed selectivity 
        evi_single = evidences_e_supp[:,1]-evidences_e_supp[:,0]
        predictions_e_pop[np.where(evi_single>=0)[0],0] = 1
        predictions_e_pop[np.where(evi_single<0)[0],0]  = 0 
        evi_mixed = evidences_e_supp[:,3]-evidences_e_supp[:,2]
        predictions_e_pop[np.where(evi_mixed>=0)[0],1] = 1
        predictions_e_pop[np.where(evi_mixed<0)[0],1]  = 0

        predictions_e[:, 4] = lin_cc.predict(Xdata_teste)

        ### modifying ylabels_testc[:,3], upcoming stimulus category 
        ytr_test_bias = np.zeros(np.shape(ylabels_teste)[0])
        for iset in range(np.shape(ylabels_teste)[0]):
            bias_labels=Counter(ylabels_teste[iset,3::6])
            ytr_test_bias[iset]=(bias_labels.most_common(1)[0][0])
        ylabels_teste[:,3] = ytr_test_bias[:]# ylabels_teste[:,2]#congruent # 
        # print('~~~~~~ error match:',ylabels_teste[:,3])
        # print('~~~~~~~~~~~~~~~:',ylabels_teste[:,2])
        # print("~~~~error :",ylabels_teste[10,2:])

        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_teste)[0])
        for iset in range(np.shape(ylabels_teste)[0]):
            cchoice_labels = np.mean(ylabels_teste[iset,4::6])#Counter(ylabels_teste[iset,4::6])
            # ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
            if(cchoice_labels>0.5):
                ycchoice_test[iset]=1
            else:
                ycchoice_test[iset]=0
        ylabels_teste[:,4] = ycchoice_test[:]

        Xtest_set_error[i, :, :], ytest_set_error[i, :, :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

        if i == 0:
            yevi_set_error[i, :, :]      = evidences_e.copy()
            yevi_set_error_supp[i, :, :] = evidences_e_supp.copy()
        else:
            yevi_set_error[i, :, :]      = evidences_e.copy()
            yevi_set_error_supp[i, :, :] = evidences_e_supp.copy()

        score_e, score_e_pop = np.zeros((3 + 1 + 1, 1)), np.zeros((1+1,1))
        for j in range(np.shape(score_e)[0]):
            score_e[j, 0] = accuracy_score(
                ylabels_teste[:, j], predictions_e[:, j])

        for j in range(2):
            score_e_pop[j,0] = accuracy_score(ylabels_teste[:,3],  predictions_e_pop[:,j])

        # print(score)
        if i == 0:
            stats_c = score_c
            stats_e = score_e
            stats_c_pop = score_c_pop 
            stats_e_pop = score_e_pop
            # print('score e:',score_c[SVMAXIS,0], ' e:',score_e[SVMAXIS,0])
        else:
            stats_c = np.hstack((stats_c, score_c))
            stats_e = np.hstack((stats_e, score_e))
            stats_c_pop = np.hstack((stats_c_pop,score_c_pop))
            stats_e_pop = np.hstack((stats_e_pop,score_e_pop))

    return mmodel,stats_c,stats_e, stats_c_pop, stats_e_pop, coeffs, intercepts,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,yevi_set_correct_supp,\
        Xtest_set_error, ytest_set_error, yevi_set_error, yevi_set_error_supp, RECORDED_TRIALS_SET

def bootstrap_linsvm_proj_step(coeffs_pool, intercepts_pool, Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, pop_correct,pop_zero, pop_error, USE_POP, type, DOREVERSE=0, n_iterations=10, N_pseudo_dec=25, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
    # NN      = np.shape(Xdata_hist_set[unique_states[0],'correct'])[1]
    nlabels = 6*(len(files)-len(false_files))
    ntrain  = int(train_percent*N_pseudo_dec)
    ntest   = (N_pseudo_dec-ntrain)*4 # state


    if(PCA_n_components>0):
        NN = PCA_n_components

    Xtest_set_correct, ytest_set_correct, =\
        np.zeros((n_iterations, ntest, NN)),\
        np.zeros((n_iterations, ntest, nlabels))

    Xtest_set_error, ytest_set_error, =\
        np.zeros((n_iterations, ntest, NN)),\
        np.zeros((n_iterations, ntest, nlabels))

    yevi_set_correct = np.zeros((n_iterations, ntest, 3+2))
    yevi_set_error   = np.zeros((n_iterations, ntest, 3+2))

    stats_c, stats_e = list(),list()

    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error,merge_trials_hist=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,N_pseudo_dec,RECORD_TRIALS, RECORDED_TRIALS_SET[i])

        Xdata_trainc,Xdata_testc=Xmerge_hist_trials_correct[4][:ntrain,:],Xmerge_hist_trials_correct[4][ntrain:,:]
        ylabels_trainc,ylabels_testc = ymerge_hist_labels_correct[4][:ntrain,:],ymerge_hist_labels_correct[4][ntrain:,:]

        Xdata_traine,Xdata_teste=Xmerge_hist_trials_error[0][:ntrain,:],Xmerge_hist_trials_error[0][ntrain:,:]
        ylabels_traine,ylabels_teste = ymerge_hist_labels_error[0][:ntrain,:],ymerge_hist_labels_error[0][ntrain:,:]
        for state in range(1,4):
            Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_hist_trials_correct[state+4][:ntrain ,:])),np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrain :,:]))
            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ymerge_hist_labels_correct[state+4][:ntrain,:])),np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrain:,:]))

            Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_hist_trials_error[state][:ntrain,:])),np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntrain:,:]))
            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ymerge_hist_labels_error[state][:ntrain,:])),np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntrain :,:]))

        if(PCA_n_components>0):
            Xdata_testc = mmodel.transform(Xdata_testc)
            Xdata_teste = mmodel.transform(Xdata_teste)

        # @YX 0910 -- weights
        linw_pact, linb_pact = coeffs_pool[:, i*5+0], intercepts_pool[0, 5*i+0]
        linw_ctxt, linb_ctxt = coeffs_pool[:, i*5+1], intercepts_pool[0, 5*i+1]
        linw_xor, linb_xor   = coeffs_pool[:, i*5+2], intercepts_pool[0, 5*i+2]
        linw_bias, linb_bias = coeffs_pool[:, i*5+3], intercepts_pool[0, 5*i+3]
        linw_cc, linb_cc     = coeffs_pool[:, i*5+4], intercepts_pool[0, 5*i+4]
        # evaluate evidence model--CORRECT
        evidences_c = np.zeros((ntest, 3 + 2))
        evidences_c[:, 0] = np.squeeze(
            Xdata_testc @ linw_pact.reshape(-1, 1) + linb_pact)
        evidences_c[:, 1] = np.squeeze(
            Xdata_testc @ linw_ctxt.reshape(-1, 1) + linb_ctxt)
        evidences_c[:, 2] = np.squeeze(
            Xdata_testc @ linw_xor.reshape(-1, 1) + linb_xor)
        evidences_c[:, 3] = np.squeeze(
            Xdata_testc @ linw_bias.reshape(-1, 1) + linb_bias)
        evidences_c[:, 4] = np.squeeze(
            Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)

        predictions_c = np.zeros((np.shape(evidences_c)[0], 3 + 2))
        for j in range(3 + 2):
            predictions_c[np.where(evidences_c[:, j] > 0)[0], j] = 1
            predictions_c[np.where(evidences_c[:, j] <= 0)[0], j] = 0


        ### modifying ylabels_testc[:,3], upcoming stimulus category
        ytr_test_bias = np.zeros(np.shape(ylabels_testc)[0])
        for iset in range(np.shape(ylabels_testc)[0]):
            bias_labels=Counter(ylabels_testc[iset,3::6])
            ytr_test_bias[iset]=(bias_labels.most_common(1)[0][0])
        ylabels_testc[:,3] = ytr_test_bias[:]

        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
        for iset in range(np.shape(ylabels_testc)[0]):
            cchoice_labels = Counter(ylabels_testc[iset,4::6])
            ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
        ylabels_testc[:,4] = ycchoice_test[:]

        score_c = np.zeros((3 + 2, 1))
        for j in range(3 + 2):
            score_c[j, 0] = accuracy_score(
                ylabels_testc[:, j], predictions_c[:, j])

        Xtest_set_correct[i, :, :], ytest_set_correct[i, :, :]=Xdata_testc[:, :].copy(), ylabels_testc[:, :].copy()

        if i == 0:
            yevi_set_correct[i, :, :] = evidences_c.copy()
        else:
            yevi_set_correct[i, :, :] = evidences_c.copy()

        #### -------- AE testing trials --------------
        # evaluate evidence model
        evidences_e = np.zeros((ntest, 3 + 2))
        evidences_e[:, 0] = np.squeeze(
            Xdata_teste @ linw_pact.reshape(-1, 1) + linb_pact)
        evidences_e[:, 1] = np.squeeze(
            Xdata_teste @ linw_ctxt.reshape(-1, 1) + linb_ctxt)
        evidences_e[:, 2] = np.squeeze(
            Xdata_teste @ linw_xor.reshape(-1, 1) + linb_xor)
        evidences_e[:, 3] = np.squeeze(
            Xdata_teste @ linw_bias.reshape(-1, 1) + linb_bias)
        evidences_e[:, 4] = np.squeeze(
            Xdata_teste @ linw_cc.reshape(-1, 1) + linb_cc)


        predictions_e = np.zeros((np.shape(evidences_e)[0], 3 + 2))
        for j in range(3 + 2):
            predictions_e[np.where(evidences_e[:, j] > 0)[0], j] = 1
            predictions_e[np.where(evidences_e[:, j] <= 0)[0], j] = 0

        ### modifying ylabels_testc[:,3], upcoming stimulus category
        ytr_test_bias = np.zeros(np.shape(ylabels_teste)[0])
        for iset in range(np.shape(ylabels_teste)[0]):
            bias_labels=Counter(ylabels_teste[iset,3::6])
            ytr_test_bias[iset]=(bias_labels.most_common(1)[0][0])
        ylabels_teste[:,3] = ytr_test_bias[:]

        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_teste)[0])
        for iset in range(np.shape(ylabels_teste)[0]):
            cchoice_labels = Counter(ylabels_teste[iset,4::6])
            ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
        ylabels_teste[:,4] = ycchoice_test[:]


        score_e = np.zeros((3 + 2, 1))
        for j in range(3 + 2):
            score_e[j, 0] = accuracy_score(
                ylabels_teste[:, j], predictions_e[:, j])

        Xtest_set_error[i, :, :], ytest_set_error[i, :, :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

        if i == 0:
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()

        if i == 0:
            stats_c = [score_c[SVMAXIS,0]]
            stats_e = [score_e[SVMAXIS,0]]
        else:
            stats_c = np.append(stats_c, score_c[SVMAXIS,0])
            stats_e = np.append(stats_e, score_e[SVMAXIS,0])
    return mmodel,stats_c, stats_e, coeffs_pool, intercepts_pool, \
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

    

# def bootstrap_linsvm_step_fixationperiod_balanced(data_tr, NN, unique_states,unique_cohs,nselect, files,false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error, pop_correct,pop_error, USE_POP, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=5, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
#     ### ac/ae ratio 
#     CRATIO   = ACE_RATIO/(1+ACE_RATIO)#0.5#
#     ERATIO   = 1-CRATIO
#     #  
#     COHAXISS = unique_cohs#[-1,0,1]#[-1]#[1]

#     nlabels  = (len(files)-len(false_files))
#     ntrain   = int(train_percent*N_pseudo_dec) #
#     ntest    = (N_pseudo_dec-ntrain)*2#*len(COHAXISS)#Nchoice#len(COHAXISS) # COHS
#     itest    = N_pseudo_dec-ntrain

#     if USE_POP==1:
#         if(CONTROL==1):
#             NN = len(pop_error)
#         elif(CONTROL==2):
#             NN= len(pop_correct)
    
#     ### ONLY USE COHERENCE = 0
#     if(PCA_n_components>0):
#         NN = PCA_n_components

#     Xtest_set_correct, ytest_set_correct, =\
#         np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
#         np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

#     Xtest_set_error, ytest_set_error, =\
#         np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
#         np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

#     yevi_set_correct = np.zeros((n_iterations, ntest*len(COHAXISS), 3+2))
#     yevi_set_error   = np.zeros((n_iterations, ntest*len(COHAXISS), 3+2))

#     lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
 
#     ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
#     ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
#     ntraine = int(ntraine)  
#     print("behaviour --- ntrainc ---",ntrainc,'ntraine ----',ntraine)

#     stats_c,stats_e = np.zeros((n_iterations,len(COHAXISS))),np.zeros((n_iterations,len(COHAXISS)))
#     i_sucess = -1
#     for i in range(n_iterations):
#         if (i+1) % PRINT_PER == 0:
#             print(i)

#         ### generate training and testing dataset independently for each fold
#         data_traintest_tr = cp.dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
#         Xdata_beh_trainset, ylabels_beh_trainset, Xdata_beh_testset, ylabels_beh_testset=data_traintest_tr['Xdata_beh_trainset'], \
#             data_traintest_tr['ylabels_beh_trainset'], data_traintest_tr['Xdata_beh_testset'], data_traintest_tr['ylabels_beh_testset']

#         if (CONTROL==0):
#             Xdata_train_correct,ylabels_train_correct,_,_,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,nselect, files, false_files,EACHSTATES=ntrainc, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS
#             _,_,Xdata_train_error,ylabels_train_error,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,nselect,files, false_files,EACHSTATES=ntraine, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to  
#         else:           
#             Xdata_train_correct,ylabels_train_correct,Xdata_train_error,ylabels_train_error,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,nselect,files, false_files,EACHSTATES=ntrain, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS

#         Xdata_test_correct,ylabels_test_correct,Xdata_test_error,ylabels_test_error,merge_trials_test=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_testset,ylabels_beh_testset,unique_states,COHAXISS,nselect,files, false_files,EACHSTATES=itest, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS

#         if RECORD_TRIALS == 1:
#             RECORDED_TRIALS_SET[i]=merge_trials_train
#         ### --------- state 0 ----------------------- 
#         Xdata_trainc, Xdata_testc, Xdata_traine, Xdata_teste = [],[],[],[]
#         for idxcoh, COHAXIS in enumerate(COHAXISS):
#             Xdata_trainc,Xdata_testc     = Xdata_train_correct[COHAXIS,0],Xdata_test_correct[COHAXIS,0]
#             ylabels_trainc,ylabels_testc = ylabels_train_correct[COHAXIS,0].T,(ylabels_test_correct[COHAXIS,0].T)
#             # print('~~~~~~000',ylabels_train_correct[COHAXIS,0].T)

#             Xdata_traine,Xdata_teste     = Xdata_train_error[COHAXIS,0],Xdata_test_error[COHAXIS,0]
#             ylabels_traine,ylabels_teste = (ylabels_train_error[COHAXIS,0].T),(ylabels_test_error[COHAXIS,0].T)   

#             Xdata_trainc,Xdata_testc     = np.vstack((Xdata_trainc,Xdata_train_correct[COHAXIS,1])),np.vstack((Xdata_testc,Xdata_test_correct[COHAXIS,1]))
#             ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ylabels_train_correct[COHAXIS,1].T)),np.vstack((ylabels_testc,ylabels_test_correct[COHAXIS,1].T))

#             Xdata_traine,Xdata_teste     = np.vstack((Xdata_traine,Xdata_train_error[COHAXIS,1])),np.vstack((Xdata_teste,Xdata_test_error[COHAXIS,1]))
#             ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ylabels_train_error[COHAXIS,1].T)),np.vstack((ylabels_teste,ylabels_test_error[COHAXIS,1].T)) 
#             # print('~~~~~~111',ylabels_train_correct[COHAXIS,1].T)
#             # print('test c:',ylabels_testc[::itest])
#             # print('test e:',ylabels_teste[::itest])


#             ###### labels have no other labels except Behaviours/Stimulus
#             if(CONTROL==1): 
#                 if DOREVERSE: 
#                     ylabels_traine[:, :] = 1-ylabels_traine[:, :]
#                 Xdata_train   = Xdata_traine.copy()
#                 ylabels_train = ylabels_traine.copy() 
#             elif(CONTROL==2): 
#                 Xdata_train   = Xdata_trainc.copy()
#                 ### already substract 2 in previous step
#                 ylabels_train = ylabels_trainc.copy() 
#             elif(CONTROL==0):   
#                 if DOREVERSE:
#                     ylabels_traine[:, :] = 1-ylabels_traine[:, :]
#                 # already substract 2 in previous step
#                 Xdata_train   = np.append(Xdata_trainc, Xdata_traine, axis=0)   
#                 ylabels_train = np.append(ylabels_trainc, ylabels_traine, axis=0)
                
#             if(PCA_n_components>0):
#                 Xdata_train = mmodel.transform(Xdata_train)

#             # print('~~~~~~size of the test dataset:', np.shape(Xdata_train))

#             #### whether use populations or not 
#             if(USE_POP):
#                 if(CONTROL==1):
#                     Xdata_train = Xdata_train[:,pop_error] 
#                     Xdata_testc = Xdata_testc[:,pop_error]
#                     Xdata_teste = Xdata_teste[:,pop_error]
#                 elif(CONTROL==2):
#                     Xdata_train = Xdata_train[:,pop_correct]
#                     Xdata_testc = Xdata_testc[:,pop_correct]
#                     Xdata_teste = Xdata_teste[:,pop_correct]

#             ### --- percentage of right choices -----
#             ycchoice = np.zeros(np.shape(ylabels_train)[0])
#             for iset in range(np.shape(ylabels_train)[0]):
#                 cchoice_labels = Counter(ylabels_train[iset,:])
#                 ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
                            
            
#             lin_cc.fit(Xdata_train, ycchoice)


#             if i==0 and idxcoh==0:
#                 intercepts = np.zeros((1, 3 + 1 + 1))
#                 intercepts[:, 4] = lin_cc.intercept_[:]

#                 coeffs = np.zeros((NN, 3 + 1 + 1))
#                 coeffs[:, 4]     = lin_cc.coef_[:]

#             else:
#                 tintercepts = np.zeros((1, 3 + 1 + 1))
#                 tintercepts[:, 4] = lin_cc.intercept_[:]
#                 intercepts        = np.append(intercepts, tintercepts, axis=1)

#                 tcoeffs           = np.zeros((NN, 3 + 1 + 1))
#                 tcoeffs[:, 4]     = lin_cc.coef_[:]
#                 coeffs = np.append(coeffs, tcoeffs, axis=1)

#             #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>
#             if(PCA_n_components>0):
#                 Xdata_testc=mmodel.transform(Xdata_testc)
#                 Xdata_teste=mmodel.transform(Xdata_teste)
#             #### -------- AC testing trials 
#             linw_cc, linb_cc     = lin_cc.coef_[:], lin_cc.intercept_[:]
#             # evaluate evidence model
#             evidences_c = np.zeros((ntest, 3 + 2))
#             # print('shape...',np.shape(Xdata_testc),np.shape(evidences_c))
#             evidences_c[:, 4] = np.squeeze(
#                 Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)


#             ### --- percentage of right choices -----
#             ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
#             for iset in range(np.shape(ylabels_testc)[0]):
#                 cchoice_labels = Counter(ylabels_testc[iset,:])
#                 ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
#                 # ycchoice_test[iset] = cchoice_labels[1]/np.shape(ylabels_testc)[1]
                
#             ylabels_testc[:,0] = ycchoice_test[:]
#             # print('~~~~~~test c:',np.mean(ylabels_testc[:,4]))
            
#             # ypredict_choice = lin_cc.predict(Xdata_testc)
#             ypredict_choice = np.zeros_like(ycchoice_test)
#             ypredict_choice[np.where(evidences_c[:,4]>0)[0]]=1
#             # print('~~~~shape:',np.shape(ylabels_testc[:,4].flatten()),np.shape(np.squeeze(ylabels_testc[:,4])),np.shape(ypredict_choice))
#             prediction_correct = accuracy_score((ylabels_testc[:,0]).flatten(),ypredict_choice)
#             stats_c[i,idxcoh] = prediction_correct

#             Xtest_set_correct[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_correct[i,2* (N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_testc[:, :].copy(), ylabels_testc[:, :].copy()
            
#             yevi_set_correct[i,2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :] = evidences_c.copy()


#             #### -------- AE testing trials --------------
#             # evaluate evidence model
#             evidences_e = np.zeros((ntest, 3 + 2))
#             evidences_e[:, 4] = np.squeeze(
#                 Xdata_teste @ linw_cc.reshape(-1, 1) + linb_cc)


#             ### --- percentage of right choices -----
#             ycchoice_test = np.zeros(np.shape(ylabels_teste)[0])
#             for iset in range(np.shape(ylabels_teste)[0]):
#                 cchoice_labels = Counter(ylabels_teste[iset,:])
#                 ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
#                 # ycchoice_test[iset] = cchoice_labels[1]/np.shape(ylabels_teste)[1]
#             ylabels_teste[:,4] = ycchoice_test[:]
#             # print('~~~~~~test e:',np.mean(ylabels_teste[:,4]))
            
#             # ypredict_choice = lin_cc.predict(Xdata_teste)
#             ypredict_choice = np.zeros_like(ycchoice_test)
#             ypredict_choice[np.where(evidences_e[:,4]>0)[0]]=1
            
#             prediction_error = accuracy_score((ylabels_teste[:,4]).flatten(),ypredict_choice)
#             stats_e[i,idxcoh] = prediction_error
#             # print('~~~~~~~~ performance:', prediction_correct,prediction_error)
            

#             Xtest_set_error[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_error[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

#             yevi_set_error[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :] = evidences_e.copy()
#     fig,ax = plt.subplots(2,1,figsize=(4,6),tight_layout=True,sharex=True,sharey=True)
#     ax[0].hist(stats_c[:,:].flatten(),facecolor='red',alpha=0.25)
#     ax[1].hist(stats_e[:,:].flatten(),facecolor='blue',alpha=0.25)
#     ax[0].set_xlim([0,1.0])
#     return stats_c, stats_e, coeffs, intercepts,\
#         Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
#         Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET


def bootstrap_linsvm_step_fixationperiod_balanced(data_tr, NN, unique_states,unique_cohs,nselect, files,false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error, pop_correct,pop_error, USE_POP, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=5, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
    ### ac/ae ratio 
    CRATIO   = ACE_RATIO/(1+ACE_RATIO)#0.5#
    ERATIO   = 1-CRATIO
    #  
    COHAXISS = unique_cohs#[-1,0,1]#[-1]#[1]

    nlabels  = (len(files)-len(false_files))
    ntrain   = int(train_percent*N_pseudo_dec) #
    ntest    = (N_pseudo_dec-ntrain)*2#*len(COHAXISS)#Nchoice#len(COHAXISS) # COHS
    itest    = N_pseudo_dec-ntrain

    if USE_POP==1:
        if(CONTROL==1):
            NN = len(pop_error)
        elif(CONTROL==2):
            NN= len(pop_correct)
    
    ### ONLY USE COHERENCE = 0
    if(PCA_n_components>0):
        NN = PCA_n_components
    ### repeating
    Xtest_set_correct_rep, ytest_set_correct_rep, =\
        np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
        np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

    Xtest_set_error_rep, ytest_set_error_rep, =\
        np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
        np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

    ### alternating
    Xtest_set_correct_alt, ytest_set_correct_alt, =\
        np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
        np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

    Xtest_set_error_alt, ytest_set_error_alt, =\
        np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
        np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

    yevi_set_correct = np.zeros((n_iterations, ntest*len(COHAXISS), 3+2)) 
    yevi_set_error   = np.zeros((n_iterations, ntest*len(COHAXISS), 3+2))### 3-rep;4-alt

    lin_cc_rep   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
    lin_cc_alt   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
 
    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("behaviour --- ntrainc ---",ntrainc,'ntraine ----',ntraine)

    stats_c_rep,stats_e_rep = np.zeros((n_iterations,len(COHAXISS))),np.zeros((n_iterations,len(COHAXISS)))
    stats_c_alt,stats_e_alt = np.zeros((n_iterations,len(COHAXISS))),np.zeros((n_iterations,len(COHAXISS)))
    i_sucess = -1
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        ### generate training and testing dataset independently for each fold
        data_traintest_tr = cp.dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
        Xdata_beh_trainset, ylabels_beh_trainset, Xdata_beh_testset, ylabels_beh_testset=data_traintest_tr['Xdata_beh_trainset'], \
            data_traintest_tr['ylabels_beh_trainset'], data_traintest_tr['Xdata_beh_testset'], data_traintest_tr['ylabels_beh_testset']

        if (CONTROL==0):
            Xdata_train_correct_rep,ylabels_train_correct_rep,_,_,Xdata_train_correct_alt,ylabels_train_correct_alt,_,_,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,nselect, files, false_files,EACHSTATES=ntrainc, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS
            _,_,Xdata_train_error_rep,ylabels_train_error_rep,_,_,Xdata_train_error_alt,ylabels_train_error_alt,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,nselect,files, false_files,EACHSTATES=ntraine, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to  
        else:           
            Xdata_train_correct_rep,ylabels_train_correct_rep,Xdata_train_error_rep,ylabels_train_error_rep,Xdata_train_correct_alt,ylabels_train_correct_alt,Xdata_train_error_alt,ylabels_train_error_alt,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,nselect,files, false_files,EACHSTATES=ntrain, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS

        Xdata_test_correct_rep,ylabels_test_correct_rep,Xdata_test_error_rep,ylabels_test_error_rep,Xdata_test_correct_alt,ylabels_test_correct_alt,Xdata_test_error_alt,ylabels_test_error_alt,merge_trials_test=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_testset,ylabels_beh_testset,unique_states,COHAXISS,nselect,files, false_files,EACHSTATES=itest, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials_train
        ### --------- state 0 ----------------------- 
        ### repeating
        Xdata_trainc_rep, Xdata_testc_rep, Xdata_traine_rep, Xdata_teste_rep = [],[],[],[]
        ### alternating
        Xdata_trainc_alt, Xdata_testc_alt, Xdata_traine_alt, Xdata_teste_alt = [],[],[],[]
        for idxcoh, COHAXIS in enumerate(COHAXISS):
            ### repeating
            Xdata_trainc_rep,Xdata_testc_rep     = Xdata_train_correct_rep[COHAXIS,0],Xdata_test_correct_rep[COHAXIS,0]
            ylabels_trainc_rep,ylabels_testc_rep = ylabels_train_correct_rep[COHAXIS,0].T,(ylabels_test_correct_rep[COHAXIS,0].T)

            Xdata_traine_rep,Xdata_teste_rep     = Xdata_train_error_rep[COHAXIS,0],Xdata_test_error_rep[COHAXIS,0]
            ylabels_traine_rep,ylabels_teste_rep = (ylabels_train_error_rep[COHAXIS,0].T),(ylabels_test_error_rep[COHAXIS,0].T)   

            Xdata_trainc_rep,Xdata_testc_rep     = np.vstack((Xdata_trainc_rep,Xdata_train_correct_rep[COHAXIS,1])),np.vstack((Xdata_testc_rep,Xdata_test_correct_rep[COHAXIS,1]))
            ylabels_trainc_rep,ylabels_testc_rep = np.vstack((ylabels_trainc_rep,ylabels_train_correct_rep[COHAXIS,1].T)),np.vstack((ylabels_testc_rep,ylabels_test_correct_rep[COHAXIS,1].T))

            Xdata_traine_rep,Xdata_teste_rep     = np.vstack((Xdata_traine_rep,Xdata_train_error_rep[COHAXIS,1])),np.vstack((Xdata_teste_rep,Xdata_test_error_rep[COHAXIS,1]))
            ylabels_traine_rep,ylabels_teste_rep = np.vstack((ylabels_traine_rep,ylabels_train_error_rep[COHAXIS,1].T)),np.vstack((ylabels_teste_rep,ylabels_test_error_rep[COHAXIS,1].T))
            
            ### alternating 
            Xdata_trainc_alt,Xdata_testc_alt     = Xdata_train_correct_alt[COHAXIS,0],Xdata_test_correct_alt[COHAXIS,0]
            ylabels_trainc_alt,ylabels_testc_alt = ylabels_train_correct_alt[COHAXIS,0].T,(ylabels_test_correct_alt[COHAXIS,0].T)

            Xdata_traine_alt,Xdata_teste_alt     = Xdata_train_error_alt[COHAXIS,0],Xdata_test_error_alt[COHAXIS,0]
            ylabels_traine_alt,ylabels_teste_alt = (ylabels_train_error_alt[COHAXIS,0].T),(ylabels_test_error_alt[COHAXIS,0].T)   

            Xdata_trainc_alt,Xdata_testc_alt     = np.vstack((Xdata_trainc_alt,Xdata_train_correct_alt[COHAXIS,1])),np.vstack((Xdata_testc_alt,Xdata_test_correct_alt[COHAXIS,1]))
            ylabels_trainc_alt,ylabels_testc_alt = np.vstack((ylabels_trainc_alt,ylabels_train_correct_alt[COHAXIS,1].T)),np.vstack((ylabels_testc_alt,ylabels_test_correct_alt[COHAXIS,1].T))

            Xdata_traine_alt,Xdata_teste_alt     = np.vstack((Xdata_traine_alt,Xdata_train_error_alt[COHAXIS,1])),np.vstack((Xdata_teste_alt,Xdata_test_error_alt[COHAXIS,1]))
            ylabels_traine_alt,ylabels_teste_alt = np.vstack((ylabels_traine_alt,ylabels_train_error_alt[COHAXIS,1].T)),np.vstack((ylabels_teste_alt,ylabels_test_error_alt[COHAXIS,1].T))


            ###### labels have no other labels except Behaviours/Stimulus
            if(CONTROL==1): 
                if DOREVERSE: 
                    ylabels_traine_rep[:, :] = 1-ylabels_traine_rep[:, :]
                    ylabels_traine_alt[:, :] = 1-ylabels_traine_alt[:, :]
                
                Xdata_train_rep   = Xdata_traine_rep.copy()
                ylabels_train_rep = ylabels_traine_rep.copy() 

                Xdata_train_alt   = Xdata_traine_alt.copy()
                ylabels_train_alt = ylabels_traine_alt.copy() 
            elif(CONTROL==2): 
                Xdata_train_rep   = Xdata_trainc_rep.copy()
                ### already substract 2 in previous step
                ylabels_train_rep = ylabels_trainc_rep.copy() 

                Xdata_train_alt   = Xdata_trainc_alt.copy()
                ylabels_train_alt = ylabels_trainc_alt.copy() 
            elif(CONTROL==0):   
                if DOREVERSE:
                    ylabels_traine_rep[:, :] = 1-ylabels_traine_rep[:, :]
                    ylabels_traine_alt[:, :] = 1-ylabels_traine_alt[:, :]
                # already substract 2 in previous step
                Xdata_train_rep   = np.append(Xdata_trainc_rep, Xdata_traine_rep, axis=0)
                ylabels_train_rep = np.append(ylabels_trainc_rep, ylabels_traine_rep, axis=0)

                Xdata_train_alt   = np.append(Xdata_trainc_alt, Xdata_traine_alt, axis=0)
                ylabels_train_alt = np.append(ylabels_trainc_alt, ylabels_traine_alt, axis=0)
                
            if(PCA_n_components>0):
                Xdata_train_rep = mmodel.transform(Xdata_train_rep)
                Xdata_train_alt = mmodel.transform(Xdata_train_alt)


            ### --- percentage of right choices -----
            # print('bl 1979 .....behaviour shape: realigned:')
            ycchoice_rep = np.zeros(np.shape(ylabels_train_rep)[0])
            for iset in range(np.shape(ylabels_train_rep)[0]):
                cchoice_labels = Counter(ylabels_train_rep[iset,:])
                ycchoice_rep[iset] = (cchoice_labels.most_common(1)[0][0])

            ycchoice_alt = np.zeros(np.shape(ylabels_train_alt)[0])
            for iset in range(np.shape(ylabels_train_alt)[0]):
                cchoice_labels = Counter(ylabels_train_alt[iset,:])
                ycchoice_alt[iset] = (cchoice_labels.most_common(1)[0][0])
                            
            ### repeating decoder 
            lin_cc_rep.fit(Xdata_train_rep, ycchoice_rep)
            ### alternating decoder
            lin_cc_alt.fit(Xdata_train_alt, ycchoice_alt)


            if i==0 and idxcoh==0:
                intercepts = np.zeros((1, 3 + 1 + 1))
                intercepts[:, 3] = lin_cc_rep.intercept_[:]
                intercepts[:, 4] = lin_cc_alt.intercept_[:]

                coeffs = np.zeros((NN, 3 + 1 + 1))
                coeffs[:, 3]     = lin_cc_rep.coef_[:]
                coeffs[:, 4]     = lin_cc_alt.coef_[:]

            else:
                tintercepts = np.zeros((1, 3 + 1 + 1))
                tintercepts[:, 3] = lin_cc_rep.intercept_[:]
                tintercepts[:, 4] = lin_cc_alt.intercept_[:]
                intercepts        = np.append(intercepts, tintercepts, axis=1)

                tcoeffs           = np.zeros((NN, 3 + 1 + 1))
                tcoeffs[:, 3]     = lin_cc_rep.coef_[:]
                tcoeffs[:, 4]     = lin_cc_alt.coef_[:]
                coeffs = np.append(coeffs, tcoeffs, axis=1)

            #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>
            #### -------- AC testing trials 
            linw_cc_rep, linb_cc_rep = lin_cc_rep.coef_[:], lin_cc_rep.intercept_[:] #rep
            linw_cc_alt, linb_cc_alt = lin_cc_alt.coef_[:], lin_cc_alt.intercept_[:] #alt
            # evaluate evidence model
            evidences_c      = np.zeros((ntest, 3 + 2))
            evidences_c_supp = np.zeros((ntest, 3 + 2))

            evidences_c[:, 3] = np.squeeze(
                Xdata_testc_rep @ linw_cc_rep.reshape(-1, 1) + linb_cc_rep)
            evidences_c[:, 4] = np.squeeze(
                Xdata_testc_alt @ linw_cc_alt.reshape(-1, 1) + linb_cc_alt)

            evidences_c[:, 1] = np.squeeze(
                Xdata_testc_rep @ linw_cc_alt.reshape(-1, 1) + linb_cc_alt)
            evidences_c[:, 2] = np.squeeze(
                Xdata_testc_alt @ linw_cc_rep.reshape(-1, 1) + linb_cc_rep)


            ### --- percentage of right choices -----
            ### repeating
            ycchoice_test_rep = np.zeros(np.shape(ylabels_testc_rep)[0])
            for iset in range(np.shape(ylabels_testc_rep)[0]):
                cchoice_labels = Counter(ylabels_testc_rep[iset,:])
                ycchoice_test_rep[iset] = (cchoice_labels.most_common(1)[0][0])      
            ylabels_testc_rep[:,0] = ycchoice_test_rep[:]
            
            ypredict_choice_rep = np.zeros_like(ycchoice_test_rep)
            ypredict_choice_rep[np.where(evidences_c[:,3]-evidences_c[:,1]>0)[0]]=1 # evidence 3 ### all about repeating
            prediction_correct_rep = accuracy_score((ylabels_testc_rep[:,0]).flatten(),ypredict_choice_rep)

            stats_c_rep[i,idxcoh] = prediction_correct_rep
            Xtest_set_correct_rep[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_correct_rep[i,2* (N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_testc_rep[:, :].copy(), ylabels_testc_rep[:, :].copy()

            ### alternating
            ycchoice_test_alt = np.zeros(np.shape(ylabels_testc_alt)[0])
            for iset in range(np.shape(ylabels_testc_alt)[0]):
                cchoice_labels = Counter(ylabels_testc_alt[iset,:])
                ycchoice_test_alt[iset] = (cchoice_labels.most_common(1)[0][0])      
            ylabels_testc_alt[:,0] = ycchoice_test_alt[:]
            
            ypredict_choice_alt = np.zeros_like(ycchoice_test_alt)
            ypredict_choice_alt[np.where(evidences_c[:,4]-evidences_c[:,2]>0)[0]]=1 ### evidence 4 ### all about alternating
            prediction_correct_alt = accuracy_score((ylabels_testc_alt[:,0]).flatten(),ypredict_choice_alt)

            stats_c_alt[i,idxcoh] = prediction_correct_alt
            Xtest_set_correct_alt[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_correct_alt[i,2* (N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_testc_alt[:, :].copy(), ylabels_testc_alt[:, :].copy()
            
            ### same ---------
            yevi_set_correct[i,2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :] = evidences_c.copy() ### 3-being rep;4-being alt


            #### -------- AE testing trials --------------
            # evaluate evidence model
            evidences_e = np.zeros((ntest, 3 + 2))
            evidences_e_supp = np.zeros((ntest, 3 + 2))
            evidences_e[:, 3] = np.squeeze(
                Xdata_teste_rep @ linw_cc_rep.reshape(-1, 1) + linb_cc_rep)
            evidences_e[:, 4] = np.squeeze(
                Xdata_teste_alt @ linw_cc_alt.reshape(-1, 1) + linb_cc_alt)

            evidences_e[:, 1] = np.squeeze(
                Xdata_teste_rep @ linw_cc_alt.reshape(-1, 1) + linb_cc_alt)
            evidences_e[:, 2] = np.squeeze(
                Xdata_teste_alt @ linw_cc_rep.reshape(-1, 1) + linb_cc_rep)


            ### --- percentage of right choices -----
            ### repeating
            ycchoice_test_rep = np.zeros(np.shape(ylabels_teste_rep)[0])
            for iset in range(np.shape(ylabels_teste_rep)[0]):
                cchoice_labels = Counter(ylabels_teste_rep[iset,:])
                ycchoice_test_rep[iset] = (cchoice_labels.most_common(1)[0][0])
            ylabels_teste_rep[:,4] = ycchoice_test_rep[:]
            
            ypredict_choice_rep = np.zeros_like(ycchoice_test_rep)
            ypredict_choice_rep[np.where(evidences_e[:,3]-evidences_e[:,1]>0)[0]]=1
            
            prediction_error_rep = accuracy_score((ylabels_teste_rep[:,4]).flatten(),ypredict_choice_rep)
            stats_e_rep[i,idxcoh] = prediction_error_rep           
            Xtest_set_error_rep[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_error_rep[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_teste_rep[:, :].copy(), ylabels_teste_rep[:, :].copy()
            ### alternating
            ycchoice_test_alt = np.zeros(np.shape(ylabels_teste_alt)[0])
            for iset in range(np.shape(ylabels_teste_alt)[0]):
                cchoice_labels = Counter(ylabels_teste_alt[iset,:])
                ycchoice_test_alt[iset] = (cchoice_labels.most_common(1)[0][0])
            ylabels_teste_alt[:,4] = ycchoice_test_alt[:]
            
            ypredict_choice_alt = np.zeros_like(ycchoice_test_alt)
            ypredict_choice_alt[np.where(evidences_e[:,4]-evidences_e[:,2]>0)[0]]=1
            
            prediction_error_alt = accuracy_score((ylabels_teste_alt[:,4]).flatten(),ypredict_choice_alt)
            stats_e_alt[i,idxcoh] = prediction_error_alt           
            Xtest_set_error_alt[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_error_alt[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_teste_alt[:, :].copy(), ylabels_teste_alt[:, :].copy()

            yevi_set_error[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :] = evidences_e.copy()

    return stats_c_rep, stats_e_rep, stats_c_alt, stats_e_alt, coeffs, intercepts,\
        Xtest_set_correct_rep, ytest_set_correct_rep, Xtest_set_correct_alt, ytest_set_correct_alt, yevi_set_correct,\
        Xtest_set_error_rep, ytest_set_error_rep, Xtest_set_error_alt, ytest_set_error_alt, yevi_set_error, RECORDED_TRIALS_SET
