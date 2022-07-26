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

def bootstrap_linsvm_step_gaincontrol(data_tr,NN, unique_states,unique_cohs,files,false_files, pop_correct, pop_zero, pop_error, CONDITION_CTXT, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[], mmodel=[],PCA_n_components=0):

    ### ac/ae ratio 
    CRATIO =ACE_RATIO/(1+ACE_RATIO)# according to theratio#0.5#share the same #  
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
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        ### generate training and testing dataset independently for each fold
        data_traintest_tr = cp.dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
        Xdata_hist_trainset, Xdata_hist_testset = data_traintest_tr['Xdata_hist_trainset'],data_traintest_tr['Xdata_hist_testset']
        ylabels_hist_trainset,ylabels_hist_testset = data_traintest_tr['ylabels_hist_trainset'],data_traintest_tr['ylabels_hist_testset']
        ### training dataset 
        Xdata_train_correct,ylabels_train_correct,Xdata_train_error,ylabels_train_error,merge_trials_hist_train=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_trainset,ylabels_hist_trainset,unique_states,unique_cohs,files,false_files,ntrain,RECORD_TRIALS, RECORDED_TRIALS_SET[i]) #individual
        ### testing dataset
        Xdata_test_correct,ylabels_test_correct,Xdata_test_error,ylabels_test_error,merge_trials_hist_test=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_testset,ylabels_hist_testset,unique_states,unique_cohs,files,false_files,itest,RECORD_TRIALS, RECORDED_TRIALS_SET[i]) #individual

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

        ##### >>>>>>> Prev.CH Right cancelled  
        # Xdata_testc[itest:2*itest,pop_error]   = Xdata_teste[itest:2*itest,pop_error]*1
        # Xdata_testc[3*itest:4*itest,pop_error] = Xdata_teste[3*itest:4*itest,pop_error]*1
        # Xdata_testc[itest:2*itest,pop_correct]   = Xdata_teste[itest:2*itest,pop_correct]*1
        # Xdata_testc[3*itest:4*itest,pop_correct] = Xdata_teste[3*itest:4*itest,pop_correct]*1
        #### >>>>>>>> Prev.CH Left cancelled 
        # Xdata_testc[:itest,pop_error]          = Xdata_teste[:itest,pop_error]
        # Xdata_testc[2*itest:3*itest,pop_error] = Xdata_teste[2*itest:3*itest,pop_error] 
        # Xdata_testc[:itest,pop_correct]          = Xdata_teste[:itest,pop_correct]
        # Xdata_testc[2*itest:3*itest,pop_correct] = Xdata_teste[2*itest:3*itest,pop_correct] 
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
            # ytr_bias[iset]=ylabels_train[iset,2] ### congruent
        if(CONDITION_CTXT):
            lin_bias_r.fit(np.vstack((Xdata_train[:ntrain,:],Xdata_train[1*ntrain:2*ntrain,:])), np.hstack((ytr_bias[:ntrain],ytr_bias[ntrain*1:ntrain*2])))
            lin_bias_a.fit(np.vstack((Xdata_train[2*ntrain:3*ntrain,:],Xdata_train[3*ntrain:4*ntrain,:])), np.hstack((ytr_bias[ntrain*2:ntrain*3],ytr_bias[ntrain*3:ntrain*4])))
        else:

            lin_bias_l.fit(np.vstack((Xdata_train[:ntrain,:],Xdata_train[2*ntrain:3*ntrain,:])), np.hstack((ytr_bias[:ntrain],ytr_bias[ntrain*2:ntrain*3])))
            lin_bias_r.fit(np.vstack((Xdata_train[1*ntrain:2*ntrain,:],Xdata_train[3*ntrain:4*ntrain,:])), np.hstack((ytr_bias[ntrain*1:ntrain*2],ytr_bias[ntrain*3:ntrain*4])))
            # lin_bias.fit(Xdata_train,ytr_bias)


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
            intercepts = np.zeros((1, 3 + 1 + 1+1))
            intercepts[:, 0] = lin_pact.intercept_[:]
            intercepts[:, 1] = lin_ctxt.intercept_[:]
            intercepts[:, 2] =  lin_xor.intercept_[:]#lin_bias_.intercept_[:]#
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
            coeffs[:, 2] =  lin_xor.coef_[:]#lin_bias_.coef_[:]#
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
            tcoeffs[:, 2] =  lin_xor.coef_[:]#lin_bias_.coef_[:]#
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
        linw_xor, linb_xor   =lin_xor.coef_[:],  lin_xor.intercept_[:]
        if(CONDITION_CTXT):
            linw_bias_r, linb_bias_r = lin_bias_r.coef_[:], lin_bias_r.intercept_[:]
            linw_bias_a, linb_bias_a = lin_bias_a.coef_[:], lin_bias_a.intercept_[:]
        else:
            linw_bias_l, linb_bias_l = lin_bias_l.coef_[:], lin_bias_l.intercept_[:]
            linw_bias_r, linb_bias_r = lin_bias_r.coef_[:], lin_bias_r.intercept_[:]
        
        linw_cc, linb_cc     =  lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
        evidences_c = np.zeros((ntest, 3 + 2))
        evidences_c[:, 0] = np.squeeze(
            Xdata_testc[:,pop] @ linw_pact.reshape(-1, 1)[pop] + linb_pact)
        evidences_c[:, 1] = np.squeeze(
            Xdata_testc[:,pop] @ linw_ctxt.reshape(-1, 1)[pop] + linb_ctxt)
        evidences_c[:, 2] = np.squeeze(
            Xdata_testc[:,pop] @ linw_xor.reshape(-1, 1)[pop] + linb_xor)

        #### gain control 
        if(CONDITION_CTXT):
            evidences_c[:itest, 3] = np.squeeze(
                    Xdata_testc[:itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c[itest:2*itest, 3] = np.squeeze(
                    Xdata_testc[itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_c[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
        else:
            evidences_c[:itest, 3] = np.squeeze(
                    Xdata_testc[:itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_c[itest:2*itest, 3] = np.squeeze(
                    Xdata_testc[itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_c[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_testc[2*itest:3*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_c[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_testc[3*itest:4*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)

        evidences_c[:, 4] = np.squeeze(
            Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)

        # evaluate model
        predictions_c = np.zeros((np.shape(evidences_c)[0], 3 + 1 + 1))
        predictions_c[:, 0] = lin_pact.predict(
            Xdata_testc)  # model.predict(X_test)
        predictions_c[:, 1] = lin_ctxt.predict(Xdata_testc)
        predictions_c[:, 2] = lin_xor.predict(Xdata_testc)

        if(CONDITION_CTXT):
            predictions_c[:itest, 3] = lin_bias_r.predict(Xdata_testc[:itest,:])
            predictions_c[itest:2*itest, 3] = lin_bias_r.predict(Xdata_testc[itest:2*itest,:])
            predictions_c[2*itest:3*itest, 3] = lin_bias_a.predict(Xdata_testc[2*itest:3*itest,:])
            predictions_c[3*itest:4*itest, 3] = lin_bias_a.predict(Xdata_testc[3*itest:4*itest,:])
        else:
            predictions_c[:itest, 3] = lin_bias_l.predict(Xdata_testc[:itest,:])
            predictions_c[itest:2*itest, 3] = lin_bias_r.predict(Xdata_testc[itest:2*itest,:])
            predictions_c[2*itest:3*itest, 3] = lin_bias_l.predict(Xdata_testc[2*itest:3*itest,:])
            predictions_c[3*itest:4*itest, 3] = lin_bias_r.predict(Xdata_testc[3*itest:4*itest,:])
        
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
            yevi_set_correct[i, :, :] = evidences_c.copy()
        else:
            yevi_set_correct[i, :, :] = evidences_c.copy()

        score_c = np.zeros((3 + 1 + 1, 1))
        for j in range(np.shape(score_c)[0]):
            score_c[j, 0] = accuracy_score(
                ylabels_testc[:, j]-2, predictions_c[:, j])

        #### -------- AE testing trials --------------
        # evaluate evidence model
        evidences_e = np.zeros((ntest, 3 + 2))
        evidences_e[:, 0] = np.squeeze(
            Xdata_teste[:,pop] @ linw_pact.reshape(-1, 1)[pop] + linb_pact)
        evidences_e[:, 1] = np.squeeze(
            Xdata_teste[:,pop] @ linw_ctxt.reshape(-1, 1)[pop] + linb_ctxt)
        evidences_e[:, 2] = np.squeeze(
            Xdata_teste[:,pop] @ linw_xor.reshape(-1, 1)[pop] + linb_xor)


        # evidences_e[:, 3] = np.squeeze(
        #         Xdata_teste[:,:] @ linw_bias.reshape(-1, 1) + linb_bias)
        #### gain control 
        if(CONDITION_CTXT):
            evidences_e[:itest, 3] = np.squeeze(
                    Xdata_teste[:itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e[itest:2*itest, 3] = np.squeeze(
                    Xdata_teste[itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
            evidences_e[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,pop] @ (linw_bias_a.reshape(-1, 1))[pop] + linb_bias_a)
        else:

            #### gain control 
            evidences_e[:itest, 3] = np.squeeze(
                    Xdata_teste[:itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_e[itest:2*itest, 3] = np.squeeze(
                    Xdata_teste[itest:2*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)
            evidences_e[2*itest:3*itest, 3] = np.squeeze(
                    Xdata_teste[2*itest:3*itest,pop] @ (linw_bias_l.reshape(-1, 1))[pop] + linb_bias_l)
            evidences_e[3*itest:4*itest, 3] = np.squeeze(
                    Xdata_teste[3*itest:4*itest,pop] @ (linw_bias_r.reshape(-1, 1))[pop] + linb_bias_r)

        evidences_e[:, 4] = np.squeeze(
            Xdata_teste @ linw_cc.reshape(-1, 1) + linb_cc)

        # evaluate model
        predictions_e = np.zeros((np.shape(evidences_e)[0], 3 + 1 + 1))
        predictions_e[:, 0] = lin_pact.predict(
            Xdata_teste)  # model.predict(X_test)
        predictions_e[:, 1] = lin_ctxt.predict(Xdata_teste)
        predictions_e[:, 2] = lin_xor.predict(Xdata_teste)
        
        if(CONDITION_CTXT):
            predictions_e[:itest, 3] = lin_bias_r.predict(Xdata_teste[:itest,:])
            predictions_e[itest:2*itest, 3] = lin_bias_r.predict(Xdata_teste[itest:2*itest,:])
            predictions_e[2*itest:3*itest, 3] = lin_bias_a.predict(Xdata_teste[2*itest:3*itest,:])
            predictions_e[3*itest:4*itest, 3] = lin_bias_a.predict(Xdata_teste[3*itest:4*itest,:])
        else:
            predictions_e[:itest, 3] = lin_bias_l.predict(Xdata_teste[:itest,:])
            predictions_e[itest:2*itest, 3] = lin_bias_r.predict(Xdata_teste[itest:2*itest,:])
            predictions_e[2*itest:3*itest, 3] = lin_bias_l.predict(Xdata_teste[2*itest:3*itest,:])
            predictions_e[3*itest:4*itest, 3] = lin_bias_r.predict(Xdata_teste[3*itest:4*itest,:])

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
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()

        score_e = np.zeros((3 + 1 + 1, 1))
        for j in range(np.shape(score_e)[0]):
            score_e[j, 0] = accuracy_score(
                ylabels_teste[:, j], predictions_e[:, j])

        # print(score)
        if i == 0:
            stats_c = [score_c[SVMAXIS,0]]
            stats_e = [score_e[SVMAXIS,0]]
            # print('score e:',score_c[SVMAXIS,0], ' e:',score_e[SVMAXIS,0])
        else:
            stats_c = np.append(stats_c, score_c[SVMAXIS,0])
            stats_e = np.append(stats_e, score_e[SVMAXIS,0])
            # print('score e:',score_c[SVMAXIS,0], ' e:',score_e[SVMAXIS,0])

    return mmodel,stats_c,stats_e,coeffs, intercepts,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

# def bootstrap_linsvm_step(Xdata_hist_trainset,Xdata_hist_testset,NN, ylabels_hist_trainset,ylabels_hist_testset, unique_states,unique_cohs,files,false_files, pop_correct, pop_zero, pop_error, USE_POP, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[], mmodel=[],PCA_n_components=0):

def bootstrap_linsvm_step(data_tr,NN, unique_states,unique_cohs,files,false_files, pop_correct, pop_zero, pop_error, USE_POP, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[], mmodel=[],PCA_n_components=0):

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

    stats = list()
    lin_pact = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_ctxt = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_xor  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                      shrinking=False, tol=1e-6)

    lin_bias   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
    lin_bias_  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
    lin_cc    = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                 shrinking=False, tol=1e-6)

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)
    
    stats_c,stats_e = list(), list()

    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        ### generate training and testing dataset independently for each fold
        data_traintest_tr = cp.dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
        Xdata_hist_trainset, Xdata_hist_testset = data_traintest_tr['Xdata_hist_trainset'],data_traintest_tr['Xdata_hist_testset']
        ylabels_hist_trainset,ylabels_hist_testset = data_traintest_tr['ylabels_hist_trainset'],data_traintest_tr['ylabels_hist_testset']
        ### training dataset 
        Xdata_train_correct,ylabels_train_correct,Xdata_train_error,ylabels_train_error,merge_trials_hist_train=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_trainset,ylabels_hist_trainset,unique_states,unique_cohs,files,false_files,ntrain,RECORD_TRIALS, RECORDED_TRIALS_SET[i]) #individual
        ### testing dataset
        Xdata_test_correct,ylabels_test_correct,Xdata_test_error,ylabels_test_error,merge_trials_hist_test=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_testset,ylabels_hist_testset,unique_states,unique_cohs,files,false_files,itest,RECORD_TRIALS, RECORDED_TRIALS_SET[i]) #individual

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

        ##### ************* sub populations ***** 
        if USE_POP==1:
            if CONTROL==1:
                pop_ez = pop_zero#np.union1d(pop_error,pop_zero)#
                Xdata_train, Xdata_testc, Xdata_teste = Xdata_train[:,pop_ez],Xdata_testc[:,pop_ez],Xdata_teste[:,pop_ez] 
            elif CONTROL==2:
                pop_cz = pop_zero#np.union1d(pop_correct,pop_zero)
                Xdata_train, Xdata_testc, Xdata_teste = Xdata_train[:,pop_cz],Xdata_testc[:,pop_cz],Xdata_teste[:,pop_cz]
            elif CONTROL==0:
                pop_ce = pop_zero#np.union1d(pop_correct,pop_zero)
                Xdata_train, Xdata_testc, Xdata_teste = Xdata_train[:,pop_ce],Xdata_testc[:,pop_ce],Xdata_teste[:,pop_ce]


        ### ------------------- Gain Control ---------------------------------------
        # pop = np.arange(NN)
        pop = np.union1d(pop_correct,pop_error)
        ##### >>>>>>> Prev.CH Right cancelled  
        # Xdata_testc[itest:2*itest,pop_error]     = Xdata_teste[itest:2*itest,pop_error]*1
        # Xdata_testc[3*itest:4*itest,pop_error]   = Xdata_teste[3*itest:4*itest,pop_error]*1
        # Xdata_testc[itest:2*itest,pop_correct]   = Xdata_teste[itest:2*itest,pop_correct]*1
        # Xdata_testc[3*itest:4*itest,pop_correct] = Xdata_teste[3*itest:4*itest,pop_correct]*1
        #### >>>>>>>> Prev.CH Left cancelled 
        # Xdata_testc[:itest,pop_error]            = Xdata_teste[:itest,pop_error]
        # Xdata_testc[2*itest:3*itest,pop_error]   = Xdata_teste[2*itest:3*itest,pop_error] 
        # Xdata_testc[:itest,pop_correct]          = Xdata_teste[:itest,pop_correct]
        # Xdata_testc[2*itest:3*itest,pop_correct] = Xdata_teste[2*itest:3*itest,pop_correct] 

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
            # ytr_bias[iset]=ylabels_train[iset,2] ### congruent
        lin_bias.fit(Xdata_train,ytr_bias)

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
            intercepts = np.zeros((1, 3 + 1 + 1))
            intercepts[:, 0] = lin_pact.intercept_[:]
            intercepts[:, 1] = lin_ctxt.intercept_[:]
            intercepts[:, 2] =  lin_xor.intercept_[:]#lin_bias_.intercept_[:]#
            intercepts[:, 3] = (lin_bias.intercept_[:])#-lin_bias_.intercept_[:])
            intercepts[:, 4] = lin_cc.intercept_[:]

            coeffs = np.zeros((NN, 3 + 1 + 1))
            coeffs[:, 0] = lin_pact.coef_[:]
            coeffs[:, 1] = lin_ctxt.coef_[:]
            coeffs[:, 2] =  lin_xor.coef_[:]#lin_bias_.coef_[:]#
            coeffs[:, 3] = (lin_bias.coef_[:])#-lin_bias_.coef_[:])
            coeffs[:, 4] = lin_cc.coef_[:]

        else:
            tintercepts = np.zeros((1, 3 + 1 + 1))
            tintercepts[:, 0] = lin_pact.intercept_[:]
            tintercepts[:, 1] = lin_ctxt.intercept_[:]
            tintercepts[:, 2] = lin_xor.intercept_[:]#lin_bias_.intercept_[:]#
            tintercepts[:, 3] = (lin_bias.intercept_[:])#-lin_bias_.intercept_[:])
            tintercepts[:, 4] = lin_cc.intercept_[:]
            intercepts = np.append(intercepts, tintercepts, axis=1)

            tcoeffs = np.zeros((NN, 3 + 1 + 1))
            tcoeffs[:, 0] = lin_pact.coef_[:]
            tcoeffs[:, 1] = lin_ctxt.coef_[:]
            tcoeffs[:, 2] =  lin_xor.coef_[:]#lin_bias_.coef_[:]#
            tcoeffs[:, 3] = (lin_bias.coef_[:])#-lin_bias_.coef_[:])
            tcoeffs[:, 4] = lin_cc.coef_[:]
            coeffs = np.append(coeffs, tcoeffs, axis=1)

        #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>

        if(PCA_n_components>0):
            Xdata_testc = mmodel.transform(Xdata_testc)
            Xdata_teste = mmodel.transform(Xdata_teste)
        #### -------- AC testing trials 
        linw_pact, linb_pact = lin_pact.coef_[:], lin_pact.intercept_[:]
        linw_ctxt, linb_ctxt = lin_ctxt.coef_[:], lin_ctxt.intercept_[:]
        linw_xor, linb_xor   = lin_xor.coef_[:],  lin_xor.intercept_[:]
        linw_bias, linb_bias = lin_bias.coef_[:], lin_bias.intercept_[:]
        linw_cc, linb_cc     =  lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
        evidences_c = np.zeros((ntest, 3 + 2))
        evidences_c[:, 0] = np.squeeze(
            Xdata_testc[:,pop] @ linw_pact.reshape(-1, 1)[pop] + linb_pact)
        evidences_c[:, 1] = np.squeeze(
            Xdata_testc[:,pop] @ linw_ctxt.reshape(-1, 1)[pop] + linb_ctxt)
        evidences_c[:, 2] = np.squeeze(
            Xdata_testc[:,pop] @ linw_xor.reshape(-1, 1)[pop] + linb_xor)
        evidences_c[:, 3] = np.squeeze(
                Xdata_testc[:,pop] @ linw_bias.reshape(-1, 1)[pop] + linb_bias)

        evidences_c[:, 4] = np.squeeze(
            Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)

        # evaluate model
        predictions_c = np.zeros((np.shape(evidences_c)[0], 3 + 1 + 1))
        predictions_c[:, 0] = lin_pact.predict(
            Xdata_testc)  # model.predict(X_test)
        predictions_c[:, 1] = lin_ctxt.predict(Xdata_testc)
        predictions_c[:, 2] = lin_xor.predict(Xdata_testc)
        predictions_c[:, 3] = lin_bias.predict(Xdata_testc)
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
            yevi_set_correct[i, :, :] = evidences_c.copy()
        else:
            yevi_set_correct[i, :, :] = evidences_c.copy()

        score_c = np.zeros((3 + 1 + 1, 1))
        for j in range(np.shape(score_c)[0]):
            score_c[j, 0] = accuracy_score(
                ylabels_testc[:, j]-2, predictions_c[:, j])

        #### -------- AE testing trials --------------
        # evaluate evidence model
        evidences_e = np.zeros((ntest, 3 + 2))
        evidences_e[:, 0] = np.squeeze(
            Xdata_teste[:,pop] @ linw_pact.reshape(-1, 1)[pop] + linb_pact)
        evidences_e[:, 1] = np.squeeze(
            Xdata_teste[:,pop] @ linw_ctxt.reshape(-1, 1)[pop] + linb_ctxt)
        evidences_e[:, 2] = np.squeeze(
            Xdata_teste[:,pop] @ linw_xor.reshape(-1, 1)[pop] + linb_xor)
        evidences_e[:, 3] = np.squeeze(
                Xdata_teste[:,pop] @ linw_bias.reshape(-1, 1)[pop] + linb_bias)
            
        evidences_e[:, 4] = np.squeeze(
            Xdata_teste[:,pop] @ linw_cc.reshape(-1, 1)[pop] + linb_cc)

        # evaluate model
        predictions_e = np.zeros((np.shape(evidences_e)[0], 3 + 1 + 1))
        predictions_e[:, 0] = lin_pact.predict(
            Xdata_teste)  # model.predict(X_test)
        predictions_e[:, 1] = lin_ctxt.predict(Xdata_teste)
        predictions_e[:, 2] = lin_xor.predict(Xdata_teste)
        predictions_e[:, 3] = lin_bias.predict(Xdata_teste)
        predictions_e[:, 4] = lin_cc.predict(Xdata_teste)


        ### modifying ylabels_testc[:,3], upcoming stimulus category 
        ytr_test_bias = np.zeros(np.shape(ylabels_teste)[0])
        for iset in range(np.shape(ylabels_teste)[0]):
            bias_labels=Counter(ylabels_teste[iset,3::6])
            ytr_test_bias[iset]=(bias_labels.most_common(1)[0][0])
        ylabels_teste[:,3] = ytr_test_bias[:]# ylabels_teste[:,2]#congruent # 

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
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()

        score_e = np.zeros((3 + 1 + 1, 1))
        for j in range(np.shape(score_e)[0]):
            score_e[j, 0] = accuracy_score(
                ylabels_teste[:, j], predictions_e[:, j])

        # print(score)
        if i == 0:
            stats_c = [score_c[SVMAXIS,0]]
            stats_e = [score_e[SVMAXIS,0]]
            # print('score e:',score_c[SVMAXIS,0], ' e:',score_e[SVMAXIS,0])
        else:
            stats_c = np.append(stats_c, score_c[SVMAXIS,0])
            stats_e = np.append(stats_e, score_e[SVMAXIS,0])
            # print('score e:',score_c[SVMAXIS,0], ' e:',score_e[SVMAXIS,0])

    return mmodel,stats_c,stats_e,coeffs, intercepts,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

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

    

def bootstrap_linsvm_step_fixationperiod_balanced(data_tr, NN, unique_states,unique_cohs,files,false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error, pop_correct,pop_error, USE_POP, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=5, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
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

    Xtest_set_correct, ytest_set_correct, =\
        np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
        np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

    Xtest_set_error, ytest_set_error, =\
        np.zeros((n_iterations, ntest*len(COHAXISS), NN)),\
        np.zeros((n_iterations, ntest*len(COHAXISS), nlabels))  

    yevi_set_correct = np.zeros((n_iterations, ntest*len(COHAXISS), 3+2))
    yevi_set_error   = np.zeros((n_iterations, ntest*len(COHAXISS), 3+2))

    lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                  shrinking=False, tol=1e-6)
 
    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)

    stats_c,stats_e = np.zeros((n_iterations,len(COHAXISS))),np.zeros((n_iterations,len(COHAXISS)))
    i_sucess = -1
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        ### generate training and testing dataset independently for each fold
        data_traintest_tr = cp.dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
        Xdata_beh_trainset, ylabels_beh_trainset, Xdata_beh_testset, ylabels_beh_testset=data_traintest_tr['Xdata_beh_trainset'], \
            data_traintest_tr['ylabels_beh_trainset'], data_traintest_tr['Xdata_beh_testset'], data_traintest_tr['ylabels_beh_testset']

        if (CONTROL==0):
            Xdata_train_correct,ylabels_train_correct,_,_,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,files, false_files,EACHSTATES=ntrainc, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS
            _,_,Xdata_train_error,ylabels_train_error,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,files, false_files,EACHSTATES=ntraine, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to  
        else:           
            Xdata_train_correct,ylabels_train_correct,Xdata_train_error,ylabels_train_error,merge_trials_train=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset,ylabels_beh_trainset,unique_states,COHAXISS,files, false_files,EACHSTATES=ntrain, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS

        Xdata_test_correct,ylabels_test_correct,Xdata_test_error,ylabels_test_error,merge_trials_test=gpt.merge_pseudo_beh_trials_individual(Xdata_beh_testset,ylabels_beh_testset,unique_states,COHAXISS,files, false_files,EACHSTATES=itest, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#### from unique_cohs to COHAXISS

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials_train
        ### --------- state 0 ----------------------- 
        Xdata_trainc, Xdata_testc, Xdata_traine, Xdata_teste = [],[],[],[]
        for idxcoh, COHAXIS in enumerate(COHAXISS):
            Xdata_trainc,Xdata_testc     = Xdata_train_correct[COHAXIS,0],Xdata_test_correct[COHAXIS,0]
            ylabels_trainc,ylabels_testc = ylabels_train_correct[COHAXIS,0].T,(ylabels_test_correct[COHAXIS,0].T)
            # print('~~~~~~000',ylabels_train_correct[COHAXIS,0].T)

            Xdata_traine,Xdata_teste     = Xdata_train_error[COHAXIS,0],Xdata_test_error[COHAXIS,0]
            ylabels_traine,ylabels_teste = (ylabels_train_error[COHAXIS,0].T),(ylabels_test_error[COHAXIS,0].T)   

            Xdata_trainc,Xdata_testc     = np.vstack((Xdata_trainc,Xdata_train_correct[COHAXIS,1])),np.vstack((Xdata_testc,Xdata_test_correct[COHAXIS,1]))
            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ylabels_train_correct[COHAXIS,1].T)),np.vstack((ylabels_testc,ylabels_test_correct[COHAXIS,1].T))

            Xdata_traine,Xdata_teste     = np.vstack((Xdata_traine,Xdata_train_error[COHAXIS,1])),np.vstack((Xdata_teste,Xdata_test_error[COHAXIS,1]))
            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ylabels_train_error[COHAXIS,1].T)),np.vstack((ylabels_teste,ylabels_test_error[COHAXIS,1].T)) 
            # print('~~~~~~111',ylabels_train_correct[COHAXIS,1].T)
            # print('test c:',ylabels_testc[::itest])
            # print('test e:',ylabels_teste[::itest])


            ###### labels have no other labels except Behaviours/Stimulus
            if(CONTROL==1): 
                if DOREVERSE: 
                    ylabels_traine[:, :] = 1-ylabels_traine[:, :]
                Xdata_train   = Xdata_traine.copy()
                ylabels_train = ylabels_traine.copy() 
            elif(CONTROL==2): 
                Xdata_train   = Xdata_trainc.copy()
                ### already substract 2 in previous step
                ylabels_train = ylabels_trainc.copy() 
            elif(CONTROL==0):   
                if DOREVERSE:
                    ylabels_traine[:, :] = 1-ylabels_traine[:, :]
                # already substract 2 in previous step
                Xdata_train   = np.append(Xdata_trainc, Xdata_traine, axis=0)   
                ylabels_train = np.append(ylabels_trainc, ylabels_traine, axis=0)
                
            if(PCA_n_components>0):
                Xdata_train = mmodel.transform(Xdata_train)

            # print('~~~~~~size of the test dataset:', np.shape(Xdata_train))

            #### whether use populations or not 
            if(USE_POP):
                if(CONTROL==1):
                    Xdata_train = Xdata_train[:,pop_error] 
                    Xdata_testc = Xdata_testc[:,pop_error]
                    Xdata_teste = Xdata_teste[:,pop_error]
                elif(CONTROL==2):
                    Xdata_train = Xdata_train[:,pop_correct]
                    Xdata_testc = Xdata_testc[:,pop_correct]
                    Xdata_teste = Xdata_teste[:,pop_correct]

            ### --- percentage of right choices -----
            ycchoice = np.zeros(np.shape(ylabels_train)[0])
            for iset in range(np.shape(ylabels_train)[0]):
                cchoice_labels = Counter(ylabels_train[iset,:])
                ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
                            
            
            lin_cc.fit(Xdata_train, ycchoice)


            if i==0 and idxcoh==0:
                intercepts = np.zeros((1, 3 + 1 + 1))
                intercepts[:, 4] = lin_cc.intercept_[:]

                coeffs = np.zeros((NN, 3 + 1 + 1))
                coeffs[:, 4]     = lin_cc.coef_[:]

            else:
                tintercepts = np.zeros((1, 3 + 1 + 1))
                tintercepts[:, 4] = lin_cc.intercept_[:]
                intercepts        = np.append(intercepts, tintercepts, axis=1)

                tcoeffs           = np.zeros((NN, 3 + 1 + 1))
                tcoeffs[:, 4]     = lin_cc.coef_[:]
                coeffs = np.append(coeffs, tcoeffs, axis=1)

            #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>
            if(PCA_n_components>0):
                Xdata_testc=mmodel.transform(Xdata_testc)
                Xdata_teste=mmodel.transform(Xdata_teste)
            #### -------- AC testing trials 
            linw_cc, linb_cc     = lin_cc.coef_[:], lin_cc.intercept_[:]
            # evaluate evidence model
            evidences_c = np.zeros((ntest, 3 + 2))
            # print('shape...',np.shape(Xdata_testc),np.shape(evidences_c))
            evidences_c[:, 4] = np.squeeze(
                Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)


            ### --- percentage of right choices -----
            ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
            for iset in range(np.shape(ylabels_testc)[0]):
                cchoice_labels = Counter(ylabels_testc[iset,:])
                ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
                # ycchoice_test[iset] = cchoice_labels[1]/np.shape(ylabels_testc)[1]
                
            ylabels_testc[:,0] = ycchoice_test[:]
            # print('~~~~~~test c:',np.mean(ylabels_testc[:,4]))
            
            # ypredict_choice = lin_cc.predict(Xdata_testc)
            ypredict_choice = np.zeros_like(ycchoice_test)
            ypredict_choice[np.where(evidences_c[:,4]>0)[0]]=1
            # print('~~~~shape:',np.shape(ylabels_testc[:,4].flatten()),np.shape(np.squeeze(ylabels_testc[:,4])),np.shape(ypredict_choice))
            prediction_correct = accuracy_score((ylabels_testc[:,0]).flatten(),ypredict_choice)
            stats_c[i,idxcoh] = prediction_correct

            Xtest_set_correct[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_correct[i,2* (N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_testc[:, :].copy(), ylabels_testc[:, :].copy()
            
            yevi_set_correct[i,2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :] = evidences_c.copy()


            #### -------- AE testing trials --------------
            # evaluate evidence model
            evidences_e = np.zeros((ntest, 3 + 2))
            evidences_e[:, 4] = np.squeeze(
                Xdata_teste @ linw_cc.reshape(-1, 1) + linb_cc)


            ### --- percentage of right choices -----
            ycchoice_test = np.zeros(np.shape(ylabels_teste)[0])
            for iset in range(np.shape(ylabels_teste)[0]):
                cchoice_labels = Counter(ylabels_teste[iset,:])
                ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
                # ycchoice_test[iset] = cchoice_labels[1]/np.shape(ylabels_teste)[1]
            ylabels_teste[:,4] = ycchoice_test[:]
            # print('~~~~~~test e:',np.mean(ylabels_teste[:,4]))
            
            # ypredict_choice = lin_cc.predict(Xdata_teste)
            ypredict_choice = np.zeros_like(ycchoice_test)
            ypredict_choice[np.where(evidences_e[:,4]>0)[0]]=1
            
            prediction_error = accuracy_score((ylabels_teste[:,4]).flatten(),ypredict_choice)
            stats_e[i,idxcoh] = prediction_error
            # print('~~~~~~~~ performance:', prediction_correct,prediction_error)
            

            Xtest_set_error[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :], ytest_set_error[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

            yevi_set_error[i, 2*(N_pseudo_dec-ntrain)*idxcoh:2*(N_pseudo_dec-ntrain)*(idxcoh+1), :] = evidences_e.copy()
    fig,ax = plt.subplots(2,1,figsize=(4,6),tight_layout=True,sharex=True,sharey=True)
    ax[0].hist(stats_c[:,:].flatten(),facecolor='red',alpha=0.25)
    ax[1].hist(stats_e[:,:].flatten(),facecolor='blue',alpha=0.25)
    ax[0].set_xlim([0,1.0])
    return stats_c, stats_e, coeffs, intercepts,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET
