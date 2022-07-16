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

PRINT_PER = 1000



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

def bootstrap_linsvm_step(Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[], mmodel=[],PCA_n_components=0):

    ### ac/ae ratio 
    CRATIO = ACE_RATIO/(1+ACE_RATIO)
    ERATIO = 1-CRATIO
    # NN      = np.shape(Xdata_hist_set[unique_states[0],'correct'])[1] 
    nlabels = 6*(len(files)-len(false_files))
    ntrain  = int(train_percent*N_pseudo_dec) # *4
    ntest   = (N_pseudo_dec-ntrain)*4 # state
    itest   = N_pseudo_dec-ntrain

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
    # if (STIM_PERIOD==1):
    # 	lin_bias = LinearRegression()
    # elif(STIM_PERIOD==0): 
    lin_bias = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',shrinking=False, tol=1e-6)
    lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                 shrinking=False, tol=1e-6)

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)
    
    stats_c,stats_e = list(), list()

    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error,merge_trials_hist=gpt.merge_pseudo_hist_trials_individual(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,2*N_pseudo_dec,RECORD_TRIALS, RECORDED_TRIALS_SET[i]) #individual

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials_hist


        #### --------- state 0 -----------------------
        if CONTROL ==0: 
            Xdata_trainc,Xdata_testc=Xmerge_hist_trials_correct[4][:ntrainc,:],Xmerge_hist_trials_correct[4][ntrainc:ntrainc+itest,:] 
            Xdata_traine,Xdata_teste=Xmerge_hist_trials_error[0][:ntraine,:],Xmerge_hist_trials_error[0][ntraine:ntraine+itest,:] 
            ylabels_trainc,ylabels_testc = ymerge_hist_labels_correct[4][:ntrainc,:],ymerge_hist_labels_correct[4][ntrainc:ntrainc+itest,:] 
            ylabels_traine,ylabels_teste = ymerge_hist_labels_error[0][:ntraine,:],ymerge_hist_labels_error[0][ntraine:ntraine+itest,:] 
            for state in range(1,4): 
                Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_hist_trials_correct[state+4][:ntrainc ,:])),np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrainc:ntrainc+itest,:]))  
                ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ymerge_hist_labels_correct[state+4][:ntrainc,:])),np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrainc:ntrainc+itest,:]))  
                Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_hist_trials_error[state][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntraine:ntraine+itest,:])) 
                ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ymerge_hist_labels_error[state][:ntraine,:])),np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntraine:ntraine+itest,:])) 
        elif CONTROL ==1: 
            Xdata_testc   = Xmerge_hist_trials_correct[4][ntrainc:ntrainc+itest,:] 
            ylabels_testc = ymerge_hist_labels_correct[4][ntrainc:ntrainc+itest,:] 
            Xdata_traine,Xdata_teste=Xmerge_hist_trials_error[0][:ntraine+ntrainc,:],Xmerge_hist_trials_error[0][ntraine+ntrainc:ntrainc+ntraine+itest,:] 
            ylabels_traine,ylabels_teste = ymerge_hist_labels_error[0][:ntraine+ntrainc,:],ymerge_hist_labels_error[0][ntraine+ntrainc:ntraine+ntrainc+itest,:] 
            for state in range(1,4):
	            Xdata_testc = np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrainc:ntrainc+itest,:]))
	            ylabels_testc = np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrainc:ntrainc+itest,:]))
	         
	            Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_hist_trials_error[state][:ntrainc+ntraine,:])),np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntraine+ntrainc:ntraine+ntrainc+itest,:]))
	            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ymerge_hist_labels_error[state][:ntraine+ntrainc,:])),np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntraine+ntrainc:ntraine+ntrainc+itest,:]))  
            unique_choice_pseudo = np.unique(ylabels_traine[:,4::6])  
            # print('~~~~~ ~~~~ unique choice here:',unique_choice_pseudo, np.shape(ylabels_traine))
        elif CONTROL ==2:
	        Xdata_trainc,Xdata_testc=Xmerge_hist_trials_correct[4][:ntrainc+ntraine,:],Xmerge_hist_trials_correct[4][ntrainc+ntraine:ntrainc+ntraine+itest,:]
	        ylabels_trainc,ylabels_testc = ymerge_hist_labels_correct[4][:ntrainc+ntraine,:],ymerge_hist_labels_correct[4][ntrainc+ntraine:ntrainc+ntraine+itest,:]

	        Xdata_teste=Xmerge_hist_trials_error[0][ntraine:ntraine+itest,:]
	        ylabels_teste = ymerge_hist_labels_error[0][ntraine:ntraine+itest,:]
	        for state in range(1,4):
	            Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_hist_trials_correct[state+4][:ntrainc+ntraine ,:])),np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrainc+ntraine:ntraine+ntrainc+itest,:]))
	            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ymerge_hist_labels_correct[state+4][:ntrainc+ntraine,:])),np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrainc+ntraine:ntrainc+ntraine+itest,:]))
	         
	            Xdata_teste   = np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntraine:ntraine+itest,:]))
	            ylabels_teste = np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntraine:ntraine+itest,:]))
        # recording support vectors
        Xsup_vec_act  = {}  # np.zeros((n_iterations,1,NN))
        Xsup_vec_ctxt = {}  # np.zeros((n_iterations,1,NN))
        Xsup_vec_xor  = {}
        Xsup_vec_bias = {}  # np.zeros((n_iterations,1,NN))
        Xsup_vec_cc   = {}



        if(CONTROL==1):
            if DOREVERSE:
                ylabels_traine[:, 3] = 1-ylabels_traine[:, 3]
            Xdata_train   = Xdata_traine.copy()
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
            Xdata_train   = np.append(Xdata_trainc, Xdata_traine, axis=0)
            ylabels_train = np.append(ylabels_trainc, ylabels_traine, axis=0)
        if(PCA_n_components>0):
            Xdata_train = mmodel.transform(Xdata_train)
            # print('~~~~~~~~~ shape of data:', np.shape(Xdata_train))
        # fit model
        # model.fit(X_train,y_train) i.e model.fit(train set, train label as it
        # is a classifier)
        lin_pact.fit(Xdata_train, np.squeeze(ylabels_train[:, 0]))
        Xsup_vec_act[i]  = lin_pact.support_vectors_

        lin_ctxt.fit(Xdata_train, np.squeeze(ylabels_train[:, 1]))
        Xsup_vec_ctxt[i] = lin_ctxt.support_vectors_

        lin_xor.fit(Xdata_train, np.squeeze(ylabels_train[:, 2]))
        Xsup_vec_xor[i]  = lin_xor.support_vectors_
        
        ### ---- most common tr. bias -----------


        # if(STIM_PERIOD==1): 
        #     ycoh_stim = np.zeros(np.shape(ylabels_train)[0]) 
        #     for iset in range(np.shape(ylabels_train)[0]): 
        #         cohs_expectation = np.mean(ylabels_train[iset,5::6]) 
        #         # print('~~~~~~~',ylabels_train[iset,5::6])
        #         ycoh_stim[iset] = cohs_expectation  
        #         # if(np.mean(ylabels_train[iset,3::6])>0.5): 
        #         #     print('~~~~~~~ycoh_stim',iset,' ', ycoh_stim[iset]) 
        #     ycoh_stim = sstats.zscore(ycoh_stim)
        #     lin_bias.fit(Xdata_train,ycoh_stim) 
        # elif(STIM_PERIOD==0): 
        ytr_bias = np.zeros(np.shape(ylabels_train)[0]) 
        for iset in range(np.shape(ylabels_train)[0]): 
            bias_labels=Counter(ylabels_train[iset,3::6]) 
            ytr_bias[iset]=(bias_labels.most_common(1)[0][0]) 
            # ytr_bias[iset]=ylabels_train[iset,2] ### congruent
        lin_bias.fit(Xdata_train, ytr_bias)
        Xsup_vec_bias[i]  = lin_bias.support_vectors_
        # print("~~~~~~~bias:",np.shape(lin_bias.coef_),np.shape(lin_bias.intercept_))

        # lin_cc.fit(Xdata_train, np.squeeze(ylabels_train[:, 4]))
        # Xsup_vec_cc[i]    = lin_cc.support_vectors_

        ### --- percentage of right choices -----
        ycchoice = np.zeros(np.shape(ylabels_train)[0])
        for iset in range(np.shape(ylabels_train)[0]):
            cchoice_labels = Counter(ylabels_train[iset,4::6])
            ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
        lin_cc.fit(Xdata_train, ycchoice)
        Xsup_vec_cc[i]    = lin_cc.support_vectors_

        if i == 0:
            intercepts = np.zeros((1, 3 + 1 + 1))
            intercepts[:, 0] = lin_pact.intercept_[:]
            intercepts[:, 1] = lin_ctxt.intercept_[:]
            intercepts[:, 2] = lin_xor.intercept_[:]
            intercepts[:, 3] = lin_bias.intercept_[:]
            intercepts[:, 4] = lin_cc.intercept_[:]

            coeffs = np.zeros((NN, 3 + 1 + 1))
            coeffs[:, 0] = lin_pact.coef_[:]
            coeffs[:, 1] = lin_ctxt.coef_[:]
            coeffs[:, 2] = lin_xor.coef_[:]
            coeffs[:, 3] = lin_bias.coef_[:]
            coeffs[:, 4] = lin_cc.coef_[:]

        else:
            tintercepts = np.zeros((1, 3 + 1 + 1))
            tintercepts[:, 0] = lin_pact.intercept_[:]
            tintercepts[:, 1] = lin_ctxt.intercept_[:]
            tintercepts[:, 2] = lin_xor.intercept_[:]
            tintercepts[:, 3] = lin_bias.intercept_[:]
            tintercepts[:, 4] = lin_cc.intercept_[:]
            intercepts = np.append(intercepts, tintercepts, axis=1)

            tcoeffs = np.zeros((NN, 3 + 1 + 1))
            tcoeffs[:, 0] = lin_pact.coef_[:]
            tcoeffs[:, 1] = lin_ctxt.coef_[:]
            tcoeffs[:, 2] = lin_xor.coef_[:]
            tcoeffs[:, 3] = lin_bias.coef_[:]
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
        linw_cc, linb_cc     = lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
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
        # print('~~~~~~ correct match:',ytr_test_bias[:5],ylabels_testc[:5,2])

        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
        for iset in range(np.shape(ylabels_testc)[0]):
            cchoice_labels = Counter(ylabels_testc[iset,4::6])
            ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
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
            Xdata_teste @ linw_pact.reshape(-1, 1) + linb_pact)
        evidences_e[:, 1] = np.squeeze(
            Xdata_teste @ linw_ctxt.reshape(-1, 1) + linb_ctxt)
        evidences_e[:, 2] = np.squeeze(
            Xdata_teste @ linw_xor.reshape(-1, 1) + linb_xor)
        evidences_e[:, 3] = np.squeeze(
            Xdata_teste @ linw_bias.reshape(-1, 1) + linb_bias)
        evidences_e[:, 4] = np.squeeze(
            Xdata_teste @ linw_cc.reshape(-1, 1) + linb_cc)

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
        # print('~~~~~~ error match:',ytr_test_bias[:5],ylabels_teste[:5,2])

        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_teste)[0])
        for iset in range(np.shape(ylabels_teste)[0]):
            cchoice_labels = Counter(ylabels_teste[iset,4::6])
            ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
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
            # print('score e:',score_c, ' e:',score_e)
            stats_c = score_c
            stats_e = score_e
        else:
            stats_c = np.append(stats_c, score_c, axis=1)
            stats_e = np.append(stats_e, score_e, axis=1)
            # print('score e:',score_c, ' e:',score_e)

    return mmodel,stats_c,stats_e,coeffs, intercepts,\
        Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias, Xsup_vec_cc,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

def bootstrap_linsvm_proj_step(coeffs_pool, intercepts_pool, Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, n_iterations=10, N_pseudo_dec=25, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
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

        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error,merge_trials_hist=gpt.merge_pseudo_hist_trials(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,N_pseudo_dec,RECORD_TRIALS, RECORDED_TRIALS_SET[i])

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
            stats_c = score_c
            stats_e = score_e
        else:
            stats_c = np.append(stats_c, score_c, axis=1)
            stats_e = np.append(stats_e, score_e, axis=1)
    return mmodel,stats_c, stats_e, coeffs_pool, intercepts_pool, \
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

def shuffle_linsvm_proj_step(coeffs_pool, intercepts_pool, Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, n_iterations=10, N_pseudo_dec=25, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
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



    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error,merge_trials_hist=gpt.shuffle_pseudo_hist_trials(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,N_pseudo_dec,RECORD_TRIALS, RECORDED_TRIALS_SET[i])

        Xdata_trainc,Xdata_testc=Xmerge_hist_trials_correct[4][:ntrain,:],Xmerge_hist_trials_correct[4][ntrain:,:]
        ylabels_trainc,ylabels_testc = ymerge_hist_labels_correct[4][:ntrain,:],ymerge_hist_labels_correct[4][ntrain:,:]

        Xdata_traine,Xdata_teste=Xmerge_hist_trials_error[0][:ntrain,:],Xmerge_hist_trials_error[0][ntrain:,:]
        ylabels_traine,ylabels_teste = ymerge_hist_labels_error[0][:ntrain,:],ymerge_hist_labels_error[0][ntrain:,:]
        for state in range(1,4):
            Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_hist_trials_correct[state+4][:ntrain ,:])),np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrain :,:]))
            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ymerge_hist_labels_correct[state+4][:ntrain,:])),np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrain:,:]))

            Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_hist_trials_error[state][:ntrain,:])),np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntrain:,:]))
            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ymerge_hist_labels_error[state][:ntrain,:])),np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntrain :,:]))

        # @YX 0910 -- weights
        linw_pact, linb_pact = coeffs_pool[:, i*5+0], intercepts_pool[0, 5*i+0]
        linw_ctxt, linb_ctxt = coeffs_pool[:, i*5+1], intercepts_pool[0, 5*i+1]
        linw_xor, linb_xor   = coeffs_pool[:, i*5+2], intercepts_pool[0, 5*i+2]
        linw_bias, linb_bias = coeffs_pool[:, i*5+3], intercepts_pool[0, 5*i+3]
        linw_cc, linb_cc     = coeffs_pool[:, i * 5+4], intercepts_pool[0, 5*i+4]

        if(PCA_n_components>0):
            Xdata_testc = mmodel.transform(Xdata_testc)
            Xdata_teste = mmodel.transform(Xdata_teste)
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

        Xtest_set_error[i, :, :], ytest_set_error[i, :, :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

        if i == 0:
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()
    return mmodel,coeffs_pool, intercepts_pool, \
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

def bootstrap_linreg_step_stimperiod(Xdata_set,NN, ylabels_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, CONTROL = 0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[]):
    ### ac/ae ratio 
    CRATIO = ACE_RATIO/(1+ACE_RATIO)
    ERATIO = 1-CRATIO
    cohs_true =  [-1.00000000e+00, -7.55200000e-01,  -4.41500000e-01,  4.41500000e-01,  7.55200000e-01, 1.00000000e+0]#[-1., -0.4816, -0.2282 ,  0.2282,  0.4816,  1.] #[-1., -0.4816, -0.2282,  0. ,  0.2282,  0.4816,  1.] 
    unique_states = np.arange(0, 8, 1)   
    nlabels   = 6*(len(files)-len(false_files))
    ntrain    = int(train_percent*N_pseudo_dec) # *4
    ntest     = (N_pseudo_dec-ntrain)*(len(cohs_true)) # state
    itest     = N_pseudo_dec-ntrain

    Xtest_set, ytest_set, =\
        np.zeros((n_iterations, ntest, NN)),\
        np.zeros((n_iterations, ntest, nlabels))  

    yevi_set = np.zeros((n_iterations, ntest, 2))
    lin_stim = LinearRegression()
    lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                 shrinking=False, tol=1e-6)

    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        Xmerge_trials, ymerge_labels,merge_trials =\
            gpt.merge_pseudo_beh_trials_stimperiod(Xdata_set, ylabels_set, unique_states,
                                        unique_cohs, files, false_files,
                                        N_pseudo_dec, RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i])
        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials

        #### --------- state 0 -----------------------
        Xdata_train,Xdata_test=Xmerge_trials[cohs_true[0]][:ntrain,:],Xmerge_trials[cohs_true[0]][ntrain:,:]
        ylabels_train,ylabels_test = ymerge_labels[cohs_true[0]][:ntrain,:],ymerge_labels[cohs_true[0]][ntrain:,:]

        for coh_t in cohs_true[1:]:
            Xdata_train,Xdata_test = np.vstack((Xdata_train,Xmerge_trials[coh_t][:ntrain ,:])),np.vstack((Xdata_test,Xmerge_trials[coh_t][ntrain:,:]))
            ylabels_train,ylabels_test = np.vstack((ylabels_train,ymerge_labels[coh_t][:ntrain,:])),np.vstack((ylabels_test,ymerge_labels[coh_t][ntrain:,:]))
        # prepare train & test sets
        if DOREVERSE:
            ylabels_train[:, 3] = 1-ylabels_train[:, 3]

        lin_stim.fit(Xdata_train,ylabels_train[:,5])
        print('~~~~~~~~Linear Reg training score:',lin_stim.score(Xdata_train,ylabels_train[:,5]))
        y_predicted = lin_stim.predict(Xdata_test)
        rmse = mean_squared_error(y_predicted,ylabels_test[:,5])
        r2   = r2_score(y_predicted,ylabels_test[:,5])        
        print('~~~~~~~~Linear Reg testing score:',lin_stim.score(Xdata_test,ylabels_test[:,5]), rmse, r2)
        
        ### --- percentage of right choices -----
        ycchoice = np.zeros(np.shape(ylabels_train)[0])
        for iset in range(np.shape(ylabels_train)[0]):
            cchoice_labels = Counter(ylabels_train[iset,4::6])
            ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
        lin_cc.fit(Xdata_train, ycchoice)
        
        
        

        if i == 0:
            intercepts = np.zeros((1, 2))
            intercepts[:, 0] = lin_stim.intercept_
            intercepts[:, 1] = lin_cc.intercept_[:]

            coeffs = np.zeros((NN,  2))
            coeffs[:, 0] = lin_stim.coef_[:]
            coeffs[:, 1] = lin_cc.coef_[:]
        else:
            tintercepts = np.zeros((1, 2))
            tintercepts[:, 0] = lin_stim.intercept_
            tintercepts[:, 1] = lin_cc.intercept_[:]
            intercepts = np.append(intercepts, tintercepts, axis=1)

            tcoeffs = np.zeros((NN, 2))
            tcoeffs[:, 0] = lin_stim.coef_[:]
            tcoeffs[:, 1] = lin_cc.coef_[:]
            coeffs = np.append(coeffs, tcoeffs, axis=1)

        #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>
        #### -------- AC testing trials 
        linw_stim, linb_stim = lin_stim.coef_[:], lin_stim.intercept_
        linw_cc, linb_cc     = lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
        evidences_ = np.zeros((ntest, 2))
        evidences_[:, 0] = lin_stim.predict(Xdata_test)#np.squeeze(
            # Xdata_test @ linw_stim.reshape(-1, 1) + linb_stim)
        evidences_[:, 1] = np.squeeze(
            Xdata_test @ linw_cc.reshape(-1, 1) + linb_cc)
        Xtest_set[i, :, :], ytest_set[i, :, :]=Xdata_test[:, :].copy(), ylabels_test[:, :].copy()
        
        if i == 0:
            yevi_set[i, :, :] = evidences_.copy()
        else:
            yevi_set[i, :, :] = evidences_.copy()

    return coeffs, intercepts,\
        Xtest_set, ytest_set, yevi_set,\
        RECORDED_TRIALS_SET
        
def bootstrap_linsvm_step_fixationperiod(Xdata_set,NN, ylabels_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=5, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
    ### ac/ae ratio 
    CRATIO = 0.5#ACE_RATIO/(1+ACE_RATIO)
    ERATIO = 1-CRATIO
    #  
    COHAXISS = [0]#[-1]#[1]
    nlabels  = (len(files)-len(false_files))
    ntrain   = int(train_percent*N_pseudo_dec) # *4
    ntest    = (N_pseudo_dec-ntrain)*4*len(COHAXISS) # COHS*state
    itest    = N_pseudo_dec-ntrain
    
    ### ONLY USE COHERENCE = 0
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

    lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                  shrinking=False, tol=1e-6)
    # lin_cc   = LinearRegression()

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)
    
    stats_c,stats_e = np.zeros(n_iterations),np.zeros(n_iterations)
    i_sucess = -1
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # print('...... iteration index:', i,'........')
        ### >>>>>> generate training and testing dataset for decoder and testing(onestep)
        # N_pseudo_dec, N_pseudo_beh = 100,25
        ##### 11 July!!! ~~~~~~~~~~~~~~~~~~~~
        # unique_cohs = [0,1]
        Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error,merge_trials=gpt.merge_pseudo_beh_trials_individual(Xdata_set, ylabels_set, unique_states,unique_cohs, files, false_files, [],N_pseudo_dec*2, RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)#individual
        
        # ###### [:, idx]!!!!!!
        # ymerge_labels_correct = ymerge_labels_correct.T 
        # ymerge_labels_error   = ymerge_labels_error.T

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials


        # #### --------- state 0 -----------------------
        # Xdata_trainc,Xdata_testc=Xmerge_trials_correct[4,COHAXIS][:ntrainc,:],Xmerge_trials_correct[4,COHAXIS][ntrainc:ntrainc+itest,:]
        # ylabels_trainc,ylabels_testc = (ymerge_labels_correct[4,COHAXIS].T)[:ntrainc,:],(ymerge_labels_correct[4,COHAXIS].T)[ntrainc:ntrainc+itest,:]
        # # print('~~~~~~~',(ymerge_labels_correct[4,COHAXIS].T)[ntrainc:ntrainc+itest,:5])

        # Xdata_traine,Xdata_teste=Xmerge_trials_error[0,COHAXIS][:ntraine,:],Xmerge_trials_error[0,COHAXIS][ntraine:ntraine+itest,:]
        # ylabels_traine,ylabels_teste = (ymerge_labels_error[0,COHAXIS].T)[:ntraine,:],(ymerge_labels_error[0,COHAXIS].T)[ntraine:ntraine+itest,:]
        # # print('~~~~~~~',(ymerge_labels_error[0,COHAXIS].T)[ntrainc:ntrainc+itest,:5])
        # for state in range(1,4):
        #     Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_trials_correct[state+4,COHAXIS][:ntrainc ,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[state+4,COHAXIS][ntrainc:ntrainc+itest,:]))
        #     ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[state+4,COHAXIS].T)[:ntrainc,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[state+4,COHAXIS].T)[ntrainc:ntrainc+itest,:]))
        #     # print(state+4,'~~~~~~~',(ymerge_labels_correct[state+4,COHAXIS].T)[ntrainc:ntrainc+itest,:5])
         
        #     Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_trials_error[state,COHAXIS][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_trials_error[state,COHAXIS][ntraine:ntraine+itest,:]))
        #     ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[state,COHAXIS].T)[:ntraine,:])),np.vstack((ylabels_teste,(ymerge_labels_error[state,COHAXIS].T)[ntraine:ntraine+itest,:]))
        ### --------- state 0 ----------------------- 
        Xdata_trainc, Xdata_testc, Xdata_traine, Xdata_teste = [],[],[],[]
        if CONTROL ==0: 
            for COHAXIS in COHAXISS:
                if len(Xdata_trainc)==0:
                    Xdata_trainc,Xdata_testc     = Xmerge_trials_correct[4,COHAXIS][:ntrainc,:],Xmerge_trials_correct[4,COHAXIS][ntrainc:ntrainc+itest,:] 
                    ylabels_trainc,ylabels_testc = (ymerge_labels_correct[4,COHAXIS].T)[:ntrainc,:],(ymerge_labels_correct[4,COHAXIS].T)[ntrainc:ntrainc+itest,:] 
                    # print('~~~~~~~shape',np.shape(Xdata_trainc),ntraine+ntrainc) 
                    Xdata_traine,Xdata_teste     = Xmerge_trials_error[0,COHAXIS][:ntraine,:],Xmerge_trials_error[0,COHAXIS][ntraine:ntraine+itest,:]
                    ylabels_traine,ylabels_teste = (ymerge_labels_error[0,COHAXIS].T)[:ntraine,:],(ymerge_labels_error[0,COHAXIS].T)[ntraine:ntraine+itest,:] 
                else:
                    Xdata_trainc,Xdata_testc     = np.vstack((Xdata_trainc,Xmerge_trials_correct[4,COHAXIS][:ntrainc,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[4,COHAXIS][ntrainc:ntrainc+itest,:]))
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[4,COHAXIS].T)[:ntrainc,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[4,COHAXIS].T)[ntrainc:ntrainc+itest,:])) 
                    Xdata_traine,Xdata_teste     = np.vstack((Xdata_traine,Xmerge_trials_error[0,COHAXIS][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_trials_error[0,COHAXIS][ntraine:ntraine+itest,:]))
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[0,COHAXIS].T)[:ntraine,:])),np.vstack((ylabels_teste,(ymerge_labels_error[0,COHAXIS].T)[ntraine:ntraine+itest,:])) 
                for state in range(1,4):  
                    Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_trials_correct[state+4,COHAXIS][:ntrainc ,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[state+4,COHAXIS][ntrainc:ntrainc+itest,:])) 
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[state+4,COHAXIS].T)[:ntrainc,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[state+4,COHAXIS].T)[ntrainc:ntrainc+itest,:]))
                    Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_trials_error[state,COHAXIS][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_trials_error[state,COHAXIS][ntraine:ntraine+itest,:])) 
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[state,COHAXIS].T)[:ntraine,:])),np.vstack((ylabels_teste,(ymerge_labels_error[state,COHAXIS].T)[ntraine:ntraine+itest,:]))  
        elif CONTROL ==1:  
            for COHAXIS in COHAXISS:
                if len(Xdata_traine)==0:
                    Xdata_testc   = Xmerge_trials_correct[4,COHAXIS][ntrainc:ntrainc+itest,:]  
                    ylabels_testc = (ymerge_labels_correct[4,COHAXIS].T)[ntrainc:ntrainc+itest,:] 
                    Xdata_traine,Xdata_teste=Xmerge_trials_error[0,COHAXIS][:ntraine+ntrainc,:],Xmerge_trials_error[0,COHAXIS][ntraine+ntrainc:ntrainc+ntraine+itest,:] 

                    print('~~~~~~~shape',np.shape(Xmerge_trials_error[0,COHAXIS]),ntraine+ntrainc)
                    ylabels_traine,ylabels_teste = (ymerge_labels_error[0,COHAXIS].T)[:ntraine+ntrainc,:],(ymerge_labels_error[0,COHAXIS].T)[ntraine+ntrainc:ntraine+ntrainc+itest,:] 
                    # print('~~~~~~~~~~~ 0 mean!!!',np.mean((ymerge_labels_error[0,COHAXIS].T),axis=1),np.shape((ymerge_labels_error[0,COHAXIS].T)))
                else:
                    Xdata_testc   = np.vstack((Xdata_testc,Xmerge_trials_correct[4,COHAXIS][ntrainc:ntrainc+itest,:])) 
                    ylabels_testc = np.vstack((ylabels_testc,(ymerge_labels_correct[4,COHAXIS].T)[ntrainc:ntrainc+itest,:])) 
                    Xdata_traine,Xdata_teste=np.vstack((Xdata_traine,Xmerge_trials_error[0,COHAXIS][:ntraine+ntrainc,:])),np.vstack((Xdata_teste,Xmerge_trials_error[0,COHAXIS][ntraine+ntrainc:ntrainc+ntraine+itest,:])) 
                    # print('~~~~~~~shape',np.shape(Xdata_traine),ntraine+ntrainc)
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[0,COHAXIS].T)[:ntraine+ntrainc,:])),np.vstack((ylabels_teste,(ymerge_labels_error[0,COHAXIS].T)[ntraine+ntrainc:ntraine+ntrainc+itest,:]))                    
                for state in range(1,4): 
                    Xdata_testc = np.vstack((Xdata_testc,Xmerge_trials_correct[state+4,COHAXIS][ntrainc:ntrainc+itest,:])) 
                    ylabels_testc = np.vstack((ylabels_testc,(ymerge_labels_correct[state+4,COHAXIS].T)[ntrainc:ntrainc+itest,:])) 
                    Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_trials_error[state,COHAXIS][:ntrainc+ntraine,:])),np.vstack((Xdata_teste,Xmerge_trials_error[state,COHAXIS][ntraine+ntrainc:ntraine+ntrainc+itest,:])) 
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[state,COHAXIS].T)[:ntraine+ntrainc,:])),np.vstack((ylabels_teste,(ymerge_labels_error[state,COHAXIS].T)[ntraine+ntrainc:ntraine+ntrainc+itest,:])) 
            
            ### unique choice 
            unique_choice_pseudo = np.unique(ylabels_traine[:,4::6])
            # print('~~~~~ 0 coh ~~~~~ unique choice here:',unique_choice_pseudo,np.shape(ylabels_traine))
        elif CONTROL ==2: 
            for COHAXIS in COHAXISS:
                if len(Xdata_trainc)==0:
                    Xdata_trainc,Xdata_testc=Xmerge_trials_correct[4,COHAXIS][:ntrainc+ntraine,:],Xmerge_trials_correct[4,COHAXIS][ntrainc+ntraine:ntrainc+ntraine+itest,:]
                    ylabels_trainc,ylabels_testc = (ymerge_labels_correct[4,COHAXIS].T)[:ntrainc+ntraine,:],(ymerge_labels_correct[4,COHAXIS].T)[ntrainc+ntraine:ntrainc+ntraine+itest,:] 
                    Xdata_teste=Xmerge_trials_error[0,COHAXIS][ntraine:ntraine+itest,:] 
                    ylabels_teste = (ymerge_labels_error[0,COHAXIS].T)[ntraine:ntraine+itest,:] 
                else:
                    Xdata_trainc,Xdata_testc=np.vstack((Xdata_trainc,Xmerge_trials_correct[4,COHAXIS][:ntrainc+ntraine,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[4,COHAXIS][ntrainc+ntraine:ntrainc+ntraine+itest,:]))
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[4,COHAXIS].T)[:ntrainc+ntraine,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[4,COHAXIS].T)[ntrainc+ntraine:ntrainc+ntraine+itest,:]))
                    Xdata_teste=np.vstack((Xdata_teste,Xmerge_trials_error[0,COHAXIS][ntraine:ntraine+itest,:])) 
                    ylabels_teste = np.vstack((ylabels_teste,(ymerge_labels_error[0,COHAXIS].T)[ntraine:ntraine+itest,:]))
                for state in range(1,4): 
                    Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_trials_correct[state+4,COHAXIS][:ntrainc+ntraine ,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[state+4,COHAXIS][ntrainc+ntraine:ntraine+ntrainc+itest,:]))  
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[state+4,COHAXIS].T)[:ntrainc+ntraine,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[state+4,COHAXIS].T)[ntrainc+ntraine:ntrainc+ntraine+itest,:])) 
                    Xdata_teste   = np.vstack((Xdata_teste,Xmerge_trials_error[state,COHAXIS][ntraine:ntraine+itest,:])) 
                    ylabels_teste = np.vstack((ylabels_teste,(ymerge_labels_error[state,COHAXIS].T)[ntraine:ntraine+itest,:]))

        
        ###### labels have no other labels except Behaviours/Stimulus
        if(CONTROL==1): 
            if DOREVERSE: 
                ylabels_traine[:, :] = 1-ylabels_traine[:, :]
            Xdata_train   = Xdata_traine.copy()
            ylabels_train = ylabels_traine.copy() 
        elif(CONTROL==2): 
            Xdata_train    = Xdata_trainc.copy()
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

        print('~~~~~~size of the test dataset:', np.shape(Xdata_train))

        

        ### --- percentage of right choices -----
        ycchoice = np.zeros(np.shape(ylabels_train)[0])
        for iset in range(np.shape(ylabels_train)[0]):
            cchoice_labels = Counter(ylabels_train[iset,:])
            ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
                        
        # ycchoice = np.zeros(np.shape(ylabels_train)[0])
        # for iset in range(np.shape(ylabels_train)[0]):
        #     cchoice_labels = Counter(ylabels_train[iset,:])
        #     ycchoice[iset] = cchoice_labels[1]/np.shape(ylabels_train)[1]
        
        try:        
            lin_cc.fit(Xdata_train, ycchoice)
            # print('sucessful',i,' iteration')
            i_sucess  +=1
        except:
            # print('The number of classes has to be greater than onfthe e; got 1 class')
            continue

        if i_sucess == 0:
            intercepts = np.zeros((1, 3 + 1 + 1))
            intercepts[:, 4] = lin_cc.intercept_[:]

            coeffs = np.zeros((NN, 3 + 1 + 1))
            coeffs[:, 4] = lin_cc.coef_[:]

        else:
            tintercepts = np.zeros((1, 3 + 1 + 1))
            tintercepts[:, 4] = lin_cc.intercept_[:]
            intercepts = np.append(intercepts, tintercepts, axis=1)

            tcoeffs = np.zeros((NN, 3 + 1 + 1))
            tcoeffs[:, 4] = lin_cc.coef_[:]
            coeffs = np.append(coeffs, tcoeffs, axis=1)

        #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>
        if(PCA_n_components>0):
            Xdata_testc=mmodel.transform(Xdata_testc)
            Xdata_teste=mmodel.transform(Xdata_teste)
        #### -------- AC testing trials 
        linw_cc, linb_cc     = lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
        evidences_c = np.zeros((ntest, 3 + 2))
        evidences_c[:, 4] = np.squeeze(
            Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)


        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
        for iset in range(np.shape(ylabels_testc)[0]):
            cchoice_labels = Counter(ylabels_testc[iset,:])
            ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
            # ycchoice_test[iset] = cchoice_labels[1]/np.shape(ylabels_testc)[1]
            
        ylabels_testc[:,4] = ycchoice_test[:]
        # print('~~~~~~test c:',np.mean(ylabels_testc[:,4]))
        
        # ypredict_choice = lin_cc.predict(Xdata_testc)
        ypredict_choice = np.zeros_like(ycchoice_test)
        ypredict_choice[np.where(evidences_c[:,4]>0)[0]]=1
        # print('~~~~shape:',np.shape(ylabels_testc[:,4].flatten()),np.shape(np.squeeze(ylabels_testc[:,4])),np.shape(ypredict_choice))
        prediction_correct = accuracy_score((ylabels_testc[:,4]).flatten(),ypredict_choice)
        stats_c[i] = prediction_correct
        # rmse_correct = mean_squared_error(ypredict_choice,ylabels_testc[:,4])
        # r2_correct   = r2_score(ypredict_choice,ylabels_testc[:,4])     
        # stats_c[i] = r2_correct

        Xtest_set_correct[i, :, :], ytest_set_correct[i, :, :]=Xdata_testc[:, :].copy(), ylabels_testc[:, :].copy()
        
        if i_sucess == 0:
            yevi_set_correct[i, :, :] = evidences_c.copy()
        else:
            yevi_set_correct[i, :, :] = evidences_c.copy()

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
        stats_e[i] = prediction_error
        # print('~~~~~~~~ performance:', prediction_correct,prediction_error)
        
        # rmse_error = mean_squared_error(ypredict_choice,ylabels_teste[:,4])
        # r2_error   = r2_score(ypredict_choice,ylabels_teste[:,4])  
        # stats_e[i] = r2_error
        
        
        # print('~~~~~~~~ performance:', rmse_correct, '  ', rmse_error)
        # print('~~~~~~~ R2:', r2_correct,'    ',r2_error)

        Xtest_set_error[i, :, :], ytest_set_error[i, :, :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

        if i_sucess  == 0:
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()
    return stats_c, stats_e, coeffs, intercepts,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

def bootstrap_linsvm_step_fixationperiod_balanced(Xdata_set,NN, ylabels_set,unique_states,unique_cohs,files,false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=5, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
    ### ac/ae ratio 
    CRATIO = 0.5#ACE_RATIO/(1+ACE_RATIO)
    ERATIO = 1-CRATIO
    #  
    COHAXISS = [-1,0,1]#[-1]#[1]
    nlabels  = (len(files)-len(false_files))
    ntrain   = int(train_percent*N_pseudo_dec) #
    ntest    = (N_pseudo_dec-ntrain)*len(COHAXISS) # COHS
    itest    = N_pseudo_dec-ntrain
    
    ### ONLY USE COHERENCE = 0
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

    lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                  shrinking=False, tol=1e-6)
    # lin_cc   = LinearRegression()

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)
    
    stats_c,stats_e = np.zeros(n_iterations),np.zeros(n_iterations)
    i_sucess = -1
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error,merge_trials=gpt.merge_pseudo_beh_trials_balanced(Xdata_set,ylabels_set,unique_states,unique_cohs,files, false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error,EACHSTATES=N_pseudo_dec, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials
        ### --------- state 0 ----------------------- 
        Xdata_trainc, Xdata_testc, Xdata_traine, Xdata_teste = [],[],[],[]
        if CONTROL ==0: 
            for COHAXIS in COHAXISS:
                if len(Xdata_trainc)==0:
                    Xdata_trainc,Xdata_testc     = Xmerge_trials_correct[COHAXIS][:ntrainc,:],Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,:] 
                    ylabels_trainc,ylabels_testc = (ymerge_labels_correct[COHAXIS].T)[:ntrainc,:],(ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:] 

                    Xdata_traine,Xdata_teste     = Xmerge_trials_error[COHAXIS][:ntraine,:],Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,:]
                    ylabels_traine,ylabels_teste = (ymerge_labels_error[COHAXIS].T)[:ntraine,:],(ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:] 
                else:
                    Xdata_trainc,Xdata_testc     = np.vstack((Xdata_trainc,Xmerge_trials_correct[COHAXIS][:ntrainc,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,:]))
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[COHAXIS].T)[:ntrainc,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:])) 
                    Xdata_traine,Xdata_teste     = np.vstack((Xdata_traine,Xmerge_trials_error[COHAXIS][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,:]))
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[COHAXIS].T)[:ntraine,:])),np.vstack((ylabels_teste,(ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:])) 
        elif CONTROL ==1:  
            for COHAXIS in COHAXISS:
                if len(Xdata_traine)==0:
                    Xdata_testc   = Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,:]  
                    ylabels_testc = (ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:] 
                    Xdata_traine,Xdata_teste=Xmerge_trials_error[COHAXIS][:ntraine+ntrainc,:],Xmerge_trials_error[COHAXIS][ntraine+ntrainc:ntrainc+ntraine+itest,:] 
                    ylabels_traine,ylabels_teste = (ymerge_labels_error[COHAXIS].T)[:ntraine+ntrainc,:],(ymerge_labels_error[COHAXIS].T)[ntraine+ntrainc:ntraine+ntrainc+itest,:] 
                else:
                    Xdata_testc   = np.vstack((Xdata_testc,Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,:])) 
                    ylabels_testc = np.vstack((ylabels_testc,(ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:])) 
                    Xdata_traine,Xdata_teste=np.vstack((Xdata_traine,Xmerge_trials_error[COHAXIS][:ntraine+ntrainc,:])),np.vstack((Xdata_teste,Xmerge_trials_error[COHAXIS][ntraine+ntrainc:ntrainc+ntraine+itest,:])) 
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[COHAXIS].T)[:ntraine+ntrainc,:])),np.vstack((ylabels_teste,(ymerge_labels_error[COHAXIS].T)[ntraine+ntrainc:ntraine+ntrainc+itest,:]))                    
        elif CONTROL ==2: 
            for COHAXIS in COHAXISS:
                if len(Xdata_trainc)==0:
                    Xdata_trainc,Xdata_testc=Xmerge_trials_correct[COHAXIS][:ntrainc+ntraine,:],Xmerge_trials_correct[COHAXIS][ntrainc+ntraine:ntrainc+ntraine+itest,:]
                    ylabels_trainc,ylabels_testc = (ymerge_labels_correct[COHAXIS].T)[:ntrainc+ntraine,:],(ymerge_labels_correct[COHAXIS].T)[ntrainc+ntraine:ntrainc+ntraine+itest,:] 
                    Xdata_teste=Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,:] 
                    ylabels_teste = (ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:] 
                else:
                    Xdata_trainc,Xdata_testc=np.vstack((Xdata_trainc,Xmerge_trials_correct[COHAXIS][:ntrainc+ntraine,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[COHAXIS][ntrainc+ntraine:ntrainc+ntraine+itest,:]))
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[COHAXIS].T)[:ntrainc+ntraine,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[COHAXIS].T)[ntrainc+ntraine:ntrainc+ntraine+itest,:]))
                    Xdata_teste=np.vstack((Xdata_teste,Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,:])) 
                    ylabels_teste = np.vstack((ylabels_teste,(ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:]))
        
        ###### labels have no other labels except Behaviours/Stimulus
        if(CONTROL==1): 
            if DOREVERSE: 
                ylabels_traine[:, :] = 1-ylabels_traine[:, :]
            Xdata_train   = Xdata_traine.copy()
            ylabels_train = ylabels_traine.copy() 
        elif(CONTROL==2): 
            Xdata_train    = Xdata_trainc.copy()
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

        print('~~~~~~size of the test dataset:', np.shape(Xdata_train))

        

        ### --- percentage of right choices -----
        ycchoice = np.zeros(np.shape(ylabels_train)[0])
        for iset in range(np.shape(ylabels_train)[0]):
            cchoice_labels = Counter(ylabels_train[iset,:])
            ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
                        
        # ycchoice = np.zeros(np.shape(ylabels_train)[0])
        # for iset in range(np.shape(ylabels_train)[0]):
        #     cchoice_labels = Counter(ylabels_train[iset,:])
        #     ycchoice[iset] = cchoice_labels[1]/np.shape(ylabels_train)[1]
        
        try:        
            lin_cc.fit(Xdata_train, ycchoice)
            # print('sucessful',i,' iteration')
            i_sucess  +=1
        except:
            # print('The number of classes has to be greater than onfthe e; got 1 class')
            continue

        if i_sucess == 0:
            intercepts = np.zeros((1, 3 + 1 + 1))
            intercepts[:, 4] = lin_cc.intercept_[:]

            coeffs = np.zeros((NN, 3 + 1 + 1))
            coeffs[:, 4] = lin_cc.coef_[:]

        else:
            tintercepts = np.zeros((1, 3 + 1 + 1))
            tintercepts[:, 4] = lin_cc.intercept_[:]
            intercepts = np.append(intercepts, tintercepts, axis=1)

            tcoeffs = np.zeros((NN, 3 + 1 + 1))
            tcoeffs[:, 4] = lin_cc.coef_[:]
            coeffs = np.append(coeffs, tcoeffs, axis=1)

        #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>
        if(PCA_n_components>0):
            Xdata_testc=mmodel.transform(Xdata_testc)
            Xdata_teste=mmodel.transform(Xdata_teste)
        #### -------- AC testing trials 
        linw_cc, linb_cc     = lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
        evidences_c = np.zeros((ntest, 3 + 2))
        evidences_c[:, 4] = np.squeeze(
            Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)


        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
        for iset in range(np.shape(ylabels_testc)[0]):
            cchoice_labels = Counter(ylabels_testc[iset,:])
            ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
            # ycchoice_test[iset] = cchoice_labels[1]/np.shape(ylabels_testc)[1]
            
        ylabels_testc[:,4] = ycchoice_test[:]
        # print('~~~~~~test c:',np.mean(ylabels_testc[:,4]))
        
        # ypredict_choice = lin_cc.predict(Xdata_testc)
        ypredict_choice = np.zeros_like(ycchoice_test)
        ypredict_choice[np.where(evidences_c[:,4]>0)[0]]=1
        # print('~~~~shape:',np.shape(ylabels_testc[:,4].flatten()),np.shape(np.squeeze(ylabels_testc[:,4])),np.shape(ypredict_choice))
        prediction_correct = accuracy_score((ylabels_testc[:,4]).flatten(),ypredict_choice)
        stats_c[i] = prediction_correct
        # rmse_correct = mean_squared_error(ypredict_choice,ylabels_testc[:,4])
        # r2_correct   = r2_score(ypredict_choice,ylabels_testc[:,4])     
        # stats_c[i] = r2_correct

        Xtest_set_correct[i, :, :], ytest_set_correct[i, :, :]=Xdata_testc[:, :].copy(), ylabels_testc[:, :].copy()
        
        if i_sucess == 0:
            yevi_set_correct[i, :, :] = evidences_c.copy()
        else:
            yevi_set_correct[i, :, :] = evidences_c.copy()

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
        stats_e[i] = prediction_error
        # print('~~~~~~~~ performance:', prediction_correct,prediction_error)
        
        # rmse_error = mean_squared_error(ypredict_choice,ylabels_teste[:,4])
        # r2_error   = r2_score(ypredict_choice,ylabels_teste[:,4])  
        # stats_e[i] = r2_error
        
        
        # print('~~~~~~~~ performance:', rmse_correct, '  ', rmse_error)
        # print('~~~~~~~ R2:', r2_correct,'    ',r2_error)

        Xtest_set_error[i, :, :], ytest_set_error[i, :, :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

        if i_sucess  == 0:
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()
    return stats_c, stats_e, coeffs, intercepts,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET


def bootstrap_linsvm_step_fixationperiod_pop(Xdata_set,NN, ylabels_set,unique_states,unique_cohs,files,false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error, pop_correct,pop_error, type, DOREVERSE=0, CONTROL = 0, STIM_PERIOD=0, n_iterations=10, N_pseudo_dec=5, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[],mmodel=[],PCA_n_components=0):
    ### ac/ae ratio 
    CRATIO = 0.5#ACE_RATIO/(1+ACE_RATIO)
    ERATIO = 1-CRATIO
    #  
    COHAXISS = [0]#[-1,0,1]#[-1]#[1]
    nlabels  = (len(files)-len(false_files))
    ntrain   = int(train_percent*N_pseudo_dec) #
    ntest    = (N_pseudo_dec-ntrain)*len(COHAXISS) # COHS
    itest    = N_pseudo_dec-ntrain

    if(CONTROL==1):
        NN = len(pop_error)
    elif(CONTROL==2):
        NN= len(pop_correct)
    
    ### ONLY USE COHERENCE = 0
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

    lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                  shrinking=False, tol=1e-6)

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)
    
    stats_c,stats_e = np.zeros(n_iterations),np.zeros(n_iterations)
    i_sucess = -1
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error,merge_trials=gpt.merge_pseudo_beh_trials_balanced(Xdata_set,ylabels_set,unique_states,unique_cohs,files, false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error,EACHSTATES=N_pseudo_dec, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[i],STIM_BEH=1)

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials
        ### --------- state 0 ----------------------- 
        Xdata_trainc, Xdata_testc, Xdata_traine, Xdata_teste = [],[],[],[]
        if CONTROL ==0: 
            for COHAXIS in COHAXISS:
                if len(Xdata_trainc)==0:
                    Xdata_trainc,Xdata_testc     = Xmerge_trials_correct[COHAXIS][:ntrainc,:],Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,:] 
                    ylabels_trainc,ylabels_testc = (ymerge_labels_correct[COHAXIS].T)[:ntrainc,:],(ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:] 

                    Xdata_traine,Xdata_teste     = Xmerge_trials_error[COHAXIS][:ntraine,:],Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,:]
                    ylabels_traine,ylabels_teste = (ymerge_labels_error[COHAXIS].T)[:ntraine,:],(ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:] 
                else:
                    Xdata_trainc,Xdata_testc     = np.vstack((Xdata_trainc,Xmerge_trials_correct[COHAXIS][:ntrainc,:])),np.vstack((Xdata_testc,Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,:]))
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[COHAXIS].T)[:ntrainc,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:])) 
                    Xdata_traine,Xdata_teste     = np.vstack((Xdata_traine,Xmerge_trials_error[COHAXIS][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,:]))
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[COHAXIS].T)[:ntraine,:])),np.vstack((ylabels_teste,(ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:])) 
        elif CONTROL ==1:  
            for COHAXIS in COHAXISS:
                if len(Xdata_traine)==0:
                    Xdata_testc   = Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,pop_error]  
                    ylabels_testc = (ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:] 
                    Xdata_traine,Xdata_teste=Xmerge_trials_error[COHAXIS][:ntraine+ntrainc,pop_error],Xmerge_trials_error[COHAXIS][ntraine+ntrainc:ntrainc+ntraine+itest,pop_error] 
                    ylabels_traine,ylabels_teste = (ymerge_labels_error[COHAXIS].T)[:ntraine+ntrainc,:],(ymerge_labels_error[COHAXIS].T)[ntraine+ntrainc:ntraine+ntrainc+itest,:] 
                else:
                    Xdata_testc   = np.vstack((Xdata_testc,Xmerge_trials_correct[COHAXIS][ntrainc:ntrainc+itest,pop_error])) 
                    ylabels_testc = np.vstack((ylabels_testc,(ymerge_labels_correct[COHAXIS].T)[ntrainc:ntrainc+itest,:])) 
                    Xdata_traine,Xdata_teste=np.vstack((Xdata_traine,Xmerge_trials_error[COHAXIS][:ntraine+ntrainc,pop_error])),np.vstack((Xdata_teste,Xmerge_trials_error[COHAXIS][ntraine+ntrainc:ntrainc+ntraine+itest,pop_error])) 
                    ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,(ymerge_labels_error[COHAXIS].T)[:ntraine+ntrainc,:])),np.vstack((ylabels_teste,(ymerge_labels_error[COHAXIS].T)[ntraine+ntrainc:ntraine+ntrainc+itest,:]))                    
        elif CONTROL ==2: 
            for COHAXIS in COHAXISS:
                if len(Xdata_trainc)==0:
                    Xdata_trainc,Xdata_testc=Xmerge_trials_correct[COHAXIS][:ntrainc+ntraine,pop_correct],Xmerge_trials_correct[COHAXIS][ntrainc+ntraine:ntrainc+ntraine+itest,pop_correct]
                    ylabels_trainc,ylabels_testc = (ymerge_labels_correct[COHAXIS].T)[:ntrainc+ntraine,:],(ymerge_labels_correct[COHAXIS].T)[ntrainc+ntraine:ntrainc+ntraine+itest,:] 
                    Xdata_teste=Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,pop_correct] 
                    ylabels_teste = (ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:] 
                else:
                    Xdata_trainc,Xdata_testc=np.vstack((Xdata_trainc,Xmerge_trials_correct[COHAXIS][:ntrainc+ntraine,pop_correct])),np.vstack((Xdata_testc,Xmerge_trials_correct[COHAXIS][ntrainc+ntraine:ntrainc+ntraine+itest,pop_correct]))
                    ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,(ymerge_labels_correct[COHAXIS].T)[:ntrainc+ntraine,:])),np.vstack((ylabels_testc,(ymerge_labels_correct[COHAXIS].T)[ntrainc+ntraine:ntrainc+ntraine+itest,:]))
                    Xdata_teste=np.vstack((Xdata_teste,Xmerge_trials_error[COHAXIS][ntraine:ntraine+itest,pop_correct])) 
                    ylabels_teste = np.vstack((ylabels_teste,(ymerge_labels_error[COHAXIS].T)[ntraine:ntraine+itest,:]))
        
        ###### labels have no other labels except Behaviours/Stimulus
        if(CONTROL==1): 
            if DOREVERSE: 
                ylabels_traine[:, :] = 1-ylabels_traine[:, :]
            Xdata_train   = Xdata_traine.copy()
            ylabels_train = ylabels_traine.copy() 
        elif(CONTROL==2): 
            Xdata_train    = Xdata_trainc.copy()
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

        

        ### --- percentage of right choices -----
        ycchoice = np.zeros(np.shape(ylabels_train)[0])
        for iset in range(np.shape(ylabels_train)[0]):
            cchoice_labels = Counter(ylabels_train[iset,:])
            ycchoice[iset] = (cchoice_labels.most_common(1)[0][0])
                        
        # ycchoice = np.zeros(np.shape(ylabels_train)[0])
        # for iset in range(np.shape(ylabels_train)[0]):
        #     cchoice_labels = Counter(ylabels_train[iset,:])
        #     ycchoice[iset] = cchoice_labels[1]/np.shape(ylabels_train)[1]
        
        try:        
            lin_cc.fit(Xdata_train, ycchoice)
            # print('sucessful',i,' iteration')
            i_sucess  +=1
        except:
            # print('The number of classes has to be greater than onfthe e; got 1 class')
            continue

        if i_sucess == 0:
            intercepts = np.zeros((1, 3 + 1 + 1))
            intercepts[:, 4] = lin_cc.intercept_[:]

            coeffs = np.zeros((NN, 3 + 1 + 1))
            coeffs[:, 4] = lin_cc.coef_[:]

        else:
            tintercepts = np.zeros((1, 3 + 1 + 1))
            tintercepts[:, 4] = lin_cc.intercept_[:]
            intercepts = np.append(intercepts, tintercepts, axis=1)

            tcoeffs = np.zeros((NN, 3 + 1 + 1))
            tcoeffs[:, 4] = lin_cc.coef_[:]
            coeffs = np.append(coeffs, tcoeffs, axis=1)

        #### >>>>>>>> testing stage >>>>>>>>>>>>>>>>
        if(PCA_n_components>0):
            Xdata_testc=mmodel.transform(Xdata_testc)
            Xdata_teste=mmodel.transform(Xdata_teste)
        #### -------- AC testing trials 
        linw_cc, linb_cc     = lin_cc.coef_[:], lin_cc.intercept_[:]
        # evaluate evidence model
        evidences_c = np.zeros((ntest, 3 + 2))
        evidences_c[:, 4] = np.squeeze(
            Xdata_testc @ linw_cc.reshape(-1, 1) + linb_cc)


        ### --- percentage of right choices -----
        ycchoice_test = np.zeros(np.shape(ylabels_testc)[0])
        for iset in range(np.shape(ylabels_testc)[0]):
            cchoice_labels = Counter(ylabels_testc[iset,:])
            ycchoice_test[iset] = (cchoice_labels.most_common(1)[0][0])
            # ycchoice_test[iset] = cchoice_labels[1]/np.shape(ylabels_testc)[1]
            
        ylabels_testc[:,4] = ycchoice_test[:]
        # print('~~~~~~test c:',np.mean(ylabels_testc[:,4]))
        
        # ypredict_choice = lin_cc.predict(Xdata_testc)
        ypredict_choice = np.zeros_like(ycchoice_test)
        ypredict_choice[np.where(evidences_c[:,4]>0)[0]]=1
        # print('~~~~shape:',np.shape(ylabels_testc[:,4].flatten()),np.shape(np.squeeze(ylabels_testc[:,4])),np.shape(ypredict_choice))
        prediction_correct = accuracy_score((ylabels_testc[:,4]).flatten(),ypredict_choice)
        stats_c[i] = prediction_correct
        # rmse_correct = mean_squared_error(ypredict_choice,ylabels_testc[:,4])
        # r2_correct   = r2_score(ypredict_choice,ylabels_testc[:,4])     
        # stats_c[i] = r2_correct

        Xtest_set_correct[i, :, :], ytest_set_correct[i, :, :]=Xdata_testc[:, :].copy(), ylabels_testc[:, :].copy()
        
        if i_sucess == 0:
            yevi_set_correct[i, :, :] = evidences_c.copy()
        else:
            yevi_set_correct[i, :, :] = evidences_c.copy()

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
        stats_e[i] = prediction_error
        print('~~~~~~~~ performance:', prediction_correct,prediction_error)
        
        # rmse_error = mean_squared_error(ypredict_choice,ylabels_teste[:,4])
        # r2_error   = r2_score(ypredict_choice,ylabels_teste[:,4])  
        # stats_e[i] = r2_error
        
        
        # print('~~~~~~~~ performance:', rmse_correct, '  ', rmse_error)
        # print('~~~~~~~ R2:', r2_correct,'    ',r2_error)

        Xtest_set_error[i, :, :], ytest_set_error[i, :, :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

        if i_sucess  == 0:
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()
    return stats_c, stats_e, coeffs, intercepts,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET