# Load packages;
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import generate_pseudo_trials as gpt
from collections import Counter

PRINT_PER = 1000

def bootstrap_linsvm_step(Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, CONTROL = 0, n_iterations=10, N_pseudo_dec=25, ACE_RATIO=0.5, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[]):

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

    yevi_set_correct = np.zeros((n_iterations, ntest, 3+2))
    yevi_set_error   = np.zeros((n_iterations, ntest, 3+2))

    stats = list()
    lin_pact = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_ctxt = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_xor  = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                      shrinking=False, tol=1e-6)
    lin_bias = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_cc   = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                     shrinking=False, tol=1e-6)

    ntrainc = int(N_pseudo_dec/0.5*train_percent*CRATIO)
    ntraine = N_pseudo_dec/0.5*train_percent-ntrainc
    ntraine = int(ntraine)  
    print("ntrainc ---",ntrainc,'ntraine ----',ntraine)
    
    

    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # print('...... iteration index:', i,'........')
        ### >>>>>> generate training and testing dataset for decoder and testing(onestep)
        # N_pseudo_dec, N_pseudo_beh = 100,25
        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error,merge_trials_hist=gpt.merge_pseudo_hist_trials(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,2*N_pseudo_dec,RECORD_TRIALS, RECORDED_TRIALS_SET[i])

        if RECORD_TRIALS == 1:
            RECORDED_TRIALS_SET[i]=merge_trials_hist


        #### --------- state 0 -----------------------
        Xdata_trainc,Xdata_testc=Xmerge_hist_trials_correct[4][:ntrainc,:],Xmerge_hist_trials_correct[4][ntrainc:ntrainc+itest,:]
        ylabels_trainc,ylabels_testc = ymerge_hist_labels_correct[4][:ntrainc,:],ymerge_hist_labels_correct[4][ntrainc:ntrainc+itest,:]

        Xdata_traine,Xdata_teste=Xmerge_hist_trials_error[0][:ntraine,:],Xmerge_hist_trials_error[0][ntraine:ntraine+itest,:]
        ylabels_traine,ylabels_teste = ymerge_hist_labels_error[0][:ntraine,:],ymerge_hist_labels_error[0][ntraine:ntraine+itest,:]
        for state in range(1,4):
            Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_hist_trials_correct[state+4][:ntrainc ,:])),np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrainc:ntrainc+itest,:]))
            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ymerge_hist_labels_correct[state+4][:ntrainc,:])),np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrainc:ntrainc+itest,:]))
         
            Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_hist_trials_error[state][:ntraine,:])),np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntraine:ntraine+itest,:]))
            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ymerge_hist_labels_error[state][:ntraine,:])),np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntraine:ntraine+itest,:]))
        # prepare train & test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        # recording support vectors
        Xsup_vec_act  = {}  # np.zeros((n_iterations,1,NN))
        Xsup_vec_ctxt = {}  # np.zeros((n_iterations,1,NN))
        Xsup_vec_xor  = {}
        Xsup_vec_bias = {}  # np.zeros((n_iterations,1,NN))
        Xsup_vec_cc   = {}

        ylabels_trainc = ylabels_trainc - 2
        if DOREVERSE:
            ylabels_traine[:, 3] = 1-ylabels_traine[:, 3]

        if(CONTROL==1):
            Xdata_train   = Xdata_traine.copy()
            ylabels_train = ylabels_traine.copy() 
        elif(CONTROL==2):
            Xdata_train   = Xdata_trainc.copy()
            ylabels_train = ylabels_trainc.copy() 
        elif(CONTROL==0):          
            Xdata_train   = np.append(Xdata_trainc, Xdata_traine, axis=0)
            ylabels_train = np.append(ylabels_trainc, ylabels_traine, axis=0)
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
        ytr_bias = np.zeros(np.shape(ylabels_train)[0])
        for iset in range(np.shape(ylabels_train)[0]):
            bias_labels=Counter(ylabels_train[iset,3::6])
            ytr_bias[iset]=(bias_labels.most_common(1)[0][0])
        lin_bias.fit(Xdata_train, ytr_bias)
        Xsup_vec_bias[i]  = lin_bias.support_vectors_

        lin_cc.fit(Xdata_train, np.squeeze(ylabels_train[:, 4]))
        Xsup_vec_cc[i]    = lin_cc.support_vectors_

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
    return coeffs, intercepts,\
        Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias, Xsup_vec_cc,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

def bootstrap_linsvm_proj_step(coeffs_pool, intercepts_pool, Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, n_iterations=10, N_pseudo_dec=25, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[]):
    # NN      = np.shape(Xdata_hist_set[unique_states[0],'correct'])[1]
    nlabels = 6*(len(files)-len(false_files))
    ntrain  = int(train_percent*N_pseudo_dec)
    ntest   = (N_pseudo_dec-ntrain)*4 # state

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
    return coeffs_pool, intercepts_pool, \
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET

def shuffle_linsvm_proj_step(coeffs_pool, intercepts_pool, Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, n_iterations=10, N_pseudo_dec=25, train_percent=0.6, RECORD_TRIALS=0, RECORDED_TRIALS_SET=[]):
    nlabels = 6*(len(files)-len(false_files))
    ntrain  = int(train_percent*N_pseudo_dec)
    ntest   = (N_pseudo_dec-ntrain)*4 # state

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
    return coeffs_pool, intercepts_pool, \
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET
