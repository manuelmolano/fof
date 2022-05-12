# Load packages;
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import generate_pseudo_trials as pseudo
PRINT_PER = 1000


def bootstrap_linsvm_decodersdef(Xtrain_poolc, ytrain_poolc, Xtrain_poole,
                                 ytrain_poole, type, DOREVERSE=0,
                                 n_iterations=10, n_percent=25):
    if(n_percent < 1):
        # Size of sample, picking only 50% of the given data in every bootstrap
        # sample
        n_size_train = int(Xtrain_poolc.shape[0] * n_percent)
        n_size_train *= 2
    else:
        n_size_train = int(n_percent / 2)
    # Lets run Bootstrap
    NN = Xtrain_poolc.shape[1]
    stats = list()
    lin_pact = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_ctxt = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_xor = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                      shrinking=False, tol=1e-6)
    lin_bias = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                       shrinking=False, tol=1e-6)
    lin_cc = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr',
                     shrinking=False, tol=1e-6)

    # recording support vectors
    Xsup_vec_act  = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_ctxt = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_xor  = {}
    Xsup_vec_bias = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_cc   = {}
    for i in range(n_iterations):
        
        if (i+1) % PRINT_PER == 0:
            print(i)
        # prepare train & test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtrainc = np.random.choice(np.arange(0, ytrain_poolc.shape[0], 1),
                                     size=n_size_train, replace=True)
        Xdata_trainc = Xtrain_poolc[idxtrainc, :]
        ydata_trainc = ytrain_poolc[idxtrainc, :]
        ydata_trainc = ydata_trainc - 2

        idxtraine = np.random.choice(np.arange(0, ytrain_poole.shape[0], 1),
                                     size=n_size_train, replace=True)
        Xdata_traine = Xtrain_poole[idxtraine, :]
        ydata_traine = ytrain_poole[idxtraine, :]
        if DOREVERSE:
            ydata_traine[:, 3] = 1-ydata_traine[:, 3]

        Xdata_train = np.append(Xdata_trainc, Xdata_traine, axis=0)

        ydata_train = np.append(ydata_trainc, ydata_traine, axis=0)
        # fit model
        # model.fit(X_train,y_train) i.e model.fit(train set, train label as it
        # is a classifier)
        lin_pact.fit(Xdata_train, np.squeeze(ydata_train[:, 0]))
        Xsup_vec_act[i] = lin_pact.support_vectors_

        lin_ctxt.fit(Xdata_train, np.squeeze(ydata_train[:, 1]))
        Xsup_vec_ctxt[i] = lin_ctxt.support_vectors_

        lin_xor.fit(Xdata_train, np.squeeze(ydata_train[:, 2]))
        Xsup_vec_xor[i] = lin_xor.support_vectors_
        
        ### ---- most common tr. bias -----------
        nset = int(np.shape(ydata_train)[0])
        ytr_bias = np.zeros(np.shape(ydata_train)[0])
        for iset in range(nset):
            bias_labels=Counter(ydata_train[iset,3::6])
            ytr_bias[iset]=(bias_labels.most_common(1)[0][0])
        lin_bias.fit(Xdata_train, ytr_bias)
        # lin_bias.fit(Xdata_train, np.squeeze(ydata_train[:, 3]))
        Xsup_vec_bias[i] = lin_bias.support_vectors_

        lin_cc.fit(Xdata_train, np.squeeze(ydata_train[:, 4]))
        Xsup_vec_cc[i] = lin_cc.support_vectors_

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
    return stats, coeffs, intercepts,\
        Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias, Xsup_vec_cc
        
def bootstrap_linsvm_proj_withtrials(coeffs_pool, intercepts_pool, Xtest_pool,
                                     ytest_pool, type, n_iterations=10,
                                     n_percent=25):
    # if n_iterations==None:
    # n_iterations = 100  #No. of bootstrap samples to be repeated (created)
    if(n_percent < 1):
        n_size_test = int(Xtest_pool.shape[0] * n_percent)
    else:
        n_size_test = n_percent
    # Lets run Bootstrap
    NN = Xtest_pool.shape[1]
    stats = list()
    nlabels = np.shape(ytest_pool)[1]
    Xtest_set, ytest_set, ypred_set =\
        np.zeros((n_iterations, n_size_test, NN)),\
        np.zeros((n_iterations, n_size_test, nlabels)),\
        np.zeros((n_iterations, n_size_test, nlabels))
    
    yevi_set = np.zeros((n_iterations, n_size_test, 3+2))
    test_set = np.zeros((n_iterations, n_size_test))
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # prepare test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtest = np.random.choice(np.arange(0, ytest_pool.shape[0], 1),
                                   size=n_size_test, replace=True)
        Xdata_test = Xtest_pool[idxtest, :]
        ydata_test = ytest_pool[idxtest, :]
        test_set[i, :] = idxtest[:]

        # @YX 0910 -- weights
        linw_pact, linb_pact = coeffs_pool[:, i*5+0], intercepts_pool[0, 5*i+0]
        linw_ctxt, linb_ctxt = coeffs_pool[:, i*5+1], intercepts_pool[0, 5*i+1]
        linw_xor, linb_xor   = coeffs_pool[:, i*5+2], intercepts_pool[0, 5*i+2]
        linw_bias, linb_bias = coeffs_pool[:, i*5+3], intercepts_pool[0, 5*i+3]
        linw_cc, linb_cc     = coeffs_pool[:, i * 5+4], intercepts_pool[0, 5*i+4]
        # evaluate evidence model
        evidences = np.zeros((len(idxtest), 3 + 2))
        evidences[:, 0] = np.squeeze(
            Xdata_test @ linw_pact.reshape(-1, 1) + linb_pact)
        evidences[:, 1] = np.squeeze(
            Xdata_test @ linw_ctxt.reshape(-1, 1) + linb_ctxt)
        evidences[:, 2] = np.squeeze(
            Xdata_test @ linw_xor.reshape(-1, 1) + linb_xor)
        evidences[:, 3] = np.squeeze(
            Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias)
        evidences[:, 4] = np.squeeze(
            Xdata_test @ linw_cc.reshape(-1, 1) + linb_cc)
        predictions = np.zeros((len(idxtest), nlabels))
        # predictions = np.zeros((len(idxtest), 3 + 2))
        for j in range(3 + 2):
            predictions[np.where(evidences[:, j] > 0)[0], j] = 1
            predictions[np.where(evidences[:, j] <= 0)[0], j] = 0

        Xtest_set[i, :, :], ytest_set[i, :, :], ypred_set[i, :, :] =\
            Xdata_test[:, :].copy(), ydata_test[:, :].copy(),\
            predictions[:, :].copy()
        score = np.zeros((3 + 2, 1))
        if(type == 'correct'):
            for j in range(3 + 2):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j] + 2, predictions[:, j])
        elif(type == 'error'):
            for j in range(3 + 2):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        elif(type == 'normal'):
            for j in range(3 + 2):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        # caution, overall accuracy score can mislead when classes are
        # imbalanced

        # print(score)
        if i == 0:
            stats = score.copy()
            # print("shape ...",stats.shape)
            yevi_set[i, :, :] = evidences.copy()
        else:
            # print("shape ...",stats.shape)
            # print("scoreshape...",score.shape)
            stats = np.append(stats, score, axis=1)
            yevi_set[i, :, :] = evidences.copy()
    return stats, Xtest_set, ytest_set, ypred_set, yevi_set, test_set

def bootstrap_linsvm_step(Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, n_iterations=10, N_pseudo_dec=25, train_percent=0.6):
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

    
    

    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        ### >>>>>> generate training and testing dataset for decoder and testing(onestep)
        # N_pseudo_dec, N_pseudo_beh = 100,25
        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error=pseudo.merge_pseudo_hist_trials(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,N_pseudo_dec)

        Xdata_trainc,Xdata_testc=Xmerge_hist_trials_correct[4][:ntrain,:],Xmerge_hist_trials_correct[4][ntrain:,:]
        ylabels_trainc,ylabels_testc = ymerge_hist_labels_correct[4][:ntrain,:],ymerge_hist_labels_correct[4][ntrain:,:]

        Xdata_traine,Xdata_teste=Xmerge_hist_trials_error[0][:ntrain,:],Xmerge_hist_trials_error[0][ntrain:,:]
        ylabels_traine,ylabels_teste = ymerge_hist_labels_error[0][:ntrain,:],ymerge_hist_labels_error[0][ntrain:,:]
        for state in range(1,4):
            Xdata_trainc,Xdata_testc = np.vstack((Xdata_trainc,Xmerge_hist_trials_correct[state+4][:ntrain ,:])),np.vstack((Xdata_testc,Xmerge_hist_trials_correct[state+4][ntrain :,:]))
            ylabels_trainc,ylabels_testc = np.vstack((ylabels_trainc,ymerge_hist_labels_correct[state+4][:ntrain,:])),np.vstack((ylabels_testc,ymerge_hist_labels_correct[state+4][ntrain:,:]))
         
            Xdata_traine,Xdata_teste = np.vstack((Xdata_traine,Xmerge_hist_trials_error[state][:ntrain,:])),np.vstack((Xdata_teste,Xmerge_hist_trials_error[state][ntrain:,:]))
            ylabels_traine,ylabels_teste = np.vstack((ylabels_traine,ymerge_hist_labels_error[state][:ntrain,:])),np.vstack((ylabels_teste,ymerge_hist_labels_error[state][ntrain :,:]))
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

        Xtest_set_error[i, :, :], ytest_set_error[i, :, :]=Xdata_teste[:, :].copy(), ylabels_teste[:, :].copy()

        if i == 0:
            yevi_set_error[i, :, :] = evidences_e.copy()
        else:
            yevi_set_error[i, :, :] = evidences_e.copy()
    return coeffs, intercepts,\
        Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias, Xsup_vec_cc,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error