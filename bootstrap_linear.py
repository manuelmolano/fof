# Load packages;
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
PRINT_PER = 1000


def bootstrap_linsvm_acc(Xtrain_pool, ytrain_pool, Xtest_pool,
                         ytest_pool, type, n_iterations=10, n_percent=25):
    # if n_iterations==None:
    # n_iterations = 100  #No. of bootstrap samples to be repeated (created)
    if(n_percent < 1):
        # Size of sample, picking only 50% of the given data in every bootstrap
        # sample
        n_size_train = int(Xtrain_pool.shape[0] * n_percent)
        n_size_test = int(Xtest_pool.shape[0] * n_percent)
    else:
        n_size_train = n_percent
        n_size_test = n_percent
    # Lets run Bootstrap

    stats = list()
    intercept = list()
    lin_clf = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-3)
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)

        # prepare train & test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtrain = np.random.choice(np.arange(0, len(ytrain_pool), 1),
                                    size=n_size_train,
                                    replace=True)
        Xdata_train = Xtrain_pool[idxtrain, :]
        ydata_train = ytrain_pool[idxtrain]
        # print("shape x:",Xdata_train.shape,ydata_train.shape)

        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtest = np.random.choice(
            np.arange(
                0,
                len(ytest_pool),
                1),
            size=n_size_test,
            replace=True)
        Xdata_test = Xtest_pool[idxtest, :]
        ydata_test = ytest_pool[idxtest]

        # fit model
        # model.fit(X_train,y_train) i.e model.fit(train set, train label as it
        # is a classifier)
        lin_clf.fit(Xdata_train, ydata_train)

        # evaluate model
        predictions = lin_clf.predict(Xdata_test)  # model.predict(X_test)
        if(type == 'correct'):
            # accuracy_score(y_test, y_pred)
            score = accuracy_score(ydata_test + 2, predictions)
        elif(type == 'error'):
            score = accuracy_score(ydata_test - 2, predictions)
        elif(type == 'normal'):
            score = accuracy_score(ydata_test, predictions)
        # caution, overall accuracy score can mislead when classes are
        # imbalanced

        # print(score)
        stats.append(score)
        intercept.append(lin_clf.intercept_)
        if i == 0:
            coeffs = np.reshape(lin_clf.coef_, (-1, 1))
        else:
            coeffs = np.append(
                coeffs, np.reshape(
                    lin_clf.coef_, (-1, 1)), axis=1)
    return stats, coeffs, intercept


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
    Xsup_vec_act = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_ctxt = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_xor = {}
    Xsup_vec_bias = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_cc = {}
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

        lin_bias.fit(Xdata_train, np.squeeze(ydata_train[:, 3]))
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


def bootstrap_linsvm_acc_withtrials(Xtrain_pool, ytrain_pool, Xtest_pool,
                                    ytest_pool, type, n_iterations=10,
                                    n_percent=25):
    '''
    @YX 05DEC -- 3+1 LABELS
    '''
    if(n_percent < 1):
        # Size of sample, picking only 50% of the given data in every bootstrap
        # sample
        n_size_train = int(Xtrain_pool.shape[0] * n_percent)
        n_size_test = int(Xtest_pool.shape[0] * n_percent)
    else:
        n_size_train = n_percent
        n_size_test = n_percent
    # Lets run Bootstrap
    NN = Xtrain_pool.shape[1]
    stats = list()
    lin_pact = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_ctxt = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_xor = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_bias = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_cc = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    Xtest_set, ytest_set, ypred_set = np.zeros((n_iterations, n_size_test, NN)),\
        np.zeros((n_iterations, n_size_test, 3 + 1 + 1)
                 ), np.zeros((n_iterations, n_size_test, 3 + 1 + 1))
    # recording support vectors
    Xsup_vec_act = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_ctxt = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_xor = {}
    Xsup_vec_bias = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_cc = {}
    yevi_set = np.zeros((n_iterations, n_size_test, 3 + 1 + 1))
    test_set = np.zeros((n_iterations, n_size_test))
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # prepare train & test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtrain = np.random.choice(
            np.arange(
                0,
                ytrain_pool.shape[0],
                1),
            size=n_size_train,
            replace=True)
        Xdata_train = Xtrain_pool[idxtrain, :]
        ydata_train = ytrain_pool[idxtrain, :]  # pact,ctxt,bias
        # print("shape x:",Xdata_train.shape,ydata_train.shape)

        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtest = np.random.choice(
            np.arange(
                0,
                ytest_pool.shape[0],
                1),
            size=n_size_test,
            replace=True)
        Xdata_test = Xtest_pool[idxtest, :]
        ydata_test = ytest_pool[idxtest, :]
        test_set[i, :] = idxtest[:]

        # fit model
        # model.fit(X_train,y_train) i.e model.fit(train set, train label as it
        # is a classifier)
        lin_pact.fit(Xdata_train, np.squeeze(ydata_train[:, 0]))
        Xsup_vec_act[i] = lin_pact.support_vectors_

        lin_ctxt.fit(Xdata_train, np.squeeze(ydata_train[:, 1]))
        Xsup_vec_ctxt[i] = lin_ctxt.support_vectors_

        lin_xor.fit(Xdata_train, np.squeeze(ydata_train[:, 2]))
        Xsup_vec_xor[i] = lin_xor.support_vectors_

        lin_bias.fit(Xdata_train, np.squeeze(ydata_train[:, 3]))
        Xsup_vec_bias[i] = lin_bias.support_vectors_

        lin_cc.fit(Xdata_train, np.squeeze(ydata_train[:, 4]))
        Xsup_vec_cc[i] = lin_cc.support_vectors_

        # evaluate model
        predictions = np.zeros((len(idxtest), 3 + 1 + 1))
        predictions[:, 0] = lin_pact.predict(
            Xdata_test)  # model.predict(X_test)
        predictions[:, 1] = lin_ctxt.predict(Xdata_test)
        predictions[:, 2] = lin_xor.predict(Xdata_test)
        predictions[:, 3] = lin_bias.predict(Xdata_test)
        predictions[:, 4] = lin_cc.predict(Xdata_test)
        # print("shape ",Xdata_test.shape,ydata_test.shape,predictions.shape)
        Xtest_set[i, :, :], ytest_set[i, :, :], ypred_set[i, :, :] =\
            Xdata_test[:, :].copy(), ydata_test[:, :].copy(),\
            predictions[:, :].copy()
        score = np.zeros((3 + 1 + 1, 1))
        if(type == 'correct'):
            for j in range(np.shape(score)[0]):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        elif(type == 'error'):
            for j in range(np.shape(score)[0]):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        elif(type == 'normal'):
            for j in range(np.shape(score)[0]):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        # caution, overall accuracy score can mislead when classes are
        # imbalanced

        # print(score)
        if i == 0:
            stats = score
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
            # print("shape...",Xdata_test.shape,coeffs.shape,intercepts.shape,coeffs.shape[0])
            yevi_set[i, :, :] = np.squeeze(Xdata_test @ coeffs +
                                           np.repeat(intercepts,
                                                     Xdata_test.shape[0], axis=0))
        else:
            stats = np.append(stats, score, axis=1)

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

            yevi_set[i, :, :] = np.squeeze(
                Xdata_test @ tcoeffs + np.repeat(tintercepts, Xdata_test.shape[0],
                                                 axis=0))
    return stats, coeffs, intercepts, Xtest_set, ytest_set, ypred_set,\
        yevi_set, test_set, Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias, Xsup_vec_cc


def bootstrap_linsvm_acc_withtrials_congruent(Xtrain_pool, ytrain_pool, Xtest_pool,
                                              ytest_pool, type, n_iterations=10,
                                              n_percent=25):
    '''
    @YX 05DEC -- 3+1 LABELS
    '''
    if(n_percent < 1):
        # Size of sample, picking only 50% of the given data in every bootstrap
        # sample
        n_size_train = int(Xtrain_pool.shape[0] * n_percent)
        n_size_test = int(Xtest_pool.shape[0] * n_percent)
    else:
        n_size_train = n_percent
        n_size_test = n_percent
    # Lets run Bootstrap
    NN = Xtrain_pool.shape[1]
    stats = list()
    lin_pact = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_ctxt = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_xor = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_bias = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    lin_cc = svm.SVC(
        C=1,
        kernel='linear',
        decision_function_shape='ovr',
        shrinking=False,
        tol=1e-6)
    Xtest_set, ytest_set, ypred_set = np.zeros((n_iterations, n_size_test, NN)),\
        np.zeros((n_iterations, n_size_test, 3 + 1 + 1)
                 ), np.zeros((n_iterations, n_size_test, 3 + 1 + 1))
    Xtest_ic_set, ytest_ic_set = np.zeros((n_iterations, n_size_test, NN)),\
        np.zeros((n_iterations, n_size_test, 3 + 1 + 1))
    # recording support vectors
    Xsup_vec_act = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_ctxt = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_xor = {}
    Xsup_vec_bias = {}  # np.zeros((n_iterations,1,NN))
    Xsup_vec_cc = {}
    yevi_set = np.zeros((n_iterations, n_size_test, 3 + 1 + 1))
    test_set = np.zeros((n_iterations, n_size_test))

    yevi_ic_set = np.zeros((n_iterations, n_size_test, 3 + 1 + 1))
    test_ic_set = np.zeros((n_iterations, n_size_test))
    # finding the congruent and incongruent
    idxcongruent = np.where(ytest_pool[:, 2] == ytest_pool[:, 3])[0]
    idxanticong = np.where(ytest_pool[:, 2] != ytest_pool[:, 3])[0]
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # prepare train & test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtrain = np.random.choice(
            np.arange(
                0,
                ytrain_pool.shape[0],
                1),
            size=n_size_train,
            replace=True)
        Xdata_train = Xtrain_pool[idxtrain, :]
        ydata_train = ytrain_pool[idxtrain, :]  # pact,ctxt,bias
        # print("shape x:",Xdata_train.shape,ydata_train.shape)

        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtest = np.random.choice(
            idxcongruent, size=n_size_test, replace=True)
        Xdata_test = Xtest_pool[idxtest, :]
        ydata_test = ytest_pool[idxtest, :]
        test_set[i, :] = idxtest[:]

        idxtest_ic = np.random.choice(
            idxanticong, size=n_size_test, replace=True)
        Xdata_test_ic = Xtest_pool[idxtest_ic, :]
        ydata_test_ic = ytest_pool[idxtest_ic, :]
        test_ic_set[i, :] = idxtest_ic[:]

        # fit model
        # model.fit(X_train,y_train) i.e model.fit(train set, train label as it
        # is a classifier)
        lin_pact.fit(Xdata_train, np.squeeze(ydata_train[:, 0]))
        Xsup_vec_act[i] = lin_pact.support_vectors_

        lin_ctxt.fit(Xdata_train, np.squeeze(ydata_train[:, 1]))
        Xsup_vec_ctxt[i] = lin_ctxt.support_vectors_

        lin_xor.fit(Xdata_train, np.squeeze(ydata_train[:, 2]))
        Xsup_vec_xor[i] = lin_xor.support_vectors_

        lin_bias.fit(Xdata_train, np.squeeze(ydata_train[:, 3]))
        Xsup_vec_bias[i] = lin_bias.support_vectors_

        lin_cc.fit(Xdata_train, np.squeeze(ydata_train[:, 4]))
        Xsup_vec_cc[i] = lin_cc.support_vectors_

        # evaluate model
        predictions = np.zeros((len(idxtest), 3 + 1 + 1))
        predictions[:, 0] = lin_pact.predict(
            Xdata_test)  # model.predict(X_test)
        predictions[:, 1] = lin_ctxt.predict(Xdata_test)
        predictions[:, 2] = lin_xor.predict(Xdata_test)
        predictions[:, 3] = lin_bias.predict(Xdata_test)
        predictions[:, 4] = lin_cc.predict(Xdata_test)
        # print("shape ",Xdata_test.shape,ydata_test.shape,predictions.shape)
        Xtest_set[i, :, :], ytest_set[i, :, :], ypred_set[i, :, :] =\
            Xdata_test[:, :].copy(), ydata_test[:, :].copy(),\
            predictions[:, :].copy()
        Xtest_ic_set[i, :, :], ytest_ic_set[i, :, :] =\
            Xdata_test_ic[:, :].copy(), ydata_test_ic[:, :].copy()

        score = np.zeros((3 + 1 + 1, 1))
        if(type == 'correct'):
            for j in range(np.shape(score)[0]):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        elif(type == 'error'):
            for j in range(np.shape(score)[0]):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        elif(type == 'normal'):
            for j in range(np.shape(score)[0]):
                # accuracy_score(y_test, y_pred)
                score[j, 0] = accuracy_score(
                    ydata_test[:, j], predictions[:, j])
        # caution, overall accuracy score can mislead when classes are
        # imbalanced

        # print(score)
        if i == 0:
            stats = score
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
            # print("shape...",Xdata_test.shape,coeffs.shape,intercepts.shape,coeffs.shape[0])
            yevi_set[i, :, :] = np.squeeze(Xdata_test @ coeffs +
                                           np.repeat(intercepts,
                                                     Xdata_test.shape[0], axis=0))
            yevi_ic_set[i, :, :] = np.squeeze(Xdata_test_ic @ coeffs +
                                              np.repeat(intercepts,
                                                        Xdata_test_ic.shape[0],
                                                        axis=0))
        else:
            stats = np.append(stats, score, axis=1)

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

            yevi_set[i, :, :] = np.squeeze(Xdata_test @ tcoeffs +
                                           np.repeat(tintercepts,
                                                     Xdata_test.shape[0], axis=0))
            yevi_ic_set[i, :, :] = np.squeeze(Xdata_test_ic @ tcoeffs +
                                              np.repeat(tintercepts,
                                                        Xdata_test_ic.shape[0],
                                                        axis=0))
    return stats, coeffs, intercepts, Xtest_set, ytest_set, ypred_set,\
        yevi_set, test_set, Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias,\
        Xsup_vec_cc, Xtest_ic_set, ytest_ic_set, yevi_ic_set, test_ic_set,


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
    Xtest_set, ytest_set, ypred_set =\
        np.zeros((n_iterations, n_size_test, NN)),\
        np.zeros((n_iterations, n_size_test, 3+2)),\
        np.zeros((n_iterations, n_size_test, 3+2))
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
        linw_xor, linb_xor = coeffs_pool[:, i*5+2], intercepts_pool[0, 5*i+2]
        linw_bias, linb_bias = coeffs_pool[:, i*5+3], intercepts_pool[0, 5*i+3]
        linw_cc, linb_cc = coeffs_pool[:, i * 5+4], intercepts_pool[0, 5*i+4]
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
        predictions = np.zeros((len(idxtest), 3 + 2))
        for j in range(3 + 2):
            predictions[np.where(evidences[:, j] > 0)[0], j] = 1
            predictions[np.where(evidences[:, j] <= 0)[0], j] = 0
        # predictions[:,0] = lin_pact.predict(Xdata_test) #model.predict(X_test)
        # predictions[:,1] = lin_ctxt.predict(Xdata_test)
        # predictions[:,2] = lin_bias.predict(Xdata_test)
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


def bootstrap_linsvm_proj_withtrials_congruent(coeffs_pool, intercepts_pool,
                                               Xtest_pool, ytest_pool, type,
                                               n_iterations=10, n_percent=25):
    if(n_percent < 1):
        n_size_test = int(Xtest_pool.shape[0] * n_percent)
    else:
        n_size_test = n_percent
    # Lets run Bootstrap
    NN = Xtest_pool.shape[1]
    stats = list()
    Xtest_set, ytest_set, ypred_set =\
        np.zeros((n_iterations, n_size_test, NN)),\
        np.zeros((n_iterations, n_size_test, 3+2)),\
        np.zeros((n_iterations, n_size_test, 3+2))
    Xtest_ic_set, ytest_ic_set = np.zeros(
        (n_iterations, n_size_test, NN)), np.zeros(
        (n_iterations, n_size_test, 3 + 2))

    yevi_set = np.zeros((n_iterations, n_size_test, 3 + 2))
    test_set = np.zeros((n_iterations, n_size_test))

    yevi_ic_set = np.zeros((n_iterations, n_size_test, 3 + 2))
    test_ic_set = np.zeros((n_iterations, n_size_test))

    # finding the congruent and incongruent
    idxcongruent = np.where(ytest_pool[:, 2] == ytest_pool[:, 3])[0]
    idxanticong = np.where(ytest_pool[:, 2] != ytest_pool[:, 3])[0]

    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # prepare test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtest = np.random.choice(
            idxcongruent, size=n_size_test, replace=True)
        Xdata_test = Xtest_pool[idxtest, :]
        ydata_test = ytest_pool[idxtest, :]
        test_set[i, :] = idxtest[:]

        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtest_ic = np.random.choice(
            idxanticong, size=n_size_test, replace=True)
        Xdata_test_ic = Xtest_pool[idxtest_ic, :]
        ydata_test_ic = ytest_pool[idxtest_ic, :]
        test_ic_set[i, :] = idxtest_ic[:]

        # @YX 0910 -- weights
        linw_pact, linb_pact = coeffs_pool[:, i*5+0], intercepts_pool[0, 5*i+0]
        linw_ctxt, linb_ctxt = coeffs_pool[:, i*5+1], intercepts_pool[0, 5*i+1]
        linw_xor, linb_xor = coeffs_pool[:, i*5+2], intercepts_pool[0, 5*i+2]
        linw_bias, linb_bias = coeffs_pool[:, i*5+3], intercepts_pool[0, 5*i+3]
        linw_cc, linb_cc = coeffs_pool[:, i*5+4], intercepts_pool[0, 5*i+4]
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

        evidences_ic = np.zeros((len(idxtest), 3 + 2))
        evidences_ic[:, 0] = np.squeeze(
            Xdata_test_ic @ linw_pact.reshape(-1, 1) + linb_pact)
        evidences_ic[:, 1] = np.squeeze(
            Xdata_test_ic @ linw_ctxt.reshape(-1, 1) + linb_ctxt)
        evidences_ic[:, 2] = np.squeeze(
            Xdata_test_ic @ linw_xor.reshape(-1, 1) + linb_xor)
        evidences_ic[:, 3] = np.squeeze(
            Xdata_test_ic @ linw_bias.reshape(-1, 1) + linb_bias)
        evidences_ic[:, 4] = np.squeeze(
            Xdata_test_ic @ linw_cc.reshape(-1, 1) + linb_cc)

        predictions = np.zeros((len(idxtest), 3 + 2))
        for j in range(3 + 2):
            predictions[np.where(evidences[:, j] > 0)[0], j] = 1
            predictions[np.where(evidences[:, j] <= 0)[0], j] = 0
        Xtest_set[i, :, :], ytest_set[i, :, :], ypred_set[i, :, :] =\
            Xdata_test[:, :].copy(), ydata_test[:, :].copy(),\
            predictions[:, :].copy()

        Xtest_ic_set[i, :, :], ytest_ic_set[i, :, :] =\
            Xdata_test_ic[:, :].copy(), ydata_test_ic[:, :].copy()

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
            yevi_ic_set[i, :, :] = evidences_ic.copy()
        else:
            # print("shape ...",stats.shape)
            # print("scoreshape...",score.shape)
            stats = np.append(stats, score, axis=1)
            yevi_set[i, :, :] = evidences.copy()
            yevi_ic_set[i, :, :] = evidences_ic.copy()
    return stats, Xtest_set, ytest_set, ypred_set, yevi_set, test_set,\
        Xtest_ic_set, ytest_ic_set, yevi_ic_set, test_ic_set


def bootstrap_linreg_accuracy(Xtrain_pool, ytrain_pool, Xtest_pool, ytest_pool,
                              type, n_iterations=10, n_percent=25):
    # if n_iterations==None:
    # n_iterations = 100  #No. of bootstrap samples to be repeated (created)
    if(n_percent < 1):
        # Size of sample, picking only 50% of the given data in every bootstrap
        # sample
        n_size_train = int(Xtrain_pool.shape[0] * n_percent)
    else:
        n_size_train = n_percent
    # Lets run Bootstrap

    stats = list()
    intercept = list()
    lin_reg = LinearRegression()
    for i in range(n_iterations):
        if (i+1) % PRINT_PER == 0:
            print(i)
        # prepare train & test sets
        # Sampling with replacement..whichever is not used in training data
        # will be used in test data
        idxtrain = np.random.choice(
            np.arange(
                0,
                len(ytrain_pool),
                1),
            size=n_size_train,
            replace=True)
        ydata_train = ytrain_pool[idxtrain, :]
        Xdata_train = np.reshape(Xtrain_pool[idxtrain], (-1, 1))

        # fit model
        # model.fit(X_train,y_train) i.e model.fit(train set, train label as it
        # is a classifier)
        lin_reg.fit(Xdata_train, ydata_train)

        if(type == 'normal'):
            score = lin_reg.score(Xdata_train, ydata_train)
        # caution, overall accuracy score can mislead when classes are
        # imbalanced

        # print(score)
        stats.append(score)
        intercept.append(lin_reg.intercept_)
        if i == 0:
            coeffs = np.reshape(lin_reg.coef_, (-1, 1))
        else:
            coeffs = np.append(
                coeffs, np.reshape(
                    lin_reg.coef_, (-1, 1)), axis=1)
    return stats, coeffs, intercept

# Function for linear regression


def linear_regression(X, Y, xgrid):

    M = np.ones(X.shape)  # Initialize design matrix M
    graph_mat = np.ones(xgrid.shape)  # Initialize matrix for graphing

    M = np.hstack((M, X))  # Build design matrix M
    # Build matrix for graphing predicted y vals
    graph_mat = np.hstack((graph_mat, xgrid))

    # Find least-sq. regression estimate for w using formula
    w = np.linalg.inv(M.T @ M) @ M.T @ Y
    return w, M, graph_mat
