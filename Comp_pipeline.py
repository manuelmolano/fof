# Load packages;
import glob
# import scipy.io as sio
# import time
import numpy as np
import matplotlib.pyplot as plt

# Import libraries
# from collections import Counter
from mpl_toolkits import mplot3d

# from sklearn import datasets, svm, pipeline
# from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
# import os
from sklearn import metrics

# from sklearn.utils import resample  # for Bootstrap sampling
# from sklearn.metrics import accuracy_score
# from scipy import stats
# from sklearn.linear_model import LinearRegression

# self-defined functions
import general_util_ctxtgt as guc
import generate_pseudo_trials as gpt
import required_data_dec as rdd
import bootstrap_linear as bl
# from collections import Counter


image_format = 'svg'  # e.g .png, .svg, etc.
dpii = 300

# %
# import get_matlab_data as gd

# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
RED = np.array((228, 26, 28)) / 255
BLUE = np.array((55, 126, 184)) / 255
ORANGE = np.array((255, 127, 0)) / 255
GREEN = np.array([0, 150, 0])/255
PURPLE = np.array([150, 0, 150])/255


def list_to_dict(lst, string):
    """
    Transform a list of variables into a dictionary.

    Parameters
    ----------
    lst : list
        list with all variables.
    string : str
        string containing the names, separated by commas.

    Returns
    -------
    d : dict
        dictionary with items in which the keys and the values are specified
        in string and lst values respectively.

    """
    string = string[0]
    string = string.replace(']', '')
    string = string.replace('[', '')
    string = string.replace('\\', '')
    string = string.replace(' ', '')
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    string = string.split(',')
    d = {s: v for s, v in zip(string, lst)}
    return d


def multivariateGrid(col_x, col_y, col_k, df, colors=[], alpha=.5, s=2):
    def rgb_to_hex(c):
        c = 255*c
        c = tuple([int(x) for x in c])
        c = '#%02x%02x%02x' % c
        return c

    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = alpha
            kwargs['s'] = s
            kwargs['edgecolor'] = 'none'
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(x=col_x, y=col_y, data=df)
    legends = []
    counter = 0
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        c = rgb_to_hex(colors[counter])
        g.plot_joint(colored_scatter(df_group[col_x], df_group[col_y], c))
        sns.distplot(df_group[col_x].values,
                     ax=g.ax_marg_x, color=c, hist=False)
        sns.distplot(df_group[col_y].values, ax=g.ax_marg_y, color=c, hist=False,
                     vertical=True)
        counter += 1
    # plt.legend(legends)
    return g


def get_all_quantities(files, numtrans=0, SKIPNAN=0):
    icount = 0
    Xdata_set = {}
    ylabels_set = {}
    Xdata_hist_set = {}
    ylabels_hist_set = {}

    metadata = {}
    pseudo_neurons = 0
    remarkfile = ""
    for i in range(len(files)):
        for T in ['correct', 'error']:
            Xdata_set[i, T] = {}
            ylabels_set[i, T] = {}

            Xdata_hist_set[i, T] = {}
            ylabels_hist_set[i, T] = {}

    # nnfiles = np.zeros(len(files))
    files = [f for f in files if f.find('data_dec') == -1]
    for idxs, f in enumerate(files):
        if icount < 0:
            break
        data = np.load(f, allow_pickle=True)
        print('file', f)
        # print('unique stimulus:',np.unique(data['obscategory'][::2]))
        tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc =\
            guc.get_RNNdata_ctxtgt(data)
        # print('responses:',np.shape(data['states']),'; gt',np.shape(data['gt']))
        if(np.shape(data['states'])[0] != np.shape(data['gt'])[0]):
            remarkfile = remarkfile+"; "+f
            continue
        if SKIPNAN == 0:
            stim_trials, idx_effect, ctxt_trials =\
                guc.transform_stim_trials_notskip(data)
        else:
            stim_trials, idx_effect, ctxt_trials =\
                guc.transform_stim_trials_ctxtgt(data)

        icount += 1

        Xdata, ydata, Xdata_idx, Xconds_2, Xacts_1,\
            Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0, Xgts_0,\
            Xcohs_0, Xdata_trialidx, Xstates =\
            rdd.req_quantities_0(stim_trials, stm, dyns, gt, choice, eff_choice,
                                 rw, obsc, BLOCK_CTXT=1)

        Xdata_correct, Xdata_error, correct_trial, error_trial, rses_correct,\
            rses_error, Xrse_6_correct, Xrse_6_error, Xcohs_0_correct,\
            Xcohs_0_error, ydata_bias_correct, ydata_bias_error,\
            ydata_xor_correct,\
            ydata_xor_error, ydata_conds_correct, ydata_conds_error,\
            ydata_choices_correct, ydata_choices_error, ydata_cchoices_correct,\
            ydata_cchoices_error, ydata_cgts_correct, ydata_cgts_error,\
            Xdata_idx_correct, Xdata_idx_error,\
            Xdata_trialidx_correct, Xdata_trialidx_error, ydata_states_correct,\
            ydata_states_error =\
            rdd.sep_correct_error(data['stimulus'], dyns, Xdata, ydata, Xdata_idx,
                                  Xconds_2, Xacts_1, Xrws_1, Xlfs_1, Xrse_6, rses,
                                  Xacts_0, Xgts_0, Xcohs_0, Xdata_trialidx,
                                  Xstates, margin=[1, 2], idd=1)

        ylabels_correct = rdd.set_ylabels(Xdata_correct, ydata_choices_correct,
                                          ydata_conds_correct, ydata_xor_correct,
                                          ydata_bias_correct,
                                          ydata_cchoices_correct, Xcohs_0_correct)
        ylabels_error = rdd.set_ylabels(Xdata_error, ydata_choices_error,
                                        ydata_conds_error, ydata_xor_error,
                                        ydata_bias_error, ydata_cchoices_error,
                                        Xcohs_0_error)

        pseudo_neurons += np.shape(Xdata_correct)[1]

        Xdata_set[idxs, 'correct'], ylabels_set[idxs, 'correct'],\
            Xdata_hist_set[idxs, 'correct'], ylabels_hist_set[idxs, 'correct'] =\
            rdd.State_trials(Xdata_correct, ydata_states_correct,
                             ydata_cchoices_correct, Xcohs_0_correct,
                             ylabels_correct, 0,)
        Xdata_set[idxs, 'error'], ylabels_set[idxs, 'error'],\
            Xdata_hist_set[idxs, 'error'], ylabels_hist_set[idxs, 'error'] =\
            rdd.State_trials(Xdata_error, ydata_states_error,
                             ydata_cchoices_error, Xcohs_0_error,
                             ylabels_error, 0,)
        metadata[idxs] = {'filename': f,
                          'totaltrials': np.shape(Xdata_correct)[0] +
                          np.shape(Xdata_error)[0],
                          'neuronnumber': np.shape(Xdata_correct)[1],
                          'ACtrials': np.shape(Xdata_correct)[0],
                          'AEtrials': np.shape(Xdata_error)[0],
                          }

        Xdata_hist_set[i, T] = {}
        ylabels_hist_set[i, T] = {}

    # nnfiles = np.zeros(len(files))
    files = [f for f in files if f.find('data_dec') == -1]
    for idxs, f in enumerate(files):
        if icount < 0:
            break
        data = np.load(f, allow_pickle=True)
        # print('unique stimulus:',np.unique(data['obscategory'][::2]))
        tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc =\
            guc.get_RNNdata_ctxtgt(data)
        print('file: ', f)
        print('responses:', np.shape(
            data['states']), '; gt', np.shape(data['gt']))
        if(np.shape(data['states'])[0] != np.shape(data['gt'])[0]):
            remarkfile = remarkfile+"; "+f
            continue
        stim_trials, idx_effect, ctxt_trials = guc.transform_stim_trials_ctxtgt(
            data)
        icount += 1

        Xdata, ydata, Xdata_idx, Xconds_2, Xacts_1,\
            Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0, Xgts_0, Xcohs_0,\
            Xdata_trialidx, Xstates =\
            rdd.req_quantities_0(stim_trials, stm, dyns, gt, choice, eff_choice,
                                 rw, obsc, BLOCK_CTXT=1)

        Xdata_correct, Xdata_error, correct_trial, error_trial, rses_correct,\
            rses_error, Xrse_6_correct, Xrse_6_error, Xcohs_0_correct,\
            Xcohs_0_error, ydata_bias_correct, ydata_bias_error,\
            ydata_xor_correct, ydata_xor_error, ydata_conds_correct,\
            ydata_conds_error, ydata_choices_correct, ydata_choices_error,\
            ydata_cchoices_correct, ydata_cchoices_error, ydata_cgts_correct,\
            ydata_cgts_error, Xdata_idx_correct, Xdata_idx_error,\
            Xdata_trialidx_correct, Xdata_trialidx_error, ydata_states_correct,\
            ydata_states_error =\
            rdd.sep_correct_error(data['stimulus'], dyns, Xdata, ydata, Xdata_idx,
                                  Xconds_2, Xacts_1, Xrws_1, Xlfs_1, Xrse_6, rses,
                                  Xacts_0, Xgts_0, Xcohs_0, Xdata_trialidx,
                                  Xstates, margin=[1, 2], idd=1)
        print('Validate trials:', np.shape(Xdata_correct))

        ylabels_correct =\
            rdd.set_ylabels(Xdata_correct, ydata_choices_correct,
                            ydata_conds_correct, ydata_xor_correct,
                            ydata_bias_correct, ydata_cchoices_correct,
                            Xcohs_0_correct)
        ylabels_error =\
            rdd.set_ylabels(Xdata_error, ydata_choices_error, ydata_conds_error,
                            ydata_xor_error, ydata_bias_error,
                            ydata_cchoices_error, Xcohs_0_error)

        pseudo_neurons += np.shape(Xdata_correct)[1]

        Xdata_set[idxs, 'correct'], ylabels_set[idxs, 'correct'],\
            Xdata_hist_set[idxs, 'correct'], ylabels_hist_set[idxs, 'correct'] =\
            rdd.State_trials(Xdata_correct, ydata_states_correct,
                             ydata_cchoices_correct, Xcohs_0_correct,
                             ylabels_correct, 0,)
        Xdata_set[idxs, 'error'], ylabels_set[idxs, 'error'],\
            Xdata_hist_set[idxs, 'error'], ylabels_hist_set[idxs, 'error'] =\
            rdd.State_trials(Xdata_error, ydata_states_error, ydata_cchoices_error,
                             Xcohs_0_error, ylabels_error, 0,)
        metadata[idxs] = {'filename': f,
                          'totaltrials': np.shape(Xdata_correct)[0] +
                          np.shape(Xdata_error)[0],
                          'neuronnumber': np.shape(Xdata_correct)[1],
                          'ACtrials': np.shape(Xdata_correct)[0],
                          'AEtrials': np.shape(Xdata_error)[0],
                          }

    lst = [Xdata_set, Xdata_hist_set,
           ylabels_set, ylabels_hist_set,
           Xcohs_0, files, metadata]
    stg = ["Xdata_set, Xdata_hist_set,"
           "ylabels_set, ylabels_hist_set,"
           "Xcohs_0, files, metadata"]
    d = list_to_dict(lst=lst, string=stg)
    return d


'''
Filtering 'good' sessions
'''


def filter_sessions(data_tr, unique_states, unique_cohs):
    Xdata_set, Xdata_hist_set, ylabels_set, ylabels_hist_set, files =\
        data_tr['Xdata_set'], data_tr['Xdata_hist_set'], data_tr['ylabels_set'],\
        data_tr['ylabels_hist_set'], data_tr['files']
    correct_false, error_false, min_hist_trials, num_hist_trials =\
        gpt.valid_hist_trials(Xdata_hist_set, ylabels_hist_set, unique_states,
                              unique_cohs, files, THRESH_TRIAL)
    _correct_false, _error_false, min_beh_trials, num_beh_trials =\
        gpt.valid_beh_trials(Xdata_set, ylabels_set, unique_states,
                             unique_cohs, files, THRESH_TRIAL)
    false_files = np.union1d(correct_false, error_false)
    false_files = np.union1d(false_files, _error_false)
    false_files = np.union1d(false_files, _correct_false)
    MIN_TRIALS = [min_hist_trials, min_beh_trials]

    return false_files, MIN_TRIALS, num_hist_trials, num_beh_trials


def get_dec_axes(data_tr, wc, bc, we, be, false_files, mode='decoding',
                 DOREVERSE=0, CONTROL=0, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[]):
    # Xdata_set, Xdata_hist_set, ylabels_set, ylabels_hist_set, files =\
    #     data_tr['Xdata_set'], data_tr['Xdata_hist_set'], data_tr['ylabels_set'],\
    #     data_tr['ylabels_hist_set'], data_tr['files']
    Xdata_hist_set, ylabels_hist_set, files = data_tr['Xdata_hist_set'], \
        data_tr['ylabels_hist_set'], data_tr['files']

    Xcohs_0 = data_tr['Xcohs_0']
    unique_states = np.arange(8)
    unique_cohs = np.sort(Xcohs_0)
    Xmerge_hist_trials_correct, ymerge_hist_labels_correct,\
        Xmerge_hist_trials_error, ymerge_hist_labels_error, _ =\
        gpt.merge_pseudo_hist_trials(Xdata_hist_set, ylabels_hist_set,
                                     unique_states, unique_cohs, files,
                                     false_files, 10,  RECORD_TRIALS=1,
                                     RECORDED_TRIALS=[])

    # finding decoding axis
    NN = np.shape(Xmerge_hist_trials_correct[4])[1]
    if(RECORD_TRIALS == 1):
        RECORDED_TRIALS_SET = {}
        for itr in range(NITERATIONS):
            RECORDED_TRIALS_SET[itr] = {}
    if(RECORD_TRIALS == 1):
        coeffs, intercepts, sup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias,\
            Xsup_vec_cc, Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
            Xtest_set_error, ytest_set_error, yevi_set_error, RECORDED_TRIALS_SET\
            = bl.bootstrap_linsvm_step(Xdata_hist_set, NN, ylabels_hist_set,
                                       unique_states, unique_cohs, files,
                                       false_files, type, DOREVERSE=DOREVERSE,
                                       CONTROL=CONTROL, n_iterations=NITERATIONS,
                                       N_pseudo_dec=NPSEUDODEC,
                                       train_percent=PERCENTTRAIN,
                                       RECORD_TRIALS=RECORD_TRIALS,
                                       RECORDED_TRIALS_SET=RECORDED_TRIALS_SET)
    else:
        coeffs, intercepts, Xtest_set_correct, ytest_set_correct,\
            yevi_set_correct, Xtest_set_error, ytest_set_error, yevi_set_error,\
            RECORDED_TRIALS_SET\
            = bl.bootstrap_linsvm_proj_step(wc, bc, Xdata_hist_set, NN,
                                            ylabels_hist_set, unique_states,
                                            unique_cohs, files, false_files, type,
                                            DOREVERSE=DOREVERSE,
                                            n_iterations=NITERATIONS,
                                            N_pseudo_dec=NPSEUDODEC,
                                            train_percent=PERCENTTRAIN,
                                            RECORD_TRIALS=RECORD_TRIALS,
                                            RECORDED_TRIALS_SET=RECORDED_TRIALS_SET)

    lst = [coeffs, intercepts,
           ytest_set_correct,
           yevi_set_correct,
           coeffs, intercepts,  # Xtest_set_error,
           ytest_set_error,  yevi_set_error,
           RECORDED_TRIALS_SET]
    stg = ["coefs_correct, intercepts_correct,"
           "ytest_set_correct, "
           "yevi_set_correct, "
           "coefs_error, intercepts_error,"  # " Xtest_set_error,"
           "ytest_set_error, yevi_set_error,"
           "RECORDED_TRIALS_SET"]
    d = list_to_dict(lst=lst, string=stg)
    return d


def flatten_data(data_tr, data_dec):
    yevi_set_correct = data_dec['yevi_set_correct']
    ytest_set_correct = data_dec['ytest_set_correct']
    IPOOLS = NITERATIONS
    # flatten data --- correct
    nlabels = np.shape(np.squeeze(ytest_set_correct[0, :, :]))[1]
    ytruthlabels_c = np.zeros((nlabels, 1))
    yevi_c = np.zeros((3 + 1 + 1, 1))
    dprimes_c = np.zeros(IPOOLS)
    dprimes_repc = np.zeros(IPOOLS)
    dprimes_altc = np.zeros(IPOOLS)
    AUCs_c = np.zeros(IPOOLS)
    AUCs_repc = np.zeros(IPOOLS)
    AUCs_altc = np.zeros(IPOOLS)
    for i in range(IPOOLS):
        # bootstrappin
        hist_evi = yevi_set_correct[i, :, :]
        test_labels = ytest_set_correct[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        ytruthlabels_c = np.append(
            ytruthlabels_c, test_labels[idx, :].T, axis=1)
        yevi_c = np.append(yevi_c, (yevi_set_correct[i, idx, :]).T, axis=1)
        dprimes_c[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, SVMAXIS]),
                                 np.squeeze(ytest_set_correct[i, :, SVMAXIS]))

        yauc_c_org = np.squeeze(ytest_set_correct[i, :, SVMAXIS])
        yauc_c = np.zeros_like(yauc_c_org)

        yauc_c[np.where(yauc_c_org == 0+2)[0]] = 1
        yauc_c[np.where(yauc_c_org == 1+2)[0]] = 2
        assert (yauc_c != 0).all()
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_c, np.squeeze(yevi_set_correct[i, :, SVMAXIS]), pos_label=2)
        auc_ac = metrics.auc(fpr, tpr)
        AUCs_c[i] = auc_ac

        # SEPARATE REP AND ALT CONTEXTS
        ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
            np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        yauc_c_ctxtrep, yauc_c_ctxtalt =\
            np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        yauc_c_evirep, yauc_c_evialt =\
            np.squeeze(yevi_set_correct[i, ctxtrep, SVMAXIS]),\
            np.squeeze(yevi_set_correct[i, ctxtalt, SVMAXIS])
        dprimes_repc[i] = guc.calculate_dprime(yauc_c_evirep, yauc_c_ctxtrep)
        dprimes_altc[i] = guc.calculate_dprime(yauc_c_evialt, yauc_c_ctxtalt)

        yauc_c_ctxtrep, yauc_c_ctxtalt = yauc_c_ctxtrep-1, yauc_c_ctxtalt-1

        fpr_rep, tpr_rep, thresholds = metrics.roc_curve(
            yauc_c_ctxtrep, yauc_c_evirep, pos_label=2)
        auc_ac_rep = metrics.auc(fpr_rep, tpr_rep)
        AUCs_repc[i] = auc_ac_rep

        fpr_alt, tpr_alt, thresholds = metrics.roc_curve(
            yauc_c_ctxtalt, yauc_c_evialt, pos_label=2)
        auc_ac_alt = metrics.auc(fpr_alt, tpr_alt)
        AUCs_altc[i] = auc_ac_alt

    ytruthlabels_c, yevi_c = ytruthlabels_c[:, 1:], yevi_c[:, 1:]
    f, ax_temp = plt.subplots(ncols=2)
    ax_temp[0].hist(AUCs_c, bins=20, alpha=0.9, facecolor='yellow')

    '''
    After Error Trials
    '''
    yevi_set_error = data_dec['yevi_set_error']
    ytest_set_error = data_dec['ytest_set_error']

    nlabels = np.shape(np.squeeze(ytest_set_error[0, :, :]))[1]
    ytruthlabels_e = np.zeros((nlabels, 1))
    yevi_e = np.zeros((3 + 1 + 1, 1))
    dprimes_e = np.zeros(IPOOLS)
    dprimes_repe = np.zeros(IPOOLS)
    dprimes_alte = np.zeros(IPOOLS)
    AUCs_e = np.zeros(IPOOLS)
    AUCs_repe = np.zeros(IPOOLS)
    AUCs_alte = np.zeros(IPOOLS)
    for i in range(IPOOLS):
        hist_evi = yevi_set_error[i, :, :]
        test_labels = ytest_set_error[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        ytruthlabels_e = np.append(
            ytruthlabels_e, test_labels[idx, :].T, axis=1)
        yevi_e = np.append(yevi_e, (yevi_set_error[i, idx, :]).T, axis=1)
        dprimes_e[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, SVMAXIS]),
                                 np.squeeze(ytest_set_error[i, :, SVMAXIS]))
        yauc_e_org = np.squeeze(ytest_set_error[i, :, SVMAXIS])
        yauc_e = np.zeros_like(yauc_e_org)

        yauc_e[np.where(yauc_e_org == 0)[0]] = 1
        yauc_e[np.where(yauc_e_org == 1)[0]] = 2
        assert (yauc_c != 0).all()
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_e, np.squeeze(yevi_set_error[i, :, SVMAXIS]), pos_label=2)
        auc_ae = metrics.auc(fpr, tpr)
        AUCs_e[i] = auc_ae

        # SEPARATE REP AND ALT CONTEXTS
        ctxtrep, ctxtalt = np.where(ytest_set_error[i, :, 1] == 0)[
            0], np.where(ytest_set_error[i, :, 1] == 1)[0]
        yauc_e_ctxtrep, yauc_e_ctxtalt =\
            np.squeeze(ytest_set_error[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_error[i, ctxtalt, SVMAXIS])
        yauc_e_evirep, yauc_e_evialt =\
            np.squeeze(yevi_set_error[i, ctxtrep, SVMAXIS]),\
            np.squeeze(yevi_set_error[i, ctxtalt, SVMAXIS])
        dprimes_repe[i] = guc.calculate_dprime(yauc_e_evirep, yauc_e_ctxtrep)
        dprimes_alte[i] = guc.calculate_dprime(yauc_e_evialt, yauc_e_ctxtalt)

        yauc_e_ctxtrep, yauc_e_ctxtalt = yauc_e_ctxtrep+1, yauc_e_ctxtalt+1

        fpr_rep, tpr_rep, thresholds = metrics.roc_curve(
            yauc_e_ctxtrep, yauc_e_evirep, pos_label=2)
        auc_ae_rep = metrics.auc(fpr_rep, tpr_rep)
        AUCs_repe[i] = auc_ae_rep

        fpr_alt, tpr_alt, thresholds = metrics.roc_curve(
            yauc_e_ctxtalt, yauc_e_evialt, pos_label=2)
        auc_ae_alt = metrics.auc(fpr_alt, tpr_alt)
        AUCs_alte[i] = auc_ae_alt

    ax_temp[1].hist(AUCs_e, bins=20, alpha=0.9, facecolor='black')

    ytruthlabels_e, yevi_e = ytruthlabels_e[:, 1:], yevi_e[:, 1:]
    lst = [ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,
           dprimes_c, dprimes_e, AUCs_c, AUCs_e,
           dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte,
           AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte]
    stg = ["ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,"
           "dprimes_c, dprimes_e, AUCs_c, AUCs_e, "
           "dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte, "
           "AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte"]
    d = list_to_dict(lst=lst, string=stg)
    return d

# visualizing the results


def projection_3D(data_flt, data_flt_light, prev_outc):
    ytruthlabels_c = data_flt['ytruthlabels_'+prev_outc]
    yevi_c = data_flt['yevi_'+prev_outc]
    ridx = np.random.choice(np.arange(len(yevi_c[1, :])),
                            size=200, replace=False)
    ridx = ridx.astype(np.int32)

    # RESAMPLE THE CONGRUENT TRIALS

    ridx_congruent =\
        np.where(ytruthlabels_c[2, :] == ytruthlabels_c[SVMAXIS, :])[0]
    ridx = np.random.choice(
        ridx_congruent, size=int(NUM_SAMPLES), replace=False)

    fig = plt.figure()  # XXX: this was in line 352 (after x, y, z = ...)
    ax = fig.add_subplot(111, projection='3d')
    # --- PLOTING CONGRUENT TRIALS, WITH CLEAR TRANSITIONS
    x, y, z = yevi_c[1, ridx], yevi_c[0, ridx], yevi_c[3, ridx]
    cms = []
    for i in ridx:
        if(ytruthlabels_c[3, i] == AX_PREV_CH_OUTC[prev_outc][0]):
            cms.append(GREEN)
        else:
            cms.append(PURPLE)
    ax.scatter(x, y, z, s=S_PLOTS, c=cms, alpha=0.9, zorder=0)
    zflat = np.full_like(z, BOTTOM_3D)  # min(ax.get_zlim()))
    ytruthlabels_c = np.array((ytruthlabels_c).copy().astype(np.int32))
    # two projections
    idxright = np.where(
        ytruthlabels_c[0, ridx] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    idxleft = np.where(ytruthlabels_c[0, ridx]
                       == AX_PREV_CH_OUTC[prev_outc][0])[0]
    igreen, iblue =\
        np.where(ytruthlabels_c[3, ridx[idxleft]] ==
                 AX_PREV_CH_OUTC[prev_outc][0])[0],\
        np.where(ytruthlabels_c[3, ridx[idxleft]] ==
                 AX_PREV_CH_OUTC[prev_outc][1])[0]
    ax.scatter(np.mean(x[idxleft[igreen]]), np.mean(y[idxleft[igreen]]), np.mean(
        z[idxleft[igreen]]), s=100, c=GREEN, edgecolor='k', zorder=1)
    ax.plot(np.mean(x[idxleft[igreen]])*np.ones(2), np.mean(y[idxleft[igreen]]) *
            np.ones(2), [zflat[0], np.mean(z[idxleft[igreen]])], 'k-', zorder=1)
    ax.scatter(np.mean(x[idxleft[iblue]]), np.mean(y[idxleft[iblue]]), np.mean(
        z[idxleft[iblue]]), s=100, c=PURPLE, edgecolor='k', zorder=1)
    ax.plot(np.mean(x[idxleft[iblue]])*np.ones(2), np.mean(y[idxleft[iblue]]) *
            np.ones(2), [zflat[0], np.mean(z[idxleft[iblue]])], 'k-', zorder=1)

    ibluehist, igreenhist = idxleft[iblue], idxleft[igreen]

    igreen, iblue =\
        np.where(ytruthlabels_c[3, ridx[idxright]] ==
                 AX_PREV_CH_OUTC[prev_outc][0])[0],\
        np.where(ytruthlabels_c[3, ridx[idxright]] ==
                 AX_PREV_CH_OUTC[prev_outc][1])[0]
    ax.scatter(np.mean(x[idxright[igreen]]), np.mean(y[idxright[igreen]]),
               np.mean(z[idxright[igreen]]), s=100, c=GREEN, edgecolor='k',
               zorder=1)
    ax.plot(np.mean(x[idxright[igreen]])*np.ones(2),
            np.mean(y[idxright[igreen]])*np.ones(2),
            [zflat[0], np.mean(z[idxright[igreen]])], 'k-', zorder=1)
    ax.scatter(np.mean(x[idxright[iblue]]), np.mean(y[idxright[iblue]]),
               np.mean(z[idxright[iblue]]), s=100, c=PURPLE, edgecolor='k',
               zorder=1)
    ax.plot(np.mean(x[idxright[iblue]])*np.ones(2), np.mean(y[idxright[iblue]]) *
            np.ones(2), [zflat[0], np.mean(z[idxright[iblue]])], 'k-', zorder=1)

    # histogram side
    fig2dd, ax2dd = plt.subplots(figsize=(6, 3))
    zrange = np.linspace(-10, 6, 30)
    ibluehist = np.append(ibluehist, idxright[iblue])
    igreenhist = np.append(igreenhist, idxright[igreen])
    ax2dd.hist(z[ibluehist], bins=zrange, density=True, facecolor=PURPLE,
               alpha=0.9)
    ax2dd.hist(z[igreenhist], bins=zrange, density=True, facecolor=GREEN,
               alpha=0.9)

    igreen, iblue =\
        np.where(ytruthlabels_c[1, ridx] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
        np.where(ytruthlabels_c[1, ridx] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    ax.scatter(x[igreen], y[igreen], zflat[igreen],
               s=S_PLOTS, c=BLUE, alpha=0.9)
    ax.scatter(x[iblue], y[iblue], zflat[iblue], s=S_PLOTS, c=RED, alpha=0.9)


def projections_2D(data_flt, prev_outc, fit=False, name=''):
    ytruthlabels = data_flt['ytruthlabels_'+prev_outc]
    yevi = data_flt['yevi_'+prev_outc]
    '''
    Four conditions (four clouds)
    '''
    idxprel = np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][0])[0]
    idxctxtr = np.where(ytruthlabels[1, :] == AX_PREV_CH_OUTC[prev_outc][0])[0]

    idxprer = np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    idxctxta = np.where(ytruthlabels[1, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]

    idxprelctxtr = np.intersect1d(idxprel, idxctxtr)
    idxprelctxta = np.intersect1d(idxprel, idxctxta)
    idxprerctxtr = np.intersect1d(idxprer, idxctxtr)
    idxprerctxta = np.intersect1d(idxprer, idxctxta)

    idxsample = np.zeros((4, NUM_SAMPLES), dtype=int)
    idxsample[0, :] = idxprelctxtr[:NUM_SAMPLES]
    idxsample[1, :] = idxprelctxta[:NUM_SAMPLES]
    idxsample[2, :] = idxprerctxtr[:NUM_SAMPLES]
    idxsample[3, :] = idxprerctxta[:NUM_SAMPLES]

    idxpreal, idxprear = np.union1d(idxsample[0, :], idxsample[1, :]), np.union1d(
        idxsample[2, :], idxsample[3, :])

    idxctxtr, idxctxta = np.union1d(idxsample[0, :], idxsample[2, :]), np.union1d(
        idxsample[1, :], idxsample[3, :])

    # -------- context versus tr. bias ----------------
    # plot samples
    # previous left
    # np.random.choice(idxpreal, size=NUM_SAMPLES, replace=False)
    # idxleft = idxpreal
    # np.random.choice(idxprear, size=NUM_SAMPLES, replace=False)
    # idxright = idxprear
    # figs = []
    for idx, prev_ch in zip([idxpreal, idxprear], ['Left', 'Right']):
        ctxt = np.squeeze(yevi[1, idx])
        tr_bias = np.squeeze(yevi[SVMAXIS, idx])
        df = {'Context encoding': ctxt, 'Transition bias encoding': tr_bias,
              'Upcoming Stimulus Category': ytruthlabels[SVMAXIS, idx]}
        df = pd.DataFrame(df)
        fig = multivariateGrid(col_x='Context encoding',
                               col_y='Transition bias encoding',
                               col_k='Upcoming Stimulus Category', df=df,
                               colors=[GREEN, PURPLE], s=S_PLOTS, alpha=.75)
        fig.ax_marg_x.set_xlim(XLIMS_2D)
        fig.ax_marg_y.set_ylim(YLIMS_2D)
        fig.ax_joint.axhline(y=0, color='k', linestyle='--', lw=0.5)
        fig.fig.suptitle('a'+prev_outc+' / Prev. Ch. '+prev_ch)
        if prev_outc == 'c':
            fig.ax_joint.set_yticks(YTICKS_2D)
        else:
            fig.ax_joint.set_yticks([])
            fig.ax_joint.set_ylabel('')
        fig.ax_joint.set_xticks(XTICKS_2D)
        fig.fig.set_figwidth(3)
        fig.fig.set_figheight(3)
        # fit
        if fit:
            coefficients = np.polyfit(ctxt, tr_bias, 1)
            poly = np.poly1d(coefficients)
            new_y = poly([np.min(ctxt), np.max(ctxt)])
            fig.ax_joint.plot([np.min(ctxt), np.max(ctxt)], new_y, color='k',
                              lw=0.5)

    # --------- previous ch. versus tr. bias
    idxrpt = idxctxtr
    idxalt = idxctxta
    # figs = []
    for idx, ctxt in zip([idxrpt, idxalt], ['Rep', 'Alt']):
        prev_ch = np.squeeze(yevi[0, idx])
        tr_bias = np.squeeze(yevi[SVMAXIS, idx])
        df = {'Prev ch. encoding': prev_ch, 'Transition bias encoding': tr_bias,
              'Upcoming Stimulus Category': ytruthlabels[SVMAXIS, idx]}
        df = pd.DataFrame(df)
        fig = multivariateGrid(col_x='Prev ch. encoding',
                               col_y='Transition bias encoding',
                               col_k='Upcoming Stimulus Category', df=df,
                               colors=[GREEN, PURPLE], s=S_PLOTS, alpha=.75)
        fig.ax_marg_x.set_xlim(XLIMS_2D)
        fig.ax_marg_y.set_ylim(YLIMS_2D)
        fig.ax_joint.axhline(y=0, color='k', linestyle='--', lw=0.5)
        fig.fig.suptitle('a'+prev_outc+' / Ctxt. '+ctxt)
        if prev_outc == 'c':
            fig.ax_joint.set_yticks(YTICKS_2D)
        else:
            fig.ax_joint.set_yticks([])
            fig.ax_joint.set_ylabel('')
        fig.ax_joint.set_xticks(XTICKS_2D)
        fig.fig.set_figwidth(3)
        fig.fig.set_figheight(3)
        # fit
        if fit:
            coefficients = np.polyfit(prev_ch, tr_bias, 1)
            poly = np.poly1d(coefficients)
            new_y = poly([np.min(prev_ch), np.max(prev_ch)])
            fig.ax_joint.plot([np.min(prev_ch), np.max(prev_ch)], new_y, color='k',
                              lw=0.5)

    # # -------- context versus tr. bias ----------------
    # idxpreal, idxprear =\
    #     np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
    #     np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    # # idxbiasl, idxbiasr =\
    # #     np.where(ytruthlabels[3, :] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
    # #     np.where(ytruthlabels[3, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]

    # # plot samples
    # # previous left
    # # np.random.choice(idxpreal, size=NUM_SAMPLES, replace=False)
    # idxleft = idxpreal[:NUM_SAMPLES]
    # idxpreal = idxleft
    # # np.random.choice(idxprear, size=NUM_SAMPLES, replace=False)
    # idxright = idxprear[:NUM_SAMPLES]
    # idxprear = idxright
    # # figs = []
    # for idx, prev_ch in zip([idxpreal, idxprear], ['Left', 'Right']):
    #     ctxt = np.squeeze(yevi[1, idx])
    #     tr_bias = np.squeeze(yevi[SVMAXIS, idx])
    #     df = {'Context encoding': ctxt, 'Transition bias encoding': tr_bias,
    #           'Upcoming Stimulus Category': ytruthlabels[SVMAXIS, idx]}
    #     df = pd.DataFrame(df)
    #     fig = multivariateGrid(col_x='Context encoding',
    #                            col_y='Transition bias encoding',
    #                            col_k='Upcoming Stimulus Category', df=df,
    #                            colors=[GREEN, PURPLE], s=S_PLOTS, alpha=.75)
    #     fig.ax_marg_x.set_xlim(XLIMS_2D)
    #     fig.ax_marg_y.set_ylim(YLIMS_2D)
    #     fig.ax_joint.axhline(y=0, color='k', linestyle='--', lw=0.5)
    #     fig.fig.suptitle('a'+prev_outc+' / Prev. Ch. '+prev_ch)
    #     if prev_outc == 'c':
    #         fig.ax_joint.set_yticks(YTICKS_2D)
    #     else:
    #         fig.ax_joint.set_yticks([])
    #         fig.ax_joint.set_ylabel('')
    #     fig.ax_joint.set_xticks(XTICKS_2D)
    #     fig.fig.set_figwidth(4)
    #     fig.fig.set_figheight(4)
    #     # fit
    #     if fit:
    #         coefficients = np.polyfit(ctxt, tr_bias, 1)
    #         poly = np.poly1d(coefficients)
    #         new_y = poly([np.min(ctxt), np.max(ctxt)])
    #         fig.ax_joint.plot([np.min(ctxt), np.max(ctxt)], new_y, color='k',
    #                           lw=0.5)

    # # --------- previous ch. versus tr. bias
    # idxctxtr, idxctxta =\
    #     np.where(ytruthlabels[1, :] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
    #     np.where(ytruthlabels[1, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    # # idxbiasl, idxbiasr =\
    # #     np.where(ytruthlabels[3, :] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
    # #     np.where(ytruthlabels[3, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]

    # # plot samples
    # # previous left
    # # np.random.choice(idxpreal, size=NUM_SAMPLES, replace=False)
    # idxrpt = idxctxtr[:NUM_SAMPLES]
    # idxrpt = idxrpt
    # # np.random.choice(idxprear, size=NUM_SAMPLES, replace=False)
    # idxalt = idxctxta[:NUM_SAMPLES]
    # idxalt = idxalt
    # # figs = []
    # for idx, ctxt in zip([idxrpt, idxalt], ['Rep', 'Alt']):
    #     prev_ch = np.squeeze(yevi[0, idx])
    #     tr_bias = np.squeeze(yevi[SVMAXIS, idx])
    #     df = {'Prev ch. encoding': prev_ch, 'Transition bias encoding': tr_bias,
    #           'Upcoming Stimulus Category': ytruthlabels[SVMAXIS, idx]}
    #     df = pd.DataFrame(df)
    #     fig = multivariateGrid(col_x='Prev ch. encoding',
    #                            col_y='Transition bias encoding',
    #                            col_k='Upcoming Stimulus Category', df=df,
    #                            colors=[GREEN, PURPLE], s=S_PLOTS, alpha=.75)
    #     fig.ax_marg_x.set_xlim(XLIMS_2D)
    #     fig.ax_marg_y.set_ylim(YLIMS_2D)
    #     fig.ax_joint.axhline(y=0, color='k', linestyle='--', lw=0.5)
    #     fig.fig.suptitle('a'+prev_outc+' / Ctxt. '+ctxt)
    #     if prev_outc == 'c':
    #         fig.ax_joint.set_yticks(YTICKS_2D)
    #     else:
    #         fig.ax_joint.set_yticks([])
    #         fig.ax_joint.set_ylabel('')
    #     fig.ax_joint.set_xticks(XTICKS_2D)
    #     fig.fig.set_figwidth(3)
    #     fig.fig.set_figheight(3)
    #     # fit
    #     if fit:
    #         coefficients = np.polyfit(prev_ch, tr_bias, 1)
    #         poly = np.poly1d(coefficients)
    #         new_y = poly([np.min(prev_ch), np.max(prev_ch)])
    # fig.ax_joint.plot([np.min(prev_ch), np.max(prev_ch)], new_y,
    #                   color='k',lw=0.5)

    # # # plot histograms
    # # binsset = np.linspace(-8, 8, 40)
    # # fig, axs = plt.subplots(figsize=(4, 3))
    # # # We can also normalize our inputs by the total number of counts
    # # axs.hist(yevi[SVMAXIS, idxbiasl], bins=binsset,
    # #          density=True, facecolor=GREEN, alpha=0.25)
    # # axs.hist(yevi[SVMAXIS, idxbiasr], bins=binsset,
    # #          density=True, facecolor='tab:purple', alpha=0.25)
    # # axs.set_ylim([0, 0.5])
    # # y = np.zeros((yevi.shape[1],))
    # # y[idxbiasl] = 1
    # # y[idxbiasr] = 2
    # # assert (y != 0).all()
    # # fpr, tpr, thresholds = metrics.roc_curve(y, yevi[SVMAXIS, :], pos_label=2)
    # # AUC = metrics.auc(fpr, tpr)
    # # axs.set_title('AUC: '+str(np.round(AUC, 3)))
    # # image_name = SAVELOC + '/'+prev_outc+'bias_hist_' + NAME + name + '.svg'
    # # fig.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
    # # plt.close(fig)
    # # if PREV_CH == 'L':
    # #     plt.close(figs[1].fig)
    # #     return figs[0]
    # # else:
    # #     plt.close(figs[0].fig)
    # #     return figs[1]


def ctxtbin_defect(data_flt):
    # all trials
    ytruthlabels_c = data_flt['ytruthlabels_c']
    yevi_c = data_flt['yevi_c']

    ytruthlabels_e = data_flt['ytruthlabels_e']
    yevi_e = data_flt['yevi_e']

    # fig2, ax2 = plt.subplots(7, 2, figsize=(12, 12), sharex=True,
    #                          sharey=True, tight_layout=True)
    nbins = len(CTXT_BIN)
    Tbias_ctxt_c, Tbias_ctxt_e = {}, {}
    Tbias_ctxt_clabel, Tbias_ctxt_elabel = {}, {}
    dprime_ctxtdp_c, dprime_ctxtdp_e = np.zeros(nbins), np.zeros(nbins)
    ACC_correct, ACC_error = np.zeros(nbins), np.zeros(nbins)
    ctxt_evi_c = np.abs(yevi_c[1, :])
    ctxt_evi_e = np.abs(yevi_e[1, :])
    # binss = np.linspace(-4.0, 4.0, 40)
    for i in range(0, nbins):
        if i == nbins-1:
            idx_c = np.where(ctxt_evi_c > CTXT_BIN[i])[0]
        else:
            idx_c = np.intersect1d(np.where(ctxt_evi_c > CTXT_BIN[i])[
                                   0], np.where(ctxt_evi_c < CTXT_BIN[i+1])[0])

        Tbias_ctxt_c[i] = (yevi_c[SVMAXIS, idx_c])
        Tbias_ctxt_clabel[i] = (ytruthlabels_c[SVMAXIS, idx_c])
        # ROC
        # True Positive (TP): predict a label of 1 (positive), the true label is 1
        TP = np.sum(np.logical_and(Tbias_ctxt_c[i] > 0,
                                   Tbias_ctxt_clabel[i] == 3))
        # True Negative (TN): predict a label of 0 (negative), the true label is 0
        TN = np.sum(np.logical_and(
            Tbias_ctxt_c[i] <= 0, Tbias_ctxt_clabel[i] == 2))
        # False Positive (FP): predict a label of 1 (positive), the true label is 0
        FP = np.sum(np.logical_and(
            Tbias_ctxt_c[i] > 0,  Tbias_ctxt_clabel[i] == 2))
        # False Negative (FN): predict a label of 0 (negative), the true label is 1
        FN = np.sum(np.logical_and(
            Tbias_ctxt_c[i] <= 0, Tbias_ctxt_clabel[i] == 3))
        ACC_correct[i] = (TP+TN)/(TP+TN+FP+FN)
        # compute dprime
        dprime_ctxtdp_c[i-1] = guc.calculate_dprime(
            Tbias_ctxt_c[i], Tbias_ctxt_clabel[i])
        if i == nbins-1:
            idx_e = np.where(ctxt_evi_c > CTXT_BIN[i])[0]
        else:
            idx_e = np.intersect1d(np.where(ctxt_evi_e > CTXT_BIN[i])[
                                   0], np.where(ctxt_evi_e < CTXT_BIN[i+1])[0])
        Tbias_ctxt_e[i] = (yevi_e[SVMAXIS, idx_e])
        # print(">>>>>>>ebias:",np.mean(np.abs(Tbias_ctxt_e[i])))
        Tbias_ctxt_elabel[i] = (ytruthlabels_e[SVMAXIS, idx_e])
        # True Positive (TP): predict a label of 1 (positive), the true label is 1
        TP = np.sum(np.logical_and(
            Tbias_ctxt_e[i] > 0,  Tbias_ctxt_elabel[i] == 1))
        # True Negative (TN): predict a label of 0 (negative), the true label is 0
        TN = np.sum(np.logical_and(
            Tbias_ctxt_e[i] <= 0, Tbias_ctxt_elabel[i] == 0))
        # False Positive (FP): predict a label of 1 (positive), the true label is 0
        FP = np.sum(np.logical_and(
            Tbias_ctxt_e[i] > 0,  Tbias_ctxt_elabel[i] == 0))
        # False Negative (FN): predict a label of 0 (negative), the true label is 1
        FN = np.sum(np.logical_and(
            Tbias_ctxt_e[i] <= 0, Tbias_ctxt_elabel[i] == 1))
        ACC_error[i] = (TP+TN)/(TP+TN+FP+FN)
        dprime_ctxtdp_e[i] = guc.calculate_dprime(
            Tbias_ctxt_e[i], Tbias_ctxt_elabel[i])
        # # ploting histogram
        # ax2[i][0].hist(Tbias_ctxt_c[i][np.where(Tbias_ctxt_clabel[i] == 2)[
        #                0]], bins=binss, facecolor=GREEN, alpha=0.5)
        # ax2[i][0].hist(Tbias_ctxt_c[i][np.where(Tbias_ctxt_clabel[i] == 3)[
        #                0]], bins=binss, facecolor='tab:purple', alpha=0.5)

        # ax2[i][1].hist(Tbias_ctxt_e[i][np.where(Tbias_ctxt_elabel[i] == 2-2)[0]],
        #                bins=binss, facecolor=GREEN, alpha=0.5)
        # ax2[i][1].hist(Tbias_ctxt_e[i][np.where(Tbias_ctxt_elabel[i] == 3-2)[0]],
        #                bins=binss, facecolor='tab:purple', alpha=0.5)

    # image_name = SAVELOC + '/hist_ctxtbin_'+METHOD+'.svg'
    # fig2.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
    # plt.close(fig2)

    # axacc.plot((CTXT_BIN[:-1]+CTXT_BIN[1:])/2.0,
    #            ACC_correct[:-1], lw=1.5, color='yellow', alpha=0.75)
    # axacc.plot((CTXT_BIN[:-1]+CTXT_BIN[1:])/2.0,
    #            ACC_error[:-1], lw=1.5, color='black', alpha=0.75)

    # calculate Pearson's correlation
    # AC trials
    prechL_AC, prechR_AC = np.where(ytruthlabels_c[0, :] == 2)[0],\
        np.where(ytruthlabels_c[0, :] == 3)[0]
    xl_ctxt_AC, xr_ctxt_AC = yevi_c[1, prechL_AC], yevi_c[1, prechR_AC]
    yl_tbias_AC, yr_tbias_AC = yevi_c[SVMAXIS, prechL_AC],\
        yevi_c[SVMAXIS, prechR_AC]

    corrl_ac = np.mean((xl_ctxt_AC-np.mean(xl_ctxt_AC)) *
                       (yl_tbias_AC-np.mean(yl_tbias_AC)))
    corrl_ac = corrl_ac/(np.std(xl_ctxt_AC)*np.std(yl_tbias_AC))
    corrr_ac = np.mean((xr_ctxt_AC-np.mean(xr_ctxt_AC)) *
                       (yr_tbias_AC-np.mean(yr_tbias_AC)))
    corrr_ac = corrr_ac/(np.std(xr_ctxt_AC)*np.std(yr_tbias_AC))

    # AE trials
    prechL_AE, prechR_AE = np.where(ytruthlabels_e[0, :] == 2-2)[0],\
        np.where(ytruthlabels_e[0, :] == 3-2)[0]
    xl_ctxt_AE, xr_ctxt_AE = yevi_e[1, prechL_AE], yevi_e[1, prechR_AE]
    yl_tbias_AE, yr_tbias_AE = yevi_e[SVMAXIS, prechL_AE],\
        yevi_e[SVMAXIS, prechR_AE]

    corrl_ae = np.mean((xl_ctxt_AE-np.mean(xl_ctxt_AE)) *
                       (yl_tbias_AE-np.mean(yl_tbias_AE)))
    corrl_ae = corrl_ae/(np.std(xl_ctxt_AE)*np.std(yl_tbias_AE))
    corrr_ae = np.mean((xr_ctxt_AE-np.mean(xr_ctxt_AE)) *
                       (yr_tbias_AE-np.mean(yr_tbias_AE)))
    corrr_ae = corrr_ae/(np.std(xr_ctxt_AE)*np.std(yr_tbias_AE))

    return corrl_ac, corrr_ac, corrl_ae, corrr_ae, [ACC_correct[:-1],
                                                    ACC_error[:-1]]


def bias_VS_prob(data_tr, data_dec, unique_cohs, num_beh_trials, EACHSTATES,
                 NITERATIONS, ax, RECORD_TRIALS=1, RECORDED_TRIALS_SET=[]):
    Xdata_set, ylabels_set = data_tr['Xdata_set'], data_tr['ylabels_set']
    metadata = data_tr['metadata']
    coeffs, intercepts = data_dec['coefs_correct'], data_dec['intercepts_correct']
    if (RECORD_TRIALS == 1):
        RECORDED_TRIALS_SET = {}
        for i in range(NITERATIONS):
            RECORDED_TRIALS_SET[i] = {}

    FIX_TRBIAS_BINS = np.array([-1, 0, 1])

    NBINS = 5

    # NTRBIAS = len(FIX_TRBIAS_BINS)
    psychometric_trbias_correct = np.zeros(
        (NITERATIONS, len(unique_cohs), NBINS))
    psychometric_trbias_error = np.zeros(
        (NITERATIONS, len(unique_cohs), NBINS))
    trbias_range_correct = np.zeros((NITERATIONS, len(unique_cohs), NBINS))
    trbias_range_error = np.zeros((NITERATIONS, len(unique_cohs), NBINS))

    curveslopes_correct, curveslopes_error = np.zeros(
        (NITERATIONS, len(unique_cohs))), np.zeros((NITERATIONS, len(unique_cohs)))
    curveintercept_correct, curveintercept_error = np.zeros(
        (NITERATIONS, len(unique_cohs))), np.zeros((NITERATIONS, len(unique_cohs)))
    for idx in range(NITERATIONS):
        unique_states = np.arange(0, 8, 1)
        Xmerge_trials_correct, ymerge_labels_correct, Xmerge_trials_error,\
            ymerge_labels_error, RECORDED_TRIALS_SET[idx] =\
            gpt.merge_pseudo_beh_trials(Xdata_set, ylabels_set, unique_states,
                                        unique_cohs, files, false_files, metadata,
                                        EACHSTATES, RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=RECORDED_TRIALS_SET[idx])

        unique_cohs = [-1, 0, 1]
        unique_states = np.arange(4, 8, 1)
        psychometric_trbias_correct[idx, :, :], trbias_range_correct[idx, :, :] =\
            gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_correct,
                                      ymerge_labels_correct, [4, 5, 6, 7],
                                      unique_cohs, [0, 1], num_beh_trials,
                                      EACHSTATES=EACHSTATES,
                                      FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS)
        unique_states = np.arange(4)
        psychometric_trbias_error[idx, :, :], trbias_range_error[idx, :, :] =\
            gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_error,
                                      ymerge_labels_error, [0, 1, 2, 3],
                                      unique_cohs, [0, 1], num_beh_trials,
                                      EACHSTATES=EACHSTATES,
                                      FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS)

        # compute the slope for zero-coherence
        for icoh in range(len(unique_cohs)):
            coh0_correct =\
                np.polyfit(trbias_range_correct[idx, icoh, 1:-1],
                           psychometric_trbias_correct[idx, icoh, 1:-1], 1)
            coh0_error = np.polyfit(trbias_range_error[idx, icoh, 1:-1],
                                    psychometric_trbias_error[idx, icoh, 1:-1], 1)
            curveslopes_correct[idx, icoh], curveintercept_correct[idx, icoh] =\
                coh0_correct[0], coh0_correct[1]
            curveslopes_error[idx, icoh],   curveintercept_error[idx, icoh] =\
                coh0_error[0],  coh0_error[1]

    colors = plt.cm.PRGn_r(np.linspace(0, 1, 3*6))
    for i in range(3):
        meanx = np.mean(trbias_range_correct[:, i, :], axis=0)
        meany = np.mean(psychometric_trbias_correct[:, i, :], axis=0)
        errory = np.std(
            psychometric_trbias_correct[:, i, :], axis=0)/np.sqrt(NITERATIONS)
        errorx = np.std(
            trbias_range_correct[:, i, :], axis=0)/np.sqrt(NITERATIONS)
        ax[0].errorbar(meanx, meany, xerr=errorx,
                       yerr=errory, color=colors[i*3], lw=1.5)

        meanx = np.mean(trbias_range_error[:, i, :], axis=0)
        meany = np.mean(psychometric_trbias_error[:, i, :], axis=0)
        errory = np.std(
            psychometric_trbias_error[:, i, :], axis=0)/np.sqrt(NITERATIONS)
        errorx = np.std(
            trbias_range_error[:, i, :], axis=0)/np.sqrt(NITERATIONS)
        ax[1].errorbar(meanx, meany, xerr=errorx,
                       yerr=errory, color=colors[i*3], lw=1.5)
    # ax[0].plot(trbias_range_correct[1,:], psychometric_trbias_correct[1,:],
    #            color='k',lw=1.5)
    # ax[1].plot(trbias_range_error[1,:], psychometric_trbias_error[1,:],
    #            color=colors[i],lw=1.5)

    lst = [RECORDED_TRIALS_SET]
    stg = ["RECORDED_TRIALS_SET"]
    d_beh = list_to_dict(lst=lst, string=stg)
    return curveslopes_correct, curveintercept_correct, curveslopes_error,\
        curveintercept_error, d_beh

    # unique_cohs   = [-1,0,1]
    # EACHSTATES    = 60
    # Xdata_set,ylabels_set = data_tr['Xdata_set'],data_tr['ylabels_set']
    # metadata = data_tr['metadata']
    # coeffs,intercepts = data_dec['coefs_correct'], data_dec['intercepts_correct']
    # Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,
    # ymerge_labels_error=gpt.merge_pseudo_beh_trials(Xdata_set,
    # ylabels_set,unique_states,unique_cohs,files,false_files,metadata,EACHSTATES)
    # unique_states = np.arange(4,8,1)
    # _=gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_correct,
    #                                          ymerge_labels_correct,
    # [4,5,6,7],[-1,0,1],[0,1], EACHSTATES=EACHSTATES)
    # unique_states = np.arange(4)
    # _=gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_error,
    #                                          ymerge_labels_error, [0,1,2,3],
    # [-1,0,1],[0,1], EACHSTATES=EACHSTATES)


'''
merging and generating pseudo trials for how tr. bias affects behaviour (stage 2)
'''
# def from_trbias_to_beh

if __name__ == '__main__':

    PREV_CH = 'L'
    NUM_SAMPLES = 200  # 200
    THRESH_TRIAL = 1
    PLOT_ALL_TRIALS_3D = False
    S_PLOTS = 5
    BOX_WDTH = 0.25
    SVMAXIS = 3
    AX_PREV_CH_OUTC = {'c': [2, 3], 'e': [0, 1]}
    NITERATIONS, NPSEUDODEC, PERCENTTRAIN = 100, 50, 0.6
    '''
    training: 30 per state, 30*4*2=240 trials per train
    testing:  20 per state, 20*4*2=160 trials per test
    '''
    RUN_ALL = True
    RERUN = True
    DOREVERSE = 0

    RECORD_TRIALS = 1
    CONTROL = 0

    # BOTTOM_3D = -6  # where to plot blue/red projected dots in 3D figure
    # XLIMS_2D = [-3, 3]
    # YLIMS_2D = [-7, 7]
    # YTICKS_2D = [-6., 0., 6.]
    # XTICKS_2D = [-2., 0., 2.]
    # CTXT_BIN = np.linspace(0, 1.65, 7)  # (0,1.8,7)
    # XLIM_CTXT = [12000, 13000]
    # YTICKS_CTXT = [-2, 0, 2]
    # YLIM_CTXT = [-2.2, 2.2]

    BOTTOM_3D = -10  # where to plot blue/red projected dots in 3D figure
    XLIMS_2D = [-15, 15]
    YLIMS_2D = [-15, 15]
    YTICKS_2D = [-15., 0., 15.]
    XTICKS_2D = [-15., 0., 15.]
    CTXT_BIN = np.linspace(0, 1.65, 7)  # (0,1.8,7)
    XLIM_CTXT = [12000, 13000]
    YTICKS_CTXT = [-15, 0, 15]
    YLIM_CTXT = [-15.2, 15.2]

    # # %%
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys/'
    # files = glob.glob(dir+'Rat15_ss_*_data_for_python.mat')
    # Rat7_ss_45_data_for_python.mat
    # for f in files:
    #     matfile = os.path.basename(f)
    #     try:num
    #         data,units = gd.get_data_file(f)
    #     except:
    #         continue
    #     # print('response:',np.shape(data['states']),np.shape(data['contexts']))
    # 'files_pop_analysis/'
    dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys/'
    # dir = '//home/molano/DMS_electro/DataEphys/pre_processed/'
    # 'files_pop_analysis/'
    IDX_RAT = 'Rat15_'  # 'LE100_'#
    # '202*.npz')  # Rat7_ss_45_data_for_python.mat
    files = glob.glob(dir+IDX_RAT+'ss*.npz')
    # dir = 'D://Yuxiu/Code/Data/Auditory/NeuralData/Rat7/Rat7/'
    # files = glob.glob(dir+'Rat7_ss_*.npz')

    # Whether to skip the NaN at the beginning or not
    SKIPNAN = 0
    data_tr = get_all_quantities(files, numtrans=0, SKIPNAN=SKIPNAN)

    unique_states = np.arange(8)
    unique_cohs = [-1, 0, 1]
    false_files, MIN_TRIALS, num_hist_trials, num_beh_trials = filter_sessions(
        data_tr, unique_states, unique_cohs)
    print(">>>>>>>>>>>>>>> Minimum Trials per state/beh_state:", MIN_TRIALS)

    unique_states = np.arange(8)
    unique_cohs = [-1, 0, 1]
    false_files, MIN_TRIALS, num_hist_trials, num_beh_trials = filter_sessions(
        data_tr, unique_states, unique_cohs)
    print(">>>>>>>>>>>>>>> Minimum Trials per state/beh_state:", MIN_TRIALS)

    wc, bc = [], []

    if(RECORD_TRIALS == 0):
        dataname = dir+IDX_RAT+'data_dec.npz'
        data_dec = np.load(dataname, allow_pickle=True)
        RECORDED_TRIALS_SET = data_dec['RECORDED_TRIALS_SET']
        RECORDED_TRIALS_SET = RECORDED_TRIALS_SET.item()
        wc, bc = data_dec['coefs_correct'], data_dec['intercepts_correct']
    else:
        RECORDED_TRIALS_SET = np.zeros(NITERATIONS)

    if(CONTROL == 1):
        data_dec = get_dec_axes(data_tr, wc, bc, [], [], false_files,
                                mode='decoding', DOREVERSE=0,
                                CONTROL=CONTROL, RECORD_TRIALS=1,
                                RECORDED_TRIALS_SET=np.zeros(NITERATIONS))
        wc, bc = data_dec['coefs_correct'], data_dec['intercepts_correct']
        data_dec = get_dec_axes(data_tr, wc, bc, [], [], false_files,
                                mode='decoding', DOREVERSE=0,
                                CONTROL=0, RECORD_TRIALS=RECORD_TRIALS,
                                RECORDED_TRIALS_SET=RECORDED_TRIALS_SET)
    else:
        data_dec = get_dec_axes(data_tr, wc, bc, [], [], false_files,
                                mode='decoding', DOREVERSE=0,
                                CONTROL=CONTROL, RECORD_TRIALS=RECORD_TRIALS,
                                RECORDED_TRIALS_SET=RECORDED_TRIALS_SET)

    if(RECORD_TRIALS == 1):
        dataname = dir+IDX_RAT+'data_dec.npz'
        np.savez(dataname, **data_dec)

    data_flt = flatten_data(data_tr, data_dec)

    projection_3D(data_flt, data_flt, 'c')
    projection_3D(data_flt, data_flt, 'e')

    projections_2D(data_flt, prev_outc='c', fit=False, name='')
    projections_2D(data_flt, prev_outc='e', fit=False, name='')

    # x-axis bin ctxt, y-axis transition bias
    corrl_ac, corrr_ac, corrl_ae, corrr_ae, ctx_tb_trcs = ctxtbin_defect(
        data_flt)
    print('>>>>>>>>>>> P-correlation, left AC: ', corrl_ac, ' right AC: ',
          corrr_ac, ' left AE: ', corrl_ae, ' right AE: ', corrr_ae)

    # transition bias to behaviour
    unique_cohs = [-1, 0, 1]
    EACHSTATES = 60
    if(RECORD_TRIALS == 0):
        dataname = dir+IDX_RAT+'data_beh.npz'
        data_beh = np.load(dataname, allow_pickle=True)
        RECORDED_TRIALS_SET = data_beh['RECORDED_TRIALS_SET']
        RECORDED_TRIALS_SET = RECORDED_TRIALS_SET.item()
    else:
        RECORDED_TRIALS_SET = np.zeros(EACHSTATES)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)
    curveslopes_correct, curveintercept_correct, curveslopes_error,\
        curveintercept_error, data_beh =\
        bias_VS_prob(data_tr, data_dec, unique_cohs, num_beh_trials, EACHSTATES,
                     NITERATIONS, ax, RECORD_TRIALS=RECORD_TRIALS,
                     RECORDED_TRIALS_SET=RECORDED_TRIALS_SET)
    # print('>>>>>>>>>>> Curve-beh, slopes AC: ',curveslopes_correct)
    # print('>>>>>>>>>>> Curve-beh, slopes AE: ',curveslopes_error)
    if(RECORD_TRIALS == 1):
        dataname = dir+IDX_RAT+'data_beh.npz'
        np.savez(dataname, **data_beh)

    auc_repc = np.mean(data_flt['AUCs_repc'])
    auc_repe = np.mean(data_flt['AUCs_repe'])
    print('rep, correct:', auc_repc, '; error:', auc_repe)
    auc_alte = np.mean(data_flt['AUCs_alte'])
    auc_altc = np.mean(data_flt['AUCs_altc'])
    print('alt, correct:', auc_altc, '; error:', auc_alte)
