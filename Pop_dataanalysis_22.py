import os
import sys
import seaborn as sns
import bootstrap_linear as bl
import required_data_dec as rdd
# import lda_util as lu
import general_util_ctxtgt as guc
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import scipy.stats as sstats
import glob
from sklearn import metrics
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.size'] = 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
RED = np.array((228, 26, 28)) / 255
BLUE = np.array((55, 126, 184)) / 255
ORANGE = np.array((255, 127, 0)) / 255
GREEN = np.array([0, 150, 0])/255
PURPLE = np.array([150, 0, 150])/255


# --- AUXILIARY FUNCTIONS

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


def rm_top_right_lines(ax):
    """
    Remove top and right lines in ax.

    Parameters
    ----------
    ax : axis
        axis to remove lines from.

    Returns
    -------
    None.

    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def box_plot(data, ax, x, lw=.5, fliersize=4, color='k', widths=0.35):
    bp = ax.boxplot(data, positions=[x], widths=widths)
    for p in ['whiskers', 'caps', 'boxes', 'medians']:
        for bpp in bp[p]:
            bpp.set(color=color, linewidth=lw)
    bp['fliers'][0].set(markeredgecolor=color, markerfacecolor=color, alpha=0.5,
                        marker='x', markersize=fliersize)
    ax.set_xticks([])


class SeabornFig2Grid():
    """
    See:
    https://stackoverflow.com/questions/35042255/
    how-to-plot-multiple-seaborn-jointplot-in-subplot/47664533#47664533
    """

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
           isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid =\
            gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid =\
            gridspec.GridSpecFromSubplotSpec(
                r + 1, r + 1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


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
        sns.distplot(df_group[col_x].values, ax=g.ax_marg_x, color=c, hist=False)
        sns.distplot(df_group[col_y].values, ax=g.ax_marg_y, color=c, hist=False,
                     vertical=True)
        counter += 1
    # plt.legend(legends)
    return g

# ---PRE-PROCESSING


def get_all_quantities(data, numtrans=3):
    tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc = guc.get_RNNdata_ctxtgt(
        data,)
    stim_trials, cohvalues, ctxt_trials = guc.transform_stim_trials_ctxtgt(
        tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc,)
    if numtrans == 3:
        Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1, Xlfs_1,\
            Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0,\
            Xdata_trialidx = rdd.req_quantities_3all(stim_trials, cohvalues, stm,
                                                     dyns, gt, choice, eff_choice,
                                                     rw, obsc, BLOCK_CTXT)
    elif numtrans == 0:
        Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1, Xlfs_1,\
            Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0,\
            Xdata_trialidx = rdd.req_quantities_0(stim_trials, cohvalues, stm,
                                                  dyns, gt, choice, eff_choice,
                                                  rw, obsc, BLOCK_CTXT)

    # label -- bias, label -- conditions
    Xdata_correct, correct_trial, Xdata_stim_correct, Xdata_dc_correct,\
        ych_stim_correct, Xdata_correct_seq, Xdata_error, error_trial,\
        Xdata_stim_error, Xdata_dc_error, ych_stim_error, Xdata_error_seq,\
        rses_correct, rses_error, Xrse_6_correct, Xrse_6_error, Xcohs_0_correct,\
        Xcohs_0_error, ydata_bias_correct, ydata_bias_error, ydata_xor_correct,\
        ydata_xor_error, ydata_conds_correct, ydata_conds_error,\
        ydata_choices_correct, ydata_choices_error, ydata_cchoices_correct,\
        ydata_cchoices_error, ydata_cgts_correct, ydata_cgts_error,\
        ydata_crws_correct, ydata_crws_error, Xdata_idx_correct,\
        Xdata_idx_error, Xdata_trialidx_correct, Xdata_trialidx_error = \
        rdd.sep_correct_error(stm, dyns, Xdata, ydata, Xdata_idx, Xdata_resp,
                              Xconds_2, Xacts_1, Xrws_1, Xlfs_1, Xrse_6, rses,
                              Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0,
                              Xdata_trialidx, margin=[1, 2], idd=1)
    lst = [Xdata_correct,
           Xdata_error,
           Xcohs_0_correct, Xcohs_0_error, ydata_bias_correct, ydata_bias_error,
           ydata_xor_correct, ydata_xor_error, ydata_conds_correct,
           ydata_conds_error, ydata_choices_correct, ydata_choices_error,
           ydata_cchoices_correct, ydata_cchoices_error, ydata_cgts_correct,
           ydata_cgts_error,
           Xdata_idx_correct, Xdata_idx_error, Xdata_trialidx_correct,
           Xdata_trialidx_error]
    stg = ["Xdata_correct,"
           "Xdata_error,"
           "Xcohs_0_correct, Xcohs_0_error, ydata_bias_correct, ydata_bias_error,"
           "ydata_xor_correct, ydata_xor_error, ydata_conds_correct,"
           "ydata_conds_error, ydata_choices_correct, ydata_choices_error,"
           "ydata_cchoices_correct, ydata_cchoices_error, ydata_cgts_correct,"
           "ydata_cgts_error,"
           "Xdata_idx_correct, Xdata_idx_error, Xdata_trialidx_correct,"
           "Xdata_trialidx_error"]
    d = list_to_dict(lst=lst, string=stg)
    return d

# --- GET AXES


def get_dec_axes(data_tr, wc, bc, we, be, mode='decoding', DOREVERSE=0):
    # after correct
    Xdata_correct = data_tr['Xdata_correct']
    ydata_choices_correct = data_tr['ydata_choices_correct']
    ydata_conds_correct = data_tr['ydata_conds_correct']
    ydata_xor_correct = data_tr['ydata_xor_correct']
    ydata_bias_correct = data_tr['ydata_bias_correct']
    ydata_cchoices_correct = data_tr['ydata_cchoices_correct']
    n_correct = Xdata_correct.shape[0]
    idxtotal = np.arange(Xdata_correct.shape[0])
    train_poolc = np.random.choice(idxtotal, size=(int)(0.6 * n_correct),
                                   replace=False)
    test_poolc = np.setdiff1d(idxtotal, train_poolc)
    ytrain_poolc = np.zeros((len(train_poolc), 3 + 1 + 1))
    ytest_poolc = np.zeros((len(test_poolc), 3 + 1 + 1))
    ytrain_poolc[:, 0], ytest_poolc[:, 0] =\
        ydata_choices_correct[train_poolc], ydata_choices_correct[test_poolc]
    ytrain_poolc[:, 1], ytest_poolc[:, 1] =\
        ydata_conds_correct[train_poolc], ydata_conds_correct[test_poolc]
    ytrain_poolc[:, 2], ytest_poolc[:, 2] =\
        ydata_xor_correct[train_poolc], ydata_xor_correct[test_poolc]
    ytrain_poolc[:, 3], ytest_poolc[:, 3] =\
        ydata_bias_correct[train_poolc], ydata_bias_correct[test_poolc]
    ytrain_poolc[:, 4], ytest_poolc[:, 4] =\
        ydata_cchoices_correct[train_poolc], ydata_cchoices_correct[test_poolc]

    # after error
    Xdata_error = data_tr['Xdata_error']
    ydata_choices_error = data_tr['ydata_choices_error']
    ydata_conds_error = data_tr['ydata_conds_error']
    ydata_xor_error = data_tr['ydata_xor_error']
    ydata_bias_error = data_tr['ydata_bias_error']
    ydata_cchoices_error = data_tr['ydata_cchoices_error']
    n_error = Xdata_error.shape[0]
    idxtotal = np.arange(Xdata_error.shape[0])
    train_poole = np.random.choice(idxtotal, size=(int)(0.6 * n_error),
                                   replace=False)
    test_poole = np.setdiff1d(idxtotal, train_poole)
    ytrain_poole = np.zeros((len(train_poole), 3 + 1 + 1))
    ytest_poole = np.zeros((len(test_poole), 3 + 1 + 1))
    ytrain_poole[:, 0], ytest_poole[:, 0] =\
        ydata_choices_error[train_poole], ydata_choices_error[test_poole]
    ytrain_poole[:, 1], ytest_poole[:, 1] =\
        ydata_conds_error[train_poole], ydata_conds_error[test_poole]
    ytrain_poole[:, 2], ytest_poole[:, 2] =\
        ydata_xor_error[train_poole], ydata_xor_error[test_poole]
    ytrain_poole[:, 3], ytest_poole[:, 3] =\
        ydata_bias_error[train_poole], ydata_bias_error[test_poole]
    ytrain_poole[:, 4], ytest_poole[:, 4] =\
        ydata_cchoices_error[train_poole], ydata_cchoices_error[test_poole]

    # --- cell 5
    if mode == 'decoding':
        if METHOD == 'single_decoder':
            _, coeffs, intercepts, Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias,\
                Xsup_vec_cc =\
                bl.bootstrap_linsvm_decodersdef(Xdata_correct[train_poolc, :],
                                                ytrain_poolc,
                                                Xdata_error[train_poole, :],
                                                ytrain_poole, 'normal',
                                                DOREVERSE=DOREVERSE,
                                                n_iterations=IPOOLS,
                                                n_percent=IEACHTRAIN)
            coefs_correct, intercepts_correct = coeffs.copy(), intercepts.copy()
            coefs_error, intercepts_error = coeffs.copy(), intercepts.copy()

            # AC
            stats_score_correct, Xtest_set_correct, ytest_set_correct,\
                ypred_set_correct, yevi_set_correct, test_set_correct =\
                bl.bootstrap_linsvm_proj_withtrials(coefs_correct,
                                                    intercepts_correct,
                                                    Xdata_correct[test_poolc, :],
                                                    ytest_poolc, 'normal',
                                                    n_iterations=IPOOLS,
                                                    n_percent=IEACHTRAIN)
            # AE
            stats_score_error, Xtest_set_error, ytest_set_error, ypred_set_error,\
                yevi_set_error, test_set_error =\
                bl.bootstrap_linsvm_proj_withtrials(coefs_error, intercepts_error,
                                                    Xdata_error[test_poole, :],
                                                    ytest_poole, 'normal',
                                                    n_iterations=IPOOLS,
                                                    n_percent=IEACHTRAIN)
        elif METHOD == 'separate_decs':
            stats_score_correct, coefs_correct, intercepts_correct,\
                Xtest_set_correct, ytest_set_correct, ypred_set_correct,\
                yevi_set_correct, test_set_correct, Xsup_vec_act_correct,\
                Xsup_vec_ctxt_correct, Xsup_vec_bias_correct,\
                Xsup_vec_cc_correct =\
                bl.bootstrap_linsvm_acc_withtrials(Xdata_correct[train_poolc, :],
                                                   ytrain_poolc,
                                                   Xdata_correct[test_poolc, :],
                                                   ytest_poolc, 'normal',
                                                   n_iterations=IPOOLS,
                                                   n_percent=IEACHTRAIN)

            stats_score_error, coefs_error, intercepts_error, Xtest_set_error,\
                ytest_set_error, ypred_set_error, yevi_set_error, test_set_error,\
                Xsup_vec_act_error, Xsup_vec_ctxt_error, Xsup_vec_bias_error,\
                Xsup_vec_cc_error =\
                bl.bootstrap_linsvm_acc_withtrials(Xdata_error[train_poole, :],
                                                   ytrain_poole,
                                                   Xdata_error[test_poole, :],
                                                   ytest_poole, 'normal',
                                                   n_iterations=IPOOLS,
                                                   n_percent=IEACHTRAIN)

    elif mode == 'projecting':
        # AC
        coefs_correct, intercepts_correct = wc.copy(), bc.copy()
        stats_score_correct, Xtest_set_correct, ytest_set_correct,\
            ypred_set_correct, yevi_set_correct, test_set_correct =\
            bl.bootstrap_linsvm_proj_withtrials(coefs_correct, intercepts_correct,
                                                Xdata_correct[test_poolc, :],
                                                ytest_poolc, 'normal',
                                                n_iterations=IPOOLS,
                                                n_percent=IEACHTRAIN)
        # AE
        coefs_error, intercepts_error = we.copy(), be.copy()
        stats_score_error, Xtest_set_error, ytest_set_error,\
            ypred_set_error, yevi_set_error, test_set_error =\
            bl.bootstrap_linsvm_proj_withtrials(coefs_error, intercepts_error,
                                                Xdata_error[test_poole, :],
                                                ytest_poole, 'nromal',
                                                n_iterations=IPOOLS,
                                                n_percent=IEACHTRAIN)

    lst = [coefs_correct, intercepts_correct,
           ytest_set_correct,  # Xtest_set_correct,
           yevi_set_correct, test_set_correct,
           coefs_error, intercepts_error,  # Xtest_set_error,
           ytest_set_error,  yevi_set_error, test_set_error,
           test_poolc, test_poole]
    stg = ["coefs_correct, intercepts_correct,"
           "ytest_set_correct, "  # " Xtest_set_correct, "
           "yevi_set_correct, test_set_correct,"
           "coefs_error, intercepts_error,"  # " Xtest_set_error,"
           "ytest_set_error, yevi_set_error, test_set_error,"
           "test_poolc, test_poole"]
    d = list_to_dict(lst=lst, string=stg)
    return d


def flatten_data(data_tr, data_dec):
    yevi_set_correct = data_dec['yevi_set_correct']
    ytest_set_correct = data_dec['ytest_set_correct']
    test_poolc = data_dec['test_poolc']
    test_set_correct = data_dec['test_set_correct']
    ydata_cchoices_correct = data_tr['ydata_cchoices_correct']
    ydata_cgts_correct = data_tr['ydata_cgts_correct']
    ydata_bias_correct = data_tr['ydata_bias_correct']

    # flatten data --- correct
    ytruthlabels_c, ycact_c = np.zeros((3 + 1 + 1, 1)), np.zeros((1, 1))
    ytruthlabels_c_ = np.zeros((1, 1))
    ycgt_c = np.zeros((1, 1))
    yevi_c = np.zeros((3 + 1 + 1, 1))
    trials_c = []
    dprimes_c = np.zeros(IPOOLS)

    for i in range(IPOOLS):  # bootstrapping
        hist_evi = yevi_set_correct[i, :, :]
        # labels: preaction, ctxt, bias
        test_labels = ytest_set_correct[i, :, :]
        # test_set records the index in test_poolc
        ttrials = test_poolc[test_set_correct[i, :].astype(np.int32)]
        idx = np.arange(np.shape(hist_evi)[0])
        idx_c = ttrials[idx]  # indices in the total Xdata_correct

        # current action
        ycact_c = np.append(
            ycact_c, ydata_cchoices_correct[idx_c].reshape(
                1, -1), axis=1)
        # current stimulus category
        ycgt_c = np.append(
            ycgt_c, ydata_cgts_correct[idx_c].reshape(
                1, -1), axis=1)

        ytruthlabels_c = np.append(
            ytruthlabels_c, test_labels[idx, :].T, axis=1)
        ytruthlabels_c_ = np.append(
            ytruthlabels_c_, ydata_bias_correct[idx_c].reshape(
                1, -1), axis=1)
        yevi_c = np.append(
            yevi_c, (yevi_set_correct[i, idx, :]).T, axis=1)  # np.squeeze
        dprimes_c[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, SVMAXIS]),
                                 np.squeeze(ytest_set_correct[i, :, SVMAXIS]))
        # trial-index in the whole Xdata_correct dataset
        if(i == 0):
            trials_c = np.reshape(idx_c, (1, -1))
        else:
            trials_c = np.append(trials_c, np.reshape(idx_c, (1, -1)), axis=1)
    ytruthlabels_c, ycact_c, yevi_c =\
        ytruthlabels_c[:, 1:], ycact_c[:, 1:], yevi_c[:, 1:]
    ycgt_c = ycgt_c[:, 1:]

    # flatten data
    yevi_set_error = data_dec['yevi_set_error']
    ytest_set_error = data_dec['ytest_set_error']
    test_poole = data_dec['test_poole']
    test_set_error = data_dec['test_set_error']
    ydata_cchoices_error = data_tr['ydata_cchoices_error']
    ydata_cgts_error = data_tr['ydata_cgts_error']
    ydata_bias_error = data_tr['ydata_bias_error']
    ytruthlabels_e, ycact_e = np.zeros((3 + 1 + 1, 1)), np.zeros((1, 1))
    ytruthlabels_e_ = np.zeros((1, 1))
    ycgt_e = np.zeros((1, 1))
    yevi_e = np.zeros((3 + 1 + 1, 1))
    trials_e = []
    dprimes_e = np.zeros(IPOOLS)
    for i in range(IPOOLS):
        hist_evi = yevi_set_error[i, :, :]
        test_labels = ytest_set_error[i, :, :]  # labels: preaction, ctxt, bias
        # test_set records the index in test_poolc
        ttrials = test_poole[test_set_error[i, :].astype(np.int32)]
        idx = np.arange(np.shape(hist_evi)[0])
        idx_e = ttrials[idx]  # indices in the total Xdata_correct
        ycact_e = np.append(
            ycact_e, ydata_cchoices_error[idx_e].reshape(
                1, -1), axis=1)
        ycgt_e = np.append(
            ycgt_e, ydata_cgts_error[idx_e].reshape(
                1, -1), axis=1)
        ytruthlabels_e = np.append(
            ytruthlabels_e, test_labels[idx, :].T, axis=1)
        ytruthlabels_e_ = np.append(
            ytruthlabels_e_, ydata_bias_error[idx_e].reshape(
                1, -1), axis=1)
        yevi_e = np.append(
            yevi_e, (yevi_set_error[i, idx, :]).T, axis=1)  # np.squeeze
        dprimes_e[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, SVMAXIS]),
                                 np.squeeze(ytest_set_error[i, :, SVMAXIS]))
        # trial-index in the whole Xdata_error dataset
        if(i == 0):
            trials_e = np.reshape(idx_e, (1, -1))
        else:
            trials_e = np.append(trials_e, np.reshape(idx_e, (1, -1)), axis=1)
    ytruthlabels_e, ycact_e, yevi_e =\
        ytruthlabels_e[:, 1:], ycact_e[:, 1:], yevi_e[:, 1:]
    ycgt_e = ycgt_e[:, 1:]
    lst = [ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e, trials_c, trials_e,
           dprimes_c, dprimes_e]
    stg = ["ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e, trials_c, trials_e,"
           "dprimes_c, dprimes_e"]
    d = list_to_dict(lst=lst, string=stg)
    return d

# --- PLOT


def context_dynamics(data, wc, bc, ax, ax_ae, ax_ac, ax_ch):
    tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc = guc.get_RNNdata_ctxtgt(
        data,)
    stim_trials, cohvalues, ctxt_trials = guc.transform_stim_trials_ctxtgt(
        tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc,)

    Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1, Xlfs_1,\
        Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0,\
        Xdata_trialidx = rdd.req_quantities_0(stim_trials, cohvalues, stm,
                                              dyns, gt, choice, eff_choice,
                                              rw, obsc)

    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,\
        _, _, _, _, _, _, _, _, _, _, Xdata_trialidx_correct,\
        Xdata_trialidx_error = \
        rdd.sep_correct_error(stm, dyns, Xdata, ydata, Xdata_idx, Xdata_resp,
                              Xconds_2, Xacts_1, Xrws_1, Xlfs_1, Xrse_6, rses,
                              Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0,
                              Xdata_trialidx, margin=[1, 2], idd=1)

    Xdata = sstats.zscore(
        dyns[:, int(dyns.shape[1] / 2):], axis=0)  # normalize

    Xdata = Xdata[Xdata_idx.astype(int), :]
    ctxt = np.zeros(len(ctxt_trials))
    for i in range(len(ctxt_trials)):
        if(ctxt_trials[i] == 1):
            ctxt[i] = 1
        else:
            ctxt[i] = -1

    # @YX 0910 -- weights
    ntrials_tt = np.shape(Xdata)[0]
    ndecoders = int(ntrials_tt / IEACHTRAIN) + 1
    decoders_idx = np.random.choice(IPOOLS, size=ndecoders, replace=True)
    coeffs_pool, intercepts_pool = wc.copy(), bc.copy()
    for iii in range(ndecoders):
        start_idx, end_idx = IEACHTRAIN * iii, IEACHTRAIN * (iii + 1)
        end_idx = min(end_idx, ntrials_tt)
        Xdata_test = Xdata[start_idx:end_idx, :]

        i = decoders_idx[iii]
        linw_pact, linb_pact = coeffs_pool[:, i * 5 + 0],\
            intercepts_pool[0, 5 * i + 0]
        linw_ctxt, linb_ctxt = coeffs_pool[:, i * 5 + 1],\
            intercepts_pool[0, 5 * i + 1]
        linw_xor, linb_xor = coeffs_pool[:, i * 5 + 2],\
            intercepts_pool[0, 5 * i + 2]
        linw_bias, linb_bias = coeffs_pool[:, i * 5 + 3],\
            intercepts_pool[0, 5 * i + 3]
        linw_cc, linb_cc = coeffs_pool[:, i *
                                       5 + 4], intercepts_pool[0, 5 * i + 4]
        sz_evi = np.shape(Xdata_test)[0]
        if(iii == 0):
            evidences = np.zeros((sz_evi, 3 + 2))
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
            # print("size context:",np.shape(evidences))
        elif(iii > 0):
            tevidences = np.zeros((sz_evi, 3 + 2))
            tevidences[:, 0] = np.squeeze(
                Xdata_test @ linw_pact.reshape(-1, 1) + linb_pact)
            tevidences[:, 1] = np.squeeze(
                Xdata_test @ linw_ctxt.reshape(-1, 1) + linb_ctxt)
            tevidences[:, 2] = np.squeeze(
                Xdata_test @ linw_xor.reshape(-1, 1) + linb_xor)
            tevidences[:, 3] = np.squeeze(
                Xdata_test @ linw_bias.reshape(-1, 1) + linb_bias)
            tevidences[:, 4] = np.squeeze(
                Xdata_test @ linw_cc.reshape(-1, 1) + linb_cc)

            evidences = np.append(evidences, tevidences, axis=0)
            # print("size context:",np.shape(evidences))

    blk_ch = np.where(np.diff(ctxt) != 0)[0]
    blk_ch = blk_ch[np.logical_and(blk_ch > Xdata_trialidx[0] - 2000,
                                   blk_ch < Xdata_trialidx[-1] + 2000)]
    # f, ax_temp = plt.subplots(ncols=1)
    for i_bc in range(len(blk_ch) - 1):
        color = BLUE if ctxt[blk_ch[i_bc] + 1] == - \
            1 else RED
        ax.axvspan(blk_ch[i_bc], blk_ch[i_bc + 1], facecolor=color, alpha=0.3)
        # ax_temp.axvspan(blk_ch[i_bc], blk_ch[i_bc + 1], facecolor=color,
        #                 alpha=0.3)

    ctxt_evi = np.zeros_like(ctxt)
    ctxt_blk_label = np.zeros_like(ctxt_evi)
    # nontrials_c,nontrials_e=nontrials_c.astype(int32),nontrials_e.astype(int32)
    for idx, idxc in enumerate(Xdata_trialidx):
        idxc = int(idxc)
        ctxt_evi[idxc] = evidences[idx, 1]
        ctxt_blk_label[idxc] = ctxt[idxc]

    # nonctxt_c_sm = np.convolve(nonctxt_c, kernelsm, mode='same')
    ax.plot(np.arange(len(ctxt)), ctxt_evi, alpha=0.75, color='k', lw=0.5)
    # ax_temp.plot(np.arange(len(ctxt)), ctxt_evi, alpha=0.75, color='k', lw=0.5)
    # ax.plot(nontrials_c, nonctxt_c_sm, c='orange', alpha=0.75)
    # ax.set_ylim(xlims)
    ax.set_yticks(YTICKS_CTXT)
    ax.set_xticks(XLIM_CTXT)  # [nontrials[0],
    # nontrials[0]+(nontrials[-1]-nontrials[0])/2,
    # nontrials[-1]])
    # ax.set_xticklabels(['0', str((nontrials_c[-1]-nontrials_c[0])/2),
    #                     str(nontrials_c[-1]-nontrials_c[0])])
    ax.set_xticklabels([str(x) for x in XLIM_CTXT])
    # ax.set_yticks([])
    ax.set_ylabel('Context encoding')
    ax.set_xlabel('Trials')
    # ax.set_xlim(nontrials[0], nontrials[-1])
    ax.set_xlim(XLIM_CTXT)  # nontrials[0], nontrials[-1])
    ax.set_ylim(YLIM_CTXT)
    rm_top_right_lines(ax)
    # projection at block change
    pre_w = 10
    post_w = 20
    blk_ch = np.where(np.diff(ctxt) != 0)[0]
    blk_ch = blk_ch[np.logical_and(blk_ch > pre_w + 1,
                                   blk_ch < len(ctxt_evi) - (post_w + 1))]
    rep_alt = []
    alt_rep = []
    for i_bc in range(len(blk_ch)):
        trace = ctxt_evi[blk_ch[i_bc] - pre_w:blk_ch[i_bc] + post_w]
        if ctxt[blk_ch[i_bc]] == -1:
            alt_rep.append(trace)
        else:
            rep_alt.append(trace)
    rep_alt = np.array(rep_alt)
    alt_rep = np.array(alt_rep)
    xs = np.arange(pre_w + post_w) - pre_w
    ax_ch.axvline(x=0, color=(.4, .4, .4), linestyle='--')
    ax_ch.errorbar(xs, np.mean(rep_alt, axis=0),
                   np.std(rep_alt, axis=0) / np.sqrt(rep_alt.shape[0]), color=RED)
    ax_ch.errorbar(xs, np.mean(alt_rep, axis=0),
                   np.std(alt_rep, axis=0) / np.sqrt(alt_rep.shape[0]), color=BLUE)
    ax_ch.set_xlabel('Trials from block change')
    ax_ch.set_ylabel('Context encoding')
    rm_top_right_lines(ax_ch)
    # histograms
    # AE
    # f, ax_temp = plt.subplots(nrows=2)
    firsttrial_idx = Xdata_trialidx[0]
    Xdata_trialidx_error_0 = Xdata_trialidx_error - firsttrial_idx
    ctxt_evi_error = evidences[Xdata_trialidx_error_0.astype(int), 1]
    ctxt_blk_label_error = ctxt[Xdata_trialidx_error.astype(int)]
    idxblue, idxred = np.where(ctxt_blk_label_error < 0)[0],\
        np.where(ctxt_blk_label_error > 0)[0]
    ax_ae.hist(ctxt_evi_error[idxblue], bins=30, alpha=0.9, facecolor=BLUE,
               orientation="horizontal")
    ax_ae.hist(ctxt_evi_error[idxred], bins=30, alpha=0.9, facecolor=RED,
               orientation="horizontal")
    # ax_temp[0].hist(ctxt_evi_error[idxblue], bins=30, alpha=0.9, facecolor=BLUE)
    # ax_temp[0].hist(ctxt_evi_error[idxred], bins=30, alpha=0.9, facecolor=RED)

    ax_ae.set_ylim(YLIM_CTXT)
    dprime_ctxt_AE = guc.calculate_dprime(np.squeeze(ctxt_evi_error[:]),
                                          np.squeeze(ctxt_blk_label_error[:]))
    ax_ae.set_xticks([])
    ax_ae.set_yticks([])
    y = np.zeros_like(ctxt_evi_error)
    y[idxblue] = 1
    y[idxred] = 2
    assert (y != 0).all()
    fpr, tpr, thresholds = metrics.roc_curve(y, ctxt_evi_error, pos_label=2)
    AUC_ae = metrics.auc(fpr, tpr)
    ax_ae.set_title('AE   dprime: '+str(np.round(dprime_ctxt_AE, 3)) +
                    ' AUC: '+str(np.round(AUC_ae, 3)))
    rm_top_right_lines(ax_ae)

    # AC
    firsttrial_idx = Xdata_trialidx[0]
    Xdata_trialidx_correct_0 = Xdata_trialidx_correct - firsttrial_idx
    ctxt_evi_correct = evidences[Xdata_trialidx_correct_0.astype(int), 1]
    ctxt_blk_label_correct = ctxt[Xdata_trialidx_correct.astype(int)]
    idxblue, idxred = np.where(ctxt_blk_label_correct < 0)[0],\
        np.where(ctxt_blk_label_correct > 0)[0]
    ax_ac.hist(ctxt_evi_correct[idxblue], bins=30, alpha=0.9, facecolor=BLUE,
               orientation="horizontal")
    ax_ac.hist(ctxt_evi_correct[idxred], bins=30, alpha=0.9, facecolor=RED,
               orientation="horizontal")
    # ax_temp[1].hist(ctxt_evi_correct[idxblue], bins=30, alpha=0.9,
    #                 facecolor=BLUE)
    # ax_temp[1].hist(ctxt_evi_correct[idxred], bins=30, alpha=0.9, facecolor=RED)

    dprime_ctxt_AC = guc.calculate_dprime(np.squeeze(ctxt_evi_correct[:]),
                                          np.squeeze(ctxt_blk_label_correct[:]))
    ax_ac.set_ylim(YLIM_CTXT)
    ax_ac.set_xticks([])
    ax_ac.set_yticks([])
    rm_top_right_lines(ax_ac)
    y = np.zeros_like(ctxt_evi_correct)
    y[idxblue] = 1
    y[idxred] = 2
    assert (y != 0).all()
    fpr, tpr, thresholds = metrics.roc_curve(y, ctxt_evi_correct, pos_label=2)
    AUC_ac = metrics.auc(fpr, tpr)
    ax_ac.set_title('AC   dprime: '+str(np.round(dprime_ctxt_AC, 3)) +
                    ' AUC: '+str(np.round(AUC_ac, 3)))
    return AUC_ac, AUC_ae


def projection_3D(data_flt, data_flt_light):
    ytruthlabels_c = data_flt['ytruthlabels_c']
    yevi_c = data_flt['yevi_c']
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
        if(ytruthlabels_c[3, i] == 2):
            cms.append(GREEN)
        else:
            cms.append(PURPLE)
    ax.scatter(x, y, z, s=S_PLOTS, c=cms, alpha=0.9, zorder=0)
    zflat = np.full_like(z, BOTTOM_3D)  # min(ax.get_zlim()))
    ytruthlabels_c = np.array((ytruthlabels_c).copy().astype(np.int32))
    # two projections
    idxright = np.where(ytruthlabels_c[0, ridx] == 3)[0]
    idxleft = np.where(ytruthlabels_c[0, ridx] == 2)[0]
    igreen, iblue = np.where(ytruthlabels_c[3, ridx[idxleft]] == 2)[
        0], np.where(ytruthlabels_c[3, ridx[idxleft]] == 3)[0]
    ax.scatter(np.mean(x[idxleft[igreen]]), np.mean(y[idxleft[igreen]]), np.mean(
        z[idxleft[igreen]]), s=100, c=GREEN, edgecolor='k', zorder=1)
    ax.plot(np.mean(x[idxleft[igreen]])*np.ones(2), np.mean(y[idxleft[igreen]]) *
            np.ones(2), [zflat[0], np.mean(z[idxleft[igreen]])], 'k-', zorder=1)
    ax.scatter(np.mean(x[idxleft[iblue]]), np.mean(y[idxleft[iblue]]), np.mean(
        z[idxleft[iblue]]), s=100, c=PURPLE, edgecolor='k', zorder=1)
    ax.plot(np.mean(x[idxleft[iblue]])*np.ones(2), np.mean(y[idxleft[iblue]]) *
            np.ones(2), [zflat[0], np.mean(z[idxleft[iblue]])], 'k-', zorder=1)

    ibluehist, igreenhist = idxleft[iblue], idxleft[igreen]

    igreen, iblue = np.where(ytruthlabels_c[3, ridx[idxright]] == 2)[
        0], np.where(ytruthlabels_c[3, ridx[idxright]] == 3)[0]
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

    igreen, iblue = np.where(ytruthlabels_c[1, ridx] == 2)[
        0], np.where(ytruthlabels_c[1, ridx] == 3)[0]
    ax.scatter(x[igreen], y[igreen], zflat[igreen],
               s=S_PLOTS, c=BLUE, alpha=0.9)
    ax.scatter(x[iblue], y[iblue], zflat[iblue], s=S_PLOTS, c=RED, alpha=0.9)

    # light one ---ALL TRIALS
    # --- PLOTING ALL TRIALS, DO NOT REQUIRE CONGRUENCY AND LOCALISED TRANSITIONS
    if PLOT_ALL_TRIALS_3D:
        ytruthlabels_c = data_flt_light['ytruthlabels_c']
        yevi_c = data_flt_light['yevi_c']
        x, y, z = yevi_c[1, ridx], yevi_c[0, ridx], yevi_c[3, ridx]
        zhistblue, zhistgreen = [], []
        cms = []
        for i in ridx:
            if(ytruthlabels_c[3, i] == 2):
                cms.append(GREEN)
                zhistblue = np.append(zhistblue, yevi_c[3, i])
            else:
                cms.append(PURPLE)
                zhistgreen = np.append(zhistgreen, yevi_c[3, i])
        ax.scatter(x, y, z, marker='^', s=5, c=cms, alpha=0.1)

        zflat = np.full_like(z, BOTTOM_3D)  # min(ax.get_zlim()))
        ytruthlabels_c = np.array((ytruthlabels_c).copy().astype(np.int32))

        ax2dd.hist(zhistblue, bins=zrange, density=True,
                   facecolor=PURPLE, alpha=0.1)
        ax2dd.hist(zhistgreen, bins=zrange, density=True, facecolor=GREEN,
                   alpha=0.1)

        igreen, iblue = np.where(ytruthlabels_c[1, ridx] == 2)[
            0], np.where(ytruthlabels_c[1, ridx] == 3)[0]
        ax.scatter(x[igreen], y[igreen], zflat[igreen],
                   s=1, c=BLUE, alpha=0.25)
        ax.scatter(x[iblue], y[iblue], zflat[iblue], s=1, c=RED, alpha=0.25)

    image_name = SAVELOC + '/3d_sidehist_' + METHOD + '.svg'
    fig2dd.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
    plt.close(fig2dd)
    ax.view_init(azim=-30, elev=30)
    # ax.yaxis.set_ticklabels([-1.5, 0, 1.5])
    # ax.xaxis.set_ticklabels(XTICKS_2D)
    # ax.zaxis.set_ticklabels(YTICKS_2D)
    ax.set_zlim3d([-7, 7])
    ax.set_xlabel('Context encoding', fontsize=14)
    ax.set_ylabel('Previous choice encoding', fontsize=14)
    ax.set_zlabel('Transition bias encoding', fontsize=14)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    image_name = SAVELOC + '/3d_plot_' + NAME + '.svg'
    fig.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
    # plt.close(fig)


def ctxtbin_defect(data_flt):
    # all trials
    ytruthlabels_c = data_flt['ytruthlabels_c']
    yevi_c = data_flt['yevi_c']

    ytruthlabels_e = data_flt['ytruthlabels_e']
    yevi_e = data_flt['yevi_e']

    # mean_ctxt = (np.mean(np.abs(yevi_c[1, :])) +
    #              np.mean(np.abs(yevi_e[1, :])))/2.0
    fig2, ax2 = plt.subplots(7, 2, figsize=(12, 12), sharex=True,
                             sharey=True, tight_layout=True)
    nbins = len(CTXT_BIN)
    Tbias_ctxt_c, Tbias_ctxt_e = {}, {}
    Tbias_ctxt_clabel, Tbias_ctxt_elabel = {}, {}
    dprime_ctxtdp_c, dprime_ctxtdp_e = np.zeros(nbins), np.zeros(nbins)
    ACC_correct, ACC_error = np.zeros(nbins), np.zeros(nbins)
    ctxt_evi_c = np.abs(yevi_c[1, :])
    ctxt_evi_e = np.abs(yevi_e[1, :])
    binss = np.linspace(-4.0, 4.0, 40)
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
        # ploting histogram
        ax2[i][0].hist(Tbias_ctxt_c[i][np.where(Tbias_ctxt_clabel[i] == 2)[
                       0]], bins=binss, facecolor=GREEN, alpha=0.5)
        ax2[i][0].hist(Tbias_ctxt_c[i][np.where(Tbias_ctxt_clabel[i] == 3)[
                       0]], bins=binss, facecolor='tab:purple', alpha=0.5)

        ax2[i][1].hist(Tbias_ctxt_e[i][np.where(Tbias_ctxt_elabel[i] == 2-2)[0]],
                       bins=binss, facecolor=GREEN, alpha=0.5)
        ax2[i][1].hist(Tbias_ctxt_e[i][np.where(Tbias_ctxt_elabel[i] == 3-2)[0]],
                       bins=binss, facecolor='tab:purple', alpha=0.5)

    image_name = SAVELOC + '/hist_ctxtbin_'+METHOD+'.svg'
    fig2.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
    plt.close(fig2)
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
    # label_tbiasl_AC, label_tbiasr_AC = ytruthlabels_c[SVMAXIS, prechL_AC],\
    #     ytruthlabels_c[SVMAXIS, prechR_AC]

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
    # label_tbiasl_AE, label_tbiasr_AE = ytruthlabels_e[SVMAXIS, prechL_AE],\
    #     ytruthlabels_e[SVMAXIS,prechR_AE]

    corrl_ae = np.mean((xl_ctxt_AE-np.mean(xl_ctxt_AE)) *
                       (yl_tbias_AE-np.mean(yl_tbias_AE)))
    corrl_ae = corrl_ae/(np.std(xl_ctxt_AE)*np.std(yl_tbias_AE))
    corrr_ae = np.mean((xr_ctxt_AE-np.mean(xr_ctxt_AE)) *
                       (yr_tbias_AE-np.mean(yr_tbias_AE)))
    corrr_ae = corrr_ae/(np.std(xr_ctxt_AE)*np.std(yr_tbias_AE))

    return corrl_ac, corrr_ac, corrl_ae, corrr_ae, [ACC_correct[:-1],
                                                    ACC_error[:-1]]


def projections_2D(data_flt, prev_outc, fit=False, name=''):
    ytruthlabels = data_flt['ytruthlabels_'+prev_outc]
    yevi = data_flt['yevi_'+prev_outc]
    idxpreal, idxprear =\
        np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
        np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    idxbiasl, idxbiasr =\
        np.where(ytruthlabels[3, :] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
        np.where(ytruthlabels[3, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]

    # plot samples
    # previous left
    idxleft = np.random.choice(idxpreal, size=NUM_SAMPLES, replace=False)
    idxpreal = idxleft
    idxright = np.random.choice(idxprear, size=NUM_SAMPLES, replace=False)
    idxprear = idxright
    figs = []
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
        fig.fig.set_figwidth(4)
        fig.fig.set_figheight(4)
        # fit
        if fit:
            coefficients = np.polyfit(ctxt, tr_bias, 1)
            poly = np.poly1d(coefficients)
            new_y = poly([np.min(ctxt), np.max(ctxt)])
            fig.ax_joint.plot([np.min(ctxt), np.max(ctxt)], new_y, color='k',
                              lw=0.5)
        image_name = SAVELOC+'/'+prev_outc+'bias_'+prev_ch+'_'+NAME+name+'.svg'
        fig.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
        figs.append(fig)
    # plot histograms
    binsset = np.linspace(-8, 8, 40)
    fig, axs = plt.subplots(figsize=(4, 3))
    # We can also normalize our inputs by the total number of counts
    axs.hist(yevi[SVMAXIS, idxbiasl], bins=binsset,
             density=True, facecolor=GREEN, alpha=0.25)
    axs.hist(yevi[SVMAXIS, idxbiasr], bins=binsset,
             density=True, facecolor='tab:purple', alpha=0.25)
    axs.set_ylim([0, 0.5])
    # axs.set_xlim(ylims)
    y = np.zeros((yevi.shape[1],))
    y[idxbiasl] = 1
    y[idxbiasr] = 2
    assert (y != 0).all()
    fpr, tpr, thresholds = metrics.roc_curve(y, yevi[SVMAXIS, :], pos_label=2)
    AUC = metrics.auc(fpr, tpr)
    axs.set_title('AUC: '+str(np.round(AUC, 3)))
    image_name = SAVELOC + '/'+prev_outc+'bias_hist_' + NAME + name + '.svg'
    fig.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
    plt.close(fig)
    if PREV_CH == 'L':
        plt.close(figs[1].fig)
        return figs[0]
    else:
        plt.close(figs[0].fig)
        return figs[1]


def bias_VS_prob(data_tr, data_flt, ax):
    Xcohs_0_correct = data_tr['Xcohs_0_correct']
    trials_c = data_flt['trials_c']
    yevi_c = data_flt['yevi_c']
    ytruthlabels_c = data_flt['ytruthlabels_c']
    # --- cell 21 OTHER COHERENCES
    cohss = np.unique(Xcohs_0_correct)
    ratio = 3
    for ii in range(0, 9, 1):
        idxcoh0_c = np.where(
            Xcohs_0_correct[np.squeeze(trials_c)] == cohss[ii])[0]
        midval = np.median(yevi_c[SVMAXIS, idxcoh0_c])
        stdval = np.std(yevi_c[SVMAXIS, idxcoh0_c])
        binscoh0_bias = np.linspace(
            midval - stdval * ratio, midval + stdval * ratio, 7)
        stimcoh0_bias = yevi_c[SVMAXIS, idxcoh0_c].copy()
        coh0_cact = ytruthlabels_c[4, idxcoh0_c]
        # binscoh0_bias = np.linspace(-3.5,3.5,8)#np.linspace(-4.5,4.5,10)
        nbinss = len(binscoh0_bias) - 1
        probbinss = np.zeros(nbinss)
        for i in range(1, len(binscoh0_bias)):
            if i == 1:
                idxbin = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
            elif i == len(binscoh0_bias) - 1:
                idxbin = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
            else:
                idxbinl = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
                idxbinh = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
                idxbin = np.intersect1d(idxbinl, idxbinh)
            cactbin = coh0_cact[idxbin]
            if (len(cactbin) == 0):
                probbinss[i - 1] = 2
            else:
                idxx = np.where(cactbin == 3)[0]
                probbinss[i - 1] = len(idxx) * 1.0 / len(cactbin)
        xxx = 0.5 * (binscoh0_bias[1:] + binscoh0_bias[0:-1])
        xxx = xxx[1:-1]
        yyy = probbinss[1:-1]
        color = colors[ii] if cohss[ii] != 0. else 'k'
        ax.plot(xxx, yyy, marker='^', markersize=3, c=color)  # ,c='black')
    # ax.set_ylim([0.0, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    # ax.set_xticks([-4, 0, 4])
    ax.set_ylabel('Prob. right choice')  # , fontsize=14)
    ax.set_xlabel('Transition bias')
    rm_top_right_lines(ax)

    idxcoh0_c = np.where(Xcohs_0_correct[np.squeeze(trials_c)] == 0)[0]
    midval = np.median(yevi_c[SVMAXIS, idxcoh0_c])
    stdval = np.std(yevi_c[SVMAXIS, idxcoh0_c])
    binscoh0_bias = np.linspace(
        midval - stdval * ratio, midval + stdval * ratio, 7)
    stimcoh0_bias = yevi_c[SVMAXIS, idxcoh0_c].copy()
    coh0_cact = ytruthlabels_c[4, idxcoh0_c]
    # binscoh0_bias = np.linspace(-3.5,3.5,8)#np.linspace(-4.5,4.5,10)
    nbinss = len(binscoh0_bias) - 1
    probbinss = np.zeros(nbinss)
    for i in range(1, len(binscoh0_bias)):
        if i == 1:
            idxbin = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
        elif i == len(binscoh0_bias) - 1:
            idxbin = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
        else:
            idxbinl = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
            idxbinh = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
            idxbin = np.intersect1d(idxbinl, idxbinh)
        cactbin = coh0_cact[idxbin]
        if (len(cactbin) == 0):
            probbinss[i - 1] = 2
        else:
            idxx = np.where(cactbin == 3)[0]
            probbinss[i - 1] = len(idxx) * 1.0 / len(cactbin)
    xxx = 0.5 * (binscoh0_bias[1:] + binscoh0_bias[0:-1])
    xxx = xxx[1:-1]
    yyy = probbinss[1:-1]
    z1d = np.polyfit(xxx, yyy, 1)
    curveslopes_correct = z1d[0]
    curveintercept_correct = z1d[1]
    return curveslopes_correct, curveintercept_correct


def bias_VS_prob_AE(data_tr, data_flt, ax):
    Xcohs_0_error = data_tr['Xcohs_0_error']
    trials_e = data_flt['trials_e']
    yevi_e = data_flt['yevi_e']
    ytruthlabels_e = data_flt['ytruthlabels_e']
    # AE OTHER COHERENCES
    cohss = np.unique(Xcohs_0_error)
    ratio = 3
    for ii in range(0, 9, 1):
        idxcoh0_e = np.where(
            Xcohs_0_error[np.squeeze(trials_e)] == cohss[ii])[0]
        midval = np.median(yevi_e[SVMAXIS, idxcoh0_e])
        stdval = np.std(yevi_e[SVMAXIS, idxcoh0_e])
        binscoh0_bias = np.linspace(
            midval - stdval * ratio, midval + stdval * ratio, 7)
        stimcoh0_bias = yevi_e[SVMAXIS, idxcoh0_e].copy()
        coh0_cact = ytruthlabels_e[4, idxcoh0_e]
        nbinss = len(binscoh0_bias) - 1
        probbinss = np.zeros(nbinss)
        for i in range(1, len(binscoh0_bias)):
            if i == 1:
                idxbin = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
            elif i == len(binscoh0_bias) - 1:
                idxbin = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
            else:
                idxbinl = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
                idxbinh = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
                idxbin = np.intersect1d(idxbinl, idxbinh)
            cactbin = coh0_cact[idxbin]
            if (len(cactbin) == 0):
                probbinss[i - 1] = 2
            else:
                idxx = np.where(cactbin == 3 - 2)[0]
                probbinss[i - 1] = len(idxx) * 1.0 / len(cactbin)
        xxx = 0.5 * (binscoh0_bias[1:] + binscoh0_bias[0:-1])
        xxx = xxx[1:-1]
        yyy = probbinss[1:-1]
        color = colors[ii] if cohss[ii] != 0. else 'k'
        ax.plot(xxx, yyy, marker='^', markersize=3, c=color)  # ,c='black')
    ax.set_ylabel('')  # , fontsize=14)
    ax.set_xlabel('Transition bias')  # , fontsize=14)
    # ax.set_ylim([0.0, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['', '', ''])
    # ax.set_xticks([-4, 0, 4])
    rm_top_right_lines(ax)
    idxcoh0_e = np.where(
        Xcohs_0_error[np.squeeze(trials_e)] == 0)[0]
    midval = np.median(yevi_e[SVMAXIS, idxcoh0_e])
    stdval = np.std(yevi_e[SVMAXIS, idxcoh0_e])
    binscoh0_bias = np.linspace(
        midval - stdval * ratio, midval + stdval * ratio, 7)
    stimcoh0_bias = yevi_e[SVMAXIS, idxcoh0_e].copy()
    coh0_cact = ytruthlabels_e[4, idxcoh0_e]
    nbinss = len(binscoh0_bias) - 1
    probbinss = np.zeros(nbinss)
    for i in range(1, len(binscoh0_bias)):
        if i == 1:
            idxbin = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
        elif i == len(binscoh0_bias) - 1:
            idxbin = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
        else:
            idxbinl = np.where(stimcoh0_bias > binscoh0_bias[i - 1])[0]
            idxbinh = np.where(stimcoh0_bias < binscoh0_bias[i])[0]
            idxbin = np.intersect1d(idxbinl, idxbinh)
        cactbin = coh0_cact[idxbin]
        if (len(cactbin) == 0):
            probbinss[i - 1] = 2
        else:
            idxx = np.where(cactbin == 3 - 2)[0]
            probbinss[i - 1] = len(idxx) * 1.0 / len(cactbin)
    xxx = 0.5 * (binscoh0_bias[1:] + binscoh0_bias[0:-1])
    xxx = xxx[1:-1]
    yyy = probbinss[1:-1]
    z1d = np.polyfit(xxx, yyy, 1)
    curveslopes_error = z1d[0]
    curveintercept_error = z1d[1]
    return curveslopes_error, curveintercept_error


# --- MAIN


if __name__ == '__main__':
    if len(sys.argv) == 1:
        METHOD = 'single_decoder'  # separate_decs
    elif len(sys.argv) == 2:
        METHOD = sys.argv[1]
    main_folder = "/home/molano/Dropbox/project_Barna/reset_paper/" +\
        "pop_analysis/models"
    sv_folder = '/home/molano/Dropbox/project_Barna/FOF_project/DMS/' +\
        'pop_analysis/RNNs'
    plt.close('all')
    FIGSIZE = (8, 7.5)
    fig_stats = plt.figure(figsize=(2, 3))
    gs_stats = gridspec.GridSpec(2, 1)
    ax_summ_acc_VS_ctx = fig_stats.add_subplot(gs_stats[0, 0])
    ax_summ_slopes = fig_stats.add_subplot(gs_stats[1, 0])
    fig_corrs = plt.figure(figsize=(7, 7))
    gs_corrs = gridspec.GridSpec(4, 4)
    ax_summ_corrs = fig_corrs.add_subplot(gs_corrs[1:3, 3])
    # figure for x-axis: bin ctxt, y-axis transition bias
    colors = plt.cm.PRGn_r(np.linspace(0, 1, 9))
    # reverse the transition bias labels in AE trials (1-->reverse)
    IMAGE_FORMAT = 'svg'  # e.g .png, .svg, etc.
    BLOCK_CTXT = 0  # TRAIN CONTEXT ON TRUE LABELS 1 OR LAST TRANSITIONS 0
    PREV_CH = 'L'
    NUM_SAMPLES = 1000  # 200
    PLOT_ALL_TRIALS_3D = False
    S_PLOTS = 1
    BOX_WDTH = 0.25
    SVMAXIS = 3
    AX_PREV_CH_OUTC = {'c': [2, 3], 'e': [0, 1]}
    IPOOLS = 500  # 100  # number of iterations in SVM (500)
    IEACHTRAIN = 500  # 100  # number of trials in each iteration (200)
    RUN_ALL = True
    RERUN = True
    DOREVERSE = 0
    NAME = METHOD+'_rev_'+str(DOREVERSE)+'_ctxt_blk_'+str(BLOCK_CTXT)+'_' +\
        str(IPOOLS)+'_'+str(IEACHTRAIN)
    print('Using ' + METHOD + ' method')
    print('Reverse: ' + str(DOREVERSE))
    print('Run all nets: ' + str(RUN_ALL))
    print('Context defined by block: ' + str(BLOCK_CTXT))
    sel_seeds = ['9', '8']  # seed
    for i_net, net in enumerate(['16', '2_BiasCorr']):
        if net == '16':
            BOTTOM_3D = -6  # where to plot blue/red projected dots in 3D figure
            XLIMS_2D = [-3, 3]
            YLIMS_2D = [-7, 7]
            YTICKS_2D = [-6., 0., 6.]
            XTICKS_2D = [-2., 0., 2.]
            CTXT_BIN = np.linspace(0, 1.65, 7)  # (0,1.8,7)
            XLIM_CTXT = [12000, 13000]
            YTICKS_CTXT = [-2, 0, 2]
            YLIM_CTXT = [-2.2, 2.2]
        elif net == '2_BiasCorr':
            BOTTOM_3D = -4  # where to plot blue/red projected dots in 3D figure
            XLIMS_2D = [-3.5, 3.5]
            YLIMS_2D = [-5, 5]
            YTICKS_2D = [-4., 0., 4.]
            XTICKS_2D = [-3., 0., 3.]
            CTXT_BIN = np.linspace(0, 1.65, 7)  # (0,1.8,7)
            XLIM_CTXT = [11000, 12000]
            YTICKS_CTXT = [-5, 0, 1]
            YLIM_CTXT = [-6., 1.5]
        fig_main = plt.figure(figsize=FIGSIZE)
        gs = gridspec.GridSpec(4, 5)
        ax_ctx = fig_main.add_subplot(gs[0, 1])
        pos = ax_ctx.get_position()
        margin = pos.width/8
        factor = 2
        ax_ctx .set_axis_off()
        ax_ctx = plt.axes([pos.x0, pos.y0, pos.width*factor, pos.height])
        ax_ctx_ae = plt.axes([pos.x0+pos.width*factor+margin, pos.y0, pos.width/3,
                              pos.height])
        ax_ctx_ac = plt.axes([pos.x0+pos.width*factor+2*margin+pos.width/3, pos.y0,
                              pos.width/3, pos.height])
        pos = ax_ctx_ac.get_position()
        ax_ctx_change = plt.axes([pos.x0+pos.width+5*margin, pos.y0, 3*pos.width,
                                  pos.height])
        # x-axis bin ctxt, y-axis transition bias
        ctx_tb_trcs_AC, ctx_tb_trcs_AE = [], []
        total_pearcorrL_correct, total_pearcorrR_correct,\
            total_pearcorrL_error, total_pearcorrR_error = [], [], [], []
        totalslope_correct, totalintercept_correct = [], []
        totalslope_error, totalintercept_error = [], []
        auc_context_AC = []
        auc_context_AE = []
        seeds = []
        prop_cl_ctxt_tr_mat = []

        folders = glob.glob(main_folder + "/n_ch_" + net + "/seed_*")
        folders = [f for f in folders if f.find('.') == -1]
        if not RUN_ALL:
            folders = [f for f in folders if f.find('d_' + sel_seeds[i_net]) != -1]
        for f in folders:
            print(f)
            seed = f[f.find('seed') + 5:]
            seeds.append(seed)
            print(seed)
            data = np.load(f + "/test_2AFC_activity/data.npz")
            SAVELOC = sv_folder + '/figures/n_ch_' + net + '_seed_' + seed + '/'
            if not os.path.exists(SAVELOC):
                os.mkdir(SAVELOC)
            # GET QUANTITIES
            print('Get quantities')
            # label -- bias, label -- conditions
            data_trials = get_all_quantities(data, numtrans=3)
            wc, bc, we, be = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),\
                np.zeros((2, 2))
            print('Get decoding axes')
            data_name = SAVELOC + '/dec_axes_' + NAME + '.npz'
            if not os.path.exists(data_name) or RERUN:
                data_dec = get_dec_axes(data_trials, wc, bc, we, be, 'decoding',
                                        DOREVERSE=DOREVERSE)
                np.savez(data_name, **data_dec)
            data_dec = np.load(data_name)
            # FLATTEN DATA
            data_flatten = flatten_data(data_tr=data_trials, data_dec=data_dec)

            # @YX all trials
            print('Get projecting data')
            data_trials_0 = get_all_quantities(data, numtrans=0)
            all_trials =\
                data_trials_0['Xdata_trialidx_correct'].shape[0] +\
                data_trials_0['Xdata_trialidx_error'].shape[0]

            clear_ctxt_trials =\
                data_trials['Xdata_trialidx_correct'].shape[0] +\
                data_trials['Xdata_trialidx_error'].shape[0]
            prop_cl_ctxt_tr = clear_ctxt_trials/all_trials
            print('Proportion of clear context trials: ' +
                  str(np.round(prop_cl_ctxt_tr, 3)))
            prop_cl_ctxt_tr_mat.append(prop_cl_ctxt_tr)
            data_dec_0 = get_dec_axes(data_trials_0, data_dec['coefs_correct'],
                                      data_dec['intercepts_correct'],
                                      data_dec['coefs_error'],
                                      data_dec['intercepts_error'],
                                      mode='projecting')
            # FLATTEN DATA
            data_flatten_0 = flatten_data(data_tr=data_trials_0,
                                          data_dec=data_dec_0)
            print('Temporal dynamics')
            # TEMPORAL DYNAMICS CONTEXT
            if seed == sel_seeds[i_net]:
                ax = ax_ctx
                ax_ae = ax_ctx_ae
                ax_ac = ax_ctx_ac
                ax_ch = ax_ctx_change
            else:
                fig_temp = plt.figure(figsize=FIGSIZE)
                gs_temp = gridspec.GridSpec(4, 5)
                ax = fig_temp.add_subplot(gs[0, 1])
                pos = ax.get_position()
                ax .set_axis_off()
                ax = plt.axes([pos.x0, pos.y0, pos.width*factor, pos.height])
                ax_ae = plt.axes([pos.x0+pos.width*factor+margin, pos.y0,
                                  pos.width/3, pos.height])
                ax_ac = plt.axes([pos.x0+pos.width*factor+2*margin+pos.width/3,
                                  pos.y0, pos.width/3, pos.height])
                pos = ax_ac.get_position()
                ax_ch = plt.axes([pos.x0+pos.width+5*margin, pos.y0,
                                  3*pos.width, pos.height])
            auc_ac, auc_ae =\
                context_dynamics(data=data, wc=data_dec['coefs_correct'],
                                 bc=data_dec['intercepts_correct'],
                                 ax=ax, ax_ae=ax_ae, ax_ac=ax_ac, ax_ch=ax_ch)
            auc_context_AC.append(auc_ac)
            auc_context_AE.append(auc_ae)
            yevi_c = data_flatten['yevi_c']

            print('3D projection')
            # PLOT 3D PROJECTION
            projection_3D(data_flt=data_flatten, data_flt_light=data_flatten_0)
            print('2D projections')
            # PLOT 2D PROJECTIONS
            ac_2d = projections_2D(data_flt=data_flatten, prev_outc='c')
            if seed == sel_seeds[i_net]:
                SeabornFig2Grid(ac_2d, fig_main, gs[2, 0])
            else:
                SeabornFig2Grid(ac_2d, fig_temp, gs_temp[2, 0])
            print('Transition bias VS prob. Right')
            # PLOT BIA VS PROBABILITY OF RIGHT CHOICE
            if seed == sel_seeds[i_net]:
                ax = fig_main.add_subplot(gs[3, 0])
            else:
                ax = fig_temp.add_subplot(gs_temp[3, 0])
            curveslopes_correct, curveintercept_correct =\
                bias_VS_prob(data_tr=data_trials_0, data_flt=data_flatten_0, ax=ax)
            print('After error analyses')
            # AFTER ERROR
            print('2D projections after error')
            # PROJECTIONS 2D AFTER ERROR
            ae_2d = projections_2D(data_flt=data_flatten, prev_outc='e')
            if seed == sel_seeds[i_net]:
                SeabornFig2Grid(ae_2d, fig_main, gs[2, 1])
            else:
                SeabornFig2Grid(ae_2d, fig_temp, gs_temp[2, 1])
            print('Transition bias VS prob. Right after error and slopes')
            # PLOT BIA VS PROBABILITY OF RIGHT CHOICE
            if seed == sel_seeds[i_net]:
                ax = fig_main.add_subplot(gs[3, 1])
            else:
                ax = fig_temp.add_subplot(gs_temp[3, 1])
            curveslopes_error, curveintercept_error =\
                bias_VS_prob_AE(data_tr=data_trials_0, data_flt=data_flatten_0,
                                ax=ax)
            totalslope_correct.append(curveslopes_correct)
            totalintercept_correct.append(curveintercept_correct)
            totalslope_error.append(curveslopes_error)
            totalintercept_error.append(curveintercept_error)
            # x-axis bin ctxt, y-axis transition bias
            corrl_ac, corrr_ac, corrl_ae, corrr_ae, ctx_tb_trcs =\
                ctxtbin_defect(data_flatten_0)
            ctx_tb_trcs_AC.append(ctx_tb_trcs[0])
            ctx_tb_trcs_AE.append(ctx_tb_trcs[1])
            total_pearcorrL_correct.append(corrl_ac)
            total_pearcorrR_correct.append(corrr_ac)
            total_pearcorrL_error.append(corrl_ae)
            total_pearcorrR_error.append(corrr_ae)
            print('2D projections all points')
            # corr all points
            # AC
            ac_2d = projections_2D(data_flt=data_flatten_0, prev_outc='c',
                                   fit=True, name='_all_pts')
            if seed == sel_seeds[i_net]:
                SeabornFig2Grid(ac_2d, fig_corrs, gs_corrs[2*i_net+1, 0])
            # AE
            print('2D projections after error')
            # PROJECTIONS 2D AFTER ERROR
            ae_2d = projections_2D(data_flt=data_flatten_0, prev_outc='e',
                                   fit=True, name='_all_pts')
            if seed == sel_seeds[i_net]:
                SeabornFig2Grid(ae_2d, fig_corrs, gs_corrs[2*i_net+1, 1])

            if seed == sel_seeds[i_net]:
                gs.tight_layout(fig_main)
                image_name = sv_folder + '/figures/main_fig_' + NAME+'_seed_' +\
                    seed+'_'+net+'.svg'
                fig_main.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
                image_name = os.path.split(SAVELOC[:-1])[0]+'/all_figs_png/n_' +\
                    net+'/'+os.path.split(SAVELOC[:-1])[1]+NAME+'.svg'
                fig_main.savefig(image_name, format='svg', dpi=300)
            else:
                gs_temp.tight_layout(fig_temp)
                image_name = SAVELOC + 'main_fig_' + NAME + '.svg'
                fig_temp.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
                image_name = os.path.split(SAVELOC[:-1])[0]+'/all_figs_png/n_' +\
                    net+'/'+os.path.split(SAVELOC[:-1])[1]+NAME+'.svg'
                fig_temp.savefig(image_name, format='svg', dpi=300)
                plt.close(fig_temp)
        if RUN_ALL:
            stats = {'total_pearcorrL_correct': total_pearcorrL_correct,
                     'total_pearcorrL_error': total_pearcorrL_error,
                     'total_pearcorrR_correct': total_pearcorrR_correct,
                     'total_pearcorrR_error': total_pearcorrR_error,
                     'totalslope_correct': totalslope_correct,
                     'totalslope_error': totalslope_error,
                     'auc_context_AC': auc_context_AC,
                     'auc_context_AE': auc_context_AE,
                     'ctx_tb_trcs_AC': ctx_tb_trcs_AC,
                     'ctx_tb_trcs_AE': ctx_tb_trcs_AE,
                     'prop_cl_ctxt_tr_mat': prop_cl_ctxt_tr_mat,
                     'seeds': seeds}
            np.savez(sv_folder+'/figures/stats_'+NAME+'_'+net+'.npz',
                     **stats)

        stats = np.load(sv_folder+'/figures/stats_'+NAME+'_'+net+'.npz')
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(net)
        print('AUC after correct')
        print(stats['auc_context_AC'])
        print('Mean: '+str(np.mean(stats['auc_context_AC'])) +
              '\nstd: '+str(np.std(stats['auc_context_AC'])))
        print('AUC after error')
        print(stats['auc_context_AE'])
        print('Mean: '+str(np.mean(stats['auc_context_AE'])) +
              '\nstd: '+str(np.std(stats['auc_context_AE'])))
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        total_pearcorrL_correct = stats['total_pearcorrL_correct']
        total_pearcorrL_error = stats['total_pearcorrL_error']
        totalslope_correct =\
            stats['totalslope_correct']/np.median(stats['totalslope_correct'])
        totalslope_error =\
            stats['totalslope_error']/np.median(stats['totalslope_correct'])
        ctx_tb_trcs_AC = stats['ctx_tb_trcs_AC']
        ctx_tb_trcs_AE = stats['ctx_tb_trcs_AE']
        box_plot(data=total_pearcorrL_correct, ax=ax_summ_corrs, x=2*i_net+0.25,
                 lw=.5, fliersize=4, color=ORANGE, widths=BOX_WDTH)
        box_plot(data=total_pearcorrL_error, ax=ax_summ_corrs, x=2*i_net+0.75,
                 lw=.5, fliersize=4, color='k', widths=BOX_WDTH)
        box_plot(data=totalslope_correct, ax=ax_summ_slopes, x=2*i_net+0.25,
                 lw=.5, fliersize=4, color=ORANGE, widths=BOX_WDTH)
        box_plot(data=totalslope_error, ax=ax_summ_slopes, x=2*i_net+0.75,
                 lw=.5, fliersize=4, color='k', widths=BOX_WDTH)
        if net == '16':
            xs = (CTXT_BIN[:-1]+CTXT_BIN[1:])/2.0
            ax_summ_acc_VS_ctx.errorbar(xs, np.mean(ctx_tb_trcs_AC, axis=0),
                                        np.std(ctx_tb_trcs_AC, axis=0),
                                        color=ORANGE)
            ax_summ_acc_VS_ctx.errorbar(xs, np.mean(ctx_tb_trcs_AE, axis=0),
                                        np.std(ctx_tb_trcs_AE, axis=0), color='k')
    # tune panels
    ax_summ_slopes.set_ylabel('Transition bias impact')
    ax_summ_slopes.set_xticks([0.5, 2.5])
    ax_summ_slopes.set_xticklabels(['Pre-trained', '2AFC-trained'])
    ax_summ_acc_VS_ctx.set_ylabel("Transition bias accuracy")
    ax_summ_acc_VS_ctx.set_xlabel("Context encoding")
    ax_summ_acc_VS_ctx.set_xticks([0, 0.5, 1, 1.5])
    ax_summ_corrs.set_ylabel("Correlation(Bias, Context)")
    ax_summ_corrs.axhline(y=0, linestyle='--', color='k', lw=0.5)
    ax_summ_corrs.set_xticks([0.5, 2.5])
    ax_summ_corrs.set_xticklabels(['Pre-trained', '2AFC-trained'])
    rm_top_right_lines(ax=ax_summ_corrs)
    rm_top_right_lines(ax=ax_summ_slopes)
    rm_top_right_lines(ax=ax_summ_acc_VS_ctx)
    gs_stats.tight_layout(fig_stats)
    image_name = sv_folder + '/figures/stats_fig_' + NAME + '.svg'
    fig_stats.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
    image_name = sv_folder + '/figures/trb_ctx_corrs_' + NAME + '.svg'
    fig_corrs.savefig(image_name, format=IMAGE_FORMAT, dpi=300)
