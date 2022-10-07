# Load packages;
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# Import libraries
from collections import Counter
from mpl_toolkits import mplot3d

# from sklearn import datasets, svm, pipeline
# from sklearn.kernel_approximation import (RBFSampler, Nystroem)
# from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import os
from sklearn import metrics
from statannot import add_stat_annotation
from collections import Counter

# self-defined functions
import general_util_ctxtgt as guc
import generate_pseudo_trials as gpt
import required_data_dec as rdd
import bootstrap_linear as bl
import gaincontrol as gc

from scipy.stats import  mannwhitneyu as mwu
from scipy import stats
import statsmodels.api as sm 

import pickle as pk
from numpy import linalg as la
import warnings
warnings.filterwarnings("ignore")

image_format = 'svg'  # e.g .png, .svg, etc.
dpii = 300

# %
# import get_data as gd

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    print(np.shape(cov))
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D()         .rotate_deg(45)         .scale(scale_x, scale_y)         .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

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
        sns.distplot(df_group[col_x].values,
                     ax=g.ax_marg_x, color=c, hist=False)
        sns.distplot(df_group[col_y].values, ax=g.ax_marg_y, color=c, hist=False,
                     vertical=True)
        counter += 1
    # plt.legend(legends)
    return g


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

    # overall_ac_ae_ratio = np.mean(ac_ae_ratio_set)
    # lst = [overall_ac_ae_ratio,Xdata_set, Xdata_hist_set,
    #        ylabels_set, ylabels_hist_set,
    #        Xcohs_0, files, metadata]
    # stg = ["overall_ac_ae_ratio, Xdata_set, Xdata_hist_set,"
    #        "ylabels_set, ylabels_hist_set,"
    #        "Xcohs_0, files, metadata"]
    # d = list_to_dict(lst=lst, string=stg)

def unpack_dicts(data_tr):
    
    Xhistset,yhistlabelsset, Xset, ylabelsset = data_tr['Xdata_hist_set'].item(), data_tr['ylabels_hist_set'].item(),data_tr['Xdata_set'].item(), data_tr['ylabels_set'].item()
    mdata,oratio =data_tr['metadata'],data_tr['overall_ac_ae_ratio'].item()
    
    lst = [oratio,Xset, Xhistset,
           ylabelsset, yhistlabelsset,
           data_tr['Xcohs_0'], data_tr['files'], mdata]
    stg = ["overall_ac_ae_ratio, Xdata_set, Xdata_hist_set,"
           "ylabels_set, ylabels_hist_set,"
           "Xcohs_0, files, metadata"]
    d = list_to_dict(lst=lst, string=stg)
    return d
    

def get_all_quantities(files, numtrans=0, SKIPNAN=0):
    """
    Obtain dataset (responses, labels) for generating pseudo-trials

    Parameters
    ----------
    files: dict
        sessions' data, containing neuron responses/env/behaviours

    SKIPNAN: bool
        skip the nan (beh/env/resp) trials or not

    Returns
    -------
    d : dict 
        dictionary, the items are dataset for generating history pseudo-trials
        and behaviour pseudo-trials

    """
    icount = 0
    ## dataset for generating behaviour pseudo-trials
    Xdata_set   = {}
    ylabels_set = {}
    ## dataset for generating history pseudo-trials
    Xdata_hist_set   = {}
    ylabels_hist_set = {}

    ## meta data for files
    metadata       = {}
    pseudo_neurons = 0
    remarkfile     = ""
    ac_ae_ratio_set = []
    for i in range(len(files)):
        for T in ['correct', 'error']:
            Xdata_set[i, T]   = {}
            ylabels_set[i, T] = {}

            Xdata_hist_set[i, T]   = {}
            ylabels_hist_set[i, T] = {}

    
    files = [f for f in files if f.find('data_dec') == -1]
    for idxs, f in enumerate(files):
        if icount < 0:
            break
        data = np.load(f, allow_pickle=True)
        print('file', f)

        tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc =\
            guc.get_RNNdata_ctxtgt(data)
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

        ### Function to extract Neuronal responses/Environment variables/Behaviour labels for individual sessions
        Xdata, ydata, Xdata_idx, Xconds_2, Xacts_1,\
            Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0, Xgts_0,\
            Xcohs_0, Xdata_trialidx, Xstates =\
            rdd.req_quantities_0(stim_trials, stm, dyns, gt, choice, eff_choice,
                                 rw, obsc, BLOCK_CTXT=1)

        ### Function to separate after correct dataset from after error dataset 
        ace_ratio,Xdata_correct, Xdata_error, correct_trial, error_trial, rses_correct,\
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
        ac_ae_ratio_set.append(ace_ratio)


        ### To integrate all the env/behaviour variables that are further used as labels together
        ylabels_correct = rdd.set_ylabels(Xdata_correct, ydata_choices_correct,
                                          ydata_conds_correct, ydata_xor_correct,
                                          ydata_bias_correct,
                                          ydata_cchoices_correct, Xcohs_0_correct)
        ylabels_error   = rdd.set_ylabels(Xdata_error, ydata_choices_error,
                                        ydata_conds_error, ydata_xor_error,
                                        ydata_bias_error, ydata_cchoices_error,
                                        Xcohs_0_error)

        pseudo_neurons += np.shape(Xdata_correct)[1]

        ### cluster:
        ###     history trials according to their history states -- previous choice, block context
        ###     behaviour trials according to their history states and the stimulus coherence
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

    overall_ac_ae_ratio = np.mean(ac_ae_ratio_set)
    lst = [overall_ac_ae_ratio,Xdata_set, Xdata_hist_set,
           ylabels_set, ylabels_hist_set,
           Xcohs_0, files, metadata]
    stg = ["overall_ac_ae_ratio, Xdata_set, Xdata_hist_set,"
           "ylabels_set, ylabels_hist_set,"
           "Xcohs_0, files, metadata"]
    d = list_to_dict(lst=lst, string=stg)
    return d


def filter_sessions(data_tr, unique_states, unique_cohs):
    """
    Filtering sessions with sufficient state-trials

    Parameters
    ----------
    data_tr: dict
        dataset for generating pseudo-trials

    unique_states: array
        history states: 4 for ac and 4 for ae

    unique_cohs: array
        stimulus coherences -- positive/zero/negative

    Returns
    -------
    false_files : array 
        some specific states (hist/beh) have zero trials in these failed sessions.

    """
    Xdata_set, Xdata_hist_set, ylabels_set, ylabels_hist_set, files =\
        data_tr['Xdata_set'], data_tr['Xdata_hist_set'], data_tr['ylabels_set'],\
        data_tr['ylabels_hist_set'], data_tr['files']
    _correct_false, _error_false, min_beh_trials, num_beh_trials =\
        gpt.valid_beh_trials(Xdata_set, ylabels_set, unique_states,
                             unique_cohs, files, THRESH_TRIAL)
    false_files = np.union1d(_correct_false, _error_false)
    MIN_TRIALS = [min_beh_trials]
    coh_ch_stateratio_correct, coh_ch_stateratio_error = gpt.coh_ch_stateratio(Xdata_set,ylabels_set,unique_states,unique_cohs,files, false_files)
    

    return false_files, MIN_TRIALS, num_beh_trials,coh_ch_stateratio_correct, coh_ch_stateratio_error

def dataset_generate(data_tr, unique_states, unique_cohs, files, falsefiles,THRESH_TRIAL): 
    Xdata_set, ylabels_set = data_tr['Xdata_set'],data_tr['ylabels_set']
    Xdata_hist_trainset, ylabels_hist_trainset, Xdata_hist_testset, ylabels_hist_testset,Xdata_psy_trainset, ylabels_psy_trainset, Xdata_psy_testset, ylabels_psy_testset,Xdata_beh_trainset, ylabels_beh_trainset, Xdata_beh_testset, ylabels_beh_testset =gpt.train_test_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files, falsefiles, THRESH_TRIAL)
    
    lst = [Xdata_hist_trainset,Xdata_hist_testset,
           ylabels_hist_trainset, ylabels_hist_testset,
           Xdata_beh_trainset, Xdata_beh_testset,
           ylabels_beh_trainset, ylabels_beh_testset,
           Xdata_psy_trainset, Xdata_psy_testset,
           ylabels_psy_trainset, ylabels_psy_testset,
           files]
    stg = ["Xdata_hist_trainset, Xdata_hist_testset,"
           "ylabels_hist_trainset, ylabels_hist_testset,"
           "Xdata_beh_trainset, Xdata_beh_testset,"
           "ylabels_beh_trainset, ylabels_beh_testset,"
           "Xdata_psy_trainset, Xdata_psy_testset,"
           "ylabels_psy_trainset, ylabels_psy_testset,"
           "files"]
    d = list_to_dict(lst=lst, string=stg)
    return d
    

def get_dec_axes(data_tr, wc, bc, we, be,nselect, false_files,pop_correct, pop_zero, pop_error, single_pop, USE_POP, mode='decoding',
                 DOREVERSE=0, CONTROL=0, STIM_PERIOD=0, RECORD_TRIALS=1, REC_TRIALS_SET=[], PCA_only=0, mmodel=[]):
    """
    Obtaining SVM decoders for history information, using neuronal responses before stimulus presentations

    Parameters
    ----------
    data_tr: dict
        dataset for generating pseudo-trials

    wc,bc,we,be: array
        (reloading) weights and bias of SVM decoders

    false_files : array 
        some specific states (hist/beh) have zero trials in these failed sessions.

    DOREVERSE: bool
        if reversing the labels for ae trials or not 

    CONTROL: bool
        training decoders using only ae trials as a CONTROL

    RECORD_TRIALS: bool
        recording selected trials and decoders (or reloading saved data)

    REC_TRIALS_SET: dict 
        reloading saved data

    Returns
    -------
    d : dict
        dictionary contains: weights and bias of the SVM decoders for history information.
        labels of the testing dataset, encodings of the testing dataset. 

    """
    
    unique_states = np.arange(8)
    unique_cohs = [-1,0,1]
    
    files = data_tr['files']
    data_traintest_tr = dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
    ### compare single neuron activity       
    Xdata_hist_trainset, ylabels_hist_trainset = data_traintest_tr['Xdata_hist_trainset'], \
    data_traintest_tr['ylabels_hist_trainset']

    #### valid neuron number  --- take the advantage of single cell selectivity 
    NN = len(nselect)# np.shape(Xmerge_hist_trials_correct[4])[1]
    if(RECORD_TRIALS == 1):
        REC_TRIALS_SET = {}
        for itr in range(NITERATIONS):
            REC_TRIALS_SET[itr] = {}
            
        
    if(RECORD_TRIALS == 1):    
        mmodel, stats_correct, stats_error, stats_correct_pop, stats_error_pop, coeffs, intercepts, Xtest_set_correct, ytest_set_correct, yevi_set_correct, yevi_set_correct_supp,\
            Xtest_set_error, ytest_set_error, yevi_set_error,  yevi_set_error_supp, REC_TRIALS_SET\
            = bl.bootstrap_linsvm_step_gaincontrol(data_tr, NN, 
                                        unique_states, unique_cohs, nselect, files,
                                        false_files, pop_correct, pop_zero, pop_error, single_pop, 1, type, DOREVERSE=DOREVERSE,
                                        CONTROL=CONTROL, STIM_PERIOD=STIM_PERIOD, n_iterations=NITERATIONS,
                                        N_pseudo_dec=NPSEUDODEC, ACE_RATIO = ACE_RATIO,
                                        train_percent=PERCENTTRAIN,
                                        RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=REC_TRIALS_SET,mmodel=mmodel,PCA_n_components=PCA_n_components)# CONDITION_CTXT
            
    else:
        mmodel, stats_correct, stats_error, coeffs, intercepts, Xtest_set_correct, ytest_set_correct,\
            yevi_set_correct, Xtest_set_error, ytest_set_error, yevi_set_error,\
            REC_TRIALS_SET\
            = bl.bootstrap_linsvm_proj_step(wc, bc, data_tr,NN,
                                            unique_states,
                                            unique_cohs, files, false_files, type,
                                            DOREVERSE=DOREVERSE,
                                            n_iterations=NITERATIONS,
                                            N_pseudo_dec=NPSEUDODEC,
                                            train_percent=PERCENTTRAIN,
                                            RECORD_TRIALS=RECORD_TRIALS,
                                            RECORDED_TRIALS_SET=REC_TRIALS_SET,mmodel=mmodel,PCA_n_components=PCA_n_components)
    if USE_POP ==1:
        guc.cross_gaincontrol(stats_correct,stats_error, coeffs, intercepts, ytest_set_correct, yevi_set_correct, ytest_set_error, yevi_set_error, label_axis=3,evi_axis=3, CONTROL=1)      


    lst = [stats_correct, stats_error, 
           stats_correct_pop, stats_error_pop, 
           coeffs, intercepts,
           ytest_set_correct,
           yevi_set_correct, yevi_set_correct_supp,
           coeffs, intercepts,
           ytest_set_error,  yevi_set_error, yevi_set_error_supp,
           REC_TRIALS_SET]
    stg = ["stats_correct, stats_error, "
           "stats_correct_pop, stats_error_pop, "
           "coefs_correct, intercepts_correct,"
           "ytest_set_correct, "
           "yevi_set_correct, yevi_set_correct_supp, "
           "coefs_error, intercepts_error,"
           "ytest_set_error, yevi_set_error, yevi_set_error_supp,"
           "REC_TRIALS_SET"]
    d = list_to_dict(lst=lst, string=stg)
    return d, mmodel, Xtest_set_correct, Xtest_set_error


def flatten_data(data_tr, data_dec):
    """
    Flatten the encoding of history information (session-by-session)

    Parameters
    ----------
    data_tr: dict
        dataset for generating pseudo-trials

    data_dec: dict
        dataset containing the encodings of history information and labels


    Returns
    -------
    d : dict
        flatten data

    """
    yevi_set_correct      = data_dec['yevi_set_correct'] # mixed-selectivity populations
    yevi_set_correct_supp = data_dec['yevi_set_correct_supp'] # 3 Sept 
    ytest_set_correct     = data_dec['ytest_set_correct']
    IPOOLS = NITERATIONS ## should be consistent with the number of iterations used in bootstrap
    
    ### flatten data --- after correct
    nlabels        = np.shape(np.squeeze(ytest_set_correct[0, :, :]))[1]
    ytruthlabels_c = np.zeros((nlabels, 1))
    yevi_c         = np.zeros((3 + 1 + 1, 1))
    ### Gaussian assumption
    dprimes_c    = np.zeros(IPOOLS)
    dprimes_repc = np.zeros(IPOOLS)
    dprimes_altc = np.zeros(IPOOLS) # overall
    
    dprimes_lc = np.zeros(IPOOLS)
    dprimes_rc = np.zeros(IPOOLS) # overall

    AUCs_c    = np.zeros(IPOOLS)
    AUCs_repc = np.zeros(IPOOLS)
    AUCs_altc = np.zeros(IPOOLS)

    ### separation between left and right categorization
    stds_repc_lr = np.zeros((IPOOLS,2))
    stds_altc_lr = np.zeros((IPOOLS,2)) # two clusters

    
    AUCs_lc = np.zeros(IPOOLS)
    AUCs_rc = np.zeros(IPOOLS)

    for i in range(IPOOLS):
        hist_evi    = yevi_set_correct[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        test_labels = ytest_set_correct[i, :, :]
        ytruthlabels_c = np.append(
            ytruthlabels_c, test_labels[idx, :].T, axis=1)
        yevi_c = np.append(yevi_c, (yevi_set_correct[i, idx, :]).T, axis=1)
        # dprimes_c[i] =\
        #     guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, SVMAXIS]),
        #                           np.squeeze(ytest_set_correct[i, :, SVMAXIS])) # aligned direction
        dprimes_c[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, SVMAXIS]-yevi_set_correct[i,:,SVMAXIS-1]),
                                  np.squeeze(ytest_set_correct[i, :, SVMAXIS]))

        ### calculate AUC
        yauc_c_org = np.squeeze(ytest_set_correct[i, :, SVMAXIS])
        yauc_c     = np.zeros_like(yauc_c_org)
        yauc_c[np.where(yauc_c_org == 0+2)[0]] = 1
        yauc_c[np.where(yauc_c_org == 1+2)[0]] = 2
        assert (yauc_c != 0).all()
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     yauc_c, np.squeeze(yevi_set_correct[i, :, SVMAXIS]), pos_label=2)
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_c, np.squeeze(yevi_set_correct[i, :, SVMAXIS]-yevi_set_correct[i,:,SVMAXIS-1]), pos_label=2)
        auc_ac = metrics.auc(fpr, tpr)
        AUCs_c[i] = auc_ac

        ### calculate AUC conditioned on Block contexts --- overall
        ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
            np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        yauc_c_ctxtrep, yauc_c_ctxtalt =\
            np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        # yauc_c_evirep, yauc_c_evialt =\
        #     np.squeeze(yevi_set_correct[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(yevi_set_correct[i, ctxtalt, SVMAXIS])
        yauc_c_evirep, yauc_c_evialt =\
            np.squeeze(yevi_set_correct[i, ctxtrep, SVMAXIS]-yevi_set_correct[i, ctxtrep, SVMAXIS-1]),\
            np.squeeze(yevi_set_correct[i, ctxtalt, SVMAXIS]-yevi_set_correct[i, ctxtalt, SVMAXIS-1]) # mental states coexist
        # print('ac overall:',len(ctxtrep),len(ctxtalt),'--------',yauc_c_evirep)
        dprimes_repc[i] = guc.calculate_dprime(yauc_c_evirep, yauc_c_ctxtrep)
        dprimes_altc[i] = guc.calculate_dprime(yauc_c_evialt, yauc_c_ctxtalt)

        #### calculate the distance between left and right previous choices
        yauc_c_ctxtrep_lr, yauc_c_ctxtalt_lr =\
            np.squeeze(ytest_set_correct[i, ctxtrep, 0]),\
            np.squeeze(ytest_set_correct[i, ctxtalt, 0])
        yauc_c_evirep_lr, yauc_c_evialt_lr =\
            np.squeeze(yevi_set_correct[i, ctxtrep, 0]),\
            np.squeeze(yevi_set_correct[i, ctxtalt, 0])
        stds_repc_lr[i,:] = guc.calculate_disperse(yauc_c_evirep_lr,yauc_c_ctxtrep_lr)
        stds_altc_lr[i,:] = guc.calculate_disperse(yauc_c_evialt_lr,yauc_c_ctxtalt_lr)


        yauc_c_ctxtrep, yauc_c_ctxtalt = yauc_c_ctxtrep-1, yauc_c_ctxtalt-1

        fpr_rep, tpr_rep, thresholds = metrics.roc_curve(
            yauc_c_ctxtrep, yauc_c_evirep, pos_label=2)
        auc_ac_rep = metrics.auc(fpr_rep, tpr_rep)
        AUCs_repc[i] = auc_ac_rep

        fpr_alt, tpr_alt, thresholds = metrics.roc_curve(
            yauc_c_ctxtalt, yauc_c_evialt, pos_label=2)
        auc_ac_alt = metrics.auc(fpr_alt, tpr_alt)
        AUCs_altc[i] = auc_ac_alt
        
        # ### calculate dprime context relevant
        # ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        # yauc_c_ctxtrep, yauc_c_ctxtalt =\
        #     np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        # yauc_c_evirep_rel, yauc_c_evialt_rel =\
        #     np.squeeze(yevi_set_correct[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(yevi_set_correct[i, ctxtalt, SVMAXIS])
        # # print('ac relevant:',len(ctxtrep),len(ctxtalt),'--------',yauc_c_evirep)
        # dprimes_repc_rel[i] = guc.calculate_dprime(yauc_c_evirep_rel, yauc_c_ctxtrep)
        # dprimes_altc_rel[i] = guc.calculate_dprime(yauc_c_evialt_rel, yauc_c_ctxtalt)
        
        
        # ### calculate dprime context cross
        # ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        # yauc_c_ctxtrep, yauc_c_ctxtalt =\
        #     np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        # yauc_c_evirep_cross, yauc_c_evialt_cross =\
        #     np.squeeze(yevi_set_correct[i, ctxtrep, 2]),\
        #     np.squeeze(yevi_set_correct[i, ctxtalt, 2])
            
        # # print('ac cross:',len(ctxtrep),len(ctxtalt),'--------',yauc_c_evirep)
        # dprimes_repc_cross[i] = guc.calculate_dprime(yauc_c_evirep_cross, yauc_c_ctxtrep)
        # dprimes_altc_cross[i] = guc.calculate_dprime(yauc_c_evialt_cross, yauc_c_ctxtalt)
        
        
        ### calculate AUC conditioned on Previous choice --- overall
        prevchl, prevchr = np.where(ytest_set_correct[i, :, 0] == 0+2)[0],\
            np.where(ytest_set_correct[i, :, 0] == 1+2)[0]
        yauc_c_prevchl, yauc_c_prevchr =\
            np.squeeze(ytest_set_correct[i, prevchl, SVMAXIS]),\
            np.squeeze(ytest_set_correct[i, prevchr, SVMAXIS])
        # yauc_c_evil, yauc_c_evir =\
        #     np.squeeze(yevi_set_correct[i, prevchl, SVMAXIS]),\
        #     np.squeeze(yevi_set_correct[i, prevchr, SVMAXIS])
        yauc_c_evil, yauc_c_evir =\
            np.squeeze(yevi_set_correct[i, prevchl, SVMAXIS]-yevi_set_correct[i, prevchl, SVMAXIS-1]),\
            np.squeeze(yevi_set_correct[i, prevchr, SVMAXIS]-yevi_set_correct[i, prevchr, SVMAXIS-1])
            
        # print('ac overall:',len(prevchr),len(prevchl),'--------',yauc_c_evil)
        dprimes_lc[i] = guc.calculate_dprime(yauc_c_evil, yauc_c_prevchl)
        dprimes_rc[i] = guc.calculate_dprime(yauc_c_evir, yauc_c_prevchr)

        yauc_c_prevchl, yauc_c_prevchr = yauc_c_prevchl-1, yauc_c_prevchr-1
        
        
        fpr_l, tpr_l, thresholds = metrics.roc_curve(
            yauc_c_prevchl, yauc_c_evil, pos_label=2)
        auc_ac_prevchl = metrics.auc(fpr_l, tpr_l)
        AUCs_lc[i] = auc_ac_prevchl

        fpr_r, tpr_r, thresholds = metrics.roc_curve(
            yauc_c_prevchr, yauc_c_evir, pos_label=2)
        auc_ac_prevchr = metrics.auc(fpr_r, tpr_r)
        AUCs_rc[i] = auc_ac_prevchr       
        
        # ### calculate dprime conditioned on Previous choice ----- relevant
        # prevchl, prevchr = np.where(ytest_set_correct[i, :, 0] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 0] == 1+2)[0]
        # yauc_c_prevchl, yauc_c_prevchr =\
        #     np.squeeze(ytest_set_correct[i, prevchl, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, prevchr, SVMAXIS])
        # yauc_c_evil_rel, yauc_c_evir_rel =\
        #     np.squeeze(yevi_set_correct[i, prevchl, SVMAXIS]),\
        #     np.squeeze(yevi_set_correct[i, prevchr, SVMAXIS])
        # # print('ac relevant:',len(prevchr),len(prevchl),'--------',yauc_c_evil)
        # dprimes_lc_rel[i] = guc.calculate_dprime(yauc_c_evil_rel, yauc_c_prevchl)
        # dprimes_rc_rel[i] = guc.calculate_dprime(yauc_c_evir_rel, yauc_c_prevchr)
        
        # ### calculate dprime conditioned on Previous choice ----- cross
        # prevchl, prevchr = np.where(ytest_set_correct[i, :, 0] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 0] == 1+2)[0]
        # yauc_c_prevchl, yauc_c_prevchr =\
        #     np.squeeze(ytest_set_correct[i, prevchl, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, prevchr, SVMAXIS])
        # yauc_c_evil_cross, yauc_c_evir_cross =\
        #     np.squeeze(yevi_set_correct[i, prevchl, 2]),\
        #     np.squeeze(yevi_set_correct[i, prevchr, 2])
        # # print('ac ctoss:',len(prevchr),len(prevchl),'--------',yauc_c_evil)
        # dprimes_lc_cross[i] = guc.calculate_dprime(yauc_c_evil_cross, yauc_c_prevchl)
        # dprimes_rc_cross[i] = guc.calculate_dprime(yauc_c_evir_cross, yauc_c_prevchr)

    ytruthlabels_c, yevi_c = ytruthlabels_c[:, 1:], yevi_c[:, 1:]
    # f, ax_temp = plt.subplots(ncols=2)
    # ax_temp[0].hist(AUCs_c, bins=20, alpha=0.9, facecolor='yellow')

    '''
    After Error Trials
    '''
    yevi_set_error  = data_dec['yevi_set_error']
    ytest_set_error = data_dec['ytest_set_error']

    nlabels = np.shape(np.squeeze(ytest_set_error[0, :, :]))[1]
    ytruthlabels_e = np.zeros((nlabels, 1))
    yevi_e         = np.zeros((3 + 1 + 1, 1))

    dprimes_e    = np.zeros(IPOOLS)
    dprimes_repe = np.zeros(IPOOLS)
    dprimes_alte = np.zeros(IPOOLS)

    
    dprimes_le = np.zeros(IPOOLS)
    dprimes_re = np.zeros(IPOOLS)

    ### separation between left and right categorization
    stds_repe_lr = np.zeros((IPOOLS,2))
    stds_alte_lr = np.zeros((IPOOLS,2)) # two clusters


    AUCs_e    = np.zeros(IPOOLS)
    AUCs_repe = np.zeros(IPOOLS)
    AUCs_alte = np.zeros(IPOOLS)
    
    AUCs_le = np.zeros(IPOOLS)
    AUCs_re = np.zeros(IPOOLS)
    for i in range(IPOOLS):
        hist_evi = yevi_set_error[i, :, :]
        test_labels = ytest_set_error[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        ytruthlabels_e = np.append(
            ytruthlabels_e, test_labels[idx, :].T, axis=1)
        yevi_e = np.append(yevi_e, (yevi_set_error[i, idx, :]).T, axis=1)
        # dprimes_e[i] =\
        #     guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, SVMAXIS]),
        #                           np.squeeze(ytest_set_error[i, :, SVMAXIS]))
        dprimes_e[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, SVMAXIS]-yevi_set_error[i,:,SVMAXIS-1]),
                                  np.squeeze(ytest_set_error[i, :, SVMAXIS])) # overall
        

        yauc_e_org = np.squeeze(ytest_set_error[i, :, SVMAXIS])
        yauc_e = np.zeros_like(yauc_e_org)
        yauc_e[np.where(yauc_e_org == 0)[0]] = 1
        yauc_e[np.where(yauc_e_org == 1)[0]] = 2
        assert (yauc_e != 0).all()
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     yauc_e, np.squeeze(yevi_set_error[i, :, SVMAXIS]), pos_label=2)
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_e, np.squeeze(yevi_set_error[i, :, SVMAXIS]-yevi_set_error[i,:,SVMAXIS-1]), pos_label=2)
        auc_ae = metrics.auc(fpr, tpr)
        AUCs_e[i] = auc_ae

        # SEPARATE REP AND ALT CONTEXTS
        ctxtrep, ctxtalt = np.where(ytest_set_error[i, :, 1] == 0)[
            0], np.where(ytest_set_error[i, :, 1] == 1)[0]
        yauc_e_ctxtrep, yauc_e_ctxtalt =\
            np.squeeze(ytest_set_error[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_error[i, ctxtalt, SVMAXIS])
        # yauc_e_evirep, yauc_e_evialt =\
        #     np.squeeze(yevi_set_error[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(yevi_set_error[i, ctxtalt, SVMAXIS])
        yauc_e_evirep, yauc_e_evialt =\
            np.squeeze(yevi_set_error[i, ctxtrep, SVMAXIS]-yevi_set_error[i, ctxtrep, SVMAXIS-1]),\
            np.squeeze(yevi_set_error[i, ctxtalt, SVMAXIS]-yevi_set_error[i, ctxtalt, SVMAXIS-1]) # overall
        
        # print('ae overall:','--------',yauc_e_evirep)
        dprimes_repe[i] = guc.calculate_dprime(yauc_e_evirep, yauc_e_ctxtrep)
        dprimes_alte[i] = guc.calculate_dprime(yauc_e_evialt, yauc_e_ctxtalt)


        #### calculate the distance between left and right previous choices
        yauc_e_ctxtrep_lr, yauc_e_ctxtalt_lr =\
            np.squeeze(ytest_set_error[i, ctxtrep, 0]),\
            np.squeeze(ytest_set_error[i, ctxtalt, 0])
        yauc_e_evirep_lr, yauc_e_evialt_lr =\
            np.squeeze(yevi_set_error[i, ctxtrep, 0]),\
            np.squeeze(yevi_set_error[i, ctxtalt, 0])
        stds_repe_lr[i,:] = guc.calculate_disperse(yauc_e_evirep_lr,yauc_e_ctxtrep_lr)
        stds_alte_lr[i,:] = guc.calculate_disperse(yauc_e_evialt_lr,yauc_e_ctxtalt_lr)

        yauc_e_ctxtrep, yauc_e_ctxtalt = yauc_e_ctxtrep+1, yauc_e_ctxtalt+1

        fpr_rep, tpr_rep, thresholds = metrics.roc_curve(
            yauc_e_ctxtrep, yauc_e_evirep, pos_label=2)
        auc_ae_rep = metrics.auc(fpr_rep, tpr_rep)
        AUCs_repe[i] = auc_ae_rep

        fpr_alt, tpr_alt, thresholds = metrics.roc_curve(
            yauc_e_ctxtalt, yauc_e_evialt, pos_label=2)
        auc_ae_alt = metrics.auc(fpr_alt, tpr_alt)
        AUCs_alte[i] = auc_ae_alt
        
        # # dprime conditioned on CONTEXTS --- relevant
        # ctxtrep, ctxtalt = np.where(ytest_set_error[i, :, 1] == 0)[
        #     0], np.where(ytest_set_error[i, :, 1] == 1)[0]
        # yauc_e_ctxtrep, yauc_e_ctxtalt =\
        #     np.squeeze(ytest_set_error[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(ytest_set_error[i, ctxtalt, SVMAXIS])
        # yauc_e_evirep_rel, yauc_e_evialt_rel =\
        #     np.squeeze(yevi_set_error[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(yevi_set_error[i, ctxtalt, SVMAXIS])
        # # print('ae rfel:','--------',yauc_e_evirep_rel)
        # dprimes_repe_rel[i] = guc.calculate_dprime(yauc_e_evirep_rel, yauc_e_ctxtrep)
        # dprimes_alte_rel[i] = guc.calculate_dprime(yauc_e_evialt_rel, yauc_e_ctxtalt)

        
        # # dprime conditioned on CONTEXTS --- cross
        # ctxtrep, ctxtalt = np.where(ytest_set_error[i, :, 1] == 0)[
        #     0], np.where(ytest_set_error[i, :, 1] == 1)[0]
        # yauc_e_ctxtrep, yauc_e_ctxtalt =\
        #     np.squeeze(ytest_set_error[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(ytest_set_error[i, ctxtalt, SVMAXIS])
        # yauc_e_evirep_cross, yauc_e_evialt_cross =\
        #     np.squeeze(yevi_set_error[i, ctxtrep, 2]),\
        #     np.squeeze(yevi_set_error[i, ctxtalt, 2])
        # # print('ae cross:','--------',yauc_e_evirep_cross)
        # dprimes_repe_cross[i] = guc.calculate_dprime(yauc_e_evirep_cross, yauc_e_ctxtrep)
        # dprimes_alte_cross[i] = guc.calculate_dprime(yauc_e_evialt_cross, yauc_e_ctxtalt)
        
        # SEPARATE previous left and previous right CONTEXTS
        prevchl, prevchr = np.where(ytest_set_error[i, :, 0] == 0)[
            0], np.where(ytest_set_error[i, :, 0] == 1)[0]
        yauc_e_prevchl, yauc_e_prevchr =\
            np.squeeze(ytest_set_error[i, prevchl, SVMAXIS]),\
            np.squeeze(ytest_set_error[i, prevchr, SVMAXIS])

        # yauc_e_evil, yauc_e_evir =\
        #     np.squeeze(yevi_set_error[i, prevchl, SVMAXIS]),\
        #     np.squeeze(yevi_set_error[i, prevchr, SVMAXIS])
        yauc_e_evil, yauc_e_evir =\
            np.squeeze(yevi_set_error[i, prevchl, SVMAXIS]-yevi_set_error[i, prevchl, SVMAXIS-1]),\
            np.squeeze(yevi_set_error[i, prevchr, SVMAXIS]-yevi_set_error[i, prevchr, SVMAXIS-1])
        # print('ae overall:','--------',yauc_e_evil)
        dprimes_le[i] = guc.calculate_dprime(yauc_e_evil, yauc_e_prevchl)
        dprimes_re[i] = guc.calculate_dprime(yauc_e_evir, yauc_e_prevchr)
        
        
        ### AUC should be larger than 0.5 
        # yauc_e_prevchl, yauc_e_prevchr = 1-yauc_e_prevchl, 1-yauc_e_prevchr
        yauc_e_prevchl, yauc_e_prevchr = yauc_e_prevchl+1, yauc_e_prevchr+1
        
        fpr_l, tpr_l, thresholds = metrics.roc_curve(
            yauc_e_prevchl, yauc_e_evil, pos_label=2)
        auc_ae_prevchl = metrics.auc(fpr_l, tpr_l)
        AUCs_le[i] = auc_ae_prevchl

        fpr_r, tpr_r, thresholds = metrics.roc_curve(
            yauc_e_prevchr, yauc_e_evir, pos_label=2)
        auc_ae_prevchr = metrics.auc(fpr_r, tpr_r)
        AUCs_re[i] = auc_ae_prevchr
        
        # # dprime conditioned on PREVIOUS CHOICE --- relevant
        # prevchl, prevchr = np.where(ytest_set_error[i, :, 0] == 0)[
        #     0], np.where(ytest_set_error[i, :, 0] == 1)[0]
        # yauc_e_prevchl, yauc_e_prevchr =\
        #     np.squeeze(ytest_set_error[i, prevchl, SVMAXIS]),\
        #     np.squeeze(ytest_set_error[i, prevchr, SVMAXIS])
        # yauc_e_evil_rel, yauc_e_evir_rel =\
        #     np.squeeze(yevi_set_error[i, prevchl, SVMAXIS]),\
        #     np.squeeze(yevi_set_error[i, prevchr, SVMAXIS])
        # # print('ae rel:','--------',yauc_e_evil_rel)
        # dprimes_le_rel[i] = guc.calculate_dprime(yauc_e_evil_rel, yauc_e_prevchl)
        # dprimes_re_rel[i] = guc.calculate_dprime(yauc_e_evir_rel, yauc_e_prevchr)
        
        # # dprime conditioned on PREVIOUS CHOICE --- CROSS
        # prevchl, prevchr = np.where(ytest_set_error[i, :, 0] == 0)[
        #     0], np.where(ytest_set_error[i, :, 0] == 1)[0]
        # yauc_e_prevchl, yauc_e_prevchr =\
        #     np.squeeze(ytest_set_error[i, prevchl, SVMAXIS]),\
        #     np.squeeze(ytest_set_error[i, prevchr, SVMAXIS])
        # yauc_e_evil_cross, yauc_e_evir_cross =\
        #     np.squeeze(yevi_set_error[i, prevchl, 2]),\
        #     np.squeeze(yevi_set_error[i, prevchr, 2])
        # # print('ae cross:','--------',yauc_e_evil_cross)
        # dprimes_le_cross[i] = guc.calculate_dprime(yauc_e_evil_cross, yauc_e_prevchl)
        # dprimes_re_cross[i] = guc.calculate_dprime(yauc_e_evir_cross, yauc_e_prevchr)

    # ax_temp[1].hist(AUCs_e, bins=20, alpha=0.9, facecolor='black')

    ytruthlabels_e, yevi_e = ytruthlabels_e[:, 1:], yevi_e[:, 1:]
    
    lst = [ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,
           dprimes_c, dprimes_e, AUCs_c, AUCs_e,
           dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte,
           stds_repc_lr, stds_altc_lr,
           AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte, 
           dprimes_lc, dprimes_rc, dprimes_le, dprimes_re,
           stds_repe_lr, stds_alte_lr,
           AUCs_lc, AUCs_rc, AUCs_le, AUCs_re]
    stg = ["ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,"
           "dprimes_c, dprimes_e, AUCs_c, AUCs_e, "
           "dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte, "
           "stds_repc_lr, stds_altc_lr, "
           "AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte, "
           "dprimes_lc, dprimes_rc, dprimes_le, dprimes_re, "
           "stds_repe_lr, stds_alte_lr, "
           "AUCs_lc, AUCs_rc, AUCs_le, AUCs_re"]
    d = list_to_dict(lst=lst, string=stg)
    return d

def flatten_data_cond_supp(data_tr, data_dec, pop_type='s_pop'):
    """
    Flatten the encoding of history information (session-by-session) -- supplementary

    Parameters
    ----------
    data_tr: dict
        dataset for generating pseudo-trials

    data_dec: dict
        dataset containing the encodings of history information and labels


    Returns
    -------
    d : dict
        flatten data

    """
    if pop_type=='s_pop':
        supi = 1
    else:
        supi = 3
    yevi_set_correct        = data_dec['yevi_set_correct_supp'] # 3 Sept 
    yevi_set_correct_select = data_dec['yevi_set_correct']
    ytest_set_correct       = data_dec['ytest_set_correct']
    IPOOLS = NITERATIONS ## should be consistent with the number of iterations used in bootstrap
    
    ### flatten data --- after correct
    nlabels        = np.shape(np.squeeze(ytest_set_correct[0, :, :]))[1]
    ytruthlabels_c = np.zeros((nlabels, 1))
    yevi_c         = np.zeros((3 +1+1, 1))
    ### Gaussian assumption
    dprimes_c    = np.zeros(IPOOLS)
    dprimes_repc = np.zeros(IPOOLS)
    dprimes_altc = np.zeros(IPOOLS) # overall
    
    dprimes_lc = np.zeros(IPOOLS)
    dprimes_rc = np.zeros(IPOOLS) # overall

    AUCs_c    = np.zeros(IPOOLS)
    AUCs_repc = np.zeros(IPOOLS)
    AUCs_altc = np.zeros(IPOOLS)
    
    AUCs_lc = np.zeros(IPOOLS)
    AUCs_rc = np.zeros(IPOOLS)

    for i in range(IPOOLS):
        hist_evi    = yevi_set_correct[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        test_labels = ytest_set_correct[i, :, :]
        ytruthlabels_c = np.append(
            ytruthlabels_c, test_labels[idx, :].T, axis=1)
        
        yevi_set_correct_select[i, idx, SVMAXIS]   = yevi_set_correct[i,idx,supi].copy()
        yevi_set_correct_select[i, idx, SVMAXIS-1] = yevi_set_correct[i,idx,supi-1].copy()
        yevi_c = np.append(yevi_c, (yevi_set_correct_select[i, idx, :]).T, axis=1) # 3 Sept
        
        # dprimes_c[i] =\
        #     guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, SVMAXIS]),
        #                           np.squeeze(ytest_set_correct[i, :, SVMAXIS])) # aligned direction
        dprimes_c[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, supi]-yevi_set_correct[i,:,supi-1]),np.squeeze(ytest_set_correct[i, :, SVMAXIS]))

        ### calculate AUC
        yauc_c_org = np.squeeze(ytest_set_correct[i, :, SVMAXIS])
        yauc_c     = np.zeros_like(yauc_c_org)
        yauc_c[np.where(yauc_c_org == 0+2)[0]] = 1
        yauc_c[np.where(yauc_c_org == 1+2)[0]] = 2
        assert (yauc_c != 0).all()
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     yauc_c, np.squeeze(yevi_set_correct[i, :, SVMAXIS]), pos_label=2)
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_c, np.squeeze(yevi_set_correct[i, :, supi]-yevi_set_correct[i,:,supi-1]), pos_label=2)
        auc_ac = metrics.auc(fpr, tpr)
        AUCs_c[i] = auc_ac

        ### calculate AUC conditioned on Block contexts --- overall
        ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
            np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        yauc_c_ctxtrep, yauc_c_ctxtalt =\
            np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        yauc_c_evirep, yauc_c_evialt =\
            np.squeeze(yevi_set_correct[i, ctxtrep, supi]-yevi_set_correct[i, ctxtrep, supi-1]),\
            np.squeeze(yevi_set_correct[i, ctxtalt, supi]-yevi_set_correct[i, ctxtalt, supi-1]) # mental states coexist
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
        
        # ### calculate dprime context relevant
        # ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        # yauc_c_ctxtrep, yauc_c_ctxtalt =\
        #     np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        # yauc_c_evirep_rel, yauc_c_evialt_rel =\
        #     np.squeeze(yevi_set_correct[i, ctxtrep, supi]),\
        #     np.squeeze(yevi_set_correct[i, ctxtalt, supi])
        # dprimes_repc_rel[i] = guc.calculate_dprime(yauc_c_evirep_rel, yauc_c_ctxtrep)
        # dprimes_altc_rel[i] = guc.calculate_dprime(yauc_c_evialt_rel, yauc_c_ctxtalt)
        
        
        # ### calculate dprime context cross
        # ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        # yauc_c_ctxtrep, yauc_c_ctxtalt =\
        #     np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        # yauc_c_evirep_cross, yauc_c_evialt_cross =\
        #     np.squeeze(yevi_set_correct[i, ctxtrep, supi-1]),\
        #     np.squeeze(yevi_set_correct[i, ctxtalt, supi-1])
        # dprimes_repc_cross[i] = guc.calculate_dprime(yauc_c_evirep_cross, yauc_c_ctxtrep)
        # dprimes_altc_cross[i] = guc.calculate_dprime(yauc_c_evialt_cross, yauc_c_ctxtalt)
        
        
        ### calculate AUC conditioned on Previous choice --- overall
        prevchl, prevchr = np.where(ytest_set_correct[i, :, 0] == 0+2)[0],\
            np.where(ytest_set_correct[i, :, 0] == 1+2)[0]
        yauc_c_prevchl, yauc_c_prevchr =\
            np.squeeze(ytest_set_correct[i, prevchl, SVMAXIS]),\
            np.squeeze(ytest_set_correct[i, prevchr, SVMAXIS])
        yauc_c_evil, yauc_c_evir =\
            np.squeeze(yevi_set_correct[i, prevchl, supi]-yevi_set_correct[i, prevchl, supi-1]),\
            np.squeeze(yevi_set_correct[i, prevchr, supi]-yevi_set_correct[i, prevchr, supi-1])
        dprimes_lc[i] = guc.calculate_dprime(yauc_c_evil, yauc_c_prevchl)
        dprimes_rc[i] = guc.calculate_dprime(yauc_c_evir, yauc_c_prevchr)

        yauc_c_prevchl, yauc_c_prevchr = yauc_c_prevchl-1, yauc_c_prevchr-1
        
        
        fpr_l, tpr_l, thresholds = metrics.roc_curve(
            yauc_c_prevchl, yauc_c_evil, pos_label=2)
        auc_ac_prevchl = metrics.auc(fpr_l, tpr_l)
        AUCs_lc[i] = auc_ac_prevchl

        fpr_r, tpr_r, thresholds = metrics.roc_curve(
            yauc_c_prevchr, yauc_c_evir, pos_label=2)
        auc_ac_prevchr = metrics.auc(fpr_r, tpr_r)
        AUCs_rc[i] = auc_ac_prevchr       
        
        # ### calculate dprime conditioned on Previous choice ----- relevant
        # prevchl, prevchr = np.where(ytest_set_correct[i, :, 0] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 0] == 1+2)[0]
        # yauc_c_prevchl, yauc_c_prevchr =\
        #     np.squeeze(ytest_set_correct[i, prevchl, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, prevchr, SVMAXIS])
        # yauc_c_evil_rel, yauc_c_evir_rel =\
        #     np.squeeze(yevi_set_correct[i, prevchl, supi]),\
        #     np.squeeze(yevi_set_correct[i, prevchr, supi])
        # dprimes_lc_rel[i] = guc.calculate_dprime(yauc_c_evil_rel, yauc_c_prevchl)
        # dprimes_rc_rel[i] = guc.calculate_dprime(yauc_c_evir_rel, yauc_c_prevchr)
        
        # ### calculate dprime conditioned on Previous choice ----- cross
        # prevchl, prevchr = np.where(ytest_set_correct[i, :, 0] == 0+2)[0],\
        #     np.where(ytest_set_correct[i, :, 0] == 1+2)[0]
        # yauc_c_prevchl, yauc_c_prevchr =\
        #     np.squeeze(ytest_set_correct[i, prevchl, SVMAXIS]),\
        #     np.squeeze(ytest_set_correct[i, prevchr, SVMAXIS])
        # yauc_c_evil_cross, yauc_c_evir_cross =\
        #     np.squeeze(yevi_set_correct[i, prevchl, supi-1]),\
        #     np.squeeze(yevi_set_correct[i, prevchr, supi-1])
        # dprimes_lc_cross[i] = guc.calculate_dprime(yauc_c_evil_cross, yauc_c_prevchl)
        # dprimes_rc_cross[i] = guc.calculate_dprime(yauc_c_evir_cross, yauc_c_prevchr)

    ytruthlabels_c, yevi_c = ytruthlabels_c[:, 1:], yevi_c[:, 1:]
    # f, ax_temp = plt.subplots(ncols=2)
    # ax_temp[0].hist(AUCs_c, bins=20, alpha=0.9, facecolor='yellow')

    '''
    After Error Trials
    '''
    yevi_set_error        = data_dec['yevi_set_error_supp']
    yevi_set_error_select = data_dec['yevi_set_error']
    ytest_set_error       = data_dec['ytest_set_error']

    nlabels        = np.shape(np.squeeze(ytest_set_error[0, :, :]))[1]
    ytruthlabels_e = np.zeros((nlabels, 1))
    yevi_e         = np.zeros((3+1+1, 1))

    dprimes_e    = np.zeros(IPOOLS)
    dprimes_repe = np.zeros(IPOOLS)
    dprimes_alte = np.zeros(IPOOLS)
    
    dprimes_le = np.zeros(IPOOLS)
    dprimes_re = np.zeros(IPOOLS)

    AUCs_e    = np.zeros(IPOOLS)
    AUCs_repe = np.zeros(IPOOLS)
    AUCs_alte = np.zeros(IPOOLS)
    
    AUCs_le = np.zeros(IPOOLS)
    AUCs_re = np.zeros(IPOOLS)
    for i in range(IPOOLS):
        hist_evi = yevi_set_error[i, :, :]
        test_labels = ytest_set_error[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        ytruthlabels_e = np.append(
            ytruthlabels_e, test_labels[idx, :].T, axis=1)
        # yevi_e = np.append(yevi_e, (yevi_set_error[i, idx, :]).T, axis=1)
        
        yevi_set_error_select[i, idx, SVMAXIS]   = yevi_set_error[i,idx,supi]
        yevi_set_error_select[i, idx, SVMAXIS-1] = yevi_set_error[i,idx,supi-1]
        yevi_e = np.append(yevi_e, (yevi_set_error_select[i, idx, :]).T, axis=1) # 3 Sept
        
        # dprimes_e[i] =\
        #     guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, SVMAXIS]),
        #                           np.squeeze(ytest_set_error[i, :, SVMAXIS]))
        dprimes_e[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, supi]-yevi_set_error[i,:,supi-1]),np.squeeze(ytest_set_error[i, :, SVMAXIS])) # overal

        yauc_e_org = np.squeeze(ytest_set_error[i, :, SVMAXIS])
        yauc_e     = np.zeros_like(yauc_e_org)
        yauc_e[np.where(yauc_e_org == 0)[0]] = 1
        yauc_e[np.where(yauc_e_org == 1)[0]] = 2
        assert (yauc_e != 0).all()
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     yauc_e, np.squeeze(yevi_set_error[i, :, SVMAXIS]), pos_label=2)
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_e, np.squeeze(yevi_set_error[i, :, supi]-yevi_set_error[i,:,supi-1]), pos_label=2)
        auc_ae = metrics.auc(fpr, tpr)
        AUCs_e[i] = auc_ae

        # SEPARATE REP AND ALT CONTEXTS
        ctxtrep, ctxtalt = np.where(ytest_set_error[i, :, 1] == 0)[
            0], np.where(ytest_set_error[i, :, 1] == 1)[0]
        yauc_e_ctxtrep, yauc_e_ctxtalt =\
            np.squeeze(ytest_set_error[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_error[i, ctxtalt, SVMAXIS])

        yauc_e_evirep, yauc_e_evialt =\
            np.squeeze(yevi_set_error[i, ctxtrep, supi]-yevi_set_error[i, ctxtrep, supi-1]),\
            np.squeeze(yevi_set_error[i, ctxtalt, supi]-yevi_set_error[i, ctxtalt, supi-1]) # overall
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
        
        # SEPARATE previous left and previous right CONTEXTS
        prevchl, prevchr = np.where(ytest_set_error[i, :, 0] == 0)[
            0], np.where(ytest_set_error[i, :, 0] == 1)[0]
        yauc_e_prevchl, yauc_e_prevchr =\
            np.squeeze(ytest_set_error[i, prevchl, SVMAXIS]),\
            np.squeeze(ytest_set_error[i, prevchr, SVMAXIS])
        yauc_e_evil, yauc_e_evir =\
            np.squeeze(yevi_set_error[i, prevchl, supi]-yevi_set_error[i, prevchl, supi-1]),\
            np.squeeze(yevi_set_error[i, prevchr, supi]-yevi_set_error[i, prevchr, supi-1])
        dprimes_le[i] = guc.calculate_dprime(yauc_e_evil, yauc_e_prevchl)
        dprimes_re[i] = guc.calculate_dprime(yauc_e_evir, yauc_e_prevchr)
        
        
        ### AUC should be larger than 0.5 
        # yauc_e_prevchl, yauc_e_prevchr = 1-yauc_e_prevchl, 1-yauc_e_prevchr
        yauc_e_prevchl, yauc_e_prevchr = yauc_e_prevchl+1, yauc_e_prevchr+1
        
        fpr_l, tpr_l, thresholds = metrics.roc_curve(
            yauc_e_prevchl, yauc_e_evil, pos_label=2)
        auc_ae_prevchl = metrics.auc(fpr_l, tpr_l)
        AUCs_le[i] = auc_ae_prevchl

        fpr_r, tpr_r, thresholds = metrics.roc_curve(
            yauc_e_prevchr, yauc_e_evir, pos_label=2)
        auc_ae_prevchr = metrics.auc(fpr_r, tpr_r)
        AUCs_re[i] = auc_ae_prevchr
        

    ytruthlabels_e, yevi_e = ytruthlabels_e[:, 1:], yevi_e[:, 1:]
    
    lst = [ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,
           dprimes_c, dprimes_e, AUCs_c, AUCs_e,
           dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte,
           AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte, 
           dprimes_lc, dprimes_rc, dprimes_le, dprimes_re,
           AUCs_lc, AUCs_rc, AUCs_le, AUCs_re]
    stg = ["ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,"
           "dprimes_c, dprimes_e, AUCs_c, AUCs_e, "
           "dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte, "
           "AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte, "
           "dprimes_lc, dprimes_rc, dprimes_le, dprimes_re, "
           "AUCs_lc, AUCs_rc, AUCs_le, AUCs_re"]
    d = list_to_dict(lst=lst, string=stg)
    return d

def flatten_data_supp(data_tr, data_dec, pop_type='s_pop'):
    """
    Flatten the encoding of history information (session-by-session) -- supplementary

    Parameters
    ----------
    data_tr: dict
        dataset for generating pseudo-trials

    data_dec: dict
        dataset containing the encodings of history information and labels


    Returns
    -------
    d : dict
        flatten data

    """
    if pop_type=='s_pop':
        supi = 0
    else:
        supi = 1
    yevi_set_correct  = data_dec['yevi_set_correct_supp'] # 3 Sept 
    yevi_set_correct_select  = data_dec['yevi_set_correct']
    ytest_set_correct = data_dec['ytest_set_correct']
    IPOOLS = NITERATIONS ## should be consistent with the number of iterations used in bootstrap
    
    ### flatten data --- after correct
    nlabels = np.shape(np.squeeze(ytest_set_correct[0, :, :]))[1]
    ytruthlabels_c = np.zeros((nlabels, 1))
    yevi_c         = np.zeros((3 +1+1, 1))
    ### Gaussian assumption
    dprimes_c    = np.zeros(IPOOLS)
    dprimes_repc = np.zeros(IPOOLS)
    dprimes_altc = np.zeros(IPOOLS) # overall

    
    dprimes_lc = np.zeros(IPOOLS)
    dprimes_rc = np.zeros(IPOOLS) # overall


    AUCs_c    = np.zeros(IPOOLS)
    AUCs_repc = np.zeros(IPOOLS)
    AUCs_altc = np.zeros(IPOOLS)
    
    AUCs_lc = np.zeros(IPOOLS)
    AUCs_rc = np.zeros(IPOOLS)

    for i in range(IPOOLS):
        hist_evi    = yevi_set_correct[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        test_labels = ytest_set_correct[i, :, :]
        ytruthlabels_c = np.append(
            ytruthlabels_c, test_labels[idx, :].T, axis=1)
        
        yevi_set_correct_select[i, idx, SVMAXIS]   = yevi_set_correct[i,idx,supi].copy()
        yevi_c = np.append(yevi_c, (yevi_set_correct_select[i, idx, :]).T, axis=1) # 3 Sept
        
        # dprimes_c[i] =\
        #     guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, SVMAXIS]),
        #                           np.squeeze(ytest_set_correct[i, :, SVMAXIS])) # aligned direction
        dprimes_c[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_correct[i, :, supi]),
                                  np.squeeze(ytest_set_correct[i, :, SVMAXIS]))

        ### calculate AUC
        yauc_c_org = np.squeeze(ytest_set_correct[i, :, SVMAXIS])
        yauc_c     = np.zeros_like(yauc_c_org)
        yauc_c[np.where(yauc_c_org == 0+2)[0]] = 1
        yauc_c[np.where(yauc_c_org == 1+2)[0]] = 2
        assert (yauc_c != 0).all()
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     yauc_c, np.squeeze(yevi_set_correct[i, :, SVMAXIS]), pos_label=2)
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_c, np.squeeze(yevi_set_correct[i, :, supi]), pos_label=2)
        auc_ac = metrics.auc(fpr, tpr)
        AUCs_c[i] = auc_ac

        ### calculate AUC conditioned on Block contexts --- overall
        ctxtrep, ctxtalt = np.where(ytest_set_correct[i, :, 1] == 0+2)[0],\
            np.where(ytest_set_correct[i, :, 1] == 1+2)[0]
        yauc_c_ctxtrep, yauc_c_ctxtalt =\
            np.squeeze(ytest_set_correct[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_correct[i, ctxtalt, SVMAXIS])
        yauc_c_evirep, yauc_c_evialt =\
            np.squeeze(yevi_set_correct[i, ctxtrep, supi]),\
            np.squeeze(yevi_set_correct[i, ctxtalt, supi]) # mental states coexist
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
        
        
        
        
        ### calculate AUC conditioned on Previous choice --- overall
        prevchl, prevchr = np.where(ytest_set_correct[i, :, 0] == 0+2)[0],\
            np.where(ytest_set_correct[i, :, 0] == 1+2)[0]
        yauc_c_prevchl, yauc_c_prevchr =\
            np.squeeze(ytest_set_correct[i, prevchl, SVMAXIS]),\
            np.squeeze(ytest_set_correct[i, prevchr, SVMAXIS])
        yauc_c_evil, yauc_c_evir =\
            np.squeeze(yevi_set_correct[i, prevchl, supi]),\
            np.squeeze(yevi_set_correct[i, prevchr, supi])
        dprimes_lc[i] = guc.calculate_dprime(yauc_c_evil, yauc_c_prevchl)
        dprimes_rc[i] = guc.calculate_dprime(yauc_c_evir, yauc_c_prevchr)

        yauc_c_prevchl, yauc_c_prevchr = yauc_c_prevchl-1, yauc_c_prevchr-1
        
        
        fpr_l, tpr_l, thresholds = metrics.roc_curve(
            yauc_c_prevchl, yauc_c_evil, pos_label=2)
        auc_ac_prevchl = metrics.auc(fpr_l, tpr_l)
        AUCs_lc[i] = auc_ac_prevchl

        fpr_r, tpr_r, thresholds = metrics.roc_curve(
            yauc_c_prevchr, yauc_c_evir, pos_label=2)
        auc_ac_prevchr = metrics.auc(fpr_r, tpr_r)
        AUCs_rc[i] = auc_ac_prevchr       
        

    ytruthlabels_c, yevi_c = ytruthlabels_c[:, 1:], yevi_c[:, 1:]
    # f, ax_temp = plt.subplots(ncols=2)
    # ax_temp[0].hist(AUCs_c, bins=20, alpha=0.9, facecolor='yellow')

    '''
    After Error Trials
    '''
    yevi_set_error  = data_dec['yevi_set_error_supp']
    yevi_set_error_select = data_dec['yevi_set_error']
    ytest_set_error = data_dec['ytest_set_error']

    nlabels = np.shape(np.squeeze(ytest_set_error[0, :, :]))[1]
    ytruthlabels_e = np.zeros((nlabels, 1))
    yevi_e         = np.zeros((3+1+1, 1))

    dprimes_e    = np.zeros(IPOOLS)
    dprimes_repe = np.zeros(IPOOLS)
    dprimes_alte = np.zeros(IPOOLS)

    
    dprimes_le = np.zeros(IPOOLS)
    dprimes_re = np.zeros(IPOOLS)


    AUCs_e    = np.zeros(IPOOLS)
    AUCs_repe = np.zeros(IPOOLS)
    AUCs_alte = np.zeros(IPOOLS)
    
    AUCs_le = np.zeros(IPOOLS)
    AUCs_re = np.zeros(IPOOLS)
    for i in range(IPOOLS):
        hist_evi = yevi_set_error[i, :, :]
        test_labels = ytest_set_error[i, :, :]
        idx = np.arange(np.shape(hist_evi)[0])
        ytruthlabels_e = np.append(
            ytruthlabels_e, test_labels[idx, :].T, axis=1)
        # yevi_e = np.append(yevi_e, (yevi_set_error[i, idx, :]).T, axis=1)
        
        yevi_set_error_select[i, idx, SVMAXIS]   = yevi_set_error[i,idx,supi]
        # yevi_set_error_select[i, idx, SVMAXIS-1] = yevi_set_error[i,idx,supi-1]
        yevi_e = np.append(yevi_e, (yevi_set_error_select[i, idx, :]).T, axis=1) # 3 Sept
        
        # dprimes_e[i] =\
        #     guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, SVMAXIS]),
        #                           np.squeeze(ytest_set_error[i, :, SVMAXIS]))
        dprimes_e[i] =\
            guc.calculate_dprime(np.squeeze(yevi_set_error[i, :, supi]),
                                  np.squeeze(ytest_set_error[i, :, SVMAXIS])) # overal
        

        yauc_e_org = np.squeeze(ytest_set_error[i, :, SVMAXIS])
        yauc_e = np.zeros_like(yauc_e_org)
        yauc_e[np.where(yauc_e_org == 0)[0]] = 1
        yauc_e[np.where(yauc_e_org == 1)[0]] = 2
        assert (yauc_e != 0).all()
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     yauc_e, np.squeeze(yevi_set_error[i, :, SVMAXIS]), pos_label=2)
        fpr, tpr, thresholds = metrics.roc_curve(
            yauc_e, np.squeeze(yevi_set_error[i, :, supi]), pos_label=2)
        auc_ae = metrics.auc(fpr, tpr)
        AUCs_e[i] = auc_ae

        # SEPARATE REP AND ALT CONTEXTS
        ctxtrep, ctxtalt = np.where(ytest_set_error[i, :, 1] == 0)[
            0], np.where(ytest_set_error[i, :, 1] == 1)[0]
        yauc_e_ctxtrep, yauc_e_ctxtalt =\
            np.squeeze(ytest_set_error[i, ctxtrep, SVMAXIS]),\
            np.squeeze(ytest_set_error[i, ctxtalt, SVMAXIS])
        # yauc_e_evirep, yauc_e_evialt =\
        #     np.squeeze(yevi_set_error[i, ctxtrep, SVMAXIS]),\
        #     np.squeeze(yevi_set_error[i, ctxtalt, SVMAXIS])
        yauc_e_evirep, yauc_e_evialt =\
            np.squeeze(yevi_set_error[i, ctxtrep, supi]),\
            np.squeeze(yevi_set_error[i, ctxtalt, supi]) # overall
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
        

        
        # SEPARATE previous left and previous right CONTEXTS
        prevchl, prevchr = np.where(ytest_set_error[i, :, 0] == 0)[
            0], np.where(ytest_set_error[i, :, 0] == 1)[0]
        yauc_e_prevchl, yauc_e_prevchr =\
            np.squeeze(ytest_set_error[i, prevchl, SVMAXIS]),\
            np.squeeze(ytest_set_error[i, prevchr, SVMAXIS])
        yauc_e_evil, yauc_e_evir =\
            np.squeeze(yevi_set_error[i, prevchl, supi]),\
            np.squeeze(yevi_set_error[i, prevchr, supi])
        dprimes_le[i] = guc.calculate_dprime(yauc_e_evil, yauc_e_prevchl)
        dprimes_re[i] = guc.calculate_dprime(yauc_e_evir, yauc_e_prevchr)
        
        
        ### AUC should be larger than 0.5 
        # yauc_e_prevchl, yauc_e_prevchr = 1-yauc_e_prevchl, 1-yauc_e_prevchr
        yauc_e_prevchl, yauc_e_prevchr = yauc_e_prevchl+1, yauc_e_prevchr+1
        
        fpr_l, tpr_l, thresholds = metrics.roc_curve(
            yauc_e_prevchl, yauc_e_evil, pos_label=2)
        auc_ae_prevchl = metrics.auc(fpr_l, tpr_l)
        AUCs_le[i] = auc_ae_prevchl

        fpr_r, tpr_r, thresholds = metrics.roc_curve(
            yauc_e_prevchr, yauc_e_evir, pos_label=2)
        auc_ae_prevchr = metrics.auc(fpr_r, tpr_r)
        AUCs_re[i] = auc_ae_prevchr
        
        

    # ax_temp[1].hist(AUCs_e, bins=20, alpha=0.9, facecolor='black')

    ytruthlabels_e, yevi_e = ytruthlabels_e[:, 1:], yevi_e[:, 1:]
    
    lst = [ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,
           dprimes_c, dprimes_e, AUCs_c, AUCs_e,
           dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte,
           AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte, 
           dprimes_lc, dprimes_rc, dprimes_le, dprimes_re,
           AUCs_lc, AUCs_rc, AUCs_le, AUCs_re]
    stg = ["ytruthlabels_c, ytruthlabels_e, yevi_c, yevi_e,"
           "dprimes_c, dprimes_e, AUCs_c, AUCs_e, "
           "dprimes_repc, dprimes_altc, dprimes_repe, dprimes_alte, "
           "AUCs_repc, AUCs_altc, AUCs_repe, AUCs_alte, "
           "dprimes_lc, dprimes_rc, dprimes_le, dprimes_re, "
           "AUCs_lc, AUCs_rc, AUCs_le, AUCs_re"]
    d = list_to_dict(lst=lst, string=stg)
    return d

'''
Results Visualizations
'''
def projection_3D(data_flt, data_flt_light, prev_outc):
    ytruthlabels_c = data_flt['ytruthlabels_'+prev_outc]
    yevi_c         = data_flt['yevi_'+prev_outc]

    ridx = np.random.choice(np.arange(len(yevi_c[1, :])),
                            size=200, replace=False)
    ridx = ridx.astype(np.int32)

    # RESAMPLE THE CONGRUENT TRIALS

    ridx_congruent =\
        np.where(ytruthlabels_c[2, :] == ytruthlabels_c[SVMAXIS, :])[0]
    ridx = np.random.choice(
        ridx_congruent, size=int(NUM_SAMPLES), replace=True)

    fig = plt.figure()  # XXX: this was in line 352 (after x, y, z = ...)
    ax  = fig.add_subplot(111, projection='3d')
    # --- PLOTING CONGRUENT TRIALS, WITH CLEAR TRANSITIONS
    x, y, z = yevi_c[1, ridx], yevi_c[0, ridx], yevi_c[3, ridx]+yevi_c[2,ridx] # yevi_c[3, ridx]
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

    # # histogram side
    # fig2dd, ax2dd = plt.subplots(figsize=(6, 3))
    # zrange = np.linspace(-10, 6, 30)
    # ibluehist = np.append(ibluehist, idxright[iblue])
    # igreenhist = np.append(igreenhist, idxright[igreen])
    # ax2dd.hist(z[ibluehist], bins=zrange, density=True, facecolor=PURPLE,
    #            alpha=0.9)
    # ax2dd.hist(z[igreenhist], bins=zrange, density=True, facecolor=GREEN,
    #            alpha=0.9)

    igreen, iblue =\
        np.where(ytruthlabels_c[1, ridx] == AX_PREV_CH_OUTC[prev_outc][0])[0],\
        np.where(ytruthlabels_c[1, ridx] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    ax.scatter(x[igreen], y[igreen], zflat[igreen],
               s=S_PLOTS*3, c=BLUE, alpha=0.9)
    ax.scatter(x[iblue], y[iblue], zflat[iblue], s=S_PLOTS*3, c=RED, alpha=0.9)


def projections_2D(data_flt, prev_outc, fit=False, name=''):
    ytruthlabels = data_flt['ytruthlabels_'+prev_outc]
    yevi         = data_flt['yevi_'+prev_outc]
    '''
    Four conditions (four clouds)
    '''
    idxprel  = np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][0])[0]
    idxctxtr = np.where(ytruthlabels[1, :] == AX_PREV_CH_OUTC[prev_outc][0])[0]

    idxprer  = np.where(ytruthlabels[0, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]
    idxctxta = np.where(ytruthlabels[1, :] == AX_PREV_CH_OUTC[prev_outc][1])[0]

    idxprelctxtr = np.intersect1d(idxprel, idxctxtr)
    idxprelctxta = np.intersect1d(idxprel, idxctxta)
    idxprerctxtr = np.intersect1d(idxprer, idxctxtr)
    idxprerctxta = np.intersect1d(idxprer, idxctxta)

    idxsample = np.zeros((4, NUM_SAMPLES), dtype=int)
    idxsample[0, :] = np.random.choice(idxprelctxtr, size=NUM_SAMPLES, replace=True)
    idxsample[1, :] = np.random.choice(idxprelctxta, size=NUM_SAMPLES, replace=True)#idxprelctxta[:NUM_SAMPLES]
    idxsample[2, :] = np.random.choice(idxprerctxtr, size=NUM_SAMPLES, replace=True)#idxprerctxtr[:NUM_SAMPLES]
    idxsample[3, :] = np.random.choice(idxprerctxta, size=NUM_SAMPLES, replace=True)#idxprerctxta[:NUM_SAMPLES]

    idxpreal, idxprear = np.union1d(idxsample[0, :], idxsample[1, :]), np.union1d(
        idxsample[2, :], idxsample[3, :])
    print('prel, ctxtr, current l:',np.unique(ytruthlabels[2,idxsample[0,:]]),Counter(ytruthlabels[SVMAXIS,idxsample[0,:]]).keys(),Counter(ytruthlabels[SVMAXIS,idxsample[0,:]]).values())#np.unique(ytruthlabels[SVMAXIS,idxsample[0,:]]))
    print('prel, ctxta, current r:',np.unique(ytruthlabels[2,idxsample[1,:]]),Counter(ytruthlabels[SVMAXIS,idxsample[1,:]]).keys(),Counter(ytruthlabels[SVMAXIS,idxsample[1,:]]).values())#,np.unique(ytruthlabels[SVMAXIS,idxsample[1,:]]))
    print('prer, ctxtr, current r:',np.unique(ytruthlabels[2,idxsample[2,:]]),Counter(ytruthlabels[SVMAXIS,idxsample[2,:]]).keys(),Counter(ytruthlabels[SVMAXIS,idxsample[2,:]]).values())#np.unique(ytruthlabels[SVMAXIS,idxsample[2,:]]))
    print('prer, ctxta, current l:',np.unique(ytruthlabels[2,idxsample[3,:]]),Counter(ytruthlabels[SVMAXIS,idxsample[3,:]]).keys(),Counter(ytruthlabels[SVMAXIS,idxsample[3,:]]).values())#np.unique(ytruthlabels[SVMAXIS,idxsample[3,:]]))

    idxctxtr, idxctxta = np.union1d(idxsample[0, :], idxsample[2, :]), np.union1d(
        idxsample[1, :], idxsample[3, :])

    # -------- context versus tr. bias ----------------
    # plot samples
    # previous left
    # figs = []
    
    # fighist,axhist=plt.subplots(2,1,figsize=(4,6),tight_layout=True, sharex=True)
    # # axhist[0].hist(yevi[SVMAXIS,idxsample[0,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # # axhist[0].hist(yevi[SVMAXIS,idxsample[1,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    
    # # axhist[1].hist(yevi[SVMAXIS,idxsample[2,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # # axhist[1].hist(yevi[SVMAXIS,idxsample[3,:]],bins=20,facecolor='tab:green',alpha=0.75)
    
    # axhist[0].hist(yevi[SVMAXIS,idxsample[0,:]]+yevi[SVMAXIS-1,idxsample[0,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[0].hist(yevi[SVMAXIS,idxsample[1,:]]+yevi[SVMAXIS-1,idxsample[1,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    
    # axhist[1].hist(yevi[SVMAXIS,idxsample[2,:]]+yevi[SVMAXIS-1,idxsample[2,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # axhist[1].hist(yevi[SVMAXIS,idxsample[3,:]]+yevi[SVMAXIS-1,idxsample[3,:]],bins=20,facecolor='tab:green',alpha=0.75)
        
    
   
    #######-----------------conditioned on previous choice--------------------
    # for idx, prev_ch in zip([idxpreal, idxprear], ['Left', 'Right']):
    #     ctxt    = np.squeeze(yevi[1, idx])
    #     tr_bias = np.squeeze(yevi[SVMAXIS, idx])#+yevi[SVMAXIS-1, idx])
    #     df = {'Context encoding': ctxt, 'Transition bias encoding': tr_bias,
    #           'Upcoming Stimulus Category': ytruthlabels[SVMAXIS, idx]}
    #     df  = pd.DataFrame(df)
    #     fig = multivariateGrid(col_x='Context encoding',
    #                             col_y='Transition bias encoding',
    #                             col_k='Upcoming Stimulus Category', df=df,
    #                             colors=[GREEN, PURPLE], s=S_PLOTS, alpha=.75)
        

    #     # fig.ax_marg_x.set_xlim(XLIMS_2D)
    #     # fig.ax_marg_y.set_ylim(YLIMS_2D)
    #     fig.ax_joint.axhline(y=0, color='k', linestyle='--', lw=0.5)
    #     fig.fig.suptitle('a'+prev_outc+' / Prev. Ch. '+prev_ch)
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
    #         coefficients = np.polyfit(ctxt, tr_bias, 1)
    #         poly = np.poly1d(coefficients)
    #         new_y = poly([np.min(ctxt), np.max(ctxt)])
    #         fig.ax_joint.plot([np.min(ctxt), np.max(ctxt)], new_y, color='k',
    #                           lw=0.5)

    # --------- previous ch. versus tr. bias
    idxrpt = idxctxtr
    idxalt = idxctxta
    
    
    # fighist,axhist=plt.subplots(4,1,figsize=(3,4),tight_layout=True, sharex=True,sharey=True)
    # axhist[0].hist(yevi[SVMAXIS,idxsample[0,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[0].hist(yevi[SVMAXIS,idxsample[2,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # axhist[1].hist(yevi[SVMAXIS-1,idxsample[0,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[1].hist(yevi[SVMAXIS-1,idxsample[2,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    
    # axhist[2].hist(yevi[SVMAXIS,idxsample[1,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # axhist[2].hist(yevi[SVMAXIS,idxsample[3,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[3].hist(yevi[SVMAXIS-1,idxsample[1,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # axhist[3].hist(yevi[SVMAXIS-1,idxsample[3,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[0].set_title('ctxt.')
    
    # fighist,axhist=plt.subplots(4,1,figsize=(3,4),tight_layout=True, sharex=True,sharey=True)
    # axhist[0].hist(yevi[SVMAXIS,idxsample[0,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[0].hist(yevi[SVMAXIS,idxsample[1,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # axhist[1].hist(yevi[SVMAXIS-1,idxsample[0,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[1].hist(yevi[SVMAXIS-1,idxsample[1,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    
    # axhist[2].hist(yevi[SVMAXIS,idxsample[2,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # axhist[2].hist(yevi[SVMAXIS,idxsample[3,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[3].hist(yevi[SVMAXIS-1,idxsample[2,:]],bins=20,facecolor='tab:purple',alpha=0.75)
    # axhist[3].hist(yevi[SVMAXIS-1,idxsample[3,:]],bins=20,facecolor='tab:green',alpha=0.75)
    # axhist[0].set_title('prev. ch.')
    
    # ##### 0608 comment --------------------------
    # # figs = []
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
    
    
    
    ####$     25 July >>>>>>>>>>>>>>>>>>>>>>

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


def hist_integration_balanced(data_tr,wc, bc, we, be, nselect,false_files, coh_ch_stateratio_correct,coh_ch_stateratio_eror, pop_correct, pop_error, USE_POP=1, mode='decoding',DOREVERSE=0, CONTROL=0, STIM_PERIOD=1, RECORD_TRIALS=1, REC_TRIALS_SET=[],PCA_only=0,mmodel=[]):
    
    unique_states,unique_cohs = np.arange(8), [-1,0,1]
    files = data_tr['files']        
    ### generate training and testing dataset independently for each fold
    ### there is no need to delete the non selective single cells
    data_traintest_tr = dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
    Xdata_beh_trainset, ylabels_beh_trainset, Xdata_beh_testset, ylabels_beh_testset, files = data_traintest_tr['Xdata_beh_trainset'], \
        data_traintest_tr['ylabels_beh_trainset'], data_traintest_tr['Xdata_beh_testset'], data_traintest_tr['ylabels_beh_testset'], data_traintest_tr['files']

    unique_states = np.arange(8)
    unique_cohs   = [-1,0,1]
    
    # Xdata_train_correct, ylabels_train_correct, Xdata_train_error,\
    #     ylabels_train_error, _ =\
    #     gpt.merge_pseudo_beh_trials_individual(Xdata_beh_trainset, ylabels_beh_trainset, unique_states,
    #                                 unique_cohs, files, false_files,
    #                                 2, RECORD_TRIALS=1,
    #                                 RECORDED_TRIALS_SET=[],STIM_BEH=1)
    
    # finding decoding axis
    NN = len(nselect)#np.shape(Xdata_train_correct[unique_cohs[0],0])[1]
    if(RECORD_TRIALS == 1):
        REC_TRIALS_SET = {}
        for itr in range(NITERATIONS):
            REC_TRIALS_SET[itr] = {}
    if(RECORD_TRIALS == 1):
        ### ~~~~ *2 ~~~~~ each choice --- 2 states (4 states in total)
        ##### unique_cohs --- only for those used for decoders --- zero coherence
        # stats_correct, stats_error, coeffs, intercepts,Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        #     Xtest_set_error, ytest_set_error, yevi_set_error, REC_TRIALS_SET\
        #     = bl.bootstrap_linsvm_step_fixationperiod_balanced(data_tr, NN, 
        #                                 unique_states, unique_cohs, nselect, files,
        #                                 false_files, coh_ch_stateratio_correct,coh_ch_stateratio_eror, pop_correct, pop_error, USE_POP, type, DOREVERSE=DOREVERSE,
        #                                 CONTROL=CONTROL, n_iterations=NITERATIONS,
        #                                 N_pseudo_dec=NPSEUDODEC_BEH, ACE_RATIO = ACE_RATIO,
        #                                 train_percent=PERCENTTRAIN,
        #                                 RECORD_TRIALS=RECORD_TRIALS,
        #                                 RECORDED_TRIALS_SET=REC_TRIALS_SET,mmodel=mmodel,PCA_n_components=PCA_n_components)
        stats_correct_rep, stats_error_rep, stats_correct_alt, stats_error_alt, coeffs, intercepts,Xtest_set_correct_rep, ytest_set_correct_rep, Xtest_set_correct_alt, ytest_set_correct_alt, yevi_set_correct,\
            Xtest_set_error_rep, ytest_set_error_rep, Xtest_set_error_alt, ytest_set_error_alt, yevi_set_error, REC_TRIALS_SET\
            = bl.bootstrap_linsvm_step_fixationperiod_balanced(data_tr, NN, 
                                        unique_states, unique_cohs, nselect, files,
                                        false_files, coh_ch_stateratio_correct,coh_ch_stateratio_eror, pop_correct, pop_error, USE_POP, type, DOREVERSE=DOREVERSE,
                                        CONTROL=CONTROL, n_iterations=NITERATIONS,
                                        N_pseudo_dec=NPSEUDODEC_BEH, ACE_RATIO = ACE_RATIO,
                                        train_percent=PERCENTTRAIN,
                                        RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=REC_TRIALS_SET,mmodel=mmodel,PCA_n_components=PCA_n_components)
    else:
        coeffs, intercepts,Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
            Xtest_set_error, ytest_set_error, yevi_set_error, REC_TRIALS_SET\
            = bl.bootstrap_linsvm_step_fixationperiod_balanced(data_tr, NN, 
                                        unique_states, unique_cohs, nselect, files,
                                        false_files, coh_ch_stateratio_correct,coh_ch_stateratio_eror,pop_correct, pop_error,USE_POP, type, DOREVERSE=DOREVERSE,
                                        CONTROL=CONTROL, n_iterations=NITERATIONS,
                                        N_pseudo_dec=NPSEUDODEC_BEH, ACE_RATIO = ACE_RATIO,
                                        train_percent=PERCENTTRAIN,
                                        RECORD_TRIALS=1,
                                        RECORDED_TRIALS_SET=REC_TRIALS_SET,mmodel=mmodel,PCA_n_components=PCA_n_components)

    # #### visualizing performance -- population gain control 
    # if USE_POP ==1:
    #     guc.cross_gaincontrol(stats_correct,stats_error, coeffs, intercepts, ytest_set_correct, yevi_set_correct, ytest_set_error, yevi_set_error, label_axis=0, evi_axis= 4, CONTROL=1)  

         
    # lst = [stats_correct, stats_error, 
    #        coeffs, intercepts,
    #        ytest_set_correct,
    #        yevi_set_correct,
    #        coeffs, intercepts,  # Xtest_set_error,
    #        ytest_set_error,  yevi_set_error,
    #        REC_TRIALS_SET]
    # stg = ["stats_correct, stats_error, "
    #        "coefs_correct, intercepts_correct,"
    #        "ytest_set_correct, "
    #        "yevi_set_correct, "
    #        "coefs_error, intercepts_error,"  # " Xtest_set_error,"
    #        "ytest_set_error, yevi_set_error,"
    #        "REC_TRIALS_SET"]

    lst = [stats_correct_rep, stats_error_rep,
           stats_correct_alt, stats_error_alt, 
           coeffs, intercepts,
           ytest_set_correct_rep, ytest_set_correct_alt,
           yevi_set_correct,
           coeffs, intercepts,  # Xtest_set_error,
           ytest_set_error_rep, ytest_set_error_alt,  
           yevi_set_error,
           REC_TRIALS_SET]
    stg = ["stats_correct_rep, stats_error_rep, "
           "stats_correct_alt, stats_error_alt, "
           "coefs_correct, intercepts_correct,"
           "ytest_set_correct_rep, ytest_set_correct_alt, "
           "yevi_set_correct, "
           "coefs_error, intercepts_error,"  # " Xtest_set_error,"
           "ytest_set_error_rep, ytest_set_error_alt, "
           "yevi_set_error,"
           "REC_TRIALS_SET"]

    d = list_to_dict(lst=lst, string=stg)
    return d, Xtest_set_correct_rep, Xtest_set_error_rep, Xtest_set_correct_alt, Xtest_set_error_alt

def bias_VS_prob(data_tr, data_dec, unique_cohs, nselect, num_beh_trials, EACHSTATES,
                 NITERATIONS, ax, RECORD_TRIALS=1, REC_TRIALS_SET=[],STIM_BEH=1,PCA_only=0,mmodel=[]):
    """
    Bootstrapping for psychometric curve generation

    Parameters
    ----------
    data_tr: dict
        dataset for generating pseudo-trials

    data_dec: dict
        dataset containing the encodings of history information and labels

    unique_cohs: array -- [-1, 0, 1]

    EACHSTATES: int
        number of pseudo trials per state

    NITERATIONS: int
        number of steps (bootstrap)

    ax: figure handle


    Returns
    -------
    d : dict
        PSYCHOMETRIC CURVES AND SLOPES

    """
    # Xdata_set, ylabels_set = data_tr['Xdata_psy_testset'], data_tr['ylabels_psy_testset']
    coeffs, intercepts     = data_dec['coefs_correct'], data_dec['intercepts_correct']
    if (RECORD_TRIALS == 1):
        REC_TRIALS_SET = {}
        for i in range(NITERATIONS):
            REC_TRIALS_SET[i] = {}

    FIX_TRBIAS_BINS = np.array([-1, 0, 1])
    NBINS = 5

    # NTRBIAS = len(FIX_TRBIAS_BINS)
    psychometric_trbias_correct = np.zeros(
        (NITERATIONS, len(unique_cohs), NBINS))
    psychometric_trbias_error = np.zeros(
        (NITERATIONS, len(unique_cohs), NBINS))
    trbias_range_correct = np.zeros((NITERATIONS, len(unique_cohs), NBINS))
    trbias_range_error   = np.zeros((NITERATIONS, len(unique_cohs), NBINS))

    curveslopes_correct, curveslopes_error = np.zeros(
        (NITERATIONS, len(unique_cohs))), np.zeros((NITERATIONS, len(unique_cohs)))
    curveintercept_correct, curveintercept_error = np.zeros(
        (NITERATIONS, len(unique_cohs))), np.zeros((NITERATIONS, len(unique_cohs)))
    for idx in range(NITERATIONS):    
        unique_states = np.arange(8)
        unique_cohs = [-1,0,1]
        ### generate training and testing dataset independently for each fold
        data_traintest_tr = dataset_generate(data_tr, unique_states, unique_cohs, files, false_files,THRESH_TRIAL)
        Xdata_trainset, Xdata_testset = data_traintest_tr['Xdata_psy_trainset'],data_traintest_tr['Xdata_psy_testset']
        ylabels_trainset,ylabels_testset = data_traintest_tr['ylabels_psy_trainset'],data_traintest_tr['ylabels_psy_testset']
        Xmerge_trials_correct_train, ymerge_labels_correct_train, Xmerge_trials_error_train,\
            ymerge_labels_error_train, REC_TRIALS_SET[idx] =\
            gpt.merge_pseudo_beh_trials(Xdata_trainset, ylabels_trainset, unique_states,
                                        unique_cohs, nselect, files, false_files,
                                        EACHSTATES, RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=REC_TRIALS_SET[idx],STIM_BEH=STIM_BEH)
        Xmerge_trials_correct_test, ymerge_labels_correct_test, Xmerge_trials_error_test,\
            ymerge_labels_error_test, REC_TRIALS_SET[idx] =\
            gpt.merge_pseudo_beh_trials(Xdata_testset, ylabels_testset, unique_states,
                                        unique_cohs, nselect, files, false_files,
                                        EACHSTATES, RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=REC_TRIALS_SET[idx],STIM_BEH=STIM_BEH)
        Xmerge_trials_correct, Xmerge_trials_error={},{}
        ymerge_labels_correct,ymerge_labels_error = {},{}
        
        for ikk in Xmerge_trials_correct_train.keys():
            Xmerge_trials_correct[ikk] = np.vstack((Xmerge_trials_correct_train[ikk],Xmerge_trials_correct_test[ikk]))
            ymerge_labels_correct[ikk] = np.hstack((ymerge_labels_correct_train[ikk],ymerge_labels_correct_test[ikk]))
        for ikk in Xmerge_trials_error_train.keys():
            Xmerge_trials_error[ikk] = np.vstack((Xmerge_trials_error_train[ikk],Xmerge_trials_error_test[ikk]))
            ymerge_labels_error[ikk] = np.hstack((ymerge_labels_error_train[ikk],ymerge_labels_error_test[ikk]))
    
        unique_states = np.arange(4, 8, 1)
        psychometric_trbias_correct[idx, :, :], trbias_range_correct[idx, :, :] =\
            gpt.behaviour_trbias_proj_new(coeffs, intercepts, Xmerge_trials_correct,
                                      ymerge_labels_correct, unique_states,
                                      unique_cohs, [0, 1], num_beh_trials,
                                      EACHSTATES=EACHSTATES,
                                      FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS,mmodel=mmodel,PCA_n_components=PCA_n_components)
    
        unique_states = np.arange(4)
        psychometric_trbias_error[idx, :, :], trbias_range_error[idx, :, :] =\
            gpt.behaviour_trbias_proj_new(coeffs, intercepts, Xmerge_trials_error,
                                      ymerge_labels_error, unique_states,
                                      unique_cohs, [0, 1], num_beh_trials,
                                      EACHSTATES=EACHSTATES,
                                      FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS,mmodel=mmodel,PCA_n_components=PCA_n_components)

        # # compute the slope for zero-coherence
        # for icoh in range(len(unique_cohs)):
        #     coh0_correct =\
        #         np.polyfit(trbias_range_correct[idx, icoh, 1:-1],
        #                    psychometric_trbias_correct[idx, icoh, 1:-1], 1)
        #     coh0_error = np.polyfit(trbias_range_error[idx, icoh, 1:-1],
        #                             psychometric_trbias_error[idx, icoh, 1:-1], 1)
        #     curveslopes_correct[idx, icoh], curveintercept_correct[idx, icoh] =\
        #         coh0_correct[0], coh0_correct[1]
        #     curveslopes_error[idx, icoh],   curveintercept_error[idx, icoh] =\
        #         coh0_error[0],  coh0_error[1]

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

    lst = [REC_TRIALS_SET]
    stg = ["REC_TRIALS_SET"]
    d_beh = list_to_dict(lst=lst, string=stg)
    return curveslopes_correct, curveintercept_correct, curveslopes_error,\
        curveintercept_error, d_beh


# def bias_VS_prob(data_tr, data_dec, unique_cohs, num_beh_trials, EACHSTATES,
#                  NITERATIONS, ax, RECORD_TRIALS=1, REC_TRIALS_SET=[],STIM_BEH=1,PCA_only=0,mmodel=[]):
#     """
#     Bootstrapping for psychometric curve generation

#     Parameters
#     ----------
#     data_tr: dict
#         dataset for generating pseudo-trials

#     data_dec: dict
#         dataset containing the encodings of history information and labels

#     unique_cohs: array -- [-1, 0, 1]

#     EACHSTATES: int
#         number of pseudo trials per state

#     NITERATIONS: int
#         number of steps (bootstrap)

#     ax: figure handle


#     Returns
#     -------
#     d : dict
#         PSYCHOMETRIC CURVES AND SLOPES

#     """
#     Xdata_set, ylabels_set = data_tr['Xdata_set'], data_tr['ylabels_set']
#     metadata = data_tr['metadata']
#     coeffs, intercepts = data_dec['coefs_correct'], data_dec['intercepts_correct']
#     if (RECORD_TRIALS == 1):
#         REC_TRIALS_SET = {}
#         for i in range(NITERATIONS):
#             REC_TRIALS_SET[i] = {}

#     FIX_TRBIAS_BINS = np.array([-1, 0, 1])

#     NBINS = 5

#     # NTRBIAS = len(FIX_TRBIAS_BINS)
#     psychometric_trbias_correct = np.zeros(
#         (NITERATIONS, len(unique_cohs), NBINS))
#     psychometric_trbias_error = np.zeros(
#         (NITERATIONS, len(unique_cohs), NBINS))
#     trbias_range_correct = np.zeros((NITERATIONS, len(unique_cohs), NBINS))
#     trbias_range_error = np.zeros((NITERATIONS, len(unique_cohs), NBINS))

#     curveslopes_correct, curveslopes_error = np.zeros(
#         (NITERATIONS, len(unique_cohs))), np.zeros((NITERATIONS, len(unique_cohs)))
#     curveintercept_correct, curveintercept_error = np.zeros(
#         (NITERATIONS, len(unique_cohs))), np.zeros((NITERATIONS, len(unique_cohs)))
#     for idx in range(NITERATIONS):
#         unique_states = np.arange(0, 8, 1)
#         Xmerge_trials_correct, ymerge_labels_correct, Xmerge_trials_error,\
#             ymerge_labels_error, REC_TRIALS_SET[idx] =\
#             gpt.merge_pseudo_beh_trials(Xdata_set, ylabels_set, unique_states,
#                                         unique_cohs, files, false_files, metadata,
#                                         EACHSTATES, RECORD_TRIALS=RECORD_TRIALS,
#                                         RECORDED_TRIALS_SET=REC_TRIALS_SET[idx],STIM_BEH=STIM_BEH)

#         unique_cohs = [-1, 0, 1]# [-0.6,-0.25,0,0.25,0.6]#
#         unique_states = np.arange(4, 8, 1)
#         psychometric_trbias_correct[idx, :, :], trbias_range_correct[idx, :, :] =\
#             gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_correct,
#                                       ymerge_labels_correct, [4, 5, 6, 7],
#                                       unique_cohs, [0, 1], num_beh_trials,
#                                       EACHSTATES=EACHSTATES,
#                                       FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS,mmodel=mmodel,PCA_n_components=PCA_n_components)
    
#         unique_states = np.arange(4)
#         psychometric_trbias_error[idx, :, :], trbias_range_error[idx, :, :] =\
#             gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_error,
#                                       ymerge_labels_error, [0, 1, 2, 3],
#                                       unique_cohs, [0, 1], num_beh_trials,
#                                       EACHSTATES=EACHSTATES,
#                                       FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS,mmodel=mmodel,PCA_n_components=PCA_n_components)

#         # # compute the slope for zero-coherence
#         # for icoh in range(len(unique_cohs)):
#         #     coh0_correct =\
#         #         np.polyfit(trbias_range_correct[idx, icoh, 1:-1],
#         #                    psychometric_trbias_correct[idx, icoh, 1:-1], 1)
#         #     coh0_error = np.polyfit(trbias_range_error[idx, icoh, 1:-1],
#         #                             psychometric_trbias_error[idx, icoh, 1:-1], 1)
#         #     curveslopes_correct[idx, icoh], curveintercept_correct[idx, icoh] =\
#         #         coh0_correct[0], coh0_correct[1]
#         #     curveslopes_error[idx, icoh],   curveintercept_error[idx, icoh] =\
#         #         coh0_error[0],  coh0_error[1]

#     colors = plt.cm.PRGn_r(np.linspace(0, 1, 3*6))
#     for i in range(3):
#         meanx = np.mean(trbias_range_correct[:, i, :], axis=0)
#         meany = np.mean(psychometric_trbias_correct[:, i, :], axis=0)
#         errory = np.std(
#             psychometric_trbias_correct[:, i, :], axis=0)/np.sqrt(NITERATIONS)
#         errorx = np.std(
#             trbias_range_correct[:, i, :], axis=0)/np.sqrt(NITERATIONS)
#         ax[0].errorbar(meanx, meany, xerr=errorx,
#                        yerr=errory, color=colors[i*3], lw=1.5)

#         meanx = np.mean(trbias_range_error[:, i, :], axis=0)
#         meany = np.mean(psychometric_trbias_error[:, i, :], axis=0)
#         errory = np.std(
#             psychometric_trbias_error[:, i, :], axis=0)/np.sqrt(NITERATIONS)
#         errorx = np.std(
#             trbias_range_error[:, i, :], axis=0)/np.sqrt(NITERATIONS)
#         ax[1].errorbar(meanx, meany, xerr=errorx,
#                        yerr=errory, color=colors[i*3], lw=1.5)
#     # ax[0].plot(trbias_range_correct[1,:], psychometric_trbias_correct[1,:],
#     #            color='k',lw=1.5)
#     # ax[1].plot(trbias_range_error[1,:], psychometric_trbias_error[1,:],
#     #            color=colors[i],lw=1.5)

#     lst = [REC_TRIALS_SET]
#     stg = ["REC_TRIALS_SET"]
#     d_beh = list_to_dict(lst=lst, string=stg)
#     return curveslopes_correct, curveintercept_correct, curveslopes_error,\
#         curveintercept_error, d_beh
        

def bias_VS_prob_evi(data_tr, data_dec, data_int_ac, data_int_ae, unique_cohs, num_beh_trials, pop_correct, pop_zero, pop_error, USE_POP, EACHSTATES,
                 NITERATIONS, ax, RECORD_TRIALS=1, REC_TRIALS_SET=[],STIM_BEH=1,PCA_only=0,mmodel=[]):
    """
    Bootstrapping for psychometric curve generation

    Parameters
    ----------
    data_tr: dict
        dataset for generating pseudo-trials

    data_dec: dict
        dataset containing the encodings of history information and labels

    unique_cohs: array -- [-1, 0, 1]

    EACHSTATES: int
        number of pseudo trials per state

    NITERATIONS: int
        number of steps (bootstrap)

    ax: figure handle


    Returns
    -------
    d : dict
        PSYCHOMETRIC CURVES AND SLOPES

    """
    Xdata_set, ylabels_set = data_tr['Xdata_set'], data_tr['ylabels_set']
    metadata               = data_tr['metadata']
    coeffs, intercepts = data_dec['coefs_correct'], data_dec['intercepts_correct']
    coeffs_beh_ac, intercepts_beh_ac = data_int_ac['coefs_correct'], data_int_ac['intercepts_correct']
    coeffs_beh_ae, intercepts_beh_ae = data_int_ae['coefs_error'], data_int_ae['intercepts_error']
    
    if (RECORD_TRIALS == 1):
        REC_TRIALS_SET = {}
        for i in range(NITERATIONS):
            REC_TRIALS_SET[i] = {}

    FIX_TRBIAS_BINS = np.array([-1, 0, 1])

    NBINS = 5

    # NTRBIAS = len(FIX_TRBIAS_BINS)
    behaviour_correct = {}
    behaviour_error   = {}
    trbias_range_correct        = {}
    trbias_range_error          = {}
    
    for idxcoh, coh in enumerate([-1,0,1]):
        behaviour_correct[coh], behaviour_error[coh], trbias_range_correct[coh], trbias_range_error[coh]=[],[],[],[]


    for idx in range(NITERATIONS):
        unique_states = np.arange(0, 8, 1)
        Xmerge_trials_correct, ymerge_labels_correct, Xmerge_trials_error,\
            ymerge_labels_error, REC_TRIALS_SET[idx] =\
            gpt.merge_pseudo_beh_trials(Xdata_set, ylabels_set, unique_states,
                                        unique_cohs, files, false_files, metadata,
                                        EACHSTATES, RECORD_TRIALS=RECORD_TRIALS,
                                        RECORDED_TRIALS_SET=REC_TRIALS_SET[idx],STIM_BEH=STIM_BEH)

        unique_cohs   = [-1, 0, 1]# [-0.6,-0.25,0,0.25,0.6]#
        unique_states = np.arange(4, 8, 1)
        beh_correct, trbias_correct =\
            gpt.behaviour_trbias_proj_evi(coeffs, intercepts, coeffs_beh_ac, intercepts_beh_ac, Xmerge_trials_correct,
                                      ymerge_labels_correct, [4, 5, 6, 7],
                                      unique_cohs, [0, 1], num_beh_trials, pop_correct,pop_zero,USE_POP,
                                      EACHSTATES=EACHSTATES,
                                      FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS,mmodel=mmodel,PCA_n_components=PCA_n_components)
    
        unique_states = np.arange(4)
        beh_error, trbias_error =\
            gpt.behaviour_trbias_proj_evi(coeffs, intercepts, coeffs_beh_ae, intercepts_beh_ae, Xmerge_trials_error,
                                      ymerge_labels_error, [0, 1, 2, 3],
                                      unique_cohs, [0, 1], num_beh_trials,pop_error,pop_zero, USE_POP,
                                      EACHSTATES=EACHSTATES,
                                      FIX_TRBIAS_BINS=FIX_TRBIAS_BINS, NBINS=NBINS,mmodel=mmodel,PCA_n_components=PCA_n_components)
        
        for idxcoh, coh in enumerate([-1,0,1]):
            if(len(behaviour_correct[coh])==0):
                behaviour_correct[coh], behaviour_error[coh] = beh_correct[coh],beh_error[coh]
                trbias_range_correct[coh], trbias_range_error[coh] = trbias_correct[coh],trbias_error[coh]
            else:
                behaviour_correct[coh], behaviour_error[coh] = np.hstack((behaviour_correct[coh],beh_correct[coh])),np.hstack((behaviour_error[coh],beh_error[coh]))
                trbias_range_correct[coh], trbias_range_error[coh] = np.hstack((trbias_range_correct[coh],trbias_correct[coh])),np.hstack((trbias_range_error[coh],trbias_error[coh]))

    colors = plt.cm.PRGn_r(np.linspace(0, 1, 3*6))
    for idxcoh, coh in enumerate([-1,0,1]):
        print(len(trbias_range_correct[coh]))
        idxsample = np.random.choice(np.arange(len(trbias_range_correct[coh])),size=NUM_SAMPLES,replace=False)
        ax[0].scatter(trbias_range_correct[coh][idxsample],behaviour_correct[coh][idxsample],s=20,c=colors[idxcoh*3],alpha=0.25)
        confidence_ellipse(trbias_range_correct[coh][idxsample],behaviour_correct[coh][idxsample],ax[0],edgecolor='gray')
        idxsample = np.random.choice(np.arange(len(trbias_range_error[coh])),size=NUM_SAMPLES,replace=False)
        ax[1].scatter(trbias_range_error[coh][idxsample],behaviour_error[coh][idxsample],s=20,c=colors[idxcoh*3],alpha=0.25)
        confidence_ellipse(trbias_range_error[coh][idxsample],behaviour_error[coh][idxsample],ax[1],edgecolor='gray')
    # ax[0].set_xlim([-8,8])
    # ax[0].set_xticks([-8,0,8])
    # ax[0].set_ylim([-3,3])
    # ax[0].set_yticks([-3,0,3])

    lst = [REC_TRIALS_SET]
    stg = ["REC_TRIALS_SET"]
    d_beh = list_to_dict(lst=lst, string=stg)
    return  d_beh
        

def neural_activity_sort(Xdata_set,ylabels_set,unique_states,files,idx_delete):
    Xflat_trials_correct,Xflat_trials_error = {},{}
    
    df_correct, df_error = {},{}  
    p_values = []
    
    ### counting overall neuron number 
    NN_error = 0
    Nstart, Nend = 0,0
    unique_cohs,unique_choices = [-1,0,1],[0,1]
    for idxf in range(len(files)):
        if(idxf in idx_delete):
            continue
        Nstart = Nend
        X_t = []
        for state in unique_states:
            if state>=4:
                break
            data_temp  = Xdata_set[idxf,'error'].copy()
            # try:
            #     totaltrials=np.shape(data_temp[state])[0]
            # except:
            #     print('exist for initial finding')
            #     continue
            # if totaltrials<1:
            #     continue
            # #### generate sampled true trials for individual neurons in each pseudo trial
            # NN, ntrials = np.shape(data_temp[state])[1], np.shape(data_temp[state])[0]
            # if(len(X_t)==0):
            #     NN_error +=np.shape(data_temp[state])[1]
            #     Nend +=np.shape(data_temp[state])[1]
            #     X_t = data_temp[state].copy()
            # else:
            #     X_t = np.vstack((X_t,data_temp[state].copy()))
                             
            for coh in unique_cohs:
                for ch in unique_choices:
                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        Nend +=np.shape(data_temp[state,coh,ch])[1]
                        NN_error +=np.shape(data_temp[state,coh,ch])[1]
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
        df_error[idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))

        X_t = []
        for state in unique_states:
            if state<4:
                continue    
            data_temp  = Xdata_set[idxf,'correct'].copy()

            # try:
            #     totaltrials=np.shape(data_temp[state])[0]
            # except:
            #     print('exist for initial finding')
            #     continue
            # if totaltrials<1:
            #     continue
            # #### generate sampled true trials for individual neurons in each pseudo trial
            # NN, ntrials = np.shape(data_temp[state])[1], np.shape(data_temp[state])[0]
            # if(len(X_t)==0):
            #     X_t = data_temp[state].copy()
            # else:
            #     X_t = np.vstack((X_t,data_temp[state].copy()))
                
            for coh in unique_cohs:
                for ch in unique_choices:

                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_correct[idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        for icol in range(Nstart,Nend):
            try:
                _,pg=mwu(df_correct[idxf][icol],df_error[idxf][icol],alternative='greater')
                _,pl=mwu(df_correct[idxf][icol],df_error[idxf][icol],alternative='less')
            except:
                pg,pl = 1,1
            correct_mean, error_mean = np.mean(df_correct[idxf][icol]),np.mean(df_error[idxf][icol])
            if (len(p_values)==0):
                p_values=np.array([pg, pl, min(pg,pl),correct_mean,error_mean])
            else:
                p_values = np.vstack((p_values,np.array([pg, pl, min(pg,pl),correct_mean,error_mean])))
    
    larger = np.where(p_values[:,0]<0.05)[0]
    less = np.where(p_values[:,1]<0.05)[0]   
    pop_correct = less.copy()
    pop_error   = larger.copy()
    pop_use = np.union1d(pop_correct, pop_error)
    pop_zero   = np.setdiff1d(np.arange(NN_error),pop_use)
    print('************ Neuron number **********')
    print(NN_error)
    return pop_correct, pop_error,pop_zero, NN_error,p_values

def single_neuron_selectivity(Xdata_set,ylabels_set,unique_states,files,idx_delete):
    Xflat_trials_correct,Xflat_trials_error = {},{}
    
    df={}
    y_prevch, y_ctxt, y_rw,y_ch = {},{},{},{}
    p_values,params = [],[]
    
    ### counting overall neuron number 
    NN_error = 0
    Nstart, Nend = 0,0
    unique_cohs,unique_choices = [-1,0,1],[0,1]

    for idxf in range(len(files)):
        if(idxf in idx_delete):
            continue
        Nstart = Nend
        X_t = []
        y_s = []
        y_c = []
        for state in unique_states:
            if state<4:
                data_temp  = Xdata_set[idxf,'error'].copy()                             
                for coh in unique_cohs:
                    for ch in unique_choices:
                        try:
                            totaltrials=np.shape(data_temp[state,coh,ch])[0]
                        except:
                            print('exist for initial finding')
                            continue
                        if totaltrials<1:
                            continue
                        #### generate sampled true trials for individual neurons in each pseudo trial
                        NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                        if(len(X_t)==0):
                            Nend +=np.shape(data_temp[state,coh,ch])[1]
                            NN_error +=np.shape(data_temp[state,coh,ch])[1]
                            X_t = data_temp[state,coh,ch].copy()
                            y_l = np.ones(np.shape(X_t)[0])*state
                            y_c = np.ones(np.shape(X_t)[0])*ch
            
                        else:
                            X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                            y_l = np.hstack((y_l,state*np.ones(np.shape(data_temp[state,coh,ch])[0])))
                            y_c = np.hstack((y_c,ch*np.ones(np.shape(data_temp[state,coh,ch])[0])))
            if state>=4:
                data_temp  = Xdata_set[idxf,'correct'].copy()                             
                for coh in unique_cohs:
                    for ch in unique_choices:
                        try:
                            totaltrials=np.shape(data_temp[state,coh,ch])[0]
                        except:
                            print('exist for initial finding')
                            continue
                        if totaltrials<1:
                            continue
                        #### generate sampled true trials for individual neurons in each pseudo trial
                        NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                        y_l = np.hstack((y_l,state*np.ones(np.shape(data_temp[state,coh,ch])[0])))
                        y_c = np.hstack((y_c,ch*np.ones(np.shape(data_temp[state,coh,ch])[0])))
        # df[idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        ntrials = np.shape(X_t)[0]
        ### prev ch
        idxprevl_e = np.union1d(np.where(y_l==0)[0],np.where(y_l==2)[0])
        idxprevl_c = np.union1d(np.where(y_l==4)[0],np.where(y_l==6)[0])
        y_prevch[idxf] = np.zeros_like(y_l) 
        idxprevl = np.union1d(idxprevl_e,idxprevl_c)
        idxprevr = np.setdiff1d(np.arange(ntrials), idxprevl)
        y_prevch[idxf][idxprevl] = 0 
        y_prevch[idxf][idxprevr] = 1
        
        ### context
        idxrep_e = np.union1d(np.where(y_l==0)[0],np.where(y_l==1)[0])
        idxrep_c = np.union1d(np.where(y_l==4)[0],np.where(y_l==5)[0])
        y_ctxt[idxf] = np.zeros_like(y_l) 
        idxrep = np.union1d(idxrep_e,idxrep_c)
        idxalt = np.setdiff1d(np.arange(ntrials), idxrep)
        y_ctxt[idxf][idxrep] = 0 
        y_ctxt[idxf][idxalt] = 1
        
        ### reward
        idxe = np.where(y_l<4)[0]
        idxc = np.where(y_l>=4)[0]
        y_rw[idxf] = np.zeros_like(y_l) 
        y_rw[idxf][idxe] = 0 
        y_rw[idxf][idxc] = 1
        
        ### choice 
        idxl = np.where(y_c==0)
        idxr = np.where(y_c==1)
        y_ch[idxf] = np.zeros_like(y_c)
        y_ch[idxf][idxl] = 0 
        y_ch[idxf][idxr] = 1

        for icol in range(Nstart,Nend):
            df = {'resp':X_t[:,icol-Nstart],'prevch':y_prevch[idxf],'ctxt':y_ctxt[idxf],'rw':y_rw[idxf],'ch':y_ch[idxf]}
            
            df = pd.DataFrame(df)
            X = df[['prevch','ctxt','rw','ch']]
            Y = df['resp']
            X = sm.add_constant(X)   
            '''
            add_constant: index-0 -- constant coefficient
            index-1 -- previous choice
            index-2 -- context
            index-3 -- reward 
            index-4 -- choice
            '''
            results = sm.OLS(Y,X).fit() 
            if len(params)==0:
                params   = results.params 
                p_values = results.pvalues
            else:
                params   = np.vstack((params,results.params))
                p_values = np.vstack((p_values,results.pvalues))
                
            
    print('************ Neuron number **********')
    print(NN_error)
    neuron_p = np.zeros(np.shape(p_values)[0])
    for i in range(np.shape(p_values)[0]):
        neuron_p[i]= len(np.where(p_values[i,1:]<0.05)[0])
    neuron_noselect = np.where(neuron_p<1)[0]
    neuron_select   = np.where(neuron_p>0)[0]
    return params, p_values,neuron_noselect, neuron_select


def neural_activity_sort_prevch(Xdata_set,ylabels_set,unique_states,files,idx_delete, nselect):
    Xflat_trials_correct,Xflat_trials_error = {},{}
    
    df_correct, df_error = {},{}  
    p_values_correct,p_values_error = [],[]
    
    ### counting overall neuron number 
    NN_error = 0
    Nstart, Nend = 0,0
    statel,stater=[4,6],[5,7]
    statele,statere=[0,2],[1,3]
    
    unique_cohs,unique_choices = [-1,0,1],[0,1]
    for idxf in range(len(files)):
        if(idxf in idx_delete):
            continue
        Nstart = Nend
        X_t = []
        for state in statel:   
            data_temp  = Xdata_set[idxf,'correct'].copy()
            
            for coh in unique_cohs:
                for ch in unique_choices:
                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        Nend +=np.shape(data_temp[state,coh,ch])[1]
                        NN_error +=np.shape(data_temp[state,coh,ch])[1]
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_correct['left',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        X_t = []
        for state in stater: 
            data_temp  = Xdata_set[idxf,'correct'].copy()
            
            for coh in unique_cohs:
                for ch in unique_choices:

                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_correct['right',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        
        
        X_t = []
        for state in (statele):   
            data_temp  = Xdata_set[idxf,'error'].copy()
            for coh in unique_cohs:
                for ch in unique_choices:
                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_error['left',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        X_t = []
        for state in (statere): 
            data_temp  = Xdata_set[idxf,'error'].copy()
            for coh in unique_cohs:
                for ch in unique_choices:
                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_error['right',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        for icol in range(Nstart,Nend):
            try:
                _,pg=mwu(df_correct['left',idxf][icol],df_correct['right',idxf][icol],alternative='greater')
                _,pl=mwu(df_correct['left',idxf][icol],df_correct['right',idxf][icol],alternative='less')
            except:
                pg,pl = 1,1
            left_mean, right_mean = np.mean(df_correct['left',idxf][icol]),np.mean(df_correct['right',idxf][icol])
            if (len(p_values_correct)==0):
                p_values_correct=np.array([pg, pl, min(pg,pl),left_mean,right_mean])
            else:
                p_values_correct = np.vstack((p_values_correct,np.array([pg, pl, min(pg,pl),left_mean,right_mean])))
                
        for icol in range(Nstart,Nend):
            try:
                _,pg=mwu(df_error['left',idxf][icol],df_error['right',idxf][icol],alternative='greater')
                _,pl=mwu(df_error['left',idxf][icol],df_error['right',idxf][icol],alternative='less')
            except:
                pg,pl = 1,1
            left_mean, right_mean = np.mean(df_error['left',idxf][icol]),np.mean(df_error['right',idxf][icol])
            if (len(p_values_error)==0):
                p_values_error=np.array([pg, pl, min(pg,pl),left_mean,right_mean])
            else:
                p_values_error = np.vstack((p_values_error,np.array([pg, pl, min(pg,pl),left_mean,right_mean])))
        
    
    larger = np.where(p_values_correct[nselect,0]<0.05)[0]
    less = np.where(p_values_correct[nselect,1]<0.05)[0]   
    pop_left_correct    = less.copy()
    pop_right_correct   = larger.copy()
    pop_use_correct     = np.union1d(pop_left_correct, pop_right_correct)
    pop_zero_correct    = np.setdiff1d(np.arange(len(nselect)),pop_use_correct) ##left/right and zero indicate the location in nselect
    
    
    larger = np.where(p_values_error[nselect,0]<0.05)[0]
    less = np.where(p_values_error[nselect,1]<0.05)[0]   
    pop_left_error    = less.copy()
    pop_right_error   = larger.copy()
    pop_use_error     = np.union1d(pop_left_error, pop_right_error)
    pop_zero_error    = np.setdiff1d(np.arange(len(nselect)),pop_use_error)
    print('************ Neuron number **********')
    print(len(nselect))
    return pop_left_correct, pop_right_correct,pop_zero_correct, pop_left_error, pop_right_error, pop_zero_error, NN_error,p_values_correct, p_values_error


def neural_activity_sort_ctxt(Xdata_set,ylabels_set,unique_states,files,idx_delete, nselect):
    Xflat_trials_correct,Xflat_trials_error = {},{}
    
    df_correct, df_error = {},{}  
    p_values_correct,p_values_error = [],[]
    
    ### counting overall neuron number 
    NN_error = 0
    Nstart, Nend = 0,0
    staterep,statealt=[4,5],[6,7]
    staterepe,statealte=[0,1],[2,3]
    
    unique_cohs,unique_choices = [-1,0,1],[0,1]
    for idxf in range(len(files)):
        if(idxf in idx_delete):
            continue
        Nstart = Nend
        X_t = []
        for state in staterep:   
            data_temp  = Xdata_set[idxf,'correct'].copy()
            
            for coh in unique_cohs:
                for ch in unique_choices:
                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        Nend +=np.shape(data_temp[state,coh,ch])[1]
                        NN_error +=np.shape(data_temp[state,coh,ch])[1]
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_correct['rep',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        X_t = []
        for state in statealt: 
            data_temp  = Xdata_set[idxf,'correct'].copy()
            
            for coh in unique_cohs:
                for ch in unique_choices:

                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_correct['alt',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        
        
        X_t = []
        for state in (staterepe):   
            data_temp  = Xdata_set[idxf,'error'].copy()
            for coh in unique_cohs:
                for ch in unique_choices:
                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_error['rep',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        X_t = []
        for state in (statealte): 
            data_temp  = Xdata_set[idxf,'error'].copy()
            for coh in unique_cohs:
                for ch in unique_choices:
                    try:
                        totaltrials=np.shape(data_temp[state,coh,ch])[0]
                    except:
                        print('exist for initial finding')
                        continue
                    if totaltrials<1:
                        continue
                    #### generate sampled true trials for individual neurons in each pseudo trial
                    NN, ntrials = np.shape(data_temp[state,coh,ch])[1], np.shape(data_temp[state,coh,ch])[0]
                    if(len(X_t)==0):
                        X_t = data_temp[state,coh,ch].copy()
                    else:
                        X_t = np.vstack((X_t,data_temp[state,coh,ch].copy()))
                
        df_error['alt',idxf] = pd.DataFrame(X_t,columns = np.arange(Nstart,Nend))
        
        for icol in range(Nstart,Nend):
            try:
                _,pg=mwu(df_correct['rep',idxf][icol],df_correct['alt',idxf][icol],alternative='greater')
                _,pl=mwu(df_correct['rep',idxf][icol],df_correct['alt',idxf][icol],alternative='less')
            except:
                pg,pl = 1,1
            rep_mean, alt_mean = np.mean(df_correct['rep',idxf][icol]),np.mean(df_correct['alt',idxf][icol])
            if (len(p_values_correct)==0):
                p_values_correct=np.array([pg, pl, min(pg,pl),rep_mean,alt_mean])
            else:
                p_values_correct = np.vstack((p_values_correct,np.array([pg, pl, min(pg,pl),rep_mean,alt_mean])))
                
        for icol in range(Nstart,Nend):
            try:
                _,pg=mwu(df_error['rep',idxf][icol],df_error['alt',idxf][icol],alternative='greater')
                _,pl=mwu(df_error['rep',idxf][icol],df_error['alt',idxf][icol],alternative='less')
            except:
                pg,pl = 1,1
            rep_mean, alt_mean = np.mean(df_error['rep',idxf][icol]),np.mean(df_error['alt',idxf][icol])
            if (len(p_values_error)==0):
                p_values_error=np.array([pg, pl, min(pg,pl),rep_mean,alt_mean])
            else:
                p_values_error = np.vstack((p_values_error,np.array([pg, pl, min(pg,pl),rep_mean,alt_mean])))
        
    
    larger = np.where(p_values_correct[nselect,0]<0.05)[0]
    less = np.where(p_values_correct[nselect,1]<0.05)[0]   
    pop_rep_correct     = less.copy()
    pop_alt_correct     = larger.copy()
    pop_use_correct     = np.union1d(pop_rep_correct, pop_alt_correct)
    pop_b_correct       = np.setdiff1d(np.arange(len(nselect)),pop_use_correct)
    
    
    larger = np.where(p_values_error[nselect,0]<0.05)[0]
    less = np.where(p_values_error[nselect,1]<0.05)[0]   
    pop_rep_error     = less.copy()
    pop_alt_error     = larger.copy()
    pop_use_error     = np.union1d(pop_rep_error, pop_alt_error)
    pop_b_error       = np.setdiff1d(np.arange(len(nselect)),pop_use_error)
    print('************ Neuron number **********')
    print(len(nselect))
    return pop_rep_correct, pop_alt_correct,pop_b_correct, pop_rep_error, pop_alt_error, pop_b_error, NN_error,p_values_correct, p_values_error   
        
  
# --- MAIN
if __name__ == '__main__':

    PREV_CH = 'L'
    NUM_SAMPLES = 500  # 200
    THRESH_TRIAL = 2

    PLOT_ALL_TRIALS_3D = False
    S_PLOTS = 5
    BOX_WDTH = 0.25
    SVMAXIS = 3
    AX_PREV_CH_OUTC = {'c': [2, 3], 'e': [0, 1]}
    NITERATIONS, NPSEUDODEC, PERCENTTRAIN = 50,50, 0.5
    NPSEUDODEC_BEH = 20
    ACE_RATIO = 0.5 #default
    '''
    training: 30 per state, 30*4*2=240 trials per train
    testing:  20 per state, 20*4*2=160 trials per test
    '''
    RUN_ALL   = True
    RERUN     = True
    DOREVERSE = 0 
    PCA_n_components = 0
    USE_POP = 0
 
    RECORD_TRIALS = 1
    CONTROL       = 0#normal#1#ae trained#2#ac trained


    BOTTOM_3D = -3  # where to plot blue/red projected dots in 3D figure
    XLIMS_2D = [-3, 3]
    YLIMS_2D = [-3, 3]
    YTICKS_2D = [-3., 0., 3.]
    XTICKS_2D = [-2., 0., 2.]
    CTXT_BIN = np.linspace(0, 1.65, 7)  # (0,1.8,7)
    XLIM_CTXT = [12000, 13000]
    YTICKS_CTXT = [-2, 0, 2]
    YLIM_CTXT = [-2.2, 2.2]
    # BOTTOM_3D = -6  # where to plot blue/red projected dots in 3D figure
    # XLIMS_2D = [-3, 3]
    # YLIMS_2D = [-7, 7]
    # YTICKS_2D = [-6., 0., 6.]
    # XTICKS_2D = [-2., 0., 2.]
    # CTXT_BIN = np.linspace(0, 1.65, 7)  # (0,1.8,7)
    # XLIM_CTXT = [12000, 13000]
    # YTICKS_CTXT = [-2, 0, 2]
    # YLIM_CTXT = [-2.2, 2.2]

    dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys/'#'-00s/'#'files_pop_analysis/'
    STIM_PERIOD = 0
    STIM_BEH    = 1 # 0 psychometric curve for upcoming stimuli/ 1 psychometric curve for choices
    PCA_only    = 0

    # dir = '/home/molano/DMS_electro/DataEphys/pre_processed/'
    # 'files_pop_analysis/'
    rats =['Rat32']#,'Rat15','Rat31','Rat32'] #,'Rat15'['Patxi',  'Rat31', 'Rat15', 'Rat7',
    for r in rats:
        # plt.close('all')
        print('xxxxxxxxxxxxxxxx')
        print(r)
        IDX_RAT = r+'_'
        # files = glob.glob(dir+IDX_RAT+'202*.npz')
        # dir = 'D://Yuxiu/Code/Data/Auditory/NeuralData/Rat7/Rat7/'
        files = glob.glob(dir+IDX_RAT+'ss_*.npz')

        # # GET QUANTITIES
        # print('Get quantities')
        
        # # Whether to skip the NaN at the beginning or not
        # SKIPNAN = 0
        # data_tr = get_all_quantities(files, numtrans=0, SKIPNAN=SKIPNAN) 
        # # dataname = dir+IDX_RAT+'data_trials.npz'
        # # np.savez(dataname, **data_tr)
        
        '''
        Sept 29
        '''
        ## reloading
        dataname    = dir+IDX_RAT+'data_trials.npz'#'_Sept.npz'
        data_trials = np.load(dataname, allow_pickle=True)
        data_tr     = unpack_dicts(data_trials) 
        

        ACE_RATIO   = data_tr['overall_ac_ae_ratio']
        
        unique_states = np.arange(8)
        unique_cohs   = [-1, 0, 1]# [-0.6,-0.25,0,0.25,0.6]#
  
        unique_states = np.arange(8)
        # unique_cohs = [-1, 0, 1]
        false_files, MIN_TRIALS, num_beh_trials,coh_ch_stateratio_correct, coh_ch_stateratio_error = filter_sessions(data_tr, unique_states, unique_cohs)
        # print(">>>>>>>>>>>>>>> Minimum Trials per state/beh_state:", MIN_TRIALS)
        
        
        RECOMPUTE_POP = 0
        ##### recompute subpopulations 
        if (RECOMPUTE_POP):
            unique_states = np.arange(8)
            single_params,single_p_values,nnonselect,nselect= single_neuron_selectivity(data_tr['Xdata_set'],data_tr['ylabels_set'],unique_states,files,false_files)
            pop_correct, pop_error, pop_zero, NNt, p_values = neural_activity_sort(data_tr['Xdata_set'],data_tr['ylabels_set'],unique_states,files,false_files)
            pop_left_correct, pop_right_correct,pop_zero_correct, pop_left_error, pop_right_error, pop_zero_error, NN_error,p_values_correct, p_values_error = neural_activity_sort_prevch(data_tr['Xdata_set'],data_tr['ylabels_set'],unique_states,files,false_files,nselect)   
            print('>>>>>> prevch >>>>>>')
            print('******* three populations: left, zero, right (AC) ***********')
            print(len(pop_left_correct),'  ',len(pop_zero_correct),'  ',len(pop_right_correct))
            print('******* three populations: left, zero, right (Ae) ***********')
            print(len(pop_left_error),'  ',len(pop_zero_error),'  ',len(pop_right_error))
            
            pop_rep_correct, pop_alt_correct,pop_b_correct, pop_rep_error, pop_alt_error, pop_b_error, NN_error,p_values_correct, p_values_error = neural_activity_sort_ctxt(data_tr['Xdata_set'],data_tr['ylabels_set'],unique_states,files,false_files,nselect)
            print('>>>>>> ctx >>>>>>')
            print('******* three populations: rep, zero, alt (AC) ***********')
            print(len(pop_rep_correct),'  ',len(pop_b_correct),'  ',len(pop_alt_correct))
            print('******* three populations: rep, zero, alt (Ae) ***********')
            print(len(pop_rep_error),'  ',len(pop_b_error),'  ',len(pop_alt_error))
            
                    
            # pop_correct = pop_rep_correct
            # pop_error   = pop_alt_correct
            # pop_zero    = pop_b_correct
            
            # pop_correct = pop_left_correct
            # pop_error   = pop_right_correct
            # pop_zero    = pop_zero_correct#np.setdiff1d(nselect,pop_correct)
            
            
            #### neuron selectivity --- mixed, subpopulations
            lst = [single_params,single_p_values,nnonselect,nselect,
                    pop_left_correct, pop_right_correct,pop_zero_correct, 
                    pop_left_error, pop_right_error, pop_zero_error,
                    pop_rep_correct, pop_alt_correct,pop_b_correct, 
                    pop_rep_error, pop_alt_error, pop_b_error, NN_error]
            stg = ["single_params,single_p_values,nnonselect,nselect, "
                    "pop_left_correct, pop_right_correct,pop_zero_correct, "
                    "pop_left_error, pop_right_error, pop_zero_error,"
                    "pop_rep_correct, pop_alt_correct,pop_b_correct, "
                    "pop_rep_error, pop_alt_error, pop_b_error, NN_error "]
            d_selectivity = list_to_dict(lst=lst, string=stg)
            dataname = dir+IDX_RAT+'neuron_selectivity_Sept.npz'
            np.savez(dataname, **d_selectivity)
        else:
            dataname  = dir+IDX_RAT+'neuron_selectivity_Sept.npz'
            d_selectivity  = np.load(dataname, allow_pickle=True)             
            
            nselect, nnonselect, pop_left_correct, pop_right_correct, single_pop_correct, correct_zero, pop_left_error, pop_right_error, single_pop_error, error_zero =guc.mixed_selectivity_pop(d_selectivity)
            
            
            
        
        
        '''
        1 Sept
        '''
        wc, bc = [], []
        if(RECORD_TRIALS == 0):
            dataname = dir+IDX_RAT+'data_dec.npz'
            data_dec = np.load(dataname, allow_pickle=True)
            REC_TRIALS_SET = data_dec['REC_TRIALS_SET']
            REC_TRIALS_SET = REC_TRIALS_SET.item()
            wc, bc = data_dec['coefs_correct'], data_dec['intercepts_correct']
        else:
            REC_TRIALS_SET = np.zeros(NITERATIONS)
            
            

        if(PCA_only ==1):
            mmodel = []
        elif(PCA_n_components>0):
            data_name = dir+IDX_RAT+'pca_mmodel.pkl'#'data_dec.npz'
            mmodel = pk.load(open(data_name,'rb'))
        else:
            mmodel = []
            
        
        # ------------- populations -- obtained using activity in ae trials--------------
        pop_correct = pop_left_error
        pop_error   = pop_right_error
        pop_zero    = error_zero
                     
        ## 11 July
        CONTROL =1
        data_dec, mmodel, Xhist_test_correct, Xhist_test_error = get_dec_axes(data_tr, wc, bc, [], [], nselect,false_files,pop_correct, pop_zero, pop_error, single_pop_error, USE_POP,
                                mode='decoding', DOREVERSE=0,
                                CONTROL=CONTROL, STIM_PERIOD=STIM_PERIOD, RECORD_TRIALS=1,
                                REC_TRIALS_SET=np.zeros(NITERATIONS), PCA_only=PCA_only, mmodel=mmodel)
        wc, bc = data_dec['coefs_correct'], data_dec['intercepts_correct']

            
        if(RECORD_TRIALS == 1 and CONTROL==1):
            dataname = dir+IDX_RAT+'data_dec_ae_cond_.npz'
            np.savez(dataname, **data_dec)
        
        print('Get AUCs (and d-primes)')
        data_flt  = flatten_data(data_tr, data_dec)

        if(RECORD_TRIALS == 1 and CONTROL==1):
            dataname = dir+IDX_RAT+'data_flt_ae_overall_cond_.npz'
            np.savez(dataname, **data_flt)
        
        # # # # print('3D projections ac and ae')
        # # # # projection_3D(data_flt, data_flt, 'c')
        # # # projection_3D(data_flt, data_flt, 'e')
        
        print('2D projections')
        # projections_2D(data_flt, prev_outc='c', fit=False, name='')
        projections_2D(data_flt, prev_outc='e', fit=False, name='')
        
        
        
        # if(PCA_only ==1):
        #     data_name = dir+IDX_RAT +'pca_mmodel.pkl'
        #     pk.dump(mmodel, open(data_name,"wb"))
        #     continue

        
        print('Get AUCs (and d-primes, mixed)')
        # data_flt         = flatten_data_supp(data_tr, data_dec, pop_type='mixed')       
        data_flt         = flatten_data_cond_supp(data_tr, data_dec, pop_type='mixed') ### for conditional decoders
          
        if(RECORD_TRIALS == 1 and CONTROL==1):
            dataname = dir+IDX_RAT+'data_flt_ae_mixed_cond_.npz'
            np.savez(dataname, **data_flt)
        

        # print('3D projections ac and ae (overall)')
        # projection_3D(data_flt, data_flt, 'e')
        print('2D projections')
        projections_2D(data_flt, prev_outc='e', fit=False, name='')
        
        print('Get AUCs (and d-primes, single_pop)')
        # data_flt         = flatten_data_supp(data_tr, data_dec, pop_type='s_pop')
        data_flt         = flatten_data_cond_supp(data_tr, data_dec, pop_type='s_pop')
          
        if(RECORD_TRIALS == 1 and CONTROL==1):
            dataname = dir+IDX_RAT+'data_flt_ae_single_cond_.npz'
            np.savez(dataname, **data_flt)   

        # print('3D projections ac and ae (single_pop)')
        # projection_3D(data_flt, data_flt, 'c')
        print('2D projections')
        projections_2D(data_flt, prev_outc='e', fit=False, name='')
        
        
        # -------------- change populations -- using ac trials ---------------- 
        pop_correct = pop_left_correct
        pop_error   = pop_right_correct
        pop_zero    = correct_zero#pop_zero_correct#np.setdiff1d(nselect,pop_correct)
            
        CONTROL =2
        if(CONTROL > 0):#==1):
            data_dec, mmodel, Xhist_test_correct, Xhist_test_error = get_dec_axes(data_tr, wc, bc, [], [], nselect, false_files,pop_correct, pop_zero, pop_error, single_pop_correct, USE_POP,
                                    mode='decoding', DOREVERSE=0,
                                    CONTROL=CONTROL, STIM_PERIOD=STIM_PERIOD, RECORD_TRIALS=1,
                                    REC_TRIALS_SET=np.zeros(NITERATIONS), PCA_only=PCA_only, mmodel=mmodel)
            wc, bc = data_dec['coefs_correct'], data_dec['intercepts_correct']
        else:
            data_dec,mmodel, Xhist_test_correct, Xhist_test_error = get_dec_axes(data_tr, wc, bc, [], [], nselect, false_files,pop_correct, pop_zero, pop_error, single_pop_correct, USE_POP,
                                    mode='decoding', DOREVERSE=0,
                                    CONTROL=CONTROL, STIM_PERIOD=STIM_PERIOD, RECORD_TRIALS=RECORD_TRIALS,
                                    REC_TRIALS_SET=REC_TRIALS_SET, PCA_only=PCA_only, mmodel=mmodel)

        
        if(RECORD_TRIALS == 1 and CONTROL==2):
            dataname = dir+IDX_RAT+'data_dec_ac_cond_.npz'
            np.savez(dataname, **data_dec)
        
            
        # if(PCA_only ==1):
        #     data_name = dir+IDX_RAT +'pca_mmodel.pkl'
        #     pk.dump(mmodel, open(data_name,"wb"))
        #     continue
            

        print('Get AUCs (and d-primes)')
        data_flt         = flatten_data(data_tr, data_dec)
        
        
        if(RECORD_TRIALS == 1 and CONTROL==2):
            dataname = dir+IDX_RAT+'data_flt_ac_overall_cond_.npz'
            np.savez(dataname, **data_flt)
        

        # # # print('3D projections ac and ae (mixed-selectivity)')
        # # # projection_3D(data_flt, data_flt, 'c')
        # # # # projection_3D(data_flt, data_flt, 'e')

        print('2D projections')
        projections_2D(data_flt, prev_outc='c', fit=False, name='')
        # projections_2D(data_flt, prev_outc='e', fit=False, name='')
        
        print('Get AUCs (and d-primes, mixed)')
        # data_flt         = flatten_data_cond_supp(data_tr, data_dec, pop_type='mixed')       
        data_flt         = flatten_data_cond_supp(data_tr, data_dec, pop_type='mixed') ### for conditional decoders
          
        if(RECORD_TRIALS == 1 and CONTROL==2):
            dataname = dir+IDX_RAT+'data_flt_ac_mixed_cond_.npz'
            np.savez(dataname, **data_flt)
        

        # print('3D projections ac and ae (overall)')
        # projection_3D(data_flt, data_flt, 'c')
        print('2D projections')
        projections_2D(data_flt, prev_outc='c', fit=False, name='')
        
        print('Get AUCs (and d-primes, single_pop)')
        # data_flt         = flatten_data_cond_supp(data_tr, data_dec, pop_type='s_pop')
        data_flt         = flatten_data_cond_supp(data_tr, data_dec, pop_type='s_pop')
          
        if(RECORD_TRIALS == 1 and CONTROL==2):
            dataname = dir+IDX_RAT+'data_flt_ac_single_cond_.npz'
            np.savez(dataname, **data_flt)   

        # # # print('3D projections ac and ae (single_pop)')
        # # # projection_3D(data_flt, data_flt, 'c')
        print('2D projections')
        projections_2D(data_flt, prev_outc='c', fit=False, name='')
        
        # #####------------------------------------------
        
        
        # # # EACHSTATES=50
        # # # fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True, tight_layout=True)
        # # # curveslopes_correct, curveintercept_correct, curveslopes_error,\
        # # #     curveintercept_error, data_beh =\
        # # #     bias_VS_prob(data_tr, data_dec, unique_cohs, nselect, num_beh_trials,EACHSTATES,
        # # #                   NITERATIONS, ax, RECORD_TRIALS=RECORD_TRIALS,
        # # #                   REC_TRIALS_SET=REC_TRIALS_SET,STIM_BEH=STIM_BEH,PCA_only=PCA_only,mmodel=mmodel)
            
            
        # #####***************** The Second Stage ------ Behaviour *****************
        # # 1) calculate the axes
        # # 2) plot the psychometric curve 
        
        ### transition bias decoder 
        CONTROL=1
        pop_correct = pop_left_error
        pop_error   = pop_right_error
        pop_zero    = pop_right_error
        data_int_ae,Xbeh_test_correct_rep, Xbeh_test_error_rep, Xbeh_test_correct_alt, Xbeh_test_error_alt  = hist_integration_balanced(data_tr,[], [], [], [], nselect, false_files, coh_ch_stateratio_correct, coh_ch_stateratio_error,  pop_correct, pop_error, USE_POP = USE_POP,mode='decoding',DOREVERSE=0,CONTROL=CONTROL, STIM_PERIOD=STIM_PERIOD, RECORD_TRIALS=1,REC_TRIALS_SET=np.zeros(NITERATIONS),PCA_only=PCA_only, mmodel=mmodel)
        
        if(RECORD_TRIALS == 1 and CONTROL==1):
            dataname = dir+IDX_RAT+'data_beh_ae_cond_.npz'
            np.savez(dataname, **data_int_ae)
            
        CONTROL=2
        pop_correct = pop_left_correct
        pop_error   = pop_right_correct
        pop_zero    = correct_zero#pop_zero_correct#np.setdiff1d(nselect,pop_correct)
        data_int_ac,Xbeh_test_correct_rep, Xbeh_test_error_rep, Xbeh_test_correct_alt, Xbeh_test_error_alt = hist_integration_balanced(data_tr,[], [], [], [],nselect, false_files, coh_ch_stateratio_correct, coh_ch_stateratio_error,  pop_correct, pop_error, USE_POP = USE_POP,mode='decoding',DOREVERSE=0,CONTROL=CONTROL, STIM_PERIOD=STIM_PERIOD, RECORD_TRIALS=1,REC_TRIALS_SET=np.zeros(NITERATIONS),PCA_only=PCA_only, mmodel=mmodel)  
        if(RECORD_TRIALS == 1  and CONTROL==2):
            dataname = dir+IDX_RAT+'data_beh_ac_cond_.npz'
            np.savez(dataname, **data_int_ac)
        
        
        # # print('Pearson Correlation: Transition bias v.s. context')
        # # # x-axis bin ctxt, y-axis transition bias
        # # corrl_ac, corrr_ac, corrl_ae, corrr_ae, ctx_tb_trcs = ctxtbin_defect(data_flt)
        # # print('>>>>>>>>>>> P-correlation, left AC: ', corrl_ac, ' right AC: ',  corrr_ac, ' left AE: ', corrl_ae, ' right AE: ', corrr_ae)
        

