# Load packages; 
import time
import numpy as np
import matplotlib.pyplot as plt

# Import libraries
from collections import Counter
from mpl_toolkits import mplot3d

from sklearn import datasets, svm, pipeline 
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.decomposition import PCA 
import seaborn as sns
import pandas as pd
import os

from sklearn.utils import resample # for Bootstrap sampling
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.linear_model import LinearRegression

# self-defined functions
import general_util_ctxtgt as guc
import generate_pseud_trails as gpt
import required_data_dec as rdd
import bootstrap_linear as bl


image_format = 'svg' # e.g .png, .svg, etc.
dpii=300

# %%
import scipy.io as sio
import get_matlab_data as gd
import glob


### loading data from one Rat
dir = ## data folder ##'/Users/yuxiushao/Public/DataML/Auditory/DataEphys/'
files = glob.glob(dir+'Rat32_ss_*.npz')#Rat7_ss_45_data_for_python.mat

# def get_all_quantities(files, numtrans=0):

icount = 0
# data clustering by state and stimulus coherence
Xdata_set = {}
ylabels_set = {}
# data clustering by state (history only)
Xdata_hist_set = {}
ylabels_hist_set = {}

metadata = {}
pseudo_neurons = 0
remarkfile=""
for i in range(len(files)):
    for T in ['correct','error']:
        Xdata_set[i,T]   = {}
        ylabels_set[i,T] = {}
        
        Xdata_hist_set[i,T]   = {}
        ylabels_hist_set[i,T] = {}

# data preprocessing, clustering trials
nnfiles = np.zeros(len(files))
for idxs, f in enumerate(files):
    if icount<0:
        break
    data = np.load(f,allow_pickle=True)
    tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc = get_RNNdata_ctxtgt(data)
    # print('responses:',np.shape(data['states']),'; gt',np.shape(data['gt']))
    if(np.shape(data['states'])[0]!=np.shape(data['gt'])[0]):
        remarkfile=remarkfile+"; "+f
        continue
    stim_trials, idx_effect, ctxt_trials = transform_stim_trials_ctxtgt(data)
    icount+=1
    
    Xdata, ydata, Xdata_idx, Xconds_2, Xacts_1,\
        Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0, Xgts_0,\
        Xcohs_0, Xdata_trialidx, Xstates = rdd.req_quantities_0(stim_trials, stm, dyns, gt, choice,eff_choice, rw, obsc, BLOCK_CTXT=1)
    
        
    Xdata_correct,Xdata_error,correct_trial, error_trial,rses_correct, rses_error, \
    Xrse_6_correct, Xrse_6_error, Xcohs_0_correct,\
    Xcohs_0_error, ydata_bias_correct, ydata_bias_error, ydata_xor_correct,\
    ydata_xor_error, ydata_conds_correct, ydata_conds_error,\
    ydata_choices_correct, ydata_choices_error, ydata_cchoices_correct,\
    ydata_cchoices_error, ydata_cgts_correct, ydata_cgts_error,\
    Xdata_idx_correct, Xdata_idx_error,\
    Xdata_trialidx_correct, Xdata_trialidx_error, ydata_states_correct,ydata_states_error= rdd.sep_correct_error(data['stimulus'], dyns, Xdata, ydata, Xdata_idx,Xconds_2, Xacts_1, Xrws_1, Xlfs_1, Xrse_6, rses,Xacts_0, Xgts_0, Xcohs_0, Xdata_trialidx, margin=[1, 2], idd=1)
    # print('file',f,' with validate trials:',np.shape(Xdata_correct))
    
    ylabels_correct = rdd.set_ylabels(Xdata_correct,ydata_choices_correct,ydata_conds_correct,ydata_xor_correct,ydata_bias_correct,ydata_cchoices_correct,Xcohs_0_correct)
    ylabels_error   = rdd.set_ylabels(Xdata_error,ydata_choices_error,ydata_conds_error,ydata_xor_error,ydata_bias_error,ydata_cchoices_error,Xcohs_0_error)
    
    pseudo_neurons +=np.shape(Xdata_correct)[1]
    
    Xdata_set[idxs,'correct'],ylabels_set[idxs,'correct'],Xdata_hist_set[idxs,'correct'],ylabels_hist_set[idxs,'correct']= State_trials(Xdata_correct,ydata_states_correct,ydata_cchoices_correct,Xcohs_0_correct,ylabels_correct,0,)
    Xdata_set[idxs,'error'],ylabels_set[idxs,'error'],Xdata_hist_set[idxs,'error'],ylabels_hist_set[idxs,'error']= State_trials(Xdata_error,ydata_states_error,ydata_cchoices_error,Xcohs_0_error,ylabels_error,0,)
    metadata[idxs]={'filename':f,
                    'totaltrials':np.shape(Xdata_correct)[0]+np.shape(Xdata_error)[0],
                    'neuronnumber':np.shape(Xdata_correct)[1],
                    'ACtrials':np.shape(Xdata_correct)[0],
                    'AEtrials':np.shape(Xdata_error)[0],       
    }
# return d

'''
Filtering 'good' sessions
'''
unique_states = np.arange(8)
unique_cohs   = [-1,0,1]
# def filter_sessions(Xdata_set,ylabels_set,Xdata_hist_set,ylabels_hist_set,files,unique_states,unique_cohs):
correct_false,error_false = gpt.valid_hist_trials(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files)

_correct_false,_error_false = gpt.valid_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files)

false_files = np.union1d(correct_false,error_false)
false_files = np.union1d(false_files,_error_false)
false_files = np.union1d(false_files,_correct_false)
print('skip files:',false_files)
# return false_files


'''
merging and generating pseudo trials for history info. encoding (stage 1)
'''
NITERATIONS, NPSEUDODEC, PERCENTTRAIN = 100,50,0.6 # NPSEUDODEC is the number of pseudo trials generated corresponding to each state (1/8)

# def get_dec_axes(data_tr, wc, bc, we, be, mode='decoding', DOREVERSE=0)
### to find out the total number of neurons validated
unique_states = np.arange(8)
unique_cohs   = np.sort(np.unique(Xcohs_0_correct))
Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error=gpt.merge_pseudo_hist_trials(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,10)

## finding decoding axis
NN=np.shape(Xmerge_hist_trials_correct[4])[1]
coeffs, intercepts,\
        Xsup_vec_act, Xsup_vec_ctxt, Xsup_vec_bias, Xsup_vec_cc,\
        Xtest_set_correct, ytest_set_correct, yevi_set_correct,\
        Xtest_set_error, ytest_set_error, yevi_set_error=bl.bootstrap_linsvm_step(Xdata_hist_set,NN, ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, n_iterations=NITERATIONS, N_pseudo_dec=NPSEUDODEC, train_percent=PERCENTTRAIN)

# return d

# def flatten_data(data_tr, data_dec):
IPOOLS = NITERATIONS
# flatten data --- correct
nlabels = np.shape(np.squeeze(ytest_set_correct[0,:,:]))[1]
# ytruthlabels_c, ycact_c = np.zeros((3 + 1 + 1, 1)), np.zeros((1, 1))
ytruthlabels_c, ycact_c = np.zeros((nlabels, 1)), np.zeros((1, 1))
ytruthlabels_c_ = np.zeros((1, 1))
yevi_c = np.zeros((3 + 1 + 1, 1))
dprimes_c = np.zeros(IPOOLS)

for i in range(IPOOLS):  # bootstrapping
    hist_evi = yevi_set_correct[i, :, :]
    # labels: preaction, ctxt, bias
    test_labels = ytest_set_correct[i, :, :]
    # test_set records the index in test_poolc
    idx = np.arange(np.shape(hist_evi)[0])

    ytruthlabels_c = np.append(
        ytruthlabels_c, test_labels[idx, :].T, axis=1)
    yevi_c = np.append(
        yevi_c, (yevi_set_correct[i, idx, :]).T, axis=1)  # np.squeeze
ytruthlabels_c, ycact_c, yevi_c =\
    ytruthlabels_c[:, 1:], ycact_c[:, 1:], yevi_c[:, 1:]

nlabels = np.shape(np.squeeze(ytest_set_error[0,:,:]))[1]    
# ytruthlabels_e, ycact_e = np.zeros((3 + 1 + 1, 1)), np.zeros((1, 1))
ytruthlabels_e, ycact_e = np.zeros((nlabels, 1)), np.zeros((1, 1))
ytruthlabels_e_ = np.zeros((1, 1))
yevi_e = np.zeros((3 + 1 + 1, 1))
dprimes_e = np.zeros(IPOOLS)
for i in range(IPOOLS):
    hist_evi = yevi_set_error[i, :, :]
    test_labels = ytest_set_error[i, :, :]  # labels: preaction, ctxt, bias
    idx = np.arange(np.shape(hist_evi)[0])
    ytruthlabels_e = np.append(
        ytruthlabels_e, test_labels[idx, :].T, axis=1)
    yevi_e = np.append(
        yevi_e, (yevi_set_error[i, idx, :]).T, axis=1)  # np.squeeze

ytruthlabels_e, ycact_e, yevi_e =\
    ytruthlabels_e[:, 1:], ycact_e[:, 1:], yevi_e[:, 1:]
# return d

## visualizing the results
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


'''
merging and generating pseudo trials for how tr. bias affects behaviour (stage 2)
'''
unique_cohs   = [-1,0,1]
EACHSTATES    = 60
Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error=gpt.merge_pseudo_beh_trials(Xdata_set,ylabels_set,unique_states,unique_cohs,files,false_files,metadata,EACHSTATES)
unique_states = np.arange(4,8,1)
_=gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_correct,
                                     ymerge_labels_correct, [4,5,6,7],[-1,0,1],[0,1], EACHSTATES=EACHSTATES)
unique_states = np.arange(4)
_=gpt.behaviour_trbias_proj(coeffs, intercepts, Xmerge_trials_error,
                                     ymerge_labels_error, [0,1,2,3],[-1,0,1],[0,1], EACHSTATES=EACHSTATES)