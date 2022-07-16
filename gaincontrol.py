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


def gaincontrol_test(data_tr,data_int, false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error,pop_correct,pop_zero,pop_error, mode='decoding',
                 DOREVERSE=0, CONTROL=0, STIM_PERIOD=1, RECORD_TRIALS=1, REC_TRIALS_SET=[],PCA_only=0,mmodel=[]):
    Xdata_set, ylabels_set, files = data_tr['Xdata_set'], \
        data_tr['ylabels_set'], data_tr['files']

    unique_states = np.arange(8)
    unique_cohs   = [0]

    Xmerge_trials_correct,ymerge_labels_correct,Xmerge_trials_error,ymerge_labels_error,merge_trials=gpt.merge_pseudo_beh_trials_balanced(Xdata_set,ylabels_set,unique_states,unique_cohs,files, false_files, coh_ch_stateratio_correct,coh_ch_stateratio_error,EACHSTATES=30, RECORD_TRIALS=RECORD_TRIALS, RECORDED_TRIALS_SET=REC_TRIALS_SET[0],STIM_BEH=1)

    wc, bc = data_int['coefs_correct'],data_int['intercepts_correct']

    subpop_beh_proj(wc,bc, Xmerge_trials_correct,ymerge_labels_correct,pop_correct,pop_zero, pop_error,unique_cohs,mmodel=mmodel, PCA_n_components=0)

def subpop_beh_proj(coeffs_beh_pool, intercepts_beh_pool, Xmerge_trials, ymerge_labels, pop_correct, pop_zero, pop_error, unique_cohs,mmodel=[],PCA_n_components=0): 
    idxdecoder = np.random.choice(np.arange(20),size=1) 
    unique_cohs= [0]  
    for idxcoh, coh in enumerate(unique_cohs): 
        ### finding left and right behaviour 
        left_trials  = np.where(ymerge_labels[coh][0,:]==0)[0]
        # print('~~~~~~lefttrials',left_trials)
        right_trials = np.where(ymerge_labels[coh][0,:]==1)[0] 
        linw_cc, linb_cc = np.mean(coeffs_beh_pool[:, 4::5],axis=1), np.mean(intercepts_beh_pool[0,4::5])
        correct_axes, error_axes = linw_cc[pop_correct],linw_cc[pop_error]
        zero_axes = linw_cc[pop_zero]
        left_trials_overall = Xmerge_trials[coh][left_trials,:] 
        right_trials_overall = Xmerge_trials[coh][right_trials,:] 
        left_trials     = left_trials_overall[:,:] 
        right_trials     = right_trials_overall[:,:] 
        left_trials_ep      = left_trials_overall[:,pop_error] 
        right_trials_ep     = right_trials_overall[:,pop_error] 
        left_proj  = left_trials @linw_cc + linb_cc  
        right_proj = right_trials@linw_cc + linb_cc  
        
        left_trials_cp      = left_trials_overall[:,pop_correct] 
        right_trials_cp     = right_trials_overall[:,pop_correct] 
        left_cp_proj  = left_trials_cp @correct_axes + linb_cc  
        right_cp_proj = right_trials_cp@correct_axes + linb_cc  
        
        left_trials_zp      = left_trials_overall[:,pop_zero] 
        right_trials_zp     = right_trials_overall[:,pop_zero] 
        left_zp_proj  = left_trials_zp @zero_axes + linb_cc  
        right_zp_proj = right_trials_zp@zero_axes + linb_cc  
        
        left_ep_proj  = left_trials_ep @error_axes + linb_cc  
        right_ep_proj = right_trials_ep@error_axes + linb_cc  
        
        
    fig,ax = plt.subplots(1,8,figsize=(12,5),sharey=True) 
    ax[0].boxplot(left_cp_proj)
    ax[1].boxplot(right_cp_proj)
    
    ax[2].boxplot(left_zp_proj)
    ax[3].boxplot(right_zp_proj)
    
    ax[2+2].boxplot(left_ep_proj)
    ax[3+2].boxplot(right_ep_proj)
    
    ax[4+2].boxplot(left_proj)
    ax[5+2].boxplot(right_proj)

		

