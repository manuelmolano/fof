'''
@YX 30 OCT 
Use the contextual information from block-level ground-truth.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from statannot import add_stat_annotation
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def get_RNNdata_ctxtgt(data, *kwargs):
    ctx = data['contexts']
    gt  = data['gt']
    stm = data['stimulus']
    dyns = data['states']
    choice = data['choice']
    eff_choice = data['prev_choice']
    rw = data['reward']
    obsc = data['obscategory']
    # design time clock
    nt = len(dyns[:, 0])
    tt = np.arange(1, nt, 1)
    return tt, stm, dyns, ctx, gt, choice, eff_choice, rw, obsc

def transform_stim_trials_ctxtgt(data, *kwargs):
    ctxseq = []
    ctx  = data['contexts']
    gt   = data['gt']
    dyns = data['states']
    for i in range(len(gt)):
        if(ctx[i]=='2'):
            ctxseq.append(1)
        else:
            ctxseq.append(0)
    ctxseq = np.array(ctxseq)

    ngt_tot = np.where(gt>0)[0]
    stim_trials = {}
    ctxt_trials = []

    trial_skip, idx_skip,idx_effect = [],[],[]
    for idx,igt in enumerate(ngt_tot[:len(ngt_tot)-2]):
        if (igt+1>=np.shape(data['states'])[0]):
            break
        ### check if there is any NaN in response data
        resps_check = dyns[igt-1,:]
        array_has_nan = np.isnan(np.sum(resps_check))
        if(array_has_nan==True):
            print('index:',igt)
            trial_skip = np.append(trial_skip,idx)
            idx_skip   = np.append(idx_skip,igt+1)## response index
        else:
            idx_effect = np.append(idx_effect,int(igt+1))
        ### generate stimulus trials 
        stim_trials[idx]={'stim_coh':[data['obscategory'][igt+1]],
                        'ctx':ctxseq[igt+1],
                        'gt':data['gt'][ngt_tot[idx+1]],
                        'resp':np.reshape(data['states'][igt+1],(1,-1)),
                        'choice':data['choice'][ngt_tot[idx+1]],
                        'rw':data['reward'][ngt_tot[idx+1]],
                        'start_end': np.array([ngt_tot[idx+1]-1, ngt_tot[idx+1]]),
                        'skip':array_has_nan,
                        }
        ctxt_trials = np.append(ctxt_trials,stim_trials[idx]['ctx'])
        
    # action (-1 and +1) -- left and right
    for i in range(len(stim_trials)):
        if (stim_trials[i]['choice'] == 1):
            stim_trials[i]['choice'] = -1
        else:
            stim_trials[i]['choice'] = 1
    # stimulus has direction (-1, +1) direct to left and right
    for i in range(len(stim_trials)):
        if (stim_trials[i]['gt'] == 1):
            stim_trials[i]['gt'] = -1
        elif(stim_trials[i]['gt'] == 2):
            stim_trials[i]['gt'] = 1         
    idx_effect = np.array(idx_effect,dtype=int)
    
    return stim_trials, idx_effect, ctxt_trials

def transform_stim_trials_notskip(data, *kwargs):
    ctxseq = []
    ctx  = data['contexts']
    gt   = data['gt']
    dyns = data['states']
    for i in range(len(gt)):
        if(ctx[i]=='2'):
            ctxseq.append(1)
        else:
            ctxseq.append(0)
    ctxseq  = np.array(ctxseq)

    ngt_tot = np.where(gt>0)[0]
    ### total trial length -- minimum(len(gt),np.shape(dyns)[0])
    total_tnum  = np.minimum(len(gt),np.shape(dyns)[0])
    ngt_tot = np.arange(ngt_tot[0],total_tnum,2)
    stim_trials = {}
    ctxt_trials = []

    trial_skip, idx_skip,idx_effect = [],[],[]
    for idx,igt in enumerate(ngt_tot[:len(ngt_tot)-2]):
        if (igt+1>=np.shape(data['states'])[0]):
            break
        ### check if there is any NaN in response data
        resps_check   = dyns[igt-1,:]
        array_has_nan = np.isnan(np.sum(resps_check))
        if(array_has_nan==True):
            print('index:',igt)
            trial_skip = np.append(trial_skip,idx)
            idx_skip   = np.append(idx_skip,igt+1)## response index
        else:
            idx_effect = np.append(idx_effect,int(igt+1))
        ### generate stimulus trials 
        stim_trials[idx]={'stim_coh':[data['obscategory'][igt+1]],
                        'ctx':ctxseq[igt+1],
                        'gt':data['gt'][ngt_tot[idx+1]],
                        'resp':np.reshape(data['states'][igt+1],(1,-1)),
                        'choice':data['choice'][ngt_tot[idx+1]],
                        'rw':data['reward'][ngt_tot[idx+1]],
                        'start_end': np.array([ngt_tot[idx+1]-1, ngt_tot[idx+1]]),
                        'skip':array_has_nan,
                        }
        ctxt_trials = np.append(ctxt_trials,stim_trials[idx]['ctx'])
        
    # action (-1 and +1) -- left and right
    for i in range(len(stim_trials)):
        if (stim_trials[i]['choice'] == 1):
            stim_trials[i]['choice'] = -1
        else:
            stim_trials[i]['choice'] = 1
    # stimulus has direction (-1, +1) direct to left and right
    for i in range(len(stim_trials)):
        if (stim_trials[i]['gt'] == 1):
            stim_trials[i]['gt'] = -1
        elif(stim_trials[i]['gt'] == 2):
            stim_trials[i]['gt'] = 1         
    idx_effect = np.array(idx_effect,dtype=int)
    
    return stim_trials, idx_effect, ctxt_trials


def calculate_dprime(Xdata, ylabel):
    uniques = np.unique(ylabel)
    if len(uniques) == 1:
        return 10000
    means, sigmas = np.zeros(len(uniques)), np.zeros(len(uniques))

    for i in range(len(uniques)):
        means[i] = np.mean(Xdata[np.where(ylabel[:] == uniques[i])[0]])
        sigmas[i] = np.std(Xdata[np.where(ylabel[:] == uniques[i])[0]])
    if(sigmas[0] == 0):
        print("-------", means[0], sigmas[0])
    dprimes = len(uniques)*(means[1]-means[0])**2/(sigmas[0]**2+sigmas[1]**2)
    return dprimes

def calculate_disperse(Xdata, ylabel):
    uniques = np.unique(ylabel)
    if len(uniques) == 1:
        return np.nan
    means, sigmas = np.zeros(len(uniques)), np.zeros(len(uniques))

    for i in range(len(uniques)):
        means[i] = np.mean(Xdata[np.where(ylabel[:] == uniques[i])[0]])
        sigmas[i] = np.std(Xdata[np.where(ylabel[:] == uniques[i])[0]])
    if(sigmas[0] == 0):
        print("-------", means[0], sigmas[0])
    return sigmas





#### visualizing performance --- population gain control
#### in hist_integration_balanced_pop

def cross_gaincontrol(stats_correct,stats_error, coeffs, intercepts, ytest_set_correct, yevi_set_correct, ytest_set_error, yevi_set_error, label_axis=0, evi_axis = 4, CONTROL=1):
    
    ### flatten data 
    n_iterations,n_eachiter = np.shape(ytest_set_correct)[0], np.shape(ytest_set_correct)[1]     
    n_labels = np.shape(ytest_set_correct)[2]        
    ##### projections 
    right_ac_projection, left_ac_projection, right_ae_projection, left_ae_projection  = [],[],[],[] 


    ### first iteration
    ylabels_ac,ylabels_ae,yevi_ac,yevi_ae = np.reshape(ytest_set_correct[0,:,label_axis],(-1,1)),np.reshape(ytest_set_error[0,:,label_axis],(-1,1)),np.reshape(yevi_set_correct[0,:,evi_axis],(-1,1)),np.reshape(yevi_set_error[0,:,evi_axis],(-1,1))
    for iiter in range(1,n_iterations):
        ylabels_ac = np.vstack((ylabels_ac,np.reshape(ytest_set_correct[iiter,:,label_axis],(-1,1))))
        ylabels_ae = np.vstack((ylabels_ae,np.reshape(ytest_set_error[iiter,:,label_axis],(-1,1))))

        yevi_ac = np.vstack((yevi_ac,np.reshape(yevi_set_correct[iiter,:,evi_axis],(-1,1))))
        yevi_ae = np.vstack((yevi_ae,np.reshape(yevi_set_error[iiter,:,evi_axis],(-1,1))))

    ##### make sure that the unique labels for the AC trials are 0,1

    unique_labels_ac = np.unique(ylabels_ac)
    if (unique_labels_ac[0]>1):
        ylabels_ac = ylabels_ac-2 ### for the transition bias encoding 


    left_ac_trials, right_ac_trials = np.where(ylabels_ac==0)[0], np.where(ylabels_ac==1)[0]
    left_ae_trials, right_ae_trials = np.where(ylabels_ae==0)[0], np.where(ylabels_ae==1)[0]
    left_ac_projection,right_ac_projection = yevi_ac[left_ac_trials],yevi_ac[right_ac_trials]
    left_ae_projection, right_ae_projection = yevi_ae[left_ae_trials], yevi_ae[right_ae_trials]

    ### max_v 
    maxv_ac,maxv_ae = np.max(yevi_ac),np.max(yevi_ae)
    maxv = np.max([maxv_ac,maxv_ae])
    maxv = int(1.25*maxv)

    fig_ac,ax_ac = plt.subplots(figsize=(3,5))
    df = {'left choice': left_ac_projection.flatten(),
          'right choice': right_ac_projection.flatten()}
    df = pd.DataFrame({ key:pd.Series(value) for key, value in df.items() })
    order = ['left choice', 'right choice']
    box_pairs = [('left choice', 'right choice')]
    sns.set(style='whitegrid')
    ax_ac = sns.boxplot(data=df, order=order, palette=['green','purple'],showmeans=True, 
                        meanprops={"marker":"+",
                                   "markeredgecolor":"black",
                                   "markersize":"10"})
    # add_stat_annotation(ax_ac, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    ax_ac.set_ylim ([-maxv,maxv])
    ax_ac.set_yticks([-maxv,0,maxv])

    fig_ae,ax_ae = plt.subplots(figsize=(3,5))
    df = {'left choice': left_ae_projection.flatten(),
          'right choice': right_ae_projection.flatten()}
    df = pd.DataFrame({ key:pd.Series(value) for key, value in df.items() })
    order = ['left choice', 'right choice']
    box_pairs = [('left choice', 'right choice')]
    sns.set(style='whitegrid')
    ax_ae = sns.boxplot(data=df, order=order, palette=['green','purple'],showmeans=True, 
                        meanprops={"marker":"+",
                                   "markeredgecolor":"black",
                                   "markersize":"10"})
    # add_stat_annotation(ax_ae, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    ax_ae.set_ylim ([-maxv,maxv])
    ax_ae.set_yticks([-maxv,0,maxv])

    fig_score,ax_score = plt.subplots(figsize=(3,5))
    df = {'AC trials': stats_correct.flatten(),
          'AE trials': stats_error.flatten()}
    df = pd.DataFrame({ key:pd.Series(value) for key, value in df.items() })
    order = ['AC trials', 'AE trials']
    box_pairs = [('AC trials', 'AE trials')]
    sns.set(style='whitegrid')
    ax_score = sns.boxplot(data=df, order=order, palette=['yellow','black'])
    # add_stat_annotation(ax_ae, data=df, order=order, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    ax_score.set_ylim ([0.0,1.0])
    ax_score.set_yticks([0.0,0.5,1.0])



def mixed_selectivity_pop(d_selectivity):
    single_params,single_p_values,nnonselect,nselect, pop_left_correct,\
        pop_right_correct,pop_zero_correct,pop_left_error, pop_right_error, pop_zero_error,pop_rep_correct,\
            pop_alt_correct,pop_b_correct, pop_rep_error, pop_alt_error, pop_b_error, NN_error  =\
                d_selectivity['single_params'],d_selectivity['single_p_values'],d_selectivity['nnonselect'],\
                    d_selectivity['nselect'],d_selectivity['pop_left_correct'],\
                        d_selectivity['pop_right_correct'],d_selectivity['pop_zero_correct'],\
                            d_selectivity['pop_left_error'],d_selectivity['pop_right_error'],\
                                d_selectivity['pop_zero_error'],d_selectivity['pop_rep_correct'],\
                                    d_selectivity['pop_alt_correct'],d_selectivity['pop_b_correct'],\
                                        d_selectivity['pop_rep_error'],d_selectivity['pop_alt_error'],\
                                            d_selectivity['pop_b_error'],d_selectivity['NN_error']
    ### populations by ac conditions
    ### mixed-selectivity
    rep_left_correct  = np.intersect1d(pop_left_correct,pop_rep_correct)
    rep_right_correct = np.intersect1d(pop_right_correct,pop_rep_correct)
    alt_left_correct  = np.intersect1d(pop_left_correct,pop_alt_correct)
    alt_right_correct = np.intersect1d(pop_right_correct,pop_alt_correct)
    
    ### context -- mental state
    rep_only_correct = np.intersect1d(pop_rep_correct,pop_zero_correct)
    alt_only_correct = np.intersect1d(pop_alt_correct,pop_zero_correct)
    ### previous choice -- external signal
    left_only_correct  = np.intersect1d(pop_left_correct,pop_b_correct)
    right_only_correct = np.intersect1d(pop_right_correct,pop_b_correct)
    ### non-selectivity 
    correct_zero = np.intersect1d(pop_zero_correct,pop_b_correct)
    
    # pop_left_correct  = (left_only).copy()
    # pop_right_correct = (right_only).copy()
    
    # pop_left_correct  = pop_rep_correct.copy()# pop_left_correct.copy()#np.union1d(rep_left, alt_left)
    # pop_right_correct = pop_alt_correct.copy()#pop_right_correct.copy()#np.union1d(rep_right, alt_right)
    
    # pop_left_correct   = np.union1d(rep_left_correct, alt_left_correct)
    # pop_right_correct  = np.union1d(rep_right_correct, alt_right_correct)
             
    single_pop_correct = np.union1d(left_only_correct, right_only_correct)

    ### populations by ae conditions
    ### mixed-selectivity
    rep_left_error  = np.intersect1d(pop_left_error,pop_rep_error)
    rep_right_error = np.intersect1d(pop_right_error,pop_rep_error)
    alt_left_error  = np.intersect1d(pop_left_error,pop_alt_error)
    alt_right_error = np.intersect1d(pop_right_error,pop_alt_error)
    
    ### context -- mental state
    rep_only_error = np.intersect1d(pop_rep_error,pop_zero_error)
    alt_only_error = np.intersect1d(pop_alt_error,pop_zero_error)
    ### previous choice -- external signal
    left_only_error  = np.intersect1d(pop_left_error,pop_b_error)
    right_only_error = np.intersect1d(pop_right_error,pop_b_error)
    ### non-selectivity 
    error_zero = np.intersect1d(pop_zero_error,pop_b_error)
    
    # pop_left_error  = (left_only).copy()
    # pop_right_error = (right_only).copy()
    
    # pop_left_error  = pop_rep_error.copy()# pop_left_error.copy()#np.union1d(rep_left, alt_left)
    # pop_right_error = pop_alt_error.copy()#pop_right_error.copy()#np.union1d(rep_right, alt_right)
    
    # pop_left_error   = np.union1d(rep_left_error, alt_left_error)
    # pop_right_error  = np.union1d(rep_right_error, alt_right_error)
    
    single_pop_error = np.union1d(left_only_error, right_only_error)

    return nselect,nnonselect, pop_left_correct, pop_right_correct, single_pop_correct, correct_zero, pop_left_error, pop_right_error, single_pop_error, error_zero
            