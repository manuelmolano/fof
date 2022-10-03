
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from statannot import add_stat_annotation
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def gmm_fit(neurons_fs, n_components, algo='bayes', n_init=50, random_state=None, mean_precision_prior=None,
            weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None):
    """
    fit a mixture of gaussians to a set of vectors
    :param neurons_fs: list of numpy arrays of shape n or numpy array of shape n x d
    :param n_components: int
    :param algo: 'em' or 'bayes'
    :param n_init: number of random seeds for the inference algorithm
    :param random_state: random seed for the rng to eliminate randomness
    :return: vector of population labels (of shape n), best fitted model
    """
    if isinstance(neurons_fs, list):
        X = np.vstack(neurons_fs).transpose()
    else:
        X = neurons_fs
    if algo == "em":
        model = GaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state)
    else:
        model = BayesianGaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state,
                                        init_params='random', mean_precision_prior=mean_precision_prior,
                                        weight_concentration_prior_type=weight_concentration_prior_type,
                                        weight_concentration_prior=weight_concentration_prior)
    model.fit(X)
    z = model.predict(X)
    return z, model

def pairwise_mtx(dir, IDX_RAT, timep, NITERATIONS):
    # ### ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat31_'
    # timep = '/'#'-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises

    dataname  = dir+timep+IDX_RAT+'data_dec_ac_prevch.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histac, bi_histac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ae_prevch.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histae, bi_histae = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_beh_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behae, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_beh_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behac, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    wiac_fix_hist,wiae_fix_hist= np.zeros((np.shape(wi_histac)[0],NITERATIONS)),np.zeros((np.shape(wi_histae)[0],NITERATIONS))
    wiac_fix_beh,wiae_fix_beh   = np.zeros((np.shape(wi_behac)[0],NITERATIONS)),np.zeros((np.shape(wi_behae)[0],NITERATIONS))
    print('--------------------',np.shape(wi_behac))
    for i in range(NITERATIONS):
        wiac_fix_hist[:,i], wiae_fix_hist[:,i]= wi_histac[:, i*6+3],wi_histac[:,i*6+4]#,wi_histae[:, i*6+3]-wi_histae[:,i*6+4]
        # wiac_fix_hist[:,i], wiae_fix_hist[:,i]= wi_histac[:, i*5+3],wi_histae[:, i*5+3]
        wiac_fix_beh[:,i],wiae_fix_beh[:,i]   = wi_behac[:, (i*5)*3+4],wi_behae[:, (i*5)*3+4]
        for icoh in range(1,3):
            wiac_fix_beh[:,i],wiae_fix_beh[:,i]   = wiac_fix_beh[:,i]+wi_behac[:, (i*5)*3+5*icoh+4],wiae_fix_beh[:,i]  +wi_behae[:, (i*5)*3+5*icoh+4]
        wiac_fix_beh[:,i],wiae_fix_beh[:,i]   = wiac_fix_beh[:,i]/3,wiae_fix_beh[:,i]/3 
            
    ### Predicting upcoming stimulus 
    ### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(NITERATIONS):
        wiac_fix_hist[:,i] = wiac_fix_hist[:,i]/np.linalg.norm(wiac_fix_hist[:,i]) 
        wiae_fix_hist[:,i] = wiae_fix_hist[:,i]/np.linalg.norm(wiae_fix_hist[:,i]) 

    for i in range(NITERATIONS):
        wiac_fix_beh[:,i] = wiac_fix_beh[:,i]/np.linalg.norm(wiac_fix_beh[:,i]) 
        wiae_fix_beh[:,i] = wiae_fix_beh[:,i]/np.linalg.norm(wiae_fix_beh[:,i])

    decoders_fix_set = {}
    dec_names = ['tr.bias-ac','tr.bias-ae','beh-ac','beh-ae']
    decoders_fix_set['tr.bias-ac'],decoders_fix_set['tr.bias-ae'] = wiac_fix_hist.copy(),wiae_fix_hist.copy()
    decoders_fix_set['beh-ac'],decoders_fix_set['beh-ae']  = wiac_fix_beh.copy(), wiae_fix_beh.copy() 

    ag_fix_set = {}
    for i1, pair1 in enumerate (dec_names):
        for i2, pair2 in enumerate (dec_names):
            if i2<=i1:
                continue 
            ag_fix_set[pair1,pair2] = (decoders_fix_set[pair1].copy().T)@(decoders_fix_set[pair2])
            ag_fix_set[pair1,pair2] = np.arccos(ag_fix_set[pair1,pair2])/np.pi*180 

    for i, selfpair in enumerate(dec_names):
        ag_fix_set[selfpair,selfpair] = (decoders_fix_set[selfpair].copy().T)@(decoders_fix_set[selfpair])
        ag_fix_set[selfpair,selfpair] = np.arccos(ag_fix_set[selfpair,selfpair])/np.pi*180 
        ag_fix_set[selfpair,selfpair]= ag_fix_set[selfpair,selfpair][~np.eye(ag_fix_set[selfpair,selfpair].shape[0],dtype=bool)].reshape(ag_fix_set[selfpair,selfpair].shape[0],-1)
        # print(ag_fix_set[selfpair,selfpair])

    # figure ploting -- distribution of bootstrap results
    fig_fix,ax_fix = plt.subplots(4,4, figsize=(4,6),sharex=True,sharey=True)
    BOX_WDTH = 0.6
    for i1, pair1 in enumerate (dec_names):
        for i2, pair2 in enumerate (dec_names):
            if i2<i1:
                continue 
            df = {pair1+' v.s. '+pair2: (ag_fix_set[pair1,pair2]).flatten()}
            df = pd.DataFrame(df)
            sns.violinplot(data=df,ax=ax_fix[i1,i2],width=BOX_WDTH)
            ax_fix[i1,i2].set_ylim([30,150])
            ax_fix[i1,i2].set_yticks([30,90,150])

    for i, pair in enumerate(dec_names):
        ax_fix[i,0].set_ylabel(pair)
        ax_fix[0,i].set_title(pair)
        
        
    fig_beh, ax_beh=plt.subplots(figsize=(3,4))    
    df = {'beh-ac v.s. trbias': (ag_fix_set['tr.bias-ac','beh-ac']).flatten(), 'beh-ae v.s. trbias': (ag_fix_set['tr.bias-ae','beh-ae']).flatten()}
    df = pd.DataFrame(df)
    sns.set(style='whitegrid')
    
    ax_beh = sns.violinplot(data=df)
    box_pairs = [('beh-ac v.s. trbias', 'beh-ae v.s. trbias')]
    add_stat_annotation(ax_beh, data=df, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)


import general_util_ctxtgt as guc

def pairwise_mtx_classical(dir, IDX_RAT, timep, NITERATIONS):
    # ### ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat31_'
    # timep = '/'#'-01-00s/'
    # NITERATIONS = 50
    # ## compute the angles between history encoding axises

    dataname  = dir+timep+IDX_RAT+'data_dec_ac_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histac, bi_histac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    print('shape:hist 302', np.shape(wi_histac))
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ae_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histae, bi_histae = data_dec['coefs_correct'], data_dec['intercepts_correct']
    print('shape:hist 302', np.shape(wi_histae))

    dataname  = dir+timep+IDX_RAT+'data_beh_ae_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behae, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_beh_ac_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behac, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname       = dir+timep+IDX_RAT+'neuron_selectivity_Sept.npz'
    d_selectivity  = np.load(dataname, allow_pickle=True) 
                      
    nselect, nnonselect, pop_left_correct, pop_right_correct, single_pop_correct, correct_zero, pop_left_error, pop_right_error, single_pop_error, error_zero =guc.mixed_selectivity_pop(d_selectivity)

    # mixed_selectivity = np.union1d(pop_left_correct, pop_right_correct)
    mixed_selectivity   = np.arange(len(nselect))#single_pop_error#np.union1d(pop_left_correct, pop_right_correct)#np.union1d(pop_left_correct, pop_right_correct)#
    
    caxis=3
    wiac_fix_hist_rep,wiae_fix_hist_rep= np.zeros((len(mixed_selectivity),NITERATIONS)),np.zeros((len(mixed_selectivity),NITERATIONS))
    wiac_fix_hist_alt,wiae_fix_hist_alt= np.zeros((len(mixed_selectivity),NITERATIONS)),np.zeros((len(mixed_selectivity),NITERATIONS))
    wiac_fix_beh_rep,wiae_fix_beh_rep   = np.zeros((len(mixed_selectivity),NITERATIONS)),np.zeros((len(mixed_selectivity),NITERATIONS))
    wiac_fix_beh_alt,wiae_fix_beh_alt   = np.zeros((len(mixed_selectivity),NITERATIONS)),np.zeros((len(mixed_selectivity),NITERATIONS))
    for i in range(NITERATIONS):
        wiac_fix_hist_rep[:,i], wiae_fix_hist_rep[:,i]= wi_histac[:, i*6+caxis],wi_histae[:,i*6+caxis]
        wiac_fix_hist_alt[:,i], wiae_fix_hist_alt[:,i]= wi_histac[:, i*6+caxis+1],wi_histae[:,i*6+caxis+1]
        ### three
        wiac_fix_beh_rep[:,i],wiae_fix_beh_rep[:,i]   = wi_behac[:, (i*5)*3+caxis],wi_behae[:, (i*5)*3+caxis]
        for icoh in range(1,3):
            wiac_fix_beh_rep[:,i],wiae_fix_beh_rep[:,i]   = wiac_fix_beh_rep[:,i]+wi_behac[mixed_selectivity, (i*5)*3+5*icoh+caxis],wiae_fix_beh_rep[:,i]  +wi_behae[mixed_selectivity, (i*5)*3+5*icoh+caxis]
        wiac_fix_beh_rep[:,i],wiae_fix_beh_rep[:,i]   = wiac_fix_beh_rep[:,i]/3,wiae_fix_beh_rep[:,i]/3 
        ### three
        wiac_fix_beh_alt[:,i],wiae_fix_beh_alt[:,i]   = wi_behac[:, (i*5)*3+caxis+1],wi_behae[:, (i*5)*3+caxis+1]
        for icoh in range(1,3):
            wiac_fix_beh_alt[:,i],wiae_fix_beh_alt[:,i]   = wiac_fix_beh_alt[:,i]+wi_behac[mixed_selectivity, (i*5)*3+5*icoh+caxis+1],wiae_fix_beh_alt[:,i]  +wi_behae[mixed_selectivity, (i*5)*3+5*icoh+caxis+1]
        wiac_fix_beh_alt[:,i],wiae_fix_beh_alt[:,i]   = wiac_fix_beh_alt[:,i]/3,wiae_fix_beh_alt[:,i]/3 
            
    for i in range(NITERATIONS):
        wiac_fix_hist_rep[:,i] = wiac_fix_hist_rep[:,i]/np.linalg.norm(wiac_fix_hist_rep[:,i]) 
        wiae_fix_hist_rep[:,i] = wiae_fix_hist_rep[:,i]/np.linalg.norm(wiae_fix_hist_rep[:,i]) 
        wiac_fix_hist_alt[:,i] = wiac_fix_hist_alt[:,i]/np.linalg.norm(wiac_fix_hist_alt[:,i]) 
        wiae_fix_hist_alt[:,i] = wiae_fix_hist_alt[:,i]/np.linalg.norm(wiae_fix_hist_alt[:,i]) 

    for i in range(NITERATIONS):
        wiac_fix_beh_rep[:,i] = wiac_fix_beh_rep[:,i]/np.linalg.norm(wiac_fix_beh_rep[:,i]) 
        wiae_fix_beh_rep[:,i] = wiae_fix_beh_rep[:,i]/np.linalg.norm(wiae_fix_beh_rep[:,i])
        wiac_fix_beh_alt[:,i] = wiac_fix_beh_alt[:,i]/np.linalg.norm(wiac_fix_beh_alt[:,i]) 
        wiae_fix_beh_alt[:,i] = wiae_fix_beh_alt[:,i]/np.linalg.norm(wiae_fix_beh_alt[:,i])

    decoders_fix_set = {}
    dec_names = ['rep-ac-h','rep-ae-h','alt-ac-h','alt-ae-h','rep-ac-b','rep-ae-b','alt-ac-b','alt-ae-b']
    decoders_fix_set['rep-ac-h'],decoders_fix_set['rep-ae-h'] = wiac_fix_hist_rep.copy(),wiae_fix_hist_rep.copy()
    decoders_fix_set['alt-ac-h'],decoders_fix_set['alt-ae-h']  = wiac_fix_hist_alt.copy(), wiae_fix_hist_alt.copy() 

    decoders_fix_set['rep-ac-b'],decoders_fix_set['rep-ae-b'] = wiac_fix_beh_rep.copy(),wiae_fix_beh_rep.copy()
    decoders_fix_set['alt-ac-b'],decoders_fix_set['alt-ae-b']  = wiac_fix_beh_alt.copy(), wiae_fix_beh_alt.copy() 

    ag_fix_set = {}
    for i1, pair1 in enumerate (dec_names):
        for i2, pair2 in enumerate (dec_names):
            if i2<=i1:
                continue 
            ag_fix_set[pair1,pair2] = (decoders_fix_set[pair1].copy().T)@(decoders_fix_set[pair2])
            ag_fix_set[pair1,pair2] = np.arccos(ag_fix_set[pair1,pair2])/np.pi*180 

    for i, selfpair in enumerate(dec_names):
        ag_fix_set[selfpair,selfpair] = (decoders_fix_set[selfpair].copy().T)@(decoders_fix_set[selfpair])
        ag_fix_set[selfpair,selfpair] = np.arccos(ag_fix_set[selfpair,selfpair])/np.pi*180 
        # ag_fix_set[selfpair,selfpair]= ag_fix_set[selfpair,selfpair][~np.eye(ag_fix_set[selfpair,selfpair].shape[0],dtype=bool)].reshape(ag_fix_set[selfpair,selfpair].shape[0],-1)
        # # print(ag_fix_set[selfpair,selfpair])

    # figure ploting -- distribution of bootstrap results
    fig_fix,ax_fix = plt.subplots(8,8, figsize=(10,10),sharex=True,sharey=True,)
    BOX_WDTH = 0.3
    for i1, pair1 in enumerate (dec_names):
        for i2, pair2 in enumerate (dec_names):
            if i2<i1:
                continue 
            df = {pair1+' v.s. '+pair2: (ag_fix_set[pair1,pair2]).flatten()}
            df = pd.DataFrame(df)
            sns.violinplot(data=df,ax=ax_fix[i1,i2],width=BOX_WDTH)
            ax_fix[i1,i2].set_ylim([20,160])
            ax_fix[i1,i2].set_yticks([20,90,160])

    for i, pair in enumerate(dec_names):
        ax_fix[i,0].set_ylabel(pair)
        ax_fix[0,i].set_title(pair)

        
    plt.subplots_adjust(wspace=None,hspace=None)
    plt.show()
        
        
    # fig_beh, ax_beh=plt.subplots(figsize=(3,4))    
    # df = {'beh-ac v.s. trbias': (ag_fix_set['tr.bias-ac','beh-ac']).flatten(), 'beh-ae v.s. trbias': (ag_fix_set['tr.bias-ae','beh-ae']).flatten()}
    # df = pd.DataFrame(df)
    # sns.set(style='whitegrid')
    
    # ax_beh = sns.violinplot(data=df)
    # box_pairs = [('beh-ac v.s. trbias', 'beh-ae v.s. trbias')]
    # add_stat_annotation(ax_beh, data=df, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    
from itertools import product
import matplotlib.gridspec as gridspec

def pairwise_mtx_hist(dir, IDX_RAT, timep, NITERATIONS):
    # ### ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat31_'
    # timep = '/'#'-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises

    dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histac, bi_histac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histae, bi_histae = data_dec['coefs_correct'], data_dec['intercepts_correct']


    wiac_fix_ctxt,wiae_fix_ctxt= np.zeros((np.shape(wi_histac)[0],NITERATIONS)),np.zeros((np.shape(wi_histae)[0],NITERATIONS))
    wiac_fix_prevch,wiae_fix_prevch= np.zeros((np.shape(wi_histac)[0],NITERATIONS)),np.zeros((np.shape(wi_histae)[0],NITERATIONS))
    for i in range(NITERATIONS):
        wiac_fix_ctxt[:,i], wiae_fix_ctxt[:,i]     = wi_histac[:, i*5+1],wi_histae[:,i*5+1]
        wiac_fix_prevch[:,i], wiae_fix_prevch[:,i] = wi_histac[:, i*5+3],wi_histae[:, i*5+3]
           
    ### Predicting upcoming stimulus 
    ### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(NITERATIONS):
        wiac_fix_ctxt[:,i] = wiac_fix_ctxt[:,i]/np.linalg.norm(wiac_fix_ctxt[:,i]) 
        wiae_fix_ctxt[:,i] = wiae_fix_ctxt[:,i]/np.linalg.norm(wiae_fix_ctxt[:,i]) 
        
        wiac_fix_prevch[:,i] = wiac_fix_prevch[:,i]/np.linalg.norm(wiac_fix_prevch[:,i]) 
        wiae_fix_prevch[:,i] = wiae_fix_prevch[:,i]/np.linalg.norm(wiae_fix_prevch[:,i]) 

    decoders_fix_ctxt, decoders_fix_prevch = {},{}
    dec_names = ['ac','ae']
    decoders_fix_ctxt['ac'],decoders_fix_ctxt['ae']     = wiac_fix_ctxt.copy(),wiae_fix_ctxt.copy()
    decoders_fix_prevch['ac'],decoders_fix_prevch['ae'] = wiac_fix_ctxt.copy(),wiae_fix_ctxt.copy()

    ag_fix_ctxt = {}
    for i1, pair1 in enumerate (dec_names[:2]):
        for i2, pair2 in enumerate (dec_names[:2]):
            ag_fix_ctxt[pair1,pair2] = (decoders_fix_ctxt[pair1].copy().T)@(decoders_fix_ctxt[pair2])
            ag_fix_ctxt[pair1,pair2] = np.arccos(ag_fix_ctxt[pair1,pair2])/np.pi*180 

    for i, selfpair in enumerate(dec_names):
        ag_fix_ctxt[selfpair,selfpair] = (decoders_fix_ctxt[selfpair].copy().T)@(decoders_fix_ctxt[selfpair])
        ag_fix_ctxt[selfpair,selfpair] = np.arccos(ag_fix_ctxt[selfpair,selfpair])/np.pi*180 
        ag_fix_ctxt[selfpair,selfpair]= ag_fix_ctxt[selfpair,selfpair][~np.eye(ag_fix_ctxt[selfpair,selfpair].shape[0],dtype=bool)].reshape(ag_fix_ctxt[selfpair,selfpair].shape[0],-1)
        
        
    ag_fix_prevch = {}
    for i1, pair1 in enumerate (dec_names[:2]):
        for i2, pair2 in enumerate (dec_names[:2]):
            ag_fix_prevch[pair1,pair2] = (decoders_fix_prevch[pair1].copy().T)@(decoders_fix_prevch[pair2])
            ag_fix_prevch[pair1,pair2] = np.arccos(ag_fix_prevch[pair1,pair2])/np.pi*180 

    for i, selfpair in enumerate(dec_names):
        ag_fix_prevch[selfpair,selfpair] = (decoders_fix_prevch[selfpair].copy().T)@(decoders_fix_prevch[selfpair])
        ag_fix_prevch[selfpair,selfpair] = np.arccos(ag_fix_prevch[selfpair,selfpair])/np.pi*180 
        ag_fix_prevch[selfpair,selfpair]= ag_fix_prevch[selfpair,selfpair][~np.eye(ag_fix_prevch[selfpair,selfpair].shape[0],dtype=bool)].reshape(ag_fix_prevch[selfpair,selfpair].shape[0],-1)

    # # figure ploting -- distribution of bootstrap results
    # fig_fix,ax_fix = plt.subplots(2,2, figsize=(4,4),tight_layout=True, sharex=True,sharey=True)
    # BOX_WDTH = 1.0
    # for i1, pair1 in enumerate (dec_names):
    #     for i2, pair2 in enumerate (dec_names):
    #         df = {pair1+' v.s. '+pair2: (ag_fix_set[pair1,pair2]).flatten()}
    #         df = pd.DataFrame(df)
    #         sns.boxplot(data=df,ax=ax_fix[i1,i2],width=BOX_WDTH)
    #         ax_fix[i1,i2].set_ylim([30,150])
    #         ax_fix[i1,i2].set_yticks([30,90,150])
    #         ax_fix[i1,i2].set_frame_on(None)
    # ax_fix[1,0].set_frame_on(None)

    fig = plt.figure(figsize=(7, 3))
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.0)
    
    for i in range(2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
          2,2, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        a, b = int(i/4)+1, i % 4+1
        for j, (c, d) in enumerate(product(range(1, 3), repeat=2)):
            print('------',j,c,d)
            ax = plt.Subplot(fig, inner_grid[j])
            if i==0:
                df = {dec_names[c-1]+' v.s. '+dec_names[d-1]: (ag_fix_ctxt[dec_names[c-1],dec_names[d-1]]).flatten()}
                df = pd.DataFrame(df)
                sns.boxplot(data=df,ax=ax,width=0.35)
            elif i==1:
                df = {dec_names[c-1]+' v.s. '+dec_names[d-1]: (ag_fix_prevch[dec_names[c-2],dec_names[d-2]]).flatten()}
                df = pd.DataFrame(df)
                sns.boxplot(data=df,ax=ax,width=0.35)
            ax.set_xticks([])
            ax.set_yticks([90])
            ax.set_ylim([30,90])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    # show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
def pairwise_mtx_hist_condition(dir, IDX_RAT, timep, NITERATIONS):
    # ### ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/'#'-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises

    dataname  = dir+timep+IDX_RAT+'data_dec_ac_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histac, bi_histac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ae_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_histae, bi_histae = data_dec['coefs_correct'], data_dec['intercepts_correct']

    wiac_fix_prevchL,wiac_fix_prevchR= np.zeros((np.shape(wi_histac)[0],NITERATIONS)),np.zeros((np.shape(wi_histac)[0],NITERATIONS))
    for i in range(NITERATIONS):
        wiac_fix_prevchL[:,i], wiac_fix_prevchR[:,i] = wi_histac[:, i*6+3],wi_histac[:, i*6+4]
        
    wiae_fix_prevchL,wiae_fix_prevchR= np.zeros((np.shape(wi_histae)[0],NITERATIONS)),np.zeros((np.shape(wi_histae)[0],NITERATIONS))
    for i in range(NITERATIONS):
        wiae_fix_prevchL[:,i], wiae_fix_prevchR[:,i] = wi_histae[:, i*6+3],wi_histae[:, i*6+4]
           
    for i in range(NITERATIONS):
        wiac_fix_prevchL[:,i] = wiac_fix_prevchL[:,i]/np.linalg.norm(wiac_fix_prevchL[:,i]) 
        wiac_fix_prevchR[:,i] = wiac_fix_prevchR[:,i]/np.linalg.norm(wiac_fix_prevchR[:,i]) 
        wiae_fix_prevchL[:,i] = wiae_fix_prevchL[:,i]/np.linalg.norm(wiae_fix_prevchL[:,i]) 
        wiae_fix_prevchR[:,i] = wiae_fix_prevchR[:,i]/np.linalg.norm(wiae_fix_prevchR[:,i]) 


    decoders_fix_ac, decoders_fix_ae = {},{}
    dec_names = ['prevchL','prevchR']
    decoders_fix_ac['prevchL'],decoders_fix_ac['prevchR'] = wiac_fix_prevchL.copy(),wiac_fix_prevchR.copy()
    decoders_fix_ae['prevchL'],decoders_fix_ae['prevchR'] = wiae_fix_prevchL.copy(),wiae_fix_prevchR.copy()

    ag_fix_ac = {}
    for i1, pair1 in enumerate (dec_names[:2]):
        for i2, pair2 in enumerate (dec_names[:2]):
            ag_fix_ac[pair1,pair2] = (decoders_fix_ac[pair1].copy().T)@(decoders_fix_ac[pair2])
            ag_fix_ac[pair1,pair2] = np.arccos(ag_fix_ac[pair1,pair2])/np.pi*180 

        
        
    ag_fix_ae = {}
    for i1, pair1 in enumerate (dec_names[:2]):
        for i2, pair2 in enumerate (dec_names[:2]):
            ag_fix_ae[pair1,pair2] = (decoders_fix_ae[pair1].copy().T)@(decoders_fix_ae[pair2])
            ag_fix_ae[pair1,pair2] = np.arccos(ag_fix_ae[pair1,pair2])/np.pi*180 


    fig = plt.figure(figsize=(7, 3))
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.0)
    
    for i in range(2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
          2,2, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        a, b = int(i/4)+1, i % 4+1
        for j, (c, d) in enumerate(product(range(1, 3), repeat=2)):
            print('------',j,c,d)
            ax = plt.Subplot(fig, inner_grid[j])
            if i==0:
                print('~~~~',dec_names[c-1],dec_names[d-1])
                df = {dec_names[c-1]+' v.s. '+dec_names[d-1]: (ag_fix_ac[dec_names[c-1],dec_names[d-1]]).flatten()}
                df = pd.DataFrame(df)
                sns.boxplot(data=df,ax=ax,width=0.35)
            elif i==1:
                print('......',dec_names[c-2],dec_names[d-2])
                df = {dec_names[c-1]+' v.s. '+dec_names[d-1]: (ag_fix_ae[dec_names[c-2],dec_names[d-2]]).flatten()}
                df = pd.DataFrame(df)
                sns.boxplot(data=df,ax=ax,width=0.35)
            ax.set_xticks([])
            ax.set_yticks([40,90,150])
            ax.set_ylim([40,150])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    # show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()

def accuracy_mtx_pop(dir, IDX_RAT, timep, NITERATIONS):
    ## ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    # dataname  = dir+timep+IDX_RAT+'data_dec_ac_prevch.npz'
    # data_dec  = np.load(dataname, allow_pickle=True)
    # score_fix_hist_ac_c  = data_dec['stats_correct_pop'][0,:]
    # score_fix_hist_ac_cc = data_dec['stats_correct_pop'][1,:]
    # score_fix_hist_ac = data_dec['stats_correct'][3,:]
    
    
    # dataname  = dir+timep+IDX_RAT+'data_dec_ae_prevch.npz'
    # data_dec  = np.load(dataname, allow_pickle=True)
    # score_fix_hist_ae_e  = data_dec['stats_error_pop'][0,:]
    # score_fix_hist_ae_ec = data_dec['stats_error_pop'][1,:]
    # score_fix_hist_ae = data_dec['stats_error'][3,:]
    
    
    # accuracy_fix_set = {}
    # accuracy_fix_set['ac-pop-dec'], accuracy_fix_set['ac-all-dec']= score_fix_hist_ac_c.copy(), score_fix_hist_ac.copy() 
    # accuracy_fix_set['ae-pop-dec'], accuracy_fix_set['ae-all-dec']= score_fix_hist_ae_e.copy(), score_fix_hist_ae.copy()  
    
    # accuracy_fix_set['ac-pop-dec(cross)'], accuracy_fix_set['ae-pop-dec(cross)']= score_fix_hist_ac_cc.copy(), score_fix_hist_ae_ec.copy() 


    # fig_score, ax_score=plt.subplots(figsize=(4,2),tight_layout=True, sharey=True)    
    # df_hist = {'ac-pop-dec': (accuracy_fix_set['ac-pop-dec']).flatten(), 'ac-pop-dec(cross)': 1-(accuracy_fix_set['ac-pop-dec(cross)']).flatten(), 'ac-all-dec': (accuracy_fix_set['ac-all-dec']).flatten(),'ae-pop-dec': (accuracy_fix_set['ae-pop-dec']).flatten(), 'ae-pop-dec(cross)': 1-(accuracy_fix_set['ae-pop-dec(cross)']).flatten(), 'ae-all-dec': (accuracy_fix_set['ae-all-dec']).flatten()}
    # df_hist = pd.DataFrame(df_hist)
    
    # ax_score = sns.boxplot(ax=ax_score,data=df_hist,width=0.35)
    # # box_pairs = [('ac-trials_ac-dec', 'ae-trials_ac-dec')]
    # # add_stat_annotation(ax_score, data=df_hist, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    
    # ax_score.set_ylim([0.3,1.2])
    # ax_score.set_yticks([0.5,1])
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac_mixs_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ac_single  = data_dec['stats_correct_pop'][0,:]
    score_fix_hist_ac_pop     = data_dec['stats_correct_pop'][1,:]
    score_fix_hist_ac_overall = data_dec['stats_correct'][3,:]

    
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ae_mixs_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ae_single  = data_dec['stats_error_pop'][0,:]
    score_fix_hist_ae_pop     = data_dec['stats_error_pop'][1,:]
    score_fix_hist_ae_overall = data_dec['stats_error'][3,:]
    
    
    accuracy_fix_set = {}
    accuracy_fix_set['ac-mixed-dec'], accuracy_fix_set['ac-overall-dec']= score_fix_hist_ac_pop.copy(), score_fix_hist_ac_overall.copy() 
    accuracy_fix_set['ae-mixed-dec'], accuracy_fix_set['ae-overall-dec']= score_fix_hist_ae_pop.copy(), score_fix_hist_ae_overall.copy()  
    
    accuracy_fix_set['ac-single-dec'], accuracy_fix_set['ae-single-dec']= score_fix_hist_ac_single.copy(), score_fix_hist_ae_single.copy()


    fig_score, ax_score=plt.subplots(figsize=(4,2),tight_layout=True, sharey=True)    
    df_hist = {'ac-overall-dec': (accuracy_fix_set['ac-overall-dec']).flatten(), 'ac-mixed-dec': (accuracy_fix_set['ac-mixed-dec']).flatten(), 'ac-single-dec': (accuracy_fix_set['ac-single-dec']).flatten(),'ae-overall-dec': (accuracy_fix_set['ae-overall-dec']).flatten(), 'ae-mixed-dec': 1-(accuracy_fix_set['ae-mixed-dec']).flatten(), 'ae-single-dec': (accuracy_fix_set['ae-single-dec']).flatten()}
    df_hist = pd.DataFrame(df_hist)
    
    ax_score = sns.boxplot(ax=ax_score,data=df_hist,width=0.35)
    # box_pairs = [('ac-trials_ac-dec', 'ae-trials_ac-dec')]
    # add_stat_annotation(ax_score, data=df_hist, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    
    ax_score.set_ylim([0.3,1.2])
    ax_score.set_yticks([0.5,1])
        

def accuracy_mtx(dir, IDX_RAT, timep, NITERATIONS):
    ## ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ae_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ae = data_dec['stats_error'][3,:]
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ac = data_dec['stats_correct'][3,:]
    
    idxcoh = 2
    dataname  = dir+timep+IDX_RAT+'data_beh_ae_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_beh_ae = data_dec['stats_error_alt'][:,idxcoh]
    
    dataname  = dir+timep+IDX_RAT+'data_beh_ac_cond_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_beh_ac = data_dec['stats_correct_alt'][:,idxcoh]
    
    
    accuracy_fix_set = {}
    accuracy_fix_set['hist-ac'], accuracy_fix_set['hist-ae']= score_fix_hist_ac.copy(), score_fix_hist_ae.copy() 
    accuracy_fix_set['beh-ac'], accuracy_fix_set['beh-ae']  = score_fix_beh_ac.flatten(), score_fix_beh_ae.flatten() 


    fig_score, ax_score=plt.subplots(1,2,figsize=(6,4), sharey=True)    
    df_hist = {'hist-ac': (accuracy_fix_set['hist-ac']).flatten(), 'hist-ae': (accuracy_fix_set['hist-ae']).flatten()}
    df_beh = {'beh-ac':accuracy_fix_set['beh-ac'],'beh-ae':accuracy_fix_set['beh-ae']}
    df_hist = pd.DataFrame(df_hist)
    df_beh = pd.DataFrame(df_beh)
    
    ax_score[0] = sns.violinplot(ax=ax_score[0],data=df_hist)
    box_pairs = [('hist-ac', 'hist-ae')]
    add_stat_annotation(ax_score[0], data=df_hist, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    ax_score[1] = sns.violinplot(ax=ax_score[1],data=df_beh)
    box_pairs = [('beh-ac', 'beh-ae')]
    add_stat_annotation(ax_score[1], data=df_beh, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    
    ax_score[0].set_ylim([0.0,1.2])
    ax_score[0].set_yticks([0,0.5,1])
    
    ax_score[1].set_ylim([0.0,1.2])
    ax_score[1].set_yticks([0,0.5,1])
    
    plt.subplots_adjust(wspace=None,hspace=None)
    plt.show()
        
    
def accuracy_mtx_cross_ctxt(dir, IDX_RAT, timep, NITERATIONS):
    ## ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ae_c = data_dec['stats_error'][0,:]
    score_fix_hist_ac_c = data_dec['stats_correct'][0,:]
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ae_e = data_dec['stats_error'][0,:]
    score_fix_hist_ac_e = data_dec['stats_correct'][0,:]
    
    
    accuracy_fix_set = {}
    accuracy_fix_set['ac-trials_ac-dec'], accuracy_fix_set['ae-trials_ac-dec']= score_fix_hist_ac_c.copy(), score_fix_hist_ae_c.copy() 
    accuracy_fix_set['ac-trials_ae-dec'], accuracy_fix_set['ae-trials_ae-dec']= score_fix_hist_ac_e.copy(), score_fix_hist_ae_e.copy()  


    fig_score, ax_score=plt.subplots(figsize=(4,4),tight_layout=True, sharey=True)    
    df_hist = {'ac-trials_ac-dec': (accuracy_fix_set['ac-trials_ac-dec']).flatten(), 'ae-trials_ac-dec': (accuracy_fix_set['ae-trials_ac-dec']).flatten(),'ac-trials_ae-dec': (accuracy_fix_set['ac-trials_ae-dec']).flatten(), 'ae-trials_ae-dec': (accuracy_fix_set['ae-trials_ae-dec']).flatten()}
    df_hist = pd.DataFrame(df_hist)
    
    ax_score = sns.boxplot(ax=ax_score,data=df_hist,width=0.35)
    box_pairs = [('ac-trials_ac-dec', 'ae-trials_ac-dec')]
    # add_stat_annotation(ax_score, data=df_hist, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    
    ax_score.set_ylim([0.5,1.2])
    ax_score.set_yticks([0,0.5,1])
    
    
def subpop_projection(dir, IDX_RAT, timep, NITERATIONS):
    # # # ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_flt_ac_overall_mixs.npz'
    data_dprime  = np.load(dataname, allow_pickle=True)
    dprime_left_overall,dprime_right_overall = data_dprime['dprimes_lc'],data_dprime['dprimes_rc']
    #data_dprime['dprimes_lc'],data_dprime['dprimes_rc']
    
    dataname  = dir+timep+IDX_RAT+'data_flt_ac_mixed_mixs.npz'
    data_dprime  = np.load(dataname, allow_pickle=True)
    dprime_left_mixed,dprime_right_mixed = data_dprime['dprimes_lc'],data_dprime['dprimes_rc']
    #data_dprime['dprimes_lc'],data_dprime['dprimes_rc']

    dataname  = dir+timep+IDX_RAT+'data_flt_ac_single_mixs.npz'
    data_dprime  = np.load(dataname, allow_pickle=True)
    dprime_left_single,dprime_right_single = data_dprime['dprimes_lc'],data_dprime['dprimes_rc']
    #data_dprime['dprimes_lc'],data_dprime['dprimes_rc']

    fig_dprime, ax_dprime=plt.subplots(1,2,figsize=(4,2),tight_layout=True,sharey=True,sharex=True)    
    df_left = {'overall': (dprime_left_overall).flatten(), 'mixed': (dprime_left_mixed).flatten(),'single': (dprime_left_single).flatten(),}
    df_left  = pd.DataFrame(df_left)   
    ax_dprime[0] = sns.boxplot(ax=ax_dprime[0],data=df_left,width=0.35)
    
    plt.xticks(rotation=45)
    
    df_right = {'overall': (dprime_right_overall).flatten(), 'mixed': (dprime_right_mixed).flatten(),'single-': (dprime_right_single).flatten(),}
    df_right  = pd.DataFrame(df_right)   
    ax_dprime[1] = sns.boxplot(ax=ax_dprime[1],data=df_right,width=0.35)
    plt.xticks(rotation=45)
    
    # ax_dprime[0].set_ylim([0.2,1.2])
    # ax_dprime[1].set_ylim([0.2,1.2])
    # ax_dprime[0].set_yticks([0.5,1.0])
    # ax_dprime[1].set_yticks([0.5,1.0])
    
    ax_dprime[0].set_ylim([-0.2,1.2])
    ax_dprime[1].set_ylim([-0.2,1.2])
    ax_dprime[0].set_yticks([0,4.5])
    ax_dprime[1].set_yticks([0,4.5])
    
    # box_pairs = [('left org', 'left cross-projection'),('right org', 'right cross-projection')]
    # add_stat_annotation(ax_dprime, data=df, box_pairs=box_pairs,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
    

def ref_accuracy_mtx(dir, IDX_RAT, timep, NITERATIONS):
    ## ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ae_mixs_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']
    score_fix_hist_ae = data_dec['stats_error'][:]
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac_mixs_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    score_fix_hist_ac = data_dec['stats_correct'][:]
    
    dataname  = dir+timep+IDX_RAT+'data_beh_ae_mixs_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behae, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    REF_UNI_VEC  = np.ones((np.shape(wi_behae)[0]))
    REF_UNI_VEC  = REF_UNI_VEC/np.linalg.norm(REF_UNI_VEC)  
    
    score_fix_beh_ae = data_dec['stats_error']
    
    dataname  = dir+timep+IDX_RAT+'data_beh_ac_mixs_.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behac, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']
    score_fix_beh_ac = data_dec['stats_correct']
    
    wiac_fix_hist,wiae_fix_hist = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
    wiac_fix_beh,wiae_fix_beh   = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
    for i in range(NITERATIONS):
        wiac_fix_hist[:,i],wiae_fix_hist[:,i] = wi_ac[:, i*5+3],wi_ae[:, i*5+3]
        wiac_fix_beh[:,i],wiae_fix_beh[:,i]   = wi_behac[:, i*5+4],wi_behae[:, i*5+4]
    
    ### Predicting upcoming stimulus 
    ### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(NITERATIONS):
        wiac_fix_hist[:,i] = wiac_fix_hist[:,i]/np.linalg.norm(wiac_fix_hist[:,i]) 
        wiae_fix_hist[:,i] = wiae_fix_hist[:,i]/np.linalg.norm(wiae_fix_hist[:,i])
    
    ref_wiac_fix_hist = np.mean(wiac_fix_hist,axis=1)
    ref_wiac_fix_hist = ref_wiac_fix_hist/np.linalg.norm(ref_wiac_fix_hist) 
    ref_wiac_fix_hist = np.reshape(ref_wiac_fix_hist,(-1,1))


    for i in range(NITERATIONS):
        wiac_fix_beh[:,i] = wiac_fix_beh[:,i]/np.linalg.norm(wiac_fix_beh[:,i]) 
        wiae_fix_beh[:,i] = wiae_fix_beh[:,i]/np.linalg.norm(wiae_fix_beh[:,i])
    
    decoders_fix_set = {}
    dec_names = ['hist-ac','hist-ae','beh-ac','beh-ae']
    decoders_fix_set['hist-ac'],decoders_fix_set['hist-ae'] = wiac_fix_hist.copy(),wiae_fix_hist.copy()
    decoders_fix_set['beh-ac'],decoders_fix_set['beh-ae']  = wiac_fix_beh.copy(), wiae_fix_beh.copy() 
    
    accuracy_fix_set = {}
    accuracy_fix_set['hist-ac'], accuracy_fix_set['hist-ae']= score_fix_hist_ac.copy(), score_fix_hist_ae.copy() 
    accuracy_fix_set['beh-ac'], accuracy_fix_set['beh-ae']  = score_fix_beh_ac.flatten(), score_fix_beh_ae.flatten() 
    
    ag_fix_set = {}
    for i1, pair1 in enumerate (dec_names):
        ag_fix_set[pair1] = (decoders_fix_set[pair1].copy().T)@ref_wiac_fix_hist #(REF_UNI_VEC)
        ag_fix_set[pair1] = np.arccos(ag_fix_set[pair1])#/np.pi*180 
    
    Nnbin       = 81
    nbin_ang    = np.linspace(0,2*np.pi,Nnbin, endpoint=False)
    acc_vs_ang  = {}
    acc_vs_ang_ste = {}
    for _,ikey in enumerate(dec_names):
        acc_vs_ang[ikey]=np.zeros(Nnbin-1)
        acc_vs_ang_ste[ikey]=np.zeros(Nnbin-1)
    
    
    for i in range(1,len(nbin_ang)):
        for i1, pair1 in enumerate(dec_names):
            idx_h = np.where(ag_fix_set[pair1]<nbin_ang[i])[0]
            idx_l = np.where(ag_fix_set[pair1]>=nbin_ang[i-1])[0]
            idx   = np.intersect1d(idx_l,idx_h)
            if(len(idx)>0):
                accuracy = np.mean(accuracy_fix_set[pair1][idx])
                acc_vs_ang[pair1][i-1]=accuracy 
                acc_vs_ang_ste[pair1][i-1] = np.std(accuracy_fix_set[pair1][idx])/np.sqrt(len(idx))
    
    width = np.pi*2/(Nnbin-1)
    
    # figure ploting -- distribution of bootstrap results
    fig_fix = plt.figure(figsize=(12,3))#subplots(1,4,figsize=(8,3),tight_layout=True,sharex=True,sharey=True)
    for i1, pair1 in enumerate (dec_names):
        ax_fix = fig_fix.add_subplot(1,4,i1+1, projection='polar')
        ax_fix.bar((nbin_ang[:-1]+nbin_ang[1:])/2.0, acc_vs_ang[pair1],yerr = acc_vs_ang_ste[pair1], width=width, bottom=0,alpha=0.5)
        # ax_fix[i1].errorbar((nbin_ang[:-1]+nbin_ang[1:])/2.0,acc_vs_ang[pair1],acc_vs_ang_ste[pair1],mfc='red', mec='green')
        ax_fix.set_ylim([0,1.0])
        ax_fix.set_yticks([0,0.5,1.0])

def mixed_selectivity(dir, IDX_RAT, timep, NITERATIONS):
    
    rats = ['Rat7','Rat15','Rat31','Rat32']
    fig, ax = plt.subplots(figsize=(4,4))
    width = 0.35  # the width of the bars: can also be len(x) sequence
    number_non,number_mix = [],[]
    labels =rats.copy()
    for IDX_RAT in rats:
        dataname  = dir+timep+IDX_RAT+'_neuron_selectivity.npz'
        d_selectivity  = np.load(dataname, allow_pickle=True)
        single_p_values, pop_left_correct, pop_right_correct, pop_zero_correct = d_selectivity['single_p_values'], d_selectivity['pop_left_correct'],d_selectivity['pop_right_correct'],d_selectivity['pop_zero_correct']
        pop_rep_correct, pop_alt_correct, pop_b_correct = d_selectivity['pop_rep_correct'],d_selectivity['pop_alt_correct'],d_selectivity['pop_b_correct']
        
        nnonselect, nselect = d_selectivity['nnonselect'], d_selectivity['nselect']
        selectivity_select = single_p_values[nselect,1:]
        
        if len(number_non)==0:
            number_non,number_mix= [len(nnonselect)],[len(nselect)]
        else:
            number_non, number_mix = np.append(number_non,len(nnonselect)),np.append(number_mix,len(nselect))
        figprevch, axprevch = plt.subplots(2,2,figsize=(6,4),tight_layout=True,sharey=True)
        dfprevch = {'pop-left': selectivity_select[pop_left_correct,0],
                    'pop-right':selectivity_select[pop_right_correct,0],
                    'pop-zero-prevch':selectivity_select[pop_zero_correct,0]}
        dfctxt = {'pop-left':selectivity_select[pop_left_correct,1],
                  'pop-right':selectivity_select[pop_right_correct,1],
                  'pop-zero-prevch':selectivity_select[pop_zero_correct,1]}
             
        dfprevch = dict([(k,pd.Series(v)) for k,v in dfprevch.items()])#pd.DataFrame(dfprevch)
        dfctxt   = dict([(k,pd.Series(v)) for k,v in dfctxt.items()])#pd.DataFrame(dfctxt)
        dfprevch = pd.DataFrame(dfprevch)
        dfctxt   = pd.DataFrame(dfctxt)
    
        prevch = sns.boxplot(ax=axprevch[0][0],data=dfprevch,width=0.35)
        box_pairs_prevch = [('pop-left', 'pop-zero-prevch'),('pop-right','pop-zero-prevch')]
        add_stat_annotation(axprevch[0][0], data=dfprevch, box_pairs=box_pairs_prevch,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
        
        ctxt = sns.boxplot(ax=axprevch[0][1],data=dfctxt,width=0.35)
        box_pairs_ctxt = [('pop-left', 'pop-right'),('pop-left', 'pop-zero-prevch'),('pop-right','pop-zero-prevch')]
        add_stat_annotation(axprevch[0][1], data=dfctxt, box_pairs=box_pairs_ctxt,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
        
        
        dfctxt = {'pop-rep': selectivity_select[pop_rep_correct,1],
                    'pop-alt':selectivity_select[pop_alt_correct,1],
                    'pop-zero-ctxt':selectivity_select[pop_b_correct,1]}
        dfprevch = {'pop-rep':selectivity_select[pop_rep_correct,0],
                  'pop-alt':selectivity_select[pop_alt_correct,0],
                  'pop-zero-ctxt':selectivity_select[pop_b_correct,0]}
        
        dfprevch = dict([(k,pd.Series(v)) for k,v in dfprevch.items()])#pd.DataFrame(dfprevch)
        dfctxt   = dict([(k,pd.Series(v)) for k,v in dfctxt.items()])#pd.DataFrame(dfctxt)
        dfprevch = pd.DataFrame(dfprevch)
        dfctxt   = pd.DataFrame(dfctxt)
    
        ctxt = sns.boxplot(ax=axprevch[1][1],data=dfctxt,width=0.35)
        box_pairs_ctxt = [('pop-rep', 'pop-zero-ctxt'),('pop-alt','pop-zero-ctxt')]
        add_stat_annotation(axprevch[1][1], data=dfctxt, box_pairs=box_pairs_ctxt,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
        
        prevch = sns.boxplot(ax=axprevch[1][0],data=dfprevch,width=0.35)
        box_pairs_prevch = [('pop-rep', 'pop-alt'),('pop-rep', 'pop-zero-ctxt'),('pop-alt','pop-zero-ctxt')]
        add_stat_annotation(axprevch[1][0], data=dfprevch, box_pairs=box_pairs_prevch,test='Mann-Whitney', text_format='star', loc='inside',verbose=2)
        
            
    ax.bar(labels,number_non, width, label='single or non')
    ax.bar(labels,number_mix, width, bottom=number_non,label='mixed')

def mixed_selectivity_acae(dir, IDX_RAT, timep, NITERATIONS):
    
    rats = ['Rat7','Rat15','Rat31','Rat32']
    labels = rats.copy()
    fig, ax = plt.subplots(figsize=(4,4))
    x = np.arange(len(rats))  # the label locations
    width = 0.35  # the width of the bars
    
    number_non,number_left, number_right = [],[],[]
    number_non_ae,number_left_ae, number_right_ae = [],[],[]
    labels =rats.copy()
    for IDX_RAT in rats:
        dataname  = dir+timep+IDX_RAT+'_neuron_selectivity.npz'
        d_selectivity  = np.load(dataname, allow_pickle=True)
        single_p_values, pop_left_correct, pop_right_correct, pop_zero_correct = d_selectivity['single_p_values'], d_selectivity['pop_left_correct'],d_selectivity['pop_right_correct'],d_selectivity['pop_zero_correct']
        
        pop_left_error, pop_right_error, pop_zero_error =  d_selectivity['pop_left_error'],d_selectivity['pop_right_error'],d_selectivity['pop_zero_error']
        pop_rep_correct, pop_alt_correct, pop_b_correct = d_selectivity['pop_rep_correct'],d_selectivity['pop_alt_correct'],d_selectivity['pop_b_correct']
        
        pop_rep_error, pop_alt_error, pop_b_error = d_selectivity['pop_rep_error'],d_selectivity['pop_alt_error'],d_selectivity['pop_b_error']
        
        nnonselect, nselect = d_selectivity['nnonselect'], d_selectivity['nselect']
        selectivity_select = single_p_values[nselect,1:]
        
        if len(number_non)==0:
            # number_non,number_left, number_right= [len(pop_zero_correct)],[len(pop_left_correct)],[len(pop_right_correct)]
            # number_non_ae,number_left_ae, number_right_ae = [len(pop_zero_error)],[len(pop_left_error)],[len(pop_right_error)]
            
            number_non,number_rep, number_alt= [len(pop_b_correct)],[len(pop_rep_correct)],[len(pop_alt_correct)]
            number_non_ae,number_rep_ae, number_alt_ae = [len(pop_b_error)],[len(pop_rep_error)],[len(pop_alt_error)]
        else:
            # number_non, number_left, number_right = np.append(number_non,len(pop_zero_correct)),np.append(number_left,len(pop_left_correct)),np.append(number_right,len(pop_right_correct))
            # number_non_ae, number_left_ae, number_right_ae = np.append(number_non_ae,len(pop_zero_error)), np.append(number_left_ae, len(pop_left_error)), np.append(number_right_ae,len(pop_right_error))
            
            number_non, number_rep, number_alt = np.append(number_non,len(pop_b_correct)),np.append(number_rep,len(pop_rep_correct)),np.append(number_alt,len(pop_alt_correct))
            number_non_ae, number_rep_ae, number_alt_ae = np.append(number_non_ae,len(pop_b_error)), np.append(number_rep_ae, len(pop_rep_error)), np.append(number_alt_ae,len(pop_alt_error))

    rects1 = ax.bar(x - width/2, number_non, width, label='non',facecolor='black')
    rects2 = ax.bar(x + width/2, number_non_ae, width, label='non(ae)',facecolor='gray')
    
    # rects1 = ax.bar(x - width/2, number_left, width,bottom=number_non,label='left',facecolor='green')
    # rects2 = ax.bar(x + width/2, number_left_ae, width, bottom=number_non_ae,label='left(ae)',facecolor='tab:green')
    
    # rects1 = ax.bar(x - width/2, number_right, width,bottom=number_left+number_non,label='right',facecolor='purple')
    # rects2 = ax.bar(x + width/2, number_right_ae, width, bottom=number_left_ae+number_non_ae,label='right(ae)',facecolor='tab:purple')
    
    rects1 = ax.bar(x - width/2, number_rep, width,bottom=number_non,label='left',facecolor='green')
    rects2 = ax.bar(x + width/2, number_rep_ae, width, bottom=number_non_ae,label='left(ae)',facecolor='tab:green')
    
    rects1 = ax.bar(x - width/2, number_alt, width,bottom=number_rep+number_non,label='right',facecolor='purple')
    rects2 = ax.bar(x + width/2, number_alt_ae, width, bottom=number_rep_ae+number_non_ae,label='right(ae)',facecolor='tab:purple')
    
    ax.set_ylabel('Number')
    ax.set_title('Neuron number by pop and rw')
    ax.set_xticks(x, labels)
    ax.set_yticks([0,200,400])
    # ax.legend()
    
    ax.set_xticklabels(['Rat7','Rat7','Rat15','Rat31','Rat32'])
    
    fig.tight_layout()
    
    plt.show()
    
    
def nine_subpops(dir, IDX_RAT, timep, NITERATIONS):
    dataname  = dir+IDX_RAT+'neuron_selectivity.npz'
    d_selectivity  = np.load(dataname, allow_pickle=True) 
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
    ### mixed-selectivity
    rep_left  = np.intersect1d(pop_left_correct,pop_rep_correct)
    rep_right = np.intersect1d(pop_right_correct,pop_rep_correct)
    alt_left  = np.intersect1d(pop_left_correct,pop_alt_correct)
    alt_right = np.intersect1d(pop_right_correct,pop_alt_correct)
    
    ### context -- mental state
    rep_only = np.intersect1d(pop_rep_correct,pop_zero_correct)
    alt_only = np.intersect1d(pop_alt_correct,pop_zero_correct)
    ### previous choice -- external signal
    left_only  = np.intersect1d(pop_left_correct,pop_b_correct)
    right_only = np.intersect1d(pop_right_correct,pop_b_correct)
    ### non-selectivity 
    correct_zero = np.intersect1d(pop_zero_correct,pop_b_correct)
    nn = len(nselect)
    
    # perc_neurons= [len(rep_left)/nn, len(rep_right)/nn, len(alt_left)/nn, len(alt_right)/nn, len(rep_only)/nn, len(alt_only)/nn, len(left_only)/nn, len(right_only)/nn, len(correct_zero)/nn]
    
    # dfall= pd.DataFrame.from_dict({'Selectivity': ['rep_left', 'rep_right', 'alt_left','alt_right', 'rep_only', 'alt_only', 'left_only', 'right_only', 'correct_zero'],'Neurons (%)':perc_neurons})
    # fig,ax = plt.subplots(figsize=(6,3))
    # ax = sns.barplot('Selectivity','Neurons (%)',data=dfall,ax=ax)
    # # ax.bar_label(ax.containers[0])
    # ax.set_ylim([0,0.25])
    # plt.xticks(rotation=-45)
    
    perc_neurons= [len(rep_left)/nn+len(rep_right)/nn+len(alt_left)/nn+len(alt_right)/nn, len(rep_only)/nn+ len(alt_only)/nn, len(left_only)/nn+len(right_only)/nn, len(correct_zero)/nn]
    
    dfall= pd.DataFrame.from_dict({'Selectivity': ['mixed', 'ctxt-only', 'prevch-only', 'zero'],'Neurons (%)':perc_neurons})
    fig,ax = plt.subplots(figsize=(3,3))
    ax = sns.barplot('Selectivity','Neurons (%)',data=dfall,ax=ax)
    # ax.bar_label(ax.containers[0])
    ax.set_ylim([0,0.5])
    plt.xticks(rotation=-45)
