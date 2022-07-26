
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


def beh_self_mtx(dir, IDX_RAT, timep, NITERATIONS):
    # ### ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat31_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_beh_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behae, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_beh_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behac, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    wiac_fix_hist,wiae_fix_hist = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
    wiac_fix_beh,wiae_fix_beh   = np.zeros((np.shape(wi_behac)[0],NITERATIONS)),np.zeros((np.shape(wi_behae)[0],NITERATIONS))
    for i in range(NITERATIONS):
        wiac_fix_hist[:,i],wiae_fix_hist[:,i] = wi_ac[:, i*5+3],wi_ae[:, i*5+3]
        wiac_fix_beh[:,i],wiae_fix_beh[:,i]   = wi_behac[:, i*5+4],wi_behae[:, i*5+4]

    ### Predicting upcoming stimulus 
    ### ~~~~~~~~~~~ individual iterations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(NITERATIONS):
        wiac_fix_hist[:,i] = wiac_fix_hist[:,i]/np.linalg.norm(wiac_fix_hist[:,i]) 
        wiae_fix_hist[:,i] = wiae_fix_hist[:,i]/np.linalg.norm(wiae_fix_hist[:,i])

    for i in range(NITERATIONS):
        wiac_fix_beh[:,i] = wiac_fix_beh[:,i]/np.linalg.norm(wiac_fix_beh[:,i]) 
        wiae_fix_beh[:,i] = wiae_fix_beh[:,i]/np.linalg.norm(wiae_fix_beh[:,i])

    decoders_fix_set = {}
    dec_names = ['hist-ac','hist-ae','beh-ac','beh-ae']
    decoders_fix_set['hist-ac'],decoders_fix_set['hist-ae'] = wiac_fix_hist.copy(),wiae_fix_hist.copy()
    decoders_fix_set['beh-ac'],decoders_fix_set['beh-ae']  = wiac_fix_beh.copy(), wiae_fix_beh.copy() 

    ag_fix_set = {}

    for i, selfpair in enumerate(dec_names):
        ag_fix_set[selfpair,selfpair] = (decoders_fix_set[selfpair].copy().T)@(decoders_fix_set[selfpair])
        ag_fix_set[selfpair,selfpair] = np.arccos(ag_fix_set[selfpair,selfpair])/np.pi*180 
        ag_fix_set[selfpair,selfpair]= ag_fix_set[selfpair,selfpair][~np.eye(ag_fix_set[selfpair,selfpair].shape[0],dtype=bool)].reshape(ag_fix_set[selfpair,selfpair].shape[0],-1)
        # print(ag_fix_set[selfpair,selfpair])

    # figure ploting -- distribution of bootstrap results
    fig_fix,ax_fix = plt.subplots(4,4, figsize=(8,8),tight_layout=True,sharex=True,sharey=True)
    BOX_WDTH = 0.25
    for i1, pair1 in enumerate (dec_names):
        df = {pair1+' v.s. '+pair1: ag_fix_set[pair1,pair1].flatten()}
        df = pd.DataFrame(df)
        sns.set(style='whitegrid')
        sns.boxplot(data=df,ax=ax_fix[i1,i1])
        ax_fix[i1,i1].set_ylim([30,150])
        ax_fix[i1,i1].set_yticks([30,90,150])
    for i, pair in enumerate(dec_names):
        ax_fix[i,0].set_ylabel(pair)
        ax_fix[0,i].set_title(pair)

def pairwise_mtx(dir, IDX_RAT, timep, NITERATIONS):
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
        # wiac_fix_hist[:,i], wiae_fix_hist[:,i]= wi_histac[:, i*6+3]-wi_histac[:,i*6+4],wi_histae[:, i*6+3]-wi_histae[:,i*6+4]
        wiac_fix_hist[:,i], wiae_fix_hist[:,i]= wi_histac[:, i*5+3],wi_histae[:, i*5+3]
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


    # timep = '/00-01s/'
    # ### compute the angles between history encoding axises    
    # dataname  = dir+timep+IDX_RAT+'data_beh.npz'
    # data_dec  = np.load(dataname, allow_pickle=True)
    # wi_comp_dc = data_dec['coefs']

    # wi_stim_stim = np.zeros((np.shape(wi_comp_dc)[0],NITERATIONS))
    # wi_stim_beh  = np.zeros((np.shape(wi_comp_dc)[0],NITERATIONS))
    # for i in range(NITERATIONS):
    #     wi_stim_stim[:,i] = wi_comp_dc[:, i*2+0]
    #     wi_stim_beh[:,i]  = wi_comp_dc[:, i*2+1]

    # ### Encoding received stimulus and determining behaviour
    # for i in range(NITERATIONS):
    #     wi_stim_stim[:,i] = wi_stim_stim[:,i]/np.linalg.norm(wi_stim_stim[:,i])
    #     wi_stim_beh[:,i]  = wi_stim_beh[:,i]/np.linalg.norm(wi_stim_beh[:,i])


    # decoders_stim_set = {}
    # dec_names = ['stim-stim','stim-beh']
    # decoders_stim_set['stim-stim'],decoders_stim_set['stim-beh'] = wi_stim_stim.copy(),wi_stim_beh.copy()

    # ag_stim_set = {}
    # for i1, pair1 in enumerate (dec_names):
    #     for i2, pair2 in enumerate (dec_names):
    #         if i2<=i1:
    #             continue 
    #         ag_stim_set[pair1,pair2] = (decoders_stim_set[pair1].copy().T)@(decoders_stim_set[pair2])
    #         ag_stim_set[pair1,pair2] = np.arccos(ag_stim_set[pair1,pair2])/np.pi*180 

    # for i, selfpair in enumerate(dec_names):
    #     ag_stim_set[selfpair,selfpair] = (decoders_stim_set[selfpair].copy().T)@(decoders_stim_set[selfpair])
    #     ag_stim_set[selfpair,selfpair] = np.arccos(ag_stim_set[selfpair,selfpair])/np.pi*180 

    # # figure ploting -- distribution of bootstrap results
    # fig_stim,ax_stim = plt.subplots(2,2, figsize=(4,4),tight_layout=True,sharex=True,sharey=True)
    # BOX_WDTH = 0.25
    # for i1, pair1 in enumerate (dec_names):
    #     for i2, pair2 in enumerate (dec_names):
    #         if i2<i1:
    #             continue 
    #         df = {pair1+' v.s. '+pair2: ag_stim_set[pair1,pair2].flatten()}
    #         df = pd.DataFrame(df)
    #         sns.set(style='whitegrid')
    #         sns.boxplot(data=df,ax=ax_stim[i1,i2])
    #         ax_stim[i1,i2].set_ylim([50,130])
    #         ax_stim[i1,i2].set_yticks([50,90,130])
    # for i, pair in enumerate(dec_names):
    #     ax_stim[i,0].set_ylabel(pair)
    #     ax_stim[0,i].set_title(pair)


def cluster_beh_acae(dir, IDX_RAT, timep, NITERATIONS, ncomps):
    # ### ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat31_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50


    dataname  = dir+timep+IDX_RAT+'data_dec.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_hist, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_beh_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behac, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']
    print(np.shape(wi_hist),np.shape(wi_behac))

    wi_fix_hist,wiac_fix_beh   = np.zeros((np.shape(wi_hist)[0],NITERATIONS)),np.zeros((np.shape(wi_behac)[0],NITERATIONS))
    for i in range(NITERATIONS):
        wi_fix_hist[:,i],wiac_fix_beh[:,i]   = wi_hist[:, i*5+3],wi_behac[:, i*5+4]

    ### Predicting upcoming stimulus 

    for i in range(NITERATIONS):
        wi_fix_hist[:,i] = wi_fix_hist[:,i]/np.linalg.norm(wi_fix_hist[:,i]) 
        wiac_fix_beh[:,i] = wiac_fix_beh[:,i]/np.linalg.norm(wiac_fix_beh[:,i])

    decoders_fix_set = {}
    decoders_fix_set['hist'],decoders_fix_set['beh-ac']  = wi_fix_hist.flatten(), wiac_fix_beh.flatten()
    vec_pair = np.zeros((len(decoders_fix_set['beh-ac']),2))
    vec_pair[:,0], vec_pair[:,1] = decoders_fix_set['hist'], decoders_fix_set['beh-ac']
    z_pop,model=gmm_fit(vec_pair, ncomps, algo='Bayes', n_init=50)


    # figure ploting -- distribution of bootstrap results
    fig_fix,ax_fix = plt.subplots(figsize=(4,4),tight_layout=True,sharex=True,sharey=True)
    for i in range(ncomps): 
        idx = np.where(z_pop==i)[0]
        ax_fix.scatter(decoders_fix_set['hist'][idx],decoders_fix_set['beh-ac'][idx],s=5,alpha=0.15)
        
    ax_fix.set_xlim([-0.2,0.2])
    ax_fix.set_ylim([-0.5,0.5])



def accuracy_mtx(dir, IDX_RAT, timep, NITERATIONS):
    ## ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ae_ctxt.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ae = data_dec['stats_error'][:]
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac_ctxt.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_hist_ac = data_dec['stats_correct'][:]
    
    dataname  = dir+timep+IDX_RAT+'data_beh_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_beh_ae = data_dec['stats_error']
    
    dataname  = dir+timep+IDX_RAT+'data_beh_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    score_fix_beh_ac = data_dec['stats_correct']
    
    
    accuracy_fix_set = {}
    accuracy_fix_set['hist-ac'], accuracy_fix_set['hist-ae']= score_fix_hist_ac.copy(), score_fix_hist_ae.copy() 
    accuracy_fix_set['beh-ac'], accuracy_fix_set['beh-ae']  = score_fix_beh_ac.flatten(), score_fix_beh_ae.flatten() 


    fig_score, ax_score=plt.subplots(1,2,figsize=(6,4),tight_layout=True, sharey=True)    
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

def ref_accuracy_mtx(dir, IDX_RAT, timep, NITERATIONS):
    ## ----------- history axis ------------------- 
    # dir = '/Users/yuxiushao/Public/DataML/Auditory/DataEphys'
    # IDX_RAT = 'Rat7_'
    # timep = '/-01-00s/'
    # NITERATIONS = 50
    ### compute the angles between history encoding axises
    dataname  = dir+timep+IDX_RAT+'data_dec_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']
    score_fix_hist_ae = data_dec['stats_error'][:]
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    score_fix_hist_ac = data_dec['stats_correct'][:]
    
    dataname  = dir+timep+IDX_RAT+'data_beh_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behae, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']
    
    REF_UNI_VEC  = np.ones((np.shape(wi_behae)[0]))
    REF_UNI_VEC  = REF_UNI_VEC/np.linalg.norm(REF_UNI_VEC)  
    
    score_fix_beh_ae = data_dec['stats_error']
    
    dataname  = dir+timep+IDX_RAT+'data_beh_ac.npz'
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


# ######## ***************** Three populations **************
# diff_act_values = p_values[:,3]-p_values[:,4]
# fig,ax=plt.subplots(figsize=(4,4))
# ax.hist(diff_act_values[pop_correct],facecolor='tab:blue',alpha=0.25)
# ax.hist(diff_act_values[pop_error],facecolor='tab:red',alpha=0.25)
# ax.hist(diff_act_values[pop_zero],facecolor='gray',alpha=0.25)
# ax.set_xlim([-0.8,0.8])
# ax.set_xticks([-0.8,0,0.8])

# pop_left_corr = np.setdiff1d(pop_left_correct,pop_left_error)
# print(len(pop_left_corr))
# pop_left_corr = np.setdiff1d(pop_left_correct,pop_zero_error)
# print(len(pop_left_corr))
# pop_right_corr = np.setdiff1d(pop_right_correct,pop_right_error)
# print(len(pop_right_corr))
# pop_right_corr = np.setdiff1d(pop_right_correct,pop_zero_error)
# print(len(pop_right_corr))

# fig,ax = plt.subplots(figsize=(4,3))
# ax.hist(p_values_correct[pop_correct,3]-p_values_correct[pop_correct,4],facecolor='tab:blue',alpha=0.25)
# ax.hist(p_values_correct[pop_error,3]-p_values_correct[pop_error,4],facecolor='tab:red',alpha=0.25)
# ax.hist(p_values_correct[pop_zero,3]-p_values_correct[pop_zero,4],facecolor='gray',alpha=0.25)

