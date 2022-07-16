
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
    wiac_fix_beh,wiae_fix_beh   = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
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
    fig_fix,ax_fix = plt.subplots(4,4, figsize=(8,8),tight_layout=True,sharex=True,sharey=True)
    BOX_WDTH = 0.25
    for i1, pair1 in enumerate (dec_names):
        for i2, pair2 in enumerate (dec_names):
            if i2<i1:
                continue 
            df = {pair1+' v.s. '+pair2: ag_fix_set[pair1,pair2].flatten()}
            df = pd.DataFrame(df)
            sns.set(style='whitegrid')
            sns.boxplot(data=df,ax=ax_fix[i1,i2])
            ax_fix[i1,i2].set_ylim([30,150])
            ax_fix[i1,i2].set_yticks([30,90,150])
    for i, pair in enumerate(dec_names):
        ax_fix[i,0].set_ylabel(pair)
        ax_fix[0,i].set_title(pair)


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



    dataname  = dir+timep+IDX_RAT+'data_beh_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behae, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    dataname  = dir+timep+IDX_RAT+'data_beh_ae.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_behac, _ = data_dec['coefs_correct'], data_dec['intercepts_correct']

    wiac_fix_beh,wiae_fix_beh   = np.zeros((np.shape(wi_behae)[0],NITERATIONS)),np.zeros((np.shape(wi_behae)[0],NITERATIONS))
    for i in range(NITERATIONS):
        wiac_fix_beh[:,i],wiae_fix_beh[:,i]   = wi_behac[:, i*5+4],wi_behae[:, i*5+4]

    ### Predicting upcoming stimulus 

    for i in range(NITERATIONS):
        wiac_fix_beh[:,i] = wiac_fix_beh[:,i]/np.linalg.norm(wiac_fix_beh[:,i]) 
        wiae_fix_beh[:,i] = wiae_fix_beh[:,i]/np.linalg.norm(wiae_fix_beh[:,i])

    decoders_fix_set = {}
    decoders_fix_set['beh-ac'],decoders_fix_set['beh-ae']  = np.mean(wiac_fix_beh.copy(),axis=1), np.mean(wiae_fix_beh.copy(),axis=1) 
    vec_pair = np.zeros((len(decoders_fix_set['beh-ac']),2))
    vec_pair[:,0], vec_pair[:,1] = decoders_fix_set['beh-ac'], decoders_fix_set['beh-ae']
    z_pop,model=gmm_fit(vec_pair, ncomps, algo='em', n_init=50)


    # figure ploting -- distribution of bootstrap results
    fig_fix,ax_fix = plt.subplots(1,2,figsize=(10,5),tight_layout=True,sharex=True,sharey=True)
    idx1,idx2 = np.where(z_pop==0)[0],np.where(z_pop==1)[0]
    
    ax_fix[0].scatter(decoders_fix_set['beh-ac'][idx1],decoders_fix_set['beh-ae'][idx1],s=10,c='tab:blue',alpha=0.25)
    ax_fix[1].scatter(decoders_fix_set['beh-ac'][idx2],decoders_fix_set['beh-ae'][idx2],s=10,c='tab:blue',alpha=0.25)




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
    score_fix_hist_ae = data_dec['stats_error'][3,:]
    
    dataname  = dir+timep+IDX_RAT+'data_dec_ac.npz'
    data_dec  = np.load(dataname, allow_pickle=True)
    wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']
    score_fix_hist_ac = data_dec['stats_correct'][3,:]
    
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
    accuracy_fix_set['beh-ac'], accuracy_fix_set['beh-ae']  = score_fix_beh_ac.copy(), score_fix_beh_ae.copy() 
    
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
