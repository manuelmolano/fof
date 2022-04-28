'''
@YX 30 OCT 
Use the contextual information from block-level ground-truth.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.decomposition import PCA

from numpy import *
from numpy.random import rand, randn, randint
import itertools

def get_RNNdata_ctxtgt(data,*kwargs): 
    # ctx = data['contexts']
    # gt  = data['gt']
    # stm = data['stimulus']
    # dyns =data['states']
    # choice=data['choice']
    # eff_choice=data['prev_choice']
    # rw  = data['reward']
    # obsc = data['obscategory']
    # ### design time clock
    # nt = len(dyns[:,0])
    # tt = np.arange(1,nt,1)
    # return tt,stm,dyns,ctx,gt,choice,eff_choice,rw,obsc
    ctx = data['contexts']
    gt  = data['gt']
    stm = data['stimulus']
    dyns =data['states']
    choice=data['choice']
    eff_choice=data['prev_choice']
    rw  = data['reward']
    obsc = data['obscategory']
    ### design time clock
    nt = len(dyns[:,0])
    tt = np.arange(1,nt,1)
    return tt,stm,dyns,ctx,gt,choice,eff_choice,rw,obsc

#### generate stim_trials
def transform_stim_trials_ctxtgt(tt,stm,dyns,ctx,gt,choice,eff_choice,rw,obsc,*kwargs):
	#### deal with contexts
    ctxseq=[]
    for i in range(len(gt)):
        if(ctx[i][0]=='2'):
            ctxseq.append(1) ### alternating
        elif(ctx[i][0]=='1'):
            ctxseq.append(0) ### repeating
    ctxseq=np.array(ctxseq)

    #### generate stimulus-structure
    # ground-truth as a bench mark
    ngt_tot=np.where(gt>0)[0]
    stim_trials={}
    ctxt_trials=[]
    for idx,igt in enumerate(ngt_tot[:len(ngt_tot)-2]):
        stim_trials[idx]={'stim':stm[ngt_tot[idx]+1:ngt_tot[idx+1]-1,1:3],
                            'stim_coh':obsc[ngt_tot[idx]+1:ngt_tot[idx+1]+1],
                            'ctx':ctxseq[ngt_tot[idx+1]],
                            'gt':gt[ngt_tot[idx+1]],
                            'resp':dyns[ngt_tot[idx]+1:ngt_tot[idx]+2,:],
                            'choice':eff_choice[ngt_tot[idx+1]+1],
                            'rw':rw[ngt_tot[idx+1]],
                            'start_end':np.array([igt+1,ngt_tot[idx+1]]),
        }
        ctxt_trials=np.append(ctxt_trials,stim_trials[idx]['ctx'])

    #### action (-1 and +1) -- left and right 
    for i in range(len(stim_trials)):
        if (stim_trials[i]['choice']==1):
            stim_trials[i]['choice']=-1;
        else:
            stim_trials[i]['choice']=1;
    #### stimulus has direction (-1, +1) direct to left and right
    for i in range(len(stim_trials)):
        if (stim_trials[i]['gt']==1):
            stim_trials[i]['gt']=-1;
        else:
            stim_trials[i]['gt']=1;

    #### extract the unique coherence value
    cohvalues = [0,0,0,0,0]
    cohvalues=sort(cohvalues)

    return stim_trials,cohvalues,ctxt_trials

def req_quantities_ctxtgt(stim_trials,cohvalues,stm,dyns,gt,choice,eff_choice,rw,obsc):
    #### ---- prepare LDA data
    Xdata,ydata = [],[]
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = [] ### shuffle
    # Xdata = dyns[stim_trials[0]['start_end'][0]:stim_trials[len(stim_trials)-1]['start_end'][1]+1,:]
    # ydata=[]
    Xconds_2 = []
    #### @YX 30OCT ctxt
    Xctxt_1  = []
    Xacts_1  = []
    Xrws_1   = []
    Xlfs_1   = []
    Xrse_6   = []
    rses = []
    Xacts_0  = []
    Xrws_0   = []
    Xgts_0   = []
    Xcohs_0  = []

    margin   = [1,2] ## before and after
    for i in range(4,len(stim_trials)):
        curr_stats  = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        #### no matter what conditions
        #### @YX 1909 note this is only the fixation
        Xdc_idx_0 = np.append(Xdc_idx_0,curr_stats['start_end'][1]-1)
        Xdata_idx = np.append(Xdata_idx,curr_stats['start_end'][0]-1)
        #### extract the block-level context information
        Xctxt_1   = np.append(Xctxt_1,prev1_stats['ctx'])
        #### correct-error ONE-HOT
        if(i==4):
            Xce_OH = np.zeros((4,1))
            for j in range(4):
                Xce_OH[j,0]=stim_trials[i-j-1]['rw']

        elif(i>4):
            ONEHOT = np.zeros((4,1))
            for j in range(4):
                ONEHOT[j,0]=stim_trials[i-j-1]['rw']
            Xce_OH = np.append(Xce_OH,ONEHOT,axis=1)

        if(prev1_stats['ctx']==0): ### REP
            nact      = np.shape(curr_stats['resp'])[0]
            Xconds_2  = np.append(Xconds_2,0*np.ones(nact),axis=0) ### LLL
            if(prev1_stats['choice']==-1):
                Xlfs_1  = np.append(Xlfs_1,0*np.ones(nact))
                Xacts_1 = np.append(Xacts_1,0*np.ones(nact)) ### REP+L --> LEFT BIAS
            elif(prev1_stats['choice']==1):
                Xlfs_1  = np.append(Xlfs_1,1*np.ones(nact))
                Xacts_1 = np.append(Xacts_1,1*np.ones(nact))
            if(prev1_stats['rw']==0):
                Xrws_1  = np.append(Xrws_1,0*np.ones(nact))
            elif(prev1_stats['rw']==1):
                Xrws_1  = np.append(Xrws_1,1*np.ones(nact))
            if prev1_stats['choice']==curr_stats['choice']:
                repeatevid = 1
                rses.append(1)
            else:
                rses.append(-1)
                repeatevid = -1
            #### previous action (choice)
            r_tpre    = prev1_stats['choice'] ### -1 or 1
            #### calculate hat{e}=e_t r_{t-1}
            gt_tcurr  = curr_stats['gt']
            coh_tcurr = curr_stats['stim_coh'][0]
            e_tcurr   = gt_tcurr*coh_tcurr

            rsevid_curr=e_tcurr*r_tpre
            Xrse_6.append(rsevid_curr)
            Xcohs_0.append(coh_tcurr*gt_tcurr)
            if(curr_stats['choice']==-1):
                Xacts_0.append(0)
            else:
                Xacts_0.append(1)
            Xrws_0.append(curr_stats['rw'])

            if(curr_stats['gt']==-1):
                Xgts_0.append(0)
            else:
                Xgts_0.append(1)

        ### case 3
        elif(prev1_stats['ctx']==1): ### ALT
            nact = np.shape(curr_stats['resp'])[0]
            Xconds_2 = np.append(Xconds_2,1*np.ones(nact),axis=0) ### LRL#2*np.ones(nact),axis=0) ### LRL
            if(prev1_stats['choice']==-1):
                Xlfs_1  = np.append(Xlfs_1,0*np.ones(nact))
                Xacts_1 = np.append(Xacts_1,1*np.ones(nact)) ### ALT+LEFT --> RIGHT BIAS
            elif(prev1_stats['choice']==1):
                Xlfs_1  = np.append(Xlfs_1,1*np.ones(nact))
                Xacts_1 = np.append(Xacts_1,0*np.ones(nact))

            if(prev1_stats['rw']==0):
                Xrws_1 = np.append(Xrws_1,0*np.ones(nact))
            elif(prev1_stats['rw']==1):
                Xrws_1 = np.append(Xrws_1,1*np.ones(nact))
            if prev1_stats['choice']==curr_stats['choice']:
                repeatevid = 1
                rses.append(1)
            else:
                rses.append(-1)
                repeatevid = -1
            #### previous action (choice)
            r_tpre    = prev1_stats['choice'] ### -1 or 1
            #### calculate hat{e}=e_t r_{t-1}
            gt_tcurr  = curr_stats['gt']
            coh_tcurr = curr_stats['stim_coh'][0]
            e_tcurr   = gt_tcurr*coh_tcurr
            #### calculate RSE = hat{e}*pre_choice
            rsevid_curr=e_tcurr*r_tpre
            Xrse_6.append(rsevid_curr)
            Xcohs_0.append(coh_tcurr*gt_tcurr)
            if(curr_stats['choice']==-1):
                Xacts_0.append(0)
            else:
                Xacts_0.append(1)
            Xrws_0.append(curr_stats['rw'])
            if(curr_stats['gt']==-1):
                Xgts_0.append(0)
            else:
                Xgts_0.append(1)

    return Xdata,ydata,Xdata_idx,Xdata_resp,Xconds_2,Xacts_1,Xrws_1,Xlfs_1,Xrse_6,\
    rses,Xacts_0,Xrws_0,Xgts_0,Xcohs_0,Xdc_idx_0,Xctxt_1,Xce_OH
# def req_quantities_ctxtgt(stim_trials,cohvalues,stm,dyns,gt,choice,eff_choice,rw,obsc):
    #### ---- prepare LDA data
    Xdata,ydata = [],[]
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = [] ### shuffle
    # Xdata = dyns[stim_trials[0]['start_end'][0]:stim_trials[len(stim_trials)-1]['start_end'][1]+1,:]
    # ydata=[]
    Xconds_2 = []
    #### @YX 30OCT ctxt
    Xctxt_1  = []
    Xacts_1  = []
    Xrws_1   = []
    Xlfs_1   = []
    Xrse_6   = []
    rses = []
    Xacts_0  = []
    Xrws_0   = []
    Xgts_0   = []
    Xcohs_0  = []

    flag_setup=0
    margin   = [1,2] ## before and after
    for i in range(4,len(stim_trials)):
        curr_stats  = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        #### no matter what conditions
        #### @YX 1909 note this is only the fixation
        if(prev2_stats['rw']==1 and prev3_stats['rw']==1 and prev4_stats['rw']==1):###prev1_stats['rw']==1 and
            if((prev4_stats['choice']==-1 and prev3_stats['choice']==-1 and prev2_stats['choice']==-1) or       (prev4_stats['choice']==1 and prev3_stats['choice']==1 and prev2_stats['choice']==1)):
                Xdc_idx_0 = np.append(Xdc_idx_0,curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx,curr_stats['start_end'][0]-1)
                #### extract the block-level context information
                Xctxt_1   = np.append(Xctxt_1,prev1_stats['ctx'])
                #### correct-error ONE-HOT
                if(flag_setup==0):
                    Xce_OH = np.zeros((4,1))
                    flag_setup=1
                    for j in range(4):
                        Xce_OH[j,0]=stim_trials[i-j-1]['rw']

                elif(flag_setup==1):
                    ONEHOT = np.zeros((4,1))
                    for j in range(4):
                        ONEHOT[j,0]=stim_trials[i-j-1]['rw']
                    Xce_OH = np.append(Xce_OH,ONEHOT,axis=1)

                nact      = np.shape(curr_stats['resp'])[0]
                Xconds_2  = np.append(Xconds_2,0*np.ones(nact),axis=0) ### LLL
                if(prev1_stats['choice']==-1):
                    Xlfs_1  = np.append(Xlfs_1,0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1,0*np.ones(nact)) ### REP+L --> LEFT BIAS
                elif(prev1_stats['choice']==1):
                    Xlfs_1  = np.append(Xlfs_1,1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1,1*np.ones(nact))
                if(prev1_stats['rw']==0):
                    Xrws_1  = np.append(Xrws_1,0*np.ones(nact))
                elif(prev1_stats['rw']==1):
                    Xrws_1  = np.append(Xrws_1,1*np.ones(nact))
                if prev1_stats['choice']==curr_stats['choice']:
                    repeatevid = 1
                    rses.append(1)
                else:
                    rses.append(-1)
                    repeatevid = -1
                #### previous action (choice)
                r_tpre    = prev1_stats['choice'] ### -1 or 1
                #### calculate hat{e}=e_t r_{t-1}
                gt_tcurr  = curr_stats['gt']
                coh_tcurr = curr_stats['stim_coh'][0]
                e_tcurr   = gt_tcurr*coh_tcurr

                rsevid_curr=e_tcurr*r_tpre
                Xrse_6.append(rsevid_curr)
                Xcohs_0.append(coh_tcurr*gt_tcurr)
                if(curr_stats['choice']==-1):
                    Xacts_0.append(0)
                else:
                    Xacts_0.append(1)
                Xrws_0.append(curr_stats['rw'])

                if(curr_stats['gt']==-1):
                    Xgts_0.append(0)
                else:
                    Xgts_0.append(1)

            if((prev4_stats['choice']==-1 and prev3_stats['choice']==1 and prev2_stats['choice']==-1) or     (prev4_stats['choice']==1 and prev3_stats['choice']==-1 and prev2_stats['choice']==1)):

                Xdc_idx_0 = np.append(Xdc_idx_0,curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx,curr_stats['start_end'][0]-1)
                #### extract the block-level context information
                Xctxt_1   = np.append(Xctxt_1,prev1_stats['ctx'])
                #### correct-error ONE-HOT
                if(flag_setup==0):
                    Xce_OH = np.zeros((4,1))
                    flag_setup=1
                    for j in range(4):
                        Xce_OH[j,0]=stim_trials[i-j-1]['rw']

                elif(flag_setup==1):
                    ONEHOT = np.zeros((4,1))
                    for j in range(4):
                        ONEHOT[j,0]=stim_trials[i-j-1]['rw']
                    Xce_OH = np.append(Xce_OH,ONEHOT,axis=1)
                    
                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2,1*np.ones(nact),axis=0) ### LRL#2*np.ones(nact),axis=0) ### LRL
                if(prev1_stats['choice']==-1):
                    Xlfs_1  = np.append(Xlfs_1,0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1,1*np.ones(nact)) ### ALT+LEFT --> RIGHT BIAS
                elif(prev1_stats['choice']==1):
                    Xlfs_1  = np.append(Xlfs_1,1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1,0*np.ones(nact))

                if(prev1_stats['rw']==0):
                    Xrws_1 = np.append(Xrws_1,0*np.ones(nact))
                elif(prev1_stats['rw']==1):
                    Xrws_1 = np.append(Xrws_1,1*np.ones(nact))
                if prev1_stats['choice']==curr_stats['choice']:
                    repeatevid = 1
                    rses.append(1)
                else:
                    rses.append(-1)
                    repeatevid = -1
                #### previous action (choice)
                r_tpre    = prev1_stats['choice'] ### -1 or 1
                #### calculate hat{e}=e_t r_{t-1}
                gt_tcurr  = curr_stats['gt']
                coh_tcurr = curr_stats['stim_coh'][0]
                e_tcurr   = gt_tcurr*coh_tcurr
                #### calculate RSE = hat{e}*pre_choice
                rsevid_curr=e_tcurr*r_tpre
                Xrse_6.append(rsevid_curr)
                Xcohs_0.append(coh_tcurr*gt_tcurr)
                if(curr_stats['choice']==-1):
                    Xacts_0.append(0)
                else:
                    Xacts_0.append(1)
                Xrws_0.append(curr_stats['rw'])
                if(curr_stats['gt']==-1):
                    Xgts_0.append(0)
                else:
                    Xgts_0.append(1)

    return Xdata,ydata,Xdata_idx,Xdata_resp,Xconds_2,Xacts_1,Xrws_1,Xlfs_1,Xrse_6,\
    rses,Xacts_0,Xrws_0,Xgts_0,Xcohs_0,Xdc_idx_0,Xctxt_1,Xce_OH


#### separate correct trials and error trials
def sep_correct_error_ctxtgt(stm,dyns,Xdata,ydata,Xdata_idx,Xdata_resp,Xctxt_1,Xce_OH,Xacts_1,Xrws_1,Xlfs_1,Xrse_6,rses,Xacts_0,Xrws_0,Xgts_0,Xcohs_0,Xdc_idx_0,margin=[1,2],idd=1):
    # margin=[1,2]
    ydata_bias    = Xrws_1*2+Xacts_1
    ydata_conds   = Xrws_1*2+Xctxt_1
    ydata_choices = Xrws_1*2+Xlfs_1

    ydata_cchoices= Xrws_1*2+Xacts_0
    ydata_cgts    = Xrws_1*2+Xgts_0
    ydata_crws    = Xrws_1*2+Xrws_0

    ## @YX 30OCT background
    ydata_bground = np.repeat(np.reshape(Xrws_1,(1,-1)),4,axis=0)
    ydata_bground = ydata_bground+Xce_OH

    rses=np.array(rses)

    Xrse_6 = np.array(Xrse_6)
    Xcohs_0 = np.array(Xcohs_0)

    import scipy.stats as sstats
    Xdata = sstats.zscore(dyns[:,int(dyns.shape[1]/2):],axis=0)

    correct_trial,error_trial= np.where(Xrws_1>0)[0].astype(int32),np.where(Xrws_1<1)[0].astype(int32)
    Xdata_correct = Xdata[Xdata_idx[correct_trial].astype(int32)+idd,:]
    #### @YX 0110 add --- RSE ---------
    rses_correct    = rses[correct_trial]
    Xrse_6_correct  = Xrse_6[correct_trial]
    Xcohs_0_correct = Xcohs_0[correct_trial]
    ### Xdata resp for stimulus
    Xdata_stim_correct   = Xdata[Xdata_idx[correct_trial].astype(int32)+1,:]
    Xdata_dc_correct     = Xdata[Xdc_idx_0[correct_trial].astype(int32),:]
    ych_stim_correct     = stm[Xdata_idx[correct_trial].astype(int32)+1,0]-\
        stm[Xdata_idx[correct_trial].astype(int32)+1,1]
    ydata_bias_correct    = ydata_bias[correct_trial]
    ydata_conds_correct   = ydata_conds[correct_trial]
    ydata_choices_correct = ydata_choices[correct_trial]

    ydata_cchoices_correct= ydata_cchoices[correct_trial]
    ydata_cgts_correct    = ydata_cgts[correct_trial]
    ydata_crws_correct    = ydata_crws[correct_trial]

    ydata_bground_correct = ydata_bground[:,correct_trial]


    Xdata_correct_pre     = np.zeros((margin[0],Xdata_correct.shape[0],Xdata_correct.shape[1]))
    for i in range(margin[0]):
        Xdata_correct_pre[i,:,:]=Xdata[Xdata_idx[correct_trial].astype(int32)-margin[0]+i,:].copy()
    Xdata_correct_post = np.zeros((margin[1],Xdata_correct.shape[0],Xdata_correct.shape[1]))
    for i in range(margin[1]):
        Xdata_correct_post[i,:,:]=Xdata[Xdata_idx[correct_trial].astype(int32)+i,:].copy()
    Xdata_correct_seq = np.zeros((len(correct_trial)*(margin[0]+margin[1]+1),Xdata.shape[1]))
    len_period = margin[0]+margin[1]+1
    for i in range(margin[0]):
        Xdata_correct_seq[i::len_period,:]=Xdata_correct_pre[i,:,:]
    Xdata_correct_seq[margin[0]::len_period,:]=Xdata_correct[:,:]
    for i in range(margin[1]):
        Xdata_correct_seq[margin[0]+1+i::len_period,:]=Xdata_correct_post[i,:,:]

    Xdata_error      = Xdata[Xdata_idx[error_trial].astype(int32)+idd,:]
    ### Xdata resp for stimulus
    Xdata_stim_error = Xdata[Xdata_idx[error_trial].astype(int32)+1,:]
    Xdata_dc_error   = Xdata[Xdc_idx_0[error_trial].astype(int32),:]
    # print('decision: ',Xdc_idx_0[error_trial[::200]],' stim: ',Xdata_idx[error_trial[::200]]+idd)
    ych_stim_error = stm[Xdata_idx[error_trial].astype(int32)+1,0]-\
        stm[Xdata_idx[error_trial].astype(int32)+1,1]

    #### @YX 0110 add --- RSE ---------
    rses_error   = rses[error_trial]
    Xrse_6_error = Xrse_6[error_trial]
    Xcohs_0_error = Xcohs_0[error_trial]

    ydata_bias_error   = ydata_bias[error_trial]
    ydata_conds_error  = ydata_conds[error_trial]
    ydata_choices_error= ydata_choices[error_trial]

    ydata_cchoices_error=ydata_cchoices[error_trial]
    ydata_cgts_error    = ydata_cgts[error_trial]
    ydata_crws_error   = ydata_crws[error_trial]

    ydata_bground_error = ydata_bground[:,error_trial]


    Xdata_error_pre  = np.zeros((margin[0],Xdata_error.shape[0],Xdata_error.shape[1]))
    for i in range(margin[0]):
        Xdata_error_pre[i,:,:]=Xdata[Xdata_idx[error_trial].astype(int32)-margin[0]+i,:].copy()
    Xdata_error_post = np.zeros((margin[1],Xdata_error.shape[0],Xdata_error.shape[1]))
    for i in range(margin[1]):
        Xdata_error_post[i,:,:]=Xdata[Xdata_idx[error_trial].astype(int32)+i,:].copy()

    Xdata_error_seq = np.zeros((len(error_trial)*(margin[0]+margin[1]+1),Xdata.shape[1]))
    for i in range(margin[0]):
        Xdata_error_seq[i::len_period,:]=Xdata_error_pre[i,:,:]
    Xdata_error_seq[margin[0]::len_period,:]=Xdata_error[:,:]
    for i in range(margin[1]):
        Xdata_error_seq[margin[0]+1+i::len_period,:]=Xdata_error_post[i,:,:]

    return Xdata_correct,correct_trial,Xdata_stim_correct,Xdata_dc_correct,\
        ych_stim_correct,Xdata_correct_seq,Xdata_error,error_trial,Xdata_stim_error, Xdata_dc_error,\
            ych_stim_error,Xdata_error_seq,rses_correct,rses_error,Xrse_6_correct,Xrse_6_error,Xcohs_0_correct,\
                Xcohs_0_error,ydata_bias_correct,ydata_bias_error,ydata_conds_correct,ydata_conds_error,\
                    ydata_choices_correct,ydata_choices_error,ydata_cchoices_correct,ydata_cchoices_error,\
                        ydata_cgts_correct,ydata_cgts_error,ydata_crws_correct,ydata_crws_error,\
                            ydata_bground_correct,ydata_bground_error


def calculate_dprime(Xdata, ylabel):
    uniques = np.unique(ylabel)
    if len(uniques)==1:
        return 10000
    means,sigmas = np.zeros(len(uniques)),np.zeros(len(uniques))
    
    for i in range(len(uniques)):
        means[i]=np.mean(Xdata[np.where(ylabel[:]==uniques[i])[0]])
        sigmas[i]=np.std(Xdata[np.where(ylabel[:]==uniques[i])[0]])
    if(sigmas[0]==0):
        print("-------",means[0],sigmas[0])    
    dprimes = len(uniques)*(means[1]-means[0])**2/(sigmas[0]**2+sigmas[1]**2)
    return dprimes