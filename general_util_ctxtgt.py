'''
@YX 30 OCT 
Use the contextual information from block-level ground-truth.
'''
import numpy as np


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
    ctx = data['contexts']
    gt  = data['gt']
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
        else:
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
