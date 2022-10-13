import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.decomposition import PCA

from numpy import *
from numpy.random import rand, randn, randint
import itertools
import scipy.stats as sstats

'''
generate list and required quantities
'''

def req_quantities_0(stim_trials, stm, dyns, gt, choice, eff_choice,
                     rw, obsc, BLOCK_CTXT=0):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdata_trialidx = []

    Xconds_2 = []
    Xacts_1 = []
    Xrws_1 = []
    Xlfs_1 = []
    Xrse_6 = []
    rses = []
    Xacts_0 = []
    Xrws_0 = []
    Xgts_0 = []
    Xgts_1 = []
    Xcohs_0 = []
    Xcohs_1 = [] #### previous coherence -- strength of previous action?
    
    Xstates  = []
    nstats   = 4

    iiist = 2
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        #### there are only four conditions for previous 2 and previous 1 choices
        #### do not require consecutive correct transitions
        #### but we do not allow NaN reward and NaN previous choice?
        
        ### NaN trials 
        if(np.isnan(prev1_stats['rw']) or np.isnan(prev1_stats['choice']) or np.isnan(curr_stats['choice']) or np.isnan(curr_stats['rw'])):
            print('NaN trials to escape')
            continue
        if(prev1_stats['gt']<-1):
            print('ground truth NaN')
            continue
        if(prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
            Xdata_idx      = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
            Xdata_trialidx = np.append(Xdata_trialidx, i)

            nact = np.shape(curr_stats['resp'])[0]
            if (BLOCK_CTXT):
                # 'ctx'
                Xconds_2 = np.append(
                    Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                if(prev1_stats['rw']==0):
                    if(curr_stats['ctx']==0):#prev1_choice=-1
                        Xstates = np.append(Xstates,0)
                        # REP+L --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
                    if(curr_stats['ctx']==1):#prev1_choice=-1
                        Xstates = np.append(Xstates,2)
                        # ALT+L --> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)
                elif(prev1_stats['rw']==1):
                    if(curr_stats['ctx']==0):#prev1_choice=-1
                        Xstates = np.append(Xstates,0+nstats)
                        # REP+L --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
                    if(curr_stats['ctx']==1):#prev1_choice=-1
                        Xstates = np.append(Xstates,2+nstats)
                        # ALT+L --> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)

            else:
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['rw']==0):
                    Xstates = np.append(Xstates,0)
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['rw']==1):
                    Xstates = np.append(Xstates,0+nstats)
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                    
            if(prev1_stats['choice'] == -1):
                Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
            elif(prev1_stats['choice'] == 1):
                Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
            if(prev1_stats['rw'] == 0):
                Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
            elif(prev1_stats['rw'] == 1):
                Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
            # @YX 0110 ADD -- CALCULATE THE RSE
            if prev1_stats['choice'] == curr_stats['choice']:
                rses.append(1)
            else:
                rses.append(-1)

            # previous action (choice)
            r_tpre = prev1_stats['choice']  # -1 or 1
            # calculate hat{e}=e_t r_{t-1}
            gt_tcurr = curr_stats['gt']
            coh_tcurr = curr_stats['stim_coh'][0]
            e_tcurr = gt_tcurr*coh_tcurr
            # print(">>>>",gt_tcurr,coh_tcurr)
            # calculate RSE = hat{e}*pre_choice
            rsevid_curr = e_tcurr*r_tpre
            Xrse_6.append(rsevid_curr)
            Xcohs_0.append(coh_tcurr*gt_tcurr)
            Xcohs_1.append(prev1_stats['stim_coh'][0])
            # Xacts_0.append(curr_stats['choice'])
            if(curr_stats['choice'] == -1):
                Xacts_0.append(0)
            else:
                Xacts_0.append(1)
            Xrws_0.append(curr_stats['rw'])

            if(curr_stats['gt'] == -1):
                Xgts_0.append(0)
            else:
                Xgts_0.append(1)
            if(prev1_stats['gt'] == -1):
                Xgts_1.append(0)
            else:
                Xgts_1.append(1)
        # case 2
        elif(prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):
            # @YX 1909 note this is only the fixation
            Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
            Xdata_trialidx = np.append(Xdata_trialidx, i)

            nact = np.shape(curr_stats['resp'])[0]
            if (BLOCK_CTXT):
                # 'ctx'
                Xconds_2 = np.append(
                    Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                if(prev1_stats['rw']==0):
                    if(curr_stats['ctx']==0):#prev1_choice=1
                        Xstates = np.append(Xstates,1)
                        # REP+R--> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)
                    if(curr_stats['ctx']==1):#prev1_choice=1
                        Xstates = np.append(Xstates,3)
                        # ALT+R --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
                elif(prev1_stats['rw']==1):
                    if(curr_stats['ctx']==0):#prev1_choice=1
                        Xstates = np.append(Xstates,1+nstats)
                        # REP+R --> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)
                    if(curr_stats['ctx']==1):#prev1_choice=1
                        Xstates = np.append(Xstates,3+nstats)
                        # ALT+R --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
            else:
                # RRR#1*np.ones(nact),axis=0) ### RRR
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)
                if(prev1_stats['rw']==0):
                    Xstates = np.append(Xstates,1)
                    # REP+R --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['rw']==1):
                    Xstates = np.append(Xstates,1+nstats)
                    # REP+R --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    
            if(prev1_stats['choice'] == -1):
                # print(curr_stats['start_end'][0]-1)
                Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
              
            elif(prev1_stats['choice'] == 1):
                Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))

            if(prev1_stats['rw'] == 0):
                Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
            elif(prev1_stats['rw'] == 1):
                Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                # if(prev1_stats['choice']==-1):
                #     print(curr_stats['start_end'][0]-1)
            # @YX 0110 ADD -- CALCULATE THE RSE
            if prev1_stats['choice'] == curr_stats['choice']:

                rses.append(1)
            else:
                rses.append(-1)

            # previous action (choice)
            r_tpre = prev1_stats['choice']  # -1 or 1
            # calculate hat{e}=e_t r_{t-1}
            gt_tcurr = curr_stats['gt']
            coh_tcurr = curr_stats['stim_coh'][0]
            e_tcurr = gt_tcurr*coh_tcurr
            # calculate RSE = hat{e}*pre_choice
            rsevid_curr = e_tcurr*r_tpre
            Xrse_6.append(rsevid_curr)
            Xcohs_0.append(coh_tcurr*gt_tcurr)
            Xcohs_1.append(prev1_stats['stim_coh'][0])
            if(curr_stats['choice'] == -1):
                Xacts_0.append(0)
            else:
                Xacts_0.append(1)
            Xrws_0.append(curr_stats['rw'])
            if(curr_stats['gt'] == -1):
                Xgts_0.append(0)
            else:
                Xgts_0.append(1)
            if(prev1_stats['gt'] == -1):
                Xgts_1.append(0)
            else:
                Xgts_1.append(1)
        # case 3
        elif(prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):
            # @YX 1909 note this is only the fixation
            Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
            Xdata_trialidx = np.append(Xdata_trialidx, i)

            nact = np.shape(curr_stats['resp'])[0]
            if (BLOCK_CTXT):
                # 'ctx'
                Xconds_2 = np.append(
                    Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                if(prev1_stats['rw']==0):
                    if(curr_stats['ctx']==0):#prev1_choice=1
                        Xstates = np.append(Xstates,1)
                        # REP+R --> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)
                    if(curr_stats['ctx']==1):#prev1_choice=1
                        Xstates = np.append(Xstates,3)
                        # ALT+R --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
                elif(prev1_stats['rw']==1):
                    if(curr_stats['ctx']==0):#prev1_choice=1
                        Xstates = np.append(Xstates,1+nstats)
                        # REP+R --> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)
                    if(curr_stats['ctx']==1):#prev1_choice=1
                        Xstates = np.append(Xstates,3+nstats)
                        # ALT+R --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
            else:
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['rw']==0):
                    Xstates = np.append(Xstates,3)
                    # ALT+R --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['rw']==1):
                    Xstates = np.append(Xstates,3+nstats)
                    # ALT+R --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
            if(prev1_stats['choice'] == -1):
                Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
            elif(prev1_stats['choice'] == 1):
                # print(curr_stats['start_end'][0]-1)
                Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))

            if(prev1_stats['rw'] == 0):
                Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
            elif(prev1_stats['rw'] == 1):
                Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
            # @YX 0110 ADD -- CALCULATE THE RSE
            if prev1_stats['choice'] == curr_stats['choice']:

                rses.append(1)
            else:
                rses.append(-1)

            # previous action (choice)
            r_tpre = prev1_stats['choice']  # -1 or 1
            # calculate hat{e}=e_t r_{t-1}
            gt_tcurr = curr_stats['gt']
            coh_tcurr = curr_stats['stim_coh'][0]
            e_tcurr = gt_tcurr*coh_tcurr
            # calculate RSE = hat{e}*pre_choice
            rsevid_curr = e_tcurr*r_tpre
            Xrse_6.append(rsevid_curr)
            Xcohs_0.append(coh_tcurr*gt_tcurr)
            Xcohs_1.append(prev1_stats['stim_coh'][0])
            if(curr_stats['choice'] == -1):
                Xacts_0.append(0)
            else:
                Xacts_0.append(1)
            Xrws_0.append(curr_stats['rw'])
            if(curr_stats['gt'] == -1):
                Xgts_0.append(0)
            else:
                Xgts_0.append(1)
            if(prev1_stats['gt'] == -1):
                Xgts_1.append(0)
            else:
                Xgts_1.append(1)
        # case 4
        elif(prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):
            Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
            Xdata_trialidx = np.append(Xdata_trialidx, i)

            nact = np.shape(curr_stats['resp'])[0]
            if (BLOCK_CTXT):
                # 'ctx'
                Xconds_2 = np.append(
                    Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                if(prev1_stats['rw']==0):
                    if(curr_stats['ctx']==0):#prev1_choice=-1
                        Xstates = np.append(Xstates,0)
                        # REP+L --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
                    if(curr_stats['ctx']==1):#prev1_choice=-1
                        Xstates = np.append(Xstates,2)
                        # ALT+L --> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)
                elif(prev1_stats['rw']==1):
                    if(curr_stats['ctx']==0):#prev1_choice=-1
                        Xstates = np.append(Xstates,0+nstats)
                        # REP+L --> LEFT BIAS
                        Xacts_1 = np.append(Xacts_1, 0)
                    if(curr_stats['ctx']==1):#prev1_choice=-1
                        Xstates = np.append(Xstates,2+nstats)
                        # ALT+L --> RIGHT BIAS
                        Xacts_1 = np.append(Xacts_1, 1)
            else:
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['rw']==0):
                    Xstates = np.append(Xstates,2)
                    # ALT+L --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['rw']==1):
                    Xstates = np.append(Xstates,2+nstats)
                    # ALT+L --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
            if(prev1_stats['choice'] == -1):
                Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                # print(curr_stats['start_end'][0]-1)
            elif(prev1_stats['choice'] == 1):
                # print(curr_stats['start_end'][0]-1)
                Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))

            if(prev1_stats['rw'] == 0):
                Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
            elif(prev1_stats['rw'] == 1):
                Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
            # @YX 0110 ADD -- CALCULATE THE RSE
            if prev1_stats['choice'] == curr_stats['choice']:
                rses.append(1)
            else:
                rses.append(-1)

            # previous action (choice)
            r_tpre = prev1_stats['choice']  # -1 or 1
            # calculate hat{e}=e_t r_{t-1}
            gt_tcurr = curr_stats['gt']
            coh_tcurr = curr_stats['stim_coh'][0]
            e_tcurr = gt_tcurr*coh_tcurr
            # calculate RSE = hat{e}*pre_choice
            rsevid_curr = e_tcurr*r_tpre
            Xrse_6.append(rsevid_curr)
            Xcohs_0.append(coh_tcurr*gt_tcurr)
            Xcohs_1.append(prev1_stats['stim_coh'][0])
            if(curr_stats['choice'] == -1):
                Xacts_0.append(0)
            else:
                Xacts_0.append(1)
            Xrws_0.append(curr_stats['rw'])
            if(curr_stats['gt'] == -1):
                Xgts_0.append(0)
            else:
                Xgts_0.append(1)
            if(prev1_stats['gt'] == -1):
                Xgts_1.append(0)
            else:
                Xgts_1.append(1)
        # print('state -----',Xstates[-1],'xor--------',Xacts_1[-1])
    return Xdata, ydata, Xdata_idx,Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xgts_0, Xcohs_0, Xcohs_1,\
        Xdata_trialidx, Xstates

def req_quantities_3all(stim_trials, stm, dyns, gt, choice,
                        eff_choice, rw, obsc, BLOCK_CTXT=0):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx    = []
    Xdata_trialidx = []

    Xconds_2 = []
    Xacts_1  = []
    Xrws_1   = []
    Xrws_0   = []
    Xlfs_1   = []
    Xrse_6   = []
    rses     = []
    Xacts_0  = []
    Xgts_0   = []
    Xgts_1   = []
    Xcohs_0  = []

    Xstates  = []
    nstats   = 4

    iiist = 5
    for i in range(iiist, len(stim_trials)):
        curr_stats  = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        # 8 labels
        # ---- only use the correct-correct conditions?

        if (prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1):
            # LLLL
            if(prev4_stats['choice'] == -1 and prev3_stats['choice'] == -1
               and prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
                Xdata_idx      = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdata_trialidx = np.append(Xdata_trialidx, i)

                nact = np.shape(curr_stats['resp'])[0]
                if (BLOCK_CTXT):
                    # 'ctx'
                    Xconds_2 = np.append(
                        Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                else:
                    Xconds_2 = np.append(
                        Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1  = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1  = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xstates = np.append(Xstates,0)
                    Xrws_1  = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1  = np.append(Xrws_1, 1*np.ones(nact))
                    Xstates = np.append(Xstates,0+nstats)
                # @YX 0110 ADD -- CALCULATE THE RSE
                if prev1_stats['choice'] == curr_stats['choice']:
                    rses.append(1)
                else:
                    rses.append(-1)
                r_tpre = prev1_stats['choice']  # -1 or 1
                # calculate hat{e}=e_t r_{t-1}
                gt_tcurr = curr_stats['gt']
                coh_tcurr = curr_stats['stim_coh'][0]
                e_tcurr = gt_tcurr*coh_tcurr
                # print(">>>>",gt_tcurr,coh_tcurr)
                # calculate RSE = hat{e}*pre_choice
                rsevid_curr = e_tcurr*r_tpre
                Xrse_6.append(rsevid_curr)
                Xcohs_0.append(coh_tcurr*gt_tcurr)
                # Xacts_0.append(curr_stats['choice'])
                if(curr_stats['choice'] == -1):
                    Xacts_0.append(0)
                else:
                    Xacts_0.append(1)
                Xrws_0.append(curr_stats['rw'])
                if(curr_stats['gt']==-1):
                    Xgts_0.append(0)
                else:
                    Xgts_0.append(1)

                if(prev1_stats['gt'] == -1):
                    Xgts_1.append(0)
                else:
                    Xgts_1.append(1)
            # case 2
            #### RRRR
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdata_trialidx = np.append(Xdata_trialidx, i)

                nact = np.shape(curr_stats['resp'])[0]
                if (BLOCK_CTXT):
                    # 'ctx'
                    Xconds_2 = np.append(
                        Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                else:
                    # RRR#1*np.ones(nact),axis=0) ### RRR
                    Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L ---> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))

                if(prev1_stats['rw'] == 0):
                    Xstates = np.append(Xstates,1)
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xstates = np.append(Xstates,1+nstats)
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                # @YX 0110 ADD -- CALCULATE THE RSE
                if prev1_stats['choice'] == curr_stats['choice']:

                    rses.append(1)
                else:
                    rses.append(-1)

                # previous action (choice)
                r_tpre = prev1_stats['choice']  # -1 or 1
                # calculate hat{e}=e_t r_{t-1}
                gt_tcurr = curr_stats['gt']
                coh_tcurr = curr_stats['stim_coh'][0]
                e_tcurr = gt_tcurr*coh_tcurr
                # calculate RSE = hat{e}*pre_choice
                rsevid_curr = e_tcurr*r_tpre
                Xrse_6.append(rsevid_curr)
                Xcohs_0.append(coh_tcurr*gt_tcurr)
                if(curr_stats['choice'] == -1):
                    Xacts_0.append(0)
                else:
                    Xacts_0.append(1)
                Xrws_0.append(curr_stats['rw'])
                if(curr_stats['gt']==-1):
                    Xgts_0.append(0)
                else:
                    Xgts_0.append(1)
                if(prev1_stats['gt'] == -1):
                    Xgts_1.append(0)
                else:
                    Xgts_1.append(1)
            # case 3
            #### LRLR
            elif(prev4_stats['choice'] == -1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdata_trialidx = np.append(Xdata_trialidx, i)

                nact = np.shape(curr_stats['resp'])[0]
                if (BLOCK_CTXT):
                    # 'ctx'
                    Xconds_2 = np.append(
                        Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                else:
                    # LRL#2*np.ones(nact),axis=0) ### LRL
                    Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

                if(prev1_stats['rw'] == 0):
                    Xstates = np.append(Xstates,2)
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xstates = np.append(Xstates,2+nstats)
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))

                if prev1_stats['choice'] == curr_stats['choice']:

                    rses.append(1)
                else:
                    rses.append(-1)

                # previous action (choice)
                r_tpre = prev1_stats['choice']  # -1 or 1
                # calculate hat{e}=e_t r_{t-1}
                gt_tcurr = curr_stats['gt']
                coh_tcurr = curr_stats['stim_coh'][0]
                e_tcurr = gt_tcurr*coh_tcurr
                # calculate RSE = hat{e}*pre_choice
                rsevid_curr = e_tcurr*r_tpre
                Xrse_6.append(rsevid_curr)
                Xcohs_0.append(coh_tcurr*gt_tcurr)
                if(curr_stats['choice'] == -1):
                    Xacts_0.append(0)
                else:
                    Xacts_0.append(1)
                Xrws_0.append(curr_stats['rw'])
                if(curr_stats['gt']==-1):
                    Xgts_0.append(0)
                else:
                    Xgts_0.append(1)

                if(prev1_stats['gt'] == -1):
                    Xgts_1.append(0)
                else:
                    Xgts_1.append(1)
            # case 4
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == -1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdata_trialidx = np.append(Xdata_trialidx, i)

                nact = np.shape(curr_stats['resp'])[0]
                if (BLOCK_CTXT):
                    # 'ctx'
                    Xconds_2 = np.append(
                        Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
                else:
                    # RLR3*np.ones(nact),axis=0) ### RLR
                    Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

                if(prev1_stats['rw'] == 0):
                    Xstates = np.append(Xstates,3)
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xstates = np.append(Xstates,3+nstats)
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                # @YX 0110 ADD -- CALCULATE THE RSE
                if prev1_stats['choice'] == curr_stats['choice']:

                    rses.append(1)
                else:
                    rses.append(-1)

                # previous action (choice)
                r_tpre = prev1_stats['choice']  # -1 or 1
                # calculate hat{e}=e_t r_{t-1}
                gt_tcurr = curr_stats['gt']
                coh_tcurr = curr_stats['stim_coh'][0]
                e_tcurr = gt_tcurr*coh_tcurr
                # calculate RSE = hat{e}*pre_choice
                rsevid_curr = e_tcurr*r_tpre
                Xrse_6.append(rsevid_curr)
                Xcohs_0.append(coh_tcurr*gt_tcurr)
                if(curr_stats['choice'] == -1):
                    Xacts_0.append(0)
                else:
                    Xacts_0.append(1)
                Xrws_0.append(curr_stats['rw'])
                if(curr_stats['gt']==-1):
                    Xgts_0.append(0)
                else:
                    Xgts_0.append(1)
                if(prev1_stats['gt'] == -1):
                    Xgts_1.append(0)
                else:
                    Xgts_1.append(1)
            
    return Xdata, ydata, Xdata_idx, Xconds_2, Xacts_1,\
        Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0, Xgts_0,\
        Xcohs_0, Xdata_trialidx, Xstates


def sep_correct_error(stm, dyns, Xdata, ydata, Xdata_idx,  Xconds_2,
                      Xacts_1, Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0,
                      Xgts_0, Xcohs_0, Xcohs_1, Xdata_trialidx, Xstates,
                      margin=[1, 2], idd=1):
    ydata_bias    = Xrws_1*2+Xgts_0  # Xacts_0#  ### this 3
    ydata_xor     = Xrws_1*2+Xacts_1  # Xgts_0#Xacts_0#### this 2
    ydata_conds   = Xrws_1*2+Xconds_2  # this 1
    ydata_choices = Xrws_1*2+Xlfs_1  # this 0

    ydata_cchoices = Xrws_1*2+Xacts_0  # this 4
    ydata_cgts     = Xrws_1*2+Xgts_0

    rses    = np.array(rses)

    Xrse_6  = np.array(Xrse_6)
    Xcohs_0 = np.array(Xcohs_0)

    import scipy.stats as sstats
    Xdata = dyns.copy()
    # Xdata[idx_effect,:] = sstats.zscore(dyns[idx_effect,:],axis=0)

    correct_trial, error_trial = np.where(Xrws_1 > 0)[0].astype(np.int32), np.where(Xrws_1 < 1)[0].astype(np.int32)
    Xdata_correct = Xdata[Xdata_idx[correct_trial].astype(np.int32)+idd, :]
    # @YX 0110 add --- RSE ---------
    rses_correct = rses[correct_trial]
    Xrse_6_correct = Xrse_6[correct_trial]
    Xcohs_0_correct = Xcohs_0[correct_trial]
    Xdata_idx_correct = Xdata_idx[correct_trial]
    Xdata_trialidx_correct = Xdata_trialidx[correct_trial]

    ych_stim_correct = []
    ydata_bias_correct    = ydata_bias[correct_trial]
    ydata_xor_correct     = ydata_xor[correct_trial]
    ydata_conds_correct   = ydata_conds[correct_trial]
    ydata_choices_correct = ydata_choices[correct_trial]

    ydata_cchoices_correct = ydata_cchoices[correct_trial]
    ydata_cgts_correct     = ydata_cgts[correct_trial]

    ydata_states_correct   = Xstates[correct_trial]


    Xdata_error      = Xdata[Xdata_idx[error_trial].astype(np.int32)+idd, :]
    ych_stim_error   = []
    # @YX 0110 add --- RSE ---------
    rses_error       = rses[error_trial]
    Xrse_6_error     = Xrse_6[error_trial]
    Xcohs_0_error    = Xcohs_0[error_trial]
    Xdata_idx_error  = Xdata_idx[error_trial]
    Xdata_trialidx_error = Xdata_trialidx[error_trial]
    ydata_bias_error     = ydata_bias[error_trial]
    ydata_xor_error      = ydata_xor[error_trial]
    ydata_conds_error    = ydata_conds[error_trial]
    ydata_choices_error  = ydata_choices[error_trial]

    ydata_cchoices_error = ydata_cchoices[error_trial]
    ydata_cgts_error     = ydata_cgts[error_trial]

    ydata_states_error   = Xstates[error_trial]

    ac_ae_ratio = len(correct_trial)/len(error_trial)


    return ac_ae_ratio,Xdata_correct, Xdata_error,correct_trial, error_trial,rses_correct, rses_error, \
        Xrse_6_correct, Xrse_6_error, Xcohs_0_correct,\
        Xcohs_0_error, Xcohs_1_correct,Xcohs_1_error, ydata_bias_correct, ydata_bias_error, ydata_xor_correct,\
        ydata_xor_error, ydata_conds_correct, ydata_conds_error,\
        ydata_choices_correct, ydata_choices_error, ydata_cchoices_correct,\
        ydata_cchoices_error, ydata_cgts_correct, ydata_cgts_error,\
        Xdata_idx_correct, Xdata_idx_error,\
        Xdata_trialidx_correct, Xdata_trialidx_error, ydata_states_correct,ydata_states_error


def set_ylabels(Xdata,ydata_choices,ydata_conds,ydata_xor,ydata_bias,ydata_cchoices,Xcohs_0, Xcohs_1):
    ytruthlabels      = np.zeros((np.shape(Xdata)[0],3+1+1+1+1)) #### 13 Oct
    ytruthlabels[:, 0]= ydata_choices.copy()
    ytruthlabels[:, 1]= ydata_conds.copy()
    ytruthlabels[:, 2]= ydata_xor.copy()
    ytruthlabels[:, 3]= ydata_bias.copy()
    ytruthlabels[:, 4]= ydata_cchoices.copy()
    ytruthlabels[:, 5]= Xcohs_0.copy()
    # ytruthlabels[:, 6]= Xcohs_1.copy()


    return ytruthlabels

def State_trials(Xdata,Xstates,Xchoices,Xcohs,ylabels,percent,):
    ### different states
    unique_states = np.sort(np.unique(Xstates))
    unique_cohs   = [-1,0,1] ### stimulus coherence -- binning into three values
    # unique_cohs   = [-0.6,-0.25,0,0.25,0.6]
    Xdata_return  = {}
    ylabel_return = {}
 
    Xdata_hist_return  = {}
    ylabel_hist_return = {}
    unique_choices= np.sort(np.unique(Xchoices))
    Xcohs_b = Xcohs.copy()
    Xcohs_b[np.where(Xcohs<0)[0]]=-1
    Xcohs_b[np.where(Xcohs==0)[0]]=0
    Xcohs_b[np.where(Xcohs>0)[0]]=1
    # Xcohs_b[np.where(Xcohs<0.3)[0]]=-0.6
    # Xcohs_b[np.where(Xcohs==-0.2282)[0]]=-0.25
    # Xcohs_b[np.where(Xcohs==0)[0]]=0
    
    # Xcohs_b[np.where(Xcohs==0.2282)[0]]=0.25
    # Xcohs_b[np.where(Xcohs>0.3)[0]]=0.6

    Xcohs=Xcohs_b.copy()
    
    if (unique_choices[0]>1):
        unique_choices = unique_choices-2
        Xchoices = Xchoices-2

    for idxs, state in enumerate(unique_states):
        Xdata_hist_return[state] =[]
        ylabel_hist_return[state]=[]
        for idxc, coh in enumerate(unique_cohs):
            Xdata_return[state,coh,0] = []
            ylabel_return[state,coh,0]= []
            Xdata_return[state,coh,1] = []
            ylabel_return[state,coh,1]= []

    for idxs, state in enumerate(unique_states):
        idxstates = np.where(Xstates==state)[0]
        Xdata_hist_return[state]  = Xdata[idxstates,:]
        ylabel_hist_return[state] = ylabels[idxstates,:]
        for idxc, coh in enumerate(unique_cohs):
            idxcohs   = np.where(Xcohs==coh)[0]
            idx       = np.intersect1d(idxstates,idxcohs)
            for choice in unique_choices:
                ichoices = np.where(Xchoices==choice)[0]
                idxf = np.intersect1d(idx,ichoices)
                if(np.shape(Xdata_return[state,coh,choice])[0]==0):
                    Xdata_return[state,coh,choice]=Xdata[idxf,:]
                    ylabel_return[state,coh,choice]=ylabels[idxf,:]
                elif(np.shape(Xdata_return[state,coh,choice])[0]>0):
                    Xdata_return[state,coh,choice]=np.vstack(Xdata_return[state,coh,choice],Xdata[idxf,:])
                    ylabel_return[state,coh,choice]=np.vstack(ylabel_return[state,coh,choice],ylabels[idxf,:])

    return Xdata_return,ylabel_return,Xdata_hist_return,ylabel_hist_return