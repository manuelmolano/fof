import numpy as np

'''
generate list and required quantities
'''


def req_quantities_congruent7(stim_trials, cohvalues, stm, dyns, gt, choice,
                              eff_choice, rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle
    # ydata=[]
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

    iiist = 8
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        prev5_stats = stim_trials[i-5]
        prev6_stats = stim_trials[i-6]
        prev7_stats = stim_trials[i-7]
        prev8_stats = stim_trials[i-8]

        # 8 labels
        # ---- only use the correct-correct conditions?

        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1 and prev5_stats['rw'] == 1 and
           prev6_stats['rw'] == 1 and prev7_stats['rw'] == 1 and
           prev8_stats['rw'] == 1):
            # case 1
            if(prev8_stats['choice'] == -1 and prev7_stats['choice'] == -1 and
               prev6_stats['choice'] == -1 and prev5_stats['choice'] == -1 and
               prev4_stats['choice'] == -1 and prev3_stats['choice'] == -1
               and prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 1):
                    continue
                # @YX 1909 note this is only the fixation
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev8_stats['choice'] == 1 and prev7_stats['choice'] == 1 and
                 prev6_stats['choice'] == 1 and prev5_stats['choice'] == 1 and
                 prev4_stats['choice'] == 1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 1):
                    continue
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev8_stats['choice'] == -1 and prev7_stats['choice'] == 1 and
                 prev6_stats['choice'] == -1 and prev5_stats['choice'] == 1 and
                 prev4_stats['choice'] == -1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 1):
                    continue
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev8_stats['choice'] == 1 and prev7_stats['choice'] == -1 and
                 prev6_stats['choice'] == 1 and prev5_stats['choice'] == -1 and
                 prev4_stats['choice'] == 1 and prev3_stats['choice'] == -1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 1):
                    continue
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_congruent3(stim_trials, cohvalues, stm, dyns, gt, choice,
                              eff_choice, rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle
    # ydata=[]
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

    for i in range(5, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        prev5_stats = stim_trials[i-5]
        # 8 labels
        # ---- only use the correct-correct conditions?

        # prev1_stats['rw']==1 and
        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1 and prev5_stats['rw'] == 0):
            # case 1
            if(prev4_stats['choice'] == -1 and prev3_stats['choice'] == -1 and
               prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 1):
                    continue

                # @YX 1909 note this is only the fixation
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == 1 and
                 prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 1):
                    continue
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev4_stats['choice'] == -1 and prev3_stats['choice'] == 1 and
                 prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 1):
                    continue
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == -1 and
                 prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):
                if (curr_stats['gt'] == 1 and prev1_stats['rw'] == 0):
                    continue
                if (curr_stats['gt'] == -1 and prev1_stats['rw'] == 1):
                    continue
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_0(stim_trials, cohvalues, stm, dyns, gt, choice, eff_choice,
                     rw, obsc, BLOCK_CTXT=0):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_trialidx = []
    Xdata_resp = []  # shuffle
    # ydata=[]
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

    iiist = 2
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]

        if(prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
            Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
            Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
            Xdata_trialidx = np.append(Xdata_trialidx, i)

            nact = np.shape(curr_stats['resp'])[0]
            if (BLOCK_CTXT):
                # 'ctx'
                Xconds_2 = np.append(
                    Xconds_2, curr_stats['ctx']*np.ones(nact), axis=0)
            else:
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
            if(prev1_stats['choice'] == -1):
                Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                # REP+L --> LEFT BIAS
                Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
            elif(prev1_stats['choice'] == 1):
                Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
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
            Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
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
            Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
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
                # print(curr_stats['start_end'][0]-1)
                Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
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
                # print(curr_stats['start_end'][0]-1)
            elif(prev1_stats['choice'] == 1):
                # print(curr_stats['start_end'][0]-1)
                Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0,\
        Xdata_trialidx


def req_quantities_1(stim_trials, cohvalues, stm, dyns, gt, choice, eff_choice,
                     rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle
    # ydata=[]
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

    iiist = 3
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]

        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 0):
            # congruent

            # case 1
            if(prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
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
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_4(stim_trials, cohvalues, stm, dyns, gt, choice, eff_choice,
                     rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle

    # ydata=[]
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

    iiist = 6
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        prev5_stats = stim_trials[i-5]
        prev6_stats = stim_trials[i-6]
        # 8 labels
        # ---- only use the correct-correct conditions?

        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1 and prev5_stats['rw'] == 1 and
           prev6_stats['rw'] == 0):
            # case 1
            if(prev5_stats['choice'] == -1 and prev4_stats['choice'] == -1 and
               prev3_stats['choice'] == -1 and prev2_stats['choice'] == -1 and
               prev1_stats['choice'] == -1):
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)

                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev5_stats['choice'] == 1 and prev4_stats['choice'] == 1 and
                 prev3_stats['choice'] == 1 and prev2_stats['choice'] == 1 and
                 prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev5_stats['choice'] == 1 and prev4_stats['choice'] == -1 and
                 prev3_stats['choice'] == 1 and prev2_stats['choice'] == -1 and
                 prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev5_stats['choice'] == -1 and prev4_stats['choice'] == 1 and
                 prev3_stats['choice'] == -1 and prev2_stats['choice'] == 1 and
                 prev1_stats['choice'] == -1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_3(stim_trials, cohvalues, stm, dyns, gt, choice, eff_choice,
                     rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle

    # ydata=[]
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

    iiist = 5
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        prev5_stats = stim_trials[i-5]
        # 8 labels
        # ---- only use the correct-correct conditions?

        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1 and prev5_stats['rw'] == 0):
            # case 1
            if(prev4_stats['choice'] == -1 and prev3_stats['choice'] == -1
               and prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)

                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev4_stats['choice'] == -1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == -1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_3all(stim_trials, cohvalues, stm, dyns, gt, choice,
                        eff_choice, rw, obsc, BLOCK_CTXT=0):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_trialidx = []
    Xdata_resp = []  # shuffle

    # ydata=[]
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

    iiist = 5
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        # 8 labels
        # ---- only use the correct-correct conditions?

        if (prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1):
            # case 1
            if(prev4_stats['choice'] == -1 and prev3_stats['choice'] == -1
               and prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
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
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
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
            elif(prev4_stats['choice'] == -1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
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
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev4_stats['choice'] == 1 and prev3_stats['choice'] == -1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
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
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1,\
        Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1,\
        Xcohs_0, Xdc_idx_0, Xdata_trialidx


def req_quantities_2(stim_trials, cohvalues, stm, dyns, gt, choice,
                     eff_choice, rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle

    # ydata=[]
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

    iiist = 4
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        # 8 labels
        # ---- only use the correct-correct conditions?
        # 1 alternationg 0 repeating
        if (prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 0):
            # case 1
            if(prev3_stats['choice'] == -1 and prev2_stats['choice'] == -1 and
               prev1_stats['choice'] == -1):

                # @YX 1909 note this is only the fixation
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)

                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev3_stats['choice'] == 1 and prev2_stats['choice'] == 1 and
                 prev1_stats['choice'] == 1):
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev3_stats['choice'] == 1 and prev2_stats['choice'] == -1 and
                 prev1_stats['choice'] == 1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev3_stats['choice'] == -1 and prev2_stats['choice'] == 1 and
                 prev1_stats['choice'] == -1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_5(stim_trials, cohvalues, stm, dyns, gt, choice,
                     eff_choice, rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle

    # ydata=[]
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

    iiist = 7
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        prev5_stats = stim_trials[i-5]
        prev6_stats = stim_trials[i-6]
        prev7_stats = stim_trials[i-7]

        # 8 labels
        # ---- only use the correct-correct conditions?

        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
            prev4_stats['rw'] == 1 and prev5_stats['rw'] == 1 and
                prev6_stats['rw'] == 1 and prev7_stats['rw'] == 0):
            # case 1
            if(prev6_stats['choice'] == -1 and prev5_stats['choice'] == -1 and
               prev4_stats['choice'] == -1 and prev3_stats['choice'] == -1 and
               prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):

                # @YX 1909 note this is only the fixation
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)

                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev6_stats['choice'] == 1 and prev5_stats['choice'] == 1 and
                 prev4_stats['choice'] == 1 and prev3_stats['choice'] == 1 and
                 prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev6_stats['choice'] == -1 and prev5_stats['choice'] == 1 and
                 prev4_stats['choice'] == -1 and prev3_stats['choice'] == 1 and
                 prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev6_stats['choice'] == 1 and prev5_stats['choice'] == -1 and
                 prev4_stats['choice'] == 1 and prev3_stats['choice'] == -1 and
                 prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_6(stim_trials, cohvalues, stm, dyns, gt, choice,
                     eff_choice, rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle

    # ydata=[]
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

    iiist = 8
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        prev5_stats = stim_trials[i-5]
        prev6_stats = stim_trials[i-6]
        prev7_stats = stim_trials[i-7]
        prev8_stats = stim_trials[i-8]

        # 8 labels
        # ---- only use the correct-correct conditions?

        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1 and prev5_stats['rw'] == 1 and
           prev6_stats['rw'] == 1 and prev7_stats['rw'] == 1 and
           prev8_stats['rw'] == 0):
            # case 1
            if(prev7_stats['choice'] == -1 and prev6_stats['choice'] == -1 and
               prev5_stats['choice'] == -1 and prev4_stats['choice'] == -1 and
               prev3_stats['choice'] == -1 and prev2_stats['choice'] == -1 and
               prev1_stats['choice'] == -1):

                # @YX 1909 note this is only the fixation
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)

                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev7_stats['choice'] == 1 and prev6_stats['choice'] == 1 and
                 prev5_stats['choice'] == 1 and prev4_stats['choice'] == 1 and
                 prev3_stats['choice'] == 1 and prev2_stats['choice'] == 1 and
                 prev1_stats['choice'] == 1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev7_stats['choice'] == 1 and prev6_stats['choice'] == -1 and
                 prev5_stats['choice'] == 1 and prev4_stats['choice'] == -1 and
                 prev3_stats['choice'] == 1 and prev2_stats['choice'] == -1 and
                 prev1_stats['choice'] == 1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev7_stats['choice'] == -1 and prev6_stats['choice'] == 1 and
                 prev5_stats['choice'] == -1 and prev4_stats['choice'] == 1 and
                 prev3_stats['choice'] == -1 and prev2_stats['choice'] == 1 and
                 prev1_stats['choice'] == -1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0


def req_quantities_7(stim_trials, cohvalues, stm, dyns, gt, choice, eff_choice,
                     rw, obsc):
    # ---- prepare LDA data
    Xdata, ydata = [], []
    Xdata_idx = []
    Xdc_idx_0 = []
    Xdata_resp = []  # shuffle

    # ydata=[]
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

    iiist = 8
    for i in range(iiist, len(stim_trials)):
        curr_stats = stim_trials[i]
        prev1_stats = stim_trials[i-1]
        prev2_stats = stim_trials[i-2]
        prev3_stats = stim_trials[i-3]
        prev4_stats = stim_trials[i-4]
        prev5_stats = stim_trials[i-5]
        prev6_stats = stim_trials[i-6]
        prev7_stats = stim_trials[i-7]
        prev8_stats = stim_trials[i-8]

        # 8 labels
        # ---- only use the correct-correct conditions?

        if(prev2_stats['rw'] == 1 and prev3_stats['rw'] == 1 and
           prev4_stats['rw'] == 1 and prev5_stats['rw'] == 1 and
           prev6_stats['rw'] == 1 and prev7_stats['rw'] == 1 and
           prev8_stats['rw'] == 1):
            # case 1
            if(prev8_stats['choice'] == -1 and prev7_stats['choice'] == -1 and
               prev6_stats['choice'] == -1 and prev5_stats['choice'] == -1 and
               prev4_stats['choice'] == -1 and prev3_stats['choice'] == -1
               and prev2_stats['choice'] == -1 and prev1_stats['choice'] == -1):
                # @YX 1909 note this is only the fixation
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)

                nact = np.shape(curr_stats['resp'])[0]
                Xconds_2 = np.append(Xconds_2, 0*np.ones(nact), axis=0)  # LLL
                if(prev1_stats['choice'] == -1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # REP+L --> LEFT BIAS
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                if(prev1_stats['rw'] == 0):
                    Xrws_1 = np.append(Xrws_1, 0*np.ones(nact))
                elif(prev1_stats['rw'] == 1):
                    Xrws_1 = np.append(Xrws_1, 1*np.ones(nact))
                    # if(prev1_stats['choice']==-1):
                    #     print(curr_stats['start_end'][0]-1)
                # print("number of steps per trial ",nact)
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
            elif(prev8_stats['choice'] == 1 and prev7_stats['choice'] == 1 and
                 prev6_stats['choice'] == 1 and prev5_stats['choice'] == 1 and
                 prev4_stats['choice'] == 1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == 1):
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
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
            elif(prev8_stats['choice'] == -1 and prev7_stats['choice'] == 1 and
                 prev6_stats['choice'] == -1 and prev5_stats['choice'] == 1 and
                 prev4_stats['choice'] == -1 and prev3_stats['choice'] == 1
                 and prev2_stats['choice'] == -1 and prev1_stats['choice'] == 1):

                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # LRL#2*np.ones(nact),axis=0) ### LRL
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    # ALT+LEFT --> RIGHT BIAS
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
            elif(prev8_stats['choice'] == 1 and prev7_stats['choice'] == -1 and
                 prev6_stats['choice'] == 1 and prev5_stats['choice'] == -1 and
                 prev4_stats['choice'] == 1 and prev3_stats['choice'] == -1
                 and prev2_stats['choice'] == 1 and prev1_stats['choice'] == -1):
                # @YX 1909 note this is only the fixation
                Xdata_idx = np.append(Xdata_idx, curr_stats['start_end'][0]-1)
                Xdc_idx_0 = np.append(Xdc_idx_0, curr_stats['start_end'][1]-1)

                nact = np.shape(curr_stats['resp'])[0]
                # RLR3*np.ones(nact),axis=0) ### RLR
                Xconds_2 = np.append(Xconds_2, 1*np.ones(nact), axis=0)
                if(prev1_stats['choice'] == -1):
                    Xlfs_1 = np.append(Xlfs_1, 0*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 1*np.ones(nact))
                    # print(curr_stats['start_end'][0]-1)
                elif(prev1_stats['choice'] == 1):
                    # print(curr_stats['start_end'][0]-1)
                    Xlfs_1 = np.append(Xlfs_1, 1*np.ones(nact))
                    Xacts_1 = np.append(Xacts_1, 0*np.ones(nact))

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
    return Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2, Xacts_1, Xrws_1,\
        Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0, Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0

# separate correct trials and error trials


def sep_correct_error(stm, dyns, Xdata, ydata, Xdata_idx, Xdata_resp, Xconds_2,
                      Xacts_1, Xrws_1, Xlfs_1, Xrse_6, rses, Xacts_0, Xrws_0,
                      Xgts_0, Xgts_1, Xcohs_0, Xdc_idx_0, Xdata_trialidx,
                      margin=[1, 2], idd=1):
    # margin=[1,2]
    ydata_bias = Xrws_1*2+Xgts_0  # Xacts_0#  ### this 3
    ydata_xor = Xrws_1*2+Xacts_1  # Xgts_0#Xacts_0#### this 2
    ydata_conds = Xrws_1*2+Xconds_2  # this 1
    ydata_choices = Xrws_1*2+Xlfs_1  # this 0

    ydata_cchoices = Xrws_1*2+Xacts_0  # this 4
    ydata_cgts = Xrws_1*2+Xgts_0
    ydata_crws = Xrws_1*2+Xrws_0

    rses = np.array(rses)

    Xrse_6 = np.array(Xrse_6)
    Xcohs_0 = np.array(Xcohs_0)

    import scipy.stats as sstats
    Xdata = sstats.zscore(dyns[:, int(dyns.shape[1]/2):], axis=0)

    # idd = 0 # fixation
    # idd = 1 # stimulus
    # @YX 1909 add --> pre and post
    correct_trial, error_trial = np.where(Xrws_1 > 0)[0].astype(
        np.int32), np.where(Xrws_1 < 1)[0].astype(np.int32)
    Xdata_correct = Xdata[Xdata_idx[correct_trial].astype(np.int32)+idd, :]
    # @YX 0110 add --- RSE ---------
    # print("rses:",rses.shape)
    rses_correct = rses[correct_trial]
    Xrse_6_correct = Xrse_6[correct_trial]
    Xcohs_0_correct = Xcohs_0[correct_trial]
    Xdata_idx_correct = Xdata_idx[correct_trial]
    Xdata_trialidx_correct = Xdata_trialidx[correct_trial]
    # blocklevel
    # Xdata resp for stimulus
    Xdata_stim_correct = Xdata[Xdata_idx[correct_trial].astype(np.int32)+1, :]
    Xdata_dc_correct = Xdata[Xdc_idx_0[correct_trial].astype(np.int32), :]
    ych_stim_correct = stm[Xdata_idx[correct_trial].astype(np.int32)+1, 0] -\
        stm[Xdata_idx[correct_trial].astype(np.int32)+1, 1]
    ydata_bias_correct = ydata_bias[correct_trial]
    ydata_xor_correct = ydata_xor[correct_trial]
    ydata_conds_correct = ydata_conds[correct_trial]
    ydata_choices_correct = ydata_choices[correct_trial]

    ydata_cchoices_correct = ydata_cchoices[correct_trial]
    ydata_cgts_correct = ydata_cgts[correct_trial]
    ydata_crws_correct = ydata_crws[correct_trial]

    Xdata_correct_pre = np.zeros(
        (margin[0], Xdata_correct.shape[0], Xdata_correct.shape[1]))
    for i in range(margin[0]):
        Xdata_correct_pre[i, :, :] = Xdata[Xdata_idx[correct_trial].astype(
            np.int32)-margin[0]+i, :].copy()
    Xdata_correct_post = np.zeros(
        (margin[1], Xdata_correct.shape[0], Xdata_correct.shape[1]))
    for i in range(margin[1]):
        Xdata_correct_post[i, :, :] = Xdata[Xdata_idx[correct_trial].astype(
            np.int32)+i, :].copy()
    Xdata_correct_seq = np.zeros(
        (len(correct_trial)*(margin[0]+margin[1]+1), Xdata.shape[1]))
    len_period = margin[0]+margin[1]+1
    for i in range(margin[0]):
        Xdata_correct_seq[i::len_period, :] = Xdata_correct_pre[i, :, :]
    Xdata_correct_seq[margin[0]::len_period, :] = Xdata_correct[:, :]
    for i in range(margin[1]):
        Xdata_correct_seq[margin[0]+1+i::len_period,
                          :] = Xdata_correct_post[i, :, :]

    Xdata_error = Xdata[Xdata_idx[error_trial].astype(np.int32)+idd, :]
    # Xdata resp for stimulus
    Xdata_stim_error = Xdata[Xdata_idx[error_trial].astype(np.int32)+1, :]
    Xdata_dc_error = Xdata[Xdc_idx_0[error_trial].astype(np.int32), :]
    ych_stim_error = stm[Xdata_idx[error_trial].astype(np.int32)+1, 0] -\
        stm[Xdata_idx[error_trial].astype(np.int32)+1, 1]

    # @YX 0110 add --- RSE ---------
    rses_error = rses[error_trial]
    Xrse_6_error = Xrse_6[error_trial]
    Xcohs_0_error = Xcohs_0[error_trial]
    Xdata_idx_error = Xdata_idx[error_trial]
    Xdata_trialidx_error = Xdata_trialidx[error_trial]
    ydata_bias_error = ydata_bias[error_trial]
    ydata_xor_error = ydata_xor[error_trial]
    ydata_conds_error = ydata_conds[error_trial]
    ydata_choices_error = ydata_choices[error_trial]

    ydata_cchoices_error = ydata_cchoices[error_trial]
    ydata_cgts_error = ydata_cgts[error_trial]
    ydata_crws_error = ydata_crws[error_trial]

    Xdata_error_pre = np.zeros(
        (margin[0], Xdata_error.shape[0], Xdata_error.shape[1]))
    for i in range(margin[0]):
        Xdata_error_pre[i, :, :] = Xdata[Xdata_idx[error_trial].astype(
            np.int32)-margin[0]+i, :].copy()
    Xdata_error_post = np.zeros(
        (margin[1], Xdata_error.shape[0], Xdata_error.shape[1]))
    for i in range(margin[1]):
        Xdata_error_post[i, :, :] = Xdata[Xdata_idx[error_trial].astype(
            np.int32)+i, :].copy()

    Xdata_error_seq = np.zeros(
        (len(error_trial)*(margin[0]+margin[1]+1), Xdata.shape[1]))
    for i in range(margin[0]):
        Xdata_error_seq[i::len_period, :] = Xdata_error_pre[i, :, :]
    Xdata_error_seq[margin[0]::len_period, :] = Xdata_error[:, :]
    for i in range(margin[1]):
        Xdata_error_seq[margin[0]+1+i::len_period,
                        :] = Xdata_error_post[i, :, :]

    return Xdata_correct, correct_trial, Xdata_stim_correct, Xdata_dc_correct,\
        ych_stim_correct, Xdata_correct_seq, Xdata_error, error_trial,\
        Xdata_stim_error, Xdata_dc_error, ych_stim_error, Xdata_error_seq,\
        rses_correct, rses_error, Xrse_6_correct, Xrse_6_error, Xcohs_0_correct,\
        Xcohs_0_error, ydata_bias_correct, ydata_bias_error, ydata_xor_correct,\
        ydata_xor_error, ydata_conds_correct, ydata_conds_error,\
        ydata_choices_correct, ydata_choices_error, ydata_cchoices_correct,\
        ydata_cchoices_error, ydata_cgts_correct, ydata_cgts_error,\
        ydata_crws_correct, ydata_crws_error, Xdata_idx_correct, Xdata_idx_error,\
        Xdata_trialidx_correct, Xdata_trialidx_error
