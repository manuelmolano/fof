#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:30:04 2021

@author: manuel
"""

import glob
import os
import utils as ut
import numpy as np


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    sfx = [int(x[-6:]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def get_all_behav_files(spks_sort_folder=None, behav_folder=None, sv_folder=None):
    def get_bhv_folder(bhv_f):
        assert len(bhv_f) > 0, 'No behavioral folder for rat '+rat_name
        if len(bhv_f) > 1:
            oks = []
            for f in bhv_f:
                oks.append(os.path.exists(f+'/sessions'))
            assert np.sum(oks) == 1
            bhv_f = np.array(bhv_f)[np.where(np.array(oks))[0]]
        return bhv_f[0]

    def get_bhv_session(b_f):
        if len(b_f) == 0:
            b_f = None
        elif len(b_f) > 1:
            sorted_files = order_by_sufix(file_list=b_f)
            b_f = sorted_files[-1]
        else:
            b_f = b_f[0]
        return b_f

    def load_behavior(b_f):
        try:
            p.load(b_f)
            p.process()
            p.trial_sess.head()  # preprocssd df stored in attr. trial_sess
            df = p.sess
            df_trials = p.trial_sess
        except (KeyError, IndexError):
            print('Could not load behavioral data')
            df = None
            df_trials = None
        return df, df_trials

    def create_folder(f):
        if not os.path.exists(f):
            os.mkdir(f)
    # folders
    if spks_sort_folder is None:
        spks_sort_folder = '/archive/lbektic/AfterClustering/'
    if behav_folder is None:
        behav_folder = '/archive/rat/behavioral_data/'
    if sv_folder is None:
        sv_folder = '/home/molano/fof_data/'
    # get rats from spike sorted files folder
    rats = glob.glob(spks_sort_folder+'LE*')
    for r in rats:
        rat_name = os.path.basename(r)
        sv_f_rat = sv_folder+'/'+rat_name+'/'
        create_folder(sv_f_rat)
        # get rat number to look for the electro and behav folders
        rat_num = r[r.find('/LE')+3:]
        e_fs = glob.glob(spks_sort_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        bhv_f = glob.glob(behav_folder+'*'+str(rat_num))
        # check that the behav folder exists
        bhv_f = get_bhv_folder(bhv_f)
        path, name = os.path.split(bhv_f)
        p = ut.get_behavior(main_folder=path+'/', subject=name)
        for e_f in e_fs:
            dt_indx = e_f.find(rat_num+'_20')+len(rat_num)+1
            date = e_f[dt_indx:dt_indx+10]
            date = date.replace('-', '')
            # initialize entry
            b_f = [f for f in p.available if f.find(date) != -1]
            # Load behavioral data
            b_f = get_bhv_session(b_f)
            if b_f is None:
                continue
            # load load behavior
            df, df_trials = load_behavior(b_f=b_f)
            if df is None:
                continue
            sv_f_sess = sv_f_rat+'/'+os.path.basename(e_f)
            create_folder(sv_f_sess)
            df.to_pickle(sv_f_sess+'/df')
            df_trials.to_pickle(sv_f_sess+'/df_trials')


if __name__ == '__main__':
    main_folder = '/home/molano/fof_data/'
    default = True
    redo = False
    if default:
        get_all_behav_files()
    else:
        get_all_behav_files(spks_sort_folder=main_folder+'/AfterClustering/',
                            behav_folder=main_folder+'/behavioral_data/',
                            sv_folder=main_folder)
