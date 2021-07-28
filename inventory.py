#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:16:43 2021

@author: manuel
"""

import glob

spks_sorting_folder = '/archive/rat/electrophysiology_recordings/'
electro_folder = '/archive/rat/electrophysiology_recordings/'
behav_folder = '/archive/rat/behavioral_data/'
rats = glob.glob(spks_sorting_folder+'LE*')

for r in rats:
    rat_num = r[r.find('/LE')+3:]
    b_f = glob.glob(electro_folder+'*rat_num/*rat_num*')
    b_f = glob.glob(behav_folder+'*rat_num/sessions/*rat_num*')
    
    


