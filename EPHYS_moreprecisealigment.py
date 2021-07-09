#!/usr/bin/env python
# coding: utf-8

# ___
# ## EPHYS analysis

# ### GET BEHAVIOR

# Load modules and data
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

#Import all needed libraries
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
from datahandler import Utils
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from statannot import add_stat_annotation
import itertools
from scipy import stats
from datahandler import Utils
from ast import literal_eval
from glob import glob
from open_ephys.analysis import Session
import pyopenephys

from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


# ### LOAD BEHAVIORAL DATA

batch = 'general'

path2 = ''
os.getcwd()
os.chdir(path2)

df_trials = pd.read_csv(path2 + '/global_trials.csv', sep=';')
df_params = pd.read_csv(path2 + '/global_params.csv', sep=';')
df_behavior = pd.merge(df_params, df_trials, on=['session', 'subject_name'])

# Rename some of the variables for a global consensus.
df_behavior = df_behavior.rename(columns={'subject_name': 'subject',
                                          'hithistory': 'hit',
                                          'probabilities': 'prob',
                                          'validhistory': 'valids'})

# #Remove those sessions that the animal wasn't in the final training step:
# STAGE 3 or above, MOTOR 6, no delay progression (delay lengths remain the same),
# good accuracy in short trials.
df_behavior = df_behavior.loc[(df_behavior['stage_number'] >= 3) &
                              (df_behavior['motor'] == 6) &
                              (df_behavior['delay_progression'] == 0) &
                              (df_behavior['accuracy_low'] >= 0.60) &
                              (df_behavior['accuracy'] >= 0.60)]

df_behavior['hit'] = df_behavior['hit'].astype(float)


# Select only the session and animal that we need
df_behavior = df_behavior.loc[(df_behavior.day == '2021-06-13') &
                              (df_behavior.subject == 'E10')]

# Compute vector of answers
vector_answer_dev = np.logical_not(np.logical_xor(df_behavior['reward_side'],
                                                  df_behavior['hit'].astype(int)))
vector_answer = np.where(vector_answer_dev, 0, 1)
df_behavior['vector_answer'] = vector_answer

# Add a new colum witht he repetition choice
df_behavior['repeat_choice'] = np.nan

for i in range(len(df_behavior)):
    if df_behavior['trials'].iloc[i] != 0:
        if df_behavior['vector_answer'].iloc[i-1] ==\
           df_behavior['vector_answer'].iloc[i]:
            df_behavior['repeat_choice'].iloc[i-1] = 1  # Repeat previous choice
        else:
            df_behavior['repeat_choice'].iloc[i-1] = 0  # Alternate previous choice

# Because the first trial has no delay, we need to shift one on the behavioral data
# in order to fit with the ttl one.
df_behavior = df_behavior[1:]

df_behavior = Utils.convert_strings_to_lists(df_behavior, ['L_s', 'C_s',
                                                           'C_e', 'L_e'])

# Add a colum for first lick
df_temp = []
for i in range(len(df_behavior)):
    if df_behavior.C_s.iloc[i][0] > df_behavior.L_s.iloc[i][0] or\
       np.isnan(df_behavior.C_s.iloc[i][0]):
        df_temp.append(df_behavior.L_s.iloc[i][0])
    elif df_behavior.C_s.iloc[i][0] < df_behavior.L_s.iloc[i][0] or\
        np.isnan(df_behavior.L_s.iloc[i][0]):
        df_temp.append(df_behavior.C_s.iloc[i][0])
    else:
        df_temp.append(np.nan)
df_behavior['lick'] = np.array(df_temp)

# Importing the data from a session in Ephys
path = 'C:/Users/Tiffany/Documents/Ephys/E10_2021-06-13_12-31-21/'
os.getcwd()
os.chdir(path)

# Recover previous timestamps session
df_ttl = pd.read_csv(path + '/timestamps.csv', sep=',')
df = pd.read_csv(path + '/spike.csv', sep=',')

# Mark onset of delays
df_ttl.loc[df_ttl['ttl'] == 1, 'Delay_ON'] = df_ttl['timestamps']
# Mark offset of delay
df_ttl.loc[df_ttl['ttl'] == 0, 'Delay_OFF_next'] = df_ttl['timestamps']


# Create new colum with delay offset to measure the delay duration and then rmv it
df_ttl['Delay_OFF'] = df_ttl['Delay_OFF_next'].shift(-1)
df_ttl['Delay_length'] = df_ttl['Delay_OFF'] - df_ttl['Delay_ON']
df_ttl.drop('Delay_OFF_next', axis='columns', inplace=True)

df_ttl = df_ttl[df_ttl['Delay_ON'].notna()]  # Remove the trials with nans

# Prepare a column with trial index. start in 1 because trial 0 doesn't have
# a delay and is not there.
df_ttl['trial'] = np.arange(len(df_ttl))+1

# Merge with cluster labels, use trial to associate each one
df_behavior.rename(columns={'trials': 'trial'}, inplace=True)
df2_behavior = pd.merge(df_behavior, df_ttl, on=['trial'])

df_final = pd.DataFrame()
# We now have the moment of the onset of the delay and what it corresponds in the
# behavioral session. If we substract this we can get the start for every session.
# Then, we will use the START for everything else.
df2_behavior['START'] = df2_behavior['Delay_ON']-df2_behavior['Delay_start']
df_final['START'] = df2_behavior['Delay_ON']-df2_behavior['Delay_start']
df_final['Delay_ON'] = df2_behavior['Delay_ON']
df_final['Delay_OFF'] = df2_behavior['Delay_OFF']

df_final['Stimulus_ON'] = df2_behavior['START'] +\
    df2_behavior['StimulusDuration_start']
df_final['Response_ON'] = df2_behavior['START'] +\
    df2_behavior['ResponseWindow_start']
df_final['Lick_ON'] = df2_behavior['START'] + df2_behavior['lick']
df2_behavior['END'] = df2_behavior['START'] + df2_behavior['Motor_out_end'] + 0.006
df_final['Motor_OUT'] = df2_behavior.END - 2
df_final['END'] = df2_behavior.END

df_final['vector_answer'] = df2_behavior['vector_answer']
df_final['reward_side'] = df2_behavior['reward_side']
df_final['hit'] = df2_behavior['hit']
df_final['repeat_choice'] = df2_behavior['repeat_choice']
df_final['miss'] = df2_behavior['misshistory']
df_final['trial'] = df2_behavior['trial']
df_final['delay'] =\
    np.around(df2_behavior['Delay_OFF']-df2_behavior['Delay_ON'], 2)

df_final.tail()

df['trial'] = 0
for i, rows in df_final.iterrows():
    # create a list of our conditions
    conditions = [(df.fixed_times > df_final['START'].iloc[i]) &
                  (df.fixed_times <= df_final['END'].iloc[i]),
                  (df.fixed_times < df_final['START'].iloc[i])]

    # create a list of the values we want to assign for each condition
    values = [df_final['trial'].iloc[i], df['trial']]

    # create a new column and use np.select to assign values to it using our
    # lists as arguments
    df['trial'] = np.select(conditions, values)

print(df_final[df_final['trial'] == 46]['START'].iloc[0],
      df_final[df_final['trial'] == 46]['END'].iloc[0])


# Merge with cluster labels, use cluster ID to associate each one
df3 = pd.merge(df, df_final, on=['trial'])

df3.head()

df3 = df3[df3.trial != df3.trial.unique()[-1]]

df3['a_Stimulus_ON'] = df3['fixed_times'] - df3['Stimulus_ON']
df3['a_Response_ON'] = df3['fixed_times'] - df3['Response_ON']
df3['a_Lick_ON'] = df3['fixed_times'] - df3['Lick_ON']
df3['a_Delay_OFF'] = df3['fixed_times'] - df3['Delay_OFF']


min(df3.a_Stimulus_ON)

len(df3)

df3 = df3.drop(['times', 'group'], axis=1)

# Save the data in a new csv.
os.getcwd()
os.chdir(path)
df3.to_csv(path+'data.csv')
