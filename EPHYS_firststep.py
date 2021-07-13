#!/usr/bin/env python
# coding: utf-8

# ## EPHYS analysis
from utilsJ.Behavior import ComPipe
# Load modules and data
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

# Import all needed libraries
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import itertools
from scipy import stats
from ast import literal_eval
from glob import glob
from open_ephys.analysis import Session
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# BEHAVIOR
p = ComPipe.chom('LE113',  # sujeto (folder name under parentpath)
    parentpath='/home/molano/fof_data/',
    analyze_trajectories=False)
p.load_available()
print(p.available[2]) # example target session / filename string is the actual arg
p.load(p.available[2])
p.process()
p.trial_sess.head() # preprocessed df stored in attr. trial_sess


# ELECTRO
# Importing the data from a session
path = '/home/molano/fof_data/LE113/LE113_2021-06-05_12-38-09/'
# Load spike sorted data
# Times of the spikes, array of lists
spike_times = np.load(path+'spike_times.npy')
# cluster number of each of the spikes, same length as before
spike_clusters = np.load(path+'spike_clusters.npy')
# Cluster labels (good, noise, mua) for the previous two arrays
df_labels = pd.read_csv(path+"cluster_group.tsv", sep='\t')
good_clusters = df_labels.loc[df_labels.group == 'good']



# Transforms array of lists into array of ints
spike_times_df = [item for sublist in spike_times for item in sublist]

# Put the data in a single dataframe, one colum spikes, one colum clusters
s1 = pd.Series(spike_times_df, name='times')
s2 = pd.Series(spike_clusters, name='cluster_id')
df_temp = pd.concat([s1, s2], axis=1)

# Merge with cluster labels, use cluster ID to associate each one
df = pd.merge(df_temp, df_labels, on=['cluster_id'])


# Select only clusters labelled good, which are presumably single units
df = df.loc[df.group == 'good']

# Transform the values per session to seconds. This takes into account the frame
# rate of the recordings, 30000Hz for virtually all the sessions.
df['fixed_times'] = (df.times/30000)
print(min(df['fixed_times']), max(df['fixed_times']))


# Plot the first minute to have an impression of how it looks like
sns.scatterplot('fixed_times', 'cluster_id', data=df.loc[(df['fixed_times'] < 60) &
                                                         (df.group == 'good')],
                s=30, color='black')

session = Session(path)

try:
    samples = session.recordings[0].continuous[0].samples[:, -8]
    timestamps = session.recordings[0].continuous[0].timestamps

    # samples = session.recordings[0].continuous[0].samples[:,-7]

    # Put the data in a single dataframe
    s1 = pd.Series(samples, name='samples')
    s2 = pd.Series(timestamps, name='timestamps')
    df_ttl = pd.concat([s1, s2], axis=1)
except:
    # Way to recover the data with recording node (Gui >0.5)
    session.recordnodes[0].recordings[0].continuous[0].metadata
    # Delay channel.
    samples_delay =\
        session.recordnodes[0].recordings[0].continuous[0].samples[:, -8]
    timestamps = session.recordnodes[0].recordings[0].continuous[0].timestamps
    # Sound node
    samples_sound =\
        session.recordnodes[0].recordings[0].continuous[0].samples[:, -6]

    # Put the data in a single dataframe
    s1 = pd.Series(samples_delay, name='samples')
    s2 = pd.Series(timestamps, name='timestamps')

    df_ttl = pd.concat([s1, s2], axis=1)

# Transform the analog channel to boolean.
df_ttl.loc[df_ttl['samples'] >= 1000, 'ttl'] = 1
df_ttl.loc[df_ttl['samples'] < 1000, 'ttl'] = 0

# Look for the places where there is a change and it goes from 0 to 1.
# 1 for on delay and -1 for off delay
df_ttl['diff'] = df_ttl.ttl.diff()
df_ttl['diff'].unique()


# On is a 1 and -1 is an off of the delay period

# Recover the first timestamp of the session. This is important because
# it is not 0 but when the recording was started.
initial = df_ttl.timestamps.iloc[0]

# Substract the first timestamp and divide by sampling frequency
df_ttl.timestamps = (df_ttl.timestamps - initial)/30000

# Remove the rest of the values that does not have a change.
df_ttl = df_ttl.loc[(df_ttl['diff'] == 1) | (df_ttl['diff'] == -1)]

# Create a new colum with the delay duration
df_ttl['delay'] =\
    np.around(df_ttl.loc[(df_ttl['diff'] == 1) |
                         (df_ttl['diff'] == -1)]['timestamps'].diff(), 2)

# Check that we recover all the delays that were used.
df_ttl.delay.unique()

# Save the data in a new csv.
os.getcwd()
os.chdir(path)
df_ttl.to_csv(path+'timestamps.csv')

# Remove all the colums which don't show the right delay lengths.
df_ttl = df_ttl.loc[(df_ttl['delay'] == 1) | (df_ttl['delay'] == 10) |
                    (df_ttl['delay'] == 3.00) | (df_ttl['delay'] == 0.1)]

np.array_equal(df.delay_times, df_ttl.delay)
