#!/usr/bin/env python
# coding: utf-8

# ## EPHYS analysis

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
from statannot import add_stat_annotation
import itertools
from scipy import stats
from datahandler import Utils
from ast import literal_eval
from glob import glob
from open_ephys.analysis import Session
import pyopenephys
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


def ApplyChannelMap(Data, ChannelMap):
    print('Retrieving channels according to ChannelMap... ', end='')
    for R, Rec in Data.items():
        if Rec.shape[1] < len(ChannelMap) or max(ChannelMap) > Rec.shape[1]-1:
            print('')
            print('Not enough channels in data to apply channel map. Skipping...')
            continue

        Data[R] = Data[R][:, ChannelMap]

    return(Data)


def BitsToVolts(Data, ChInfo, Unit):
    print('Converting to uV... ', end='')
    Data = {R: Rec.astype('float32') for R, Rec in Data.items()}

    if Unit.lower() == 'uv':
        U = 1
    elif Unit.lower() == 'mv':
        U = 10**-3

    for R in Data.keys():
        for C in range(len(ChInfo)):
            Data[R][:, C] = Data[R][:, C] * ChInfo[C]['bit_volts'] * U
            if 'ADC' in ChInfo[C]['channel_name']:
                Data[R][:, C] *= 10**6

    return(Data)


def Load(Folder, Processor=None, Experiment=None, Recording=None,
         Unit='uV', ChannelMap=[]):
    Files = sorted(glob(Folder+'/**/*.dat', recursive=True))
    InfoFiles = sorted(glob(Folder+'/*/*/structure.oebin'))

    Data, Rate = {}, {}
    for F, File in enumerate(Files):
        File = File.replace('\\', '/')  # Replace windows file delims
        Exp, Rec, _, Proc = File.split('/')[-5:-1]
        Exp = str(int(Exp[10:])-1)
        Rec = str(int(Rec[9:])-1)
        Proc = Proc.split('.')[0].split('-')[-1]
        if '_' in Proc:
            Proc = Proc.split('_')[0]

        if Proc not in Data.keys():
            Data[Proc], Rate[Proc] = {}, {}

        if Experiment:
            if int(Exp) != Experiment-1:
                continue

        if Recording:
            if int(Rec) != Recording-1:
                continue

        if Processor:
            if Proc != Processor:
                continue

        print('Loading recording', int(Rec)+1, '...')
        if Exp not in Data[Proc]:
            Data[Proc][Exp] = {}
        Data[Proc][Exp][Rec] = np.memmap(File, dtype='int16', mode='c')

        Info = literal_eval(open(InfoFiles[F]).read())
        ProcIndex = [Info['continuous'].index(_) for _ in Info['continuous']
                     # Changed to source_processor_id from recorded_processor_id
                     if str(_['source_processor_id']) == Proc][0]

        ChNo = Info['continuous'][ProcIndex]['num_channels']
        if Data[Proc][Exp][Rec].shape[0] % ChNo:
            print('Rec', Rec, 'is broken')
            del(Data[Proc][Exp][Rec])
            continue

        SamplesPerCh = Data[Proc][Exp][Rec].shape[0]//ChNo
        Data[Proc][Exp][Rec] = Data[Proc][Exp][Rec].reshape((SamplesPerCh, ChNo))
        Rate[Proc][Exp] = Info['continuous'][ProcIndex]['sample_rate']

    for Proc in Data.keys():
        for Exp in Data[Proc].keys():
            if Unit.lower() in ['uv', 'mv']:
                ChInfo = Info['continuous'][ProcIndex]['channels']
                Data[Proc][Exp] = BitsToVolts(Data[Proc][Exp], ChInfo, Unit)

            if ChannelMap:
                Data[Proc][Exp] = ApplyChannelMap(Data[Proc][Exp], ChannelMap)

    print('Done.')

    return(Data, Rate)


# Importing the data from a session
path = '/home/molano/fof_data/LE113/LE113_2021-06-05_12-38-09/'
# Load spike sorted data
# Times of the spikes, array of lists
spike_times = np.load(path+'spike_times.npy')
# cluster number of each of the spikes, same length as before
spike_clusters = np.load(path+'spike_clusters.npy')
# Cluster labels (good, noise, mua) for the previous two arrays
df_labels = pd.read_csv(path+"cluster_group.tsv", sep='\t')

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


# BEHAVIOR
os.getcwd()
os.chdir(path)
df.to_csv(path+'spike.csv')

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

# Remove those sessions that the animal wasn't in the final training step:
# STAGE 3 or above, MOTOR 6, no delay progression (delay lengths remain the same),
# good accuracy in short trials.
df_behavior = df_behavior.loc[(df_behavior['stage_number'] >= 3) &
                              (df_behavior['motor'] == 6) &
                              (df_behavior['delay_progression'] == 0) &
                              (df_behavior['accuracy_low'] >= 0.60) &
                              (df_behavior['accuracy'] >= 0.60)]

df_behavior['hit'] = df_behavior['hit'].astype(float)

# df.groupby(['subject','day']).count()

session = Session(path)

df_behavior = df_behavior.loc[(df_behavior.day == '2021-06-13') &
                              (df_behavior.subject == 'E10')]

# Because the first trial has no delay, we need to shift one on the behavioral data
# in order to fit with the ttl one.
df_behavior.delay_times[1:]

vector_answer_dev = np.logical_not(np.logical_xor(df_behavior['reward_side'],
                                                  df_behavior['hit'].astype(int)))
vector_answer = np.where(vector_answer_dev, 0, 1)
df_behavior['vector_answer'] = vector_answer

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
