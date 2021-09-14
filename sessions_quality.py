#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:28:28 2021

@author: molano
"""
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.backends.backend_pdf import PdfPages
# import utils as ut
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')
# import utils as ut
colors = sns.color_palette()
AX_SIZE = 0.17  # for hist and psth axes
MARGIN = .06  # for hist and psth axes
ISSUES = np.array(['', ' ', 'noise 1', 'noise 2', 'noise 3', 'no ttl', 'no signal',
                   'sil per'])
ISSS_CLR = np.array(['k', 'k', 'r', 'm', 'm', 'b', 'c', 'g'])


def set_title(ax, session, inv, inv_sbsmpld, i):
    """
    Set title and check inv and subsampled inv are equivalent.

    Parameters
    ----------
    ax : axis
        where to add the tittle.
    inv : dict
        original inventory.
    inv_sbsmpld : dict
        inventory obtained from subsampled ttl signals (sil-per seems to be diff).

    Returns
    -------
    None.

    """
    ax.set_title('Sess:'+session +
                 ' /// . #evs. csv: ' +
                 str(np.round(inv['num_stms_csv'][i], 3))+' / Sil. per.: ' +
                 str(np.round(inv['sil_per'][i], 3)) +
                 ' /// #evs. ttl: ' +
                 str(np.round(inv['num_stim_ttl'][i], 3))+' / ' +
                 str(np.round(inv['stim_ttl_dists_med'][i], 3))+' / ' +
                 str(np.round(inv['stim_ttl_dists_max'][i], 3)) +
                 ' /// #evs. anlg: ' +
                 str(np.round(inv_sbsmpld['num_stim_analogue'][i], 3))+' / ' +
                 str(np.round(inv['stim_analogue_dists_med'][i], 3))+' / ' +
                 str(np.round(inv['stim_analogue_dists_max'][i], 3)))
    ks_to_check = [k for k in inv.keys() if k not in ['num_stim_analogue',
                                                      'sil_per', 'rat', 'session',
                                                      'bhv_session', 'sgnl_stts',
                                                      'state', 'date']]
    for k in ks_to_check:
        # print(k)
        # print(inv[k][i])
        # print(inv_sbsmpld[k][i])
        if not np.isnan(inv[k][i]) and not np.isnan(inv_sbsmpld[k][i]):
            assert inv[k][i] == inv_sbsmpld[k][i], str(inv[k][i]-inv_sbsmpld[k][i])


def plot_psths(samples, e_data, offset, margin_psth):
    """
    Plot peri-event histograms for the different events (stim, fix, outcome,
                                                         analogue-stim).

    Parameters
    ----------
    samples : array
        TTL signals corresponding to the last 4 channels (if there is no image).
    e_data : dict
        dictionary containing the event times.
    offset : float
        offset (in second) that was subtracted from the event times. It corresponds
        to the time of the first TTL stim event - first CSV stim event.
    margin_psth : int
        time (in ms) to plot before and after the event.
    AX_SIZE : float, optional
        size of panels(0.17)

    Returns
    -------
    None.

    """
    # Xs to plot the peri-ev hist
    xs = np.arange(2*margin_psth)-margin_psth
    xs = xs/1000
    # this is just to adjust the panels position
    # PSTHs
    evs_lbls = ['stim_ttl_strt', 'fix_strt', 'outc_strt', 'stim_anlg_strt']
    for i_e, ev in enumerate(evs_lbls):
        aux1 = i_e % 2 != 0
        aux2 = i_e > 1
        ax_loc = [.55+(AX_SIZE+MARGIN)*aux1, .05+(AX_SIZE+MARGIN)*aux2,
                  AX_SIZE, AX_SIZE]
        ax_psth = plt.axes(ax_loc)
        ax_psth.set_title(ev)
        chnls = [0, 1] if i_e < 3 else [2, 3]
        lbls = [' (36)', ' (37)'] if i_e < 3 else [' (38)', ' (39)']
        evs = e_data[ev]
        if len(evs) > 0:
            evs = e_data['s_rate_eff']*(evs+offset)
            evs = evs.astype(int)
            peri_evs_1 = np.array([samples[x-margin_psth:x+margin_psth,
                                           chnls[0]] for x in evs])
            peri_evs_2 = np.array([samples[x-margin_psth:x+margin_psth,
                                           chnls[1]] for x in evs])
            try:
                for i_ex in range(10):
                    ax_psth.plot(xs, peri_evs_1[i_ex], color=colors[0],
                                 lw=.5, alpha=.2)
                    ax_psth.plot(xs, peri_evs_2[i_ex], color=colors[1],
                                 lw=.5, alpha=.2)
                psth_1 = np.mean(peri_evs_1, axis=0)
                psth_2 = np.mean(peri_evs_2, axis=0)
                ax_psth.plot(xs, psth_1, color=colors[0], lw=1,
                             label='ch '+lbls[0])
                ax_psth.plot(xs, psth_2, color=colors[1], lw=1,
                             label='ch '+lbls[1])
                ax_psth.legend()
            except (ValueError, IndexError) as e:
                print(e)


def plot_traces_and_hists(samples, ax_traces, num_ps=int(1e5)):
    """
    Plot two different samples from the TTL signals and their histograms.

    Parameters
    ----------
    samples : array
        TTL signals corresponding to the last 4 channels (if there is no image).
    ax_traces : axis
        where to plot the sample traces.
    num_ps : int, optional
        number of points to plot per sample (int(1e5)).

    Returns
    -------
    idx_max : int
        index for the maximum value of samples[:, 0].

    """
    idx_max = np.where(samples[:, 0] == np.max(samples[:, 0]))[0][0]
    idx_midd = int(samples.shape[0]/2)
    for ttl in range(4):
        aux1 = ttl % 2 != 0
        aux2 = ttl > 1
        ax_loc = [.05+(AX_SIZE+MARGIN)*aux1, .05+(AX_SIZE+MARGIN)*aux2,
                  AX_SIZE, AX_SIZE]
        ax_hist = plt.axes(ax_loc)
        sample = samples[:, ttl]
        ax_hist.hist(sample, 100)
        ax_hist.set_title('TTL: '+str(ttl+36))
        ax_hist.set_yscale('log')
        sample = sample/np.max(sample)
        # plot around a max event
        ax_traces.plot(np.arange(num_ps)+idx_max,
                       sample[idx_max:idx_max+num_ps]+ttl,
                       label=str(ttl+36), color=colors[ttl])
        # plot around the middle point of the recording
        ax_traces.plot(np.arange(num_ps)+idx_max+num_ps+1e3,
                       sample[idx_midd:idx_midd+num_ps]+ttl,
                       color=colors[ttl])
    ax_traces.legend(loc='lower right')
    return idx_max


def get_input(ignore=False, defaults={'class':'', 'issue': '', 'obs': ''}):
    """
    Get feedback from user about: classification of session, issue, observations.

    Parameters
    ----------
    ignore : boolean, optional
        if True, the fn will return 'revisit', '', '' (False)

    Raises
    ------
    ValueError
        error if the classification is not: 'y' (good), 'n' (bad), ' ' (revisit)

    Returns
    -------
    fldr : str
        classification of the session.
    prob : str
        issue.
    obs : str
        observations.

    """
    if ignore:
        if defaults['class'] == 'y':
            fldr = 'good'
        elif defaults['class'] == 'n':
            fldr = 'bad'
        elif defaults['class'] == ' ':
            fldr = 'revisit'
        else:
            raise ValueError('Specify the quality of the session with y/n')
        prob = defaults['issue']
        obs = defaults['obs']
    else:
        sess_class = input("Is this session good? (def: "+defaults['class']+') ')
        if sess_class == '':
            sess_class = defaults['class']
        if sess_class == 'y':
            fldr = 'good'
        elif sess_class == 'n':
            fldr = 'bad'
        elif sess_class == ' ':
            fldr = 'revisit'
        else:
            raise ValueError('Specify the quality of the session with y/n')
        prob = input("issue (def: "+defaults['issue']+') ')
        if prob == '':
            prob = defaults['issue']
        obs = input("Observations (def: "+defaults['obs']+') ')
        if obs == '':
            obs = defaults['obs']
    return fldr, prob, obs


def get_extended_inv(inv, sess_classif, issue, observations):
    """
    Build extended inventory from inv and the classifications.

    Parameters
    ----------
    inv : dict
        original inventory.
    sess_classif : str
        classification of the session.
    issue : str
        issue.
    observations : str
        observations.

    Returns
    -------
    extended_inv : dict
        extended inventory.

    """
    extended_inv = {}
    for it in inv.items():
        extended_inv[it[0]] = it[1]
    extended_inv['sess_class'] = sess_classif
    extended_inv['issue'] = issue
    extended_inv['observations'] = observations
    return extended_inv


def build_figure(samples, e_data, offset, session, inv, inv_sbsmpld, margin_psth,
                 sv_folder, num_ps, indx):
    """
    Build figure, plot traces, histograms and peri-ev histograms, and save.

    Parameters
    ----------
    samples : array
        TTL signals corresponding to the last 4 channels (if there is no image).
    e_data : dict
        dictionary containing the event times.
    offset : float
        offset (in second) that was subtracted from the event times. It corresponds
        to the time of the first TTL stim event - first CSV stim event.
    session : str
        current session.
    inv : dict
        inventory.
    inv_sbsmpld : TYPE
        DESCRIPTION.
    margin_psth : int
        time (in ms) to plot before and after the event (2e3)
    sv_folder : str
        where to save the figures and pdfs.
    num_ps : int, optional
        number of points to plot per sample (int(1e5)).
    indx : int
        index in inv of associated session.

    Returns
    -------
    f : fig
        figure.
    ax_traces : axis
        ax to plot traces.
    idx_max : int
        index for the maximum value of samples[:, 0].

    """
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.remove()
    ax_traces = plt.axes([.05, 0.55, 0.9, .4])
    set_title(ax=ax_traces, session=session, inv=inv, inv_sbsmpld=inv_sbsmpld,
              i=indx)
    # PLOT TRACES AND HISTOGRAMS
    idx_max = plot_traces_and_hists(samples=samples, ax_traces=ax_traces,
                                    num_ps=num_ps)
    # PLOT TTL PSTHs
    plot_psths(samples=samples, e_data=e_data, offset=offset,
               margin_psth=margin_psth)
    f.savefig(sv_folder+'/'+session+'.png')
    return f, ax_traces, idx_max


def batch_sessions(main_folder, sv_folder, inv, redo=False, sel_sess=[],
                   sel_rats=[], plot_fig=True, inv_sbsmpld=None, margin_psth=2e3,
                   num_ps=int(1e5), ignore_input=False):
    """
    Check and classify sessions and store figures in png and pdfs.

    Parameters
    ----------
    main_folder : str
        folder where the data is.
    sv_folder : str
        where to save the figures and pdfs.
    inv : dict
        inventory.
    redo : boolean, optional
        whether to update the comments (False)
    sel_sess : list, optional
        list of selected sessions. ([])
    sel_rats : list, optional
        list of selected rats ([])
    plot_fig : boolean, optional
        whether to plot the figures (True)
    inv_sbsmpld : dict, optional
        inventory obtained from subsampled TTL signals (None)
    margin_psth : int
        time (in ms) to plot before and after the event (2e3)
    num_ps : int, optional
        number of points to plot per sample (int(1e5)).
    ignore_input : boolean, optional
        whether to ask for input from user (False)

    Returns
    -------
    None.

    """
    inv_sbsmpld = inv if inv_sbsmpld is None else inv_sbsmpld
    if not redo and os.path.exists(main_folder+'/sess_inv_extended.npz'):
        old_inv_extended = np.load(main_folder+'/sess_inv_extended.npz',
                                   allow_pickle=1)
        sess_classif = old_inv_extended['sess_class']
        issue = old_inv_extended['issue']
        observations = old_inv_extended['observations']
    else:
        sess_classif = ['n.c.']*len(inv['session'])
        issue = ['']*len(inv['session'])
        observations = ['']*len(inv['session'])

    pdf_issues = PdfPages(sv_folder+"issues.pdf")
    pdf_selected = PdfPages(sv_folder+"selected.pdf")
    rats = glob.glob(main_folder+'LE*')
    rats = [x for x in rats if x[-4] != '.']
    used_indx = []
    dates_eq = []
    f_tmln, ax_tmln = plt.subplots()
    counter = 0
    for i_r, r in enumerate(rats):
        rat = os.path.basename(r)
        sessions = glob.glob(r+'/LE*')
        for sess in sessions:
            counter += 1
            session = os.path.basename(sess)
            date = session[session.find('202'):session.find('202')+10]
            days = 365*int(date[2:4])+30.4*int(date[5:7])+int(date[8:10])
            dates_eq.append([days, date])
            print('----')
            print(session)
            print(counter)
            if session not in sel_sess and rat not in sel_rats and\
               (len(sel_sess) != 0 or len(sel_rats) != 0):
                continue
            idx = [i for i, x in enumerate(inv['session']) if x.endswith(session)]
            idx_ss = idx[0]
            if len(idx) != 1:
                print('Could not find associated session in inventory')
                print(idx)
                continue
            assert idx_ss not in used_indx, str(idx_ss)
            used_indx.append(idx_ss)
            if not redo:
                fldr = sess_classif[idx_ss]
                prob = issue[idx_ss]
                obs = observations[idx_ss]
                if obs.endswith('EXIT'):
                    obs = obs[:-4]
            else:
                fldr, prob, obs = 'n.c.', '', ''
            plt_f = (fldr == 'n.c.' and plot_fig) or (ignore_input and plot_fig)
            if plt_f:
                # GET DATA
                offset = inv['offset'][idx_ss]
                e_file = sess+'/e_data.npz'
                e_data = np.load(e_file, allow_pickle=1)
                samples = np.load(sess+'/ttls_sbsmpl.npz', allow_pickle=1)
                samples = samples['samples']
                # BUILD FIGURE
                f, ax_traces, idx_max =\
                    build_figure(samples=samples, e_data=e_data, offset=offset,
                                 session=session, inv=inv, inv_sbsmpld=inv_sbsmpld,
                                 margin_psth=margin_psth, num_ps=num_ps,
                                 sv_folder=sv_folder, indx=idx_ss)
                if ignore_input:
                    plt.show(block=False)
            # INPUT INFO
            if fldr == 'n.c.':
                defs = {'class': '', 'issue': '', 'obs': ''}
                if inv['sil_per'][idx_ss] > 0.01:
                    defs['issue'] = 'sil per'
                elif inv['num_stim_ttl'][idx_ss] < inv['num_stms_csv'][idx_ss]/4:
                    defs['issue'] = 'no ttl'
                elif np.max(samples) > 1000:
                    defs['issue'] = 'noise 1'
                defs['class'] = 'y' if defs['issue'] == '' else 'n'
                fldr, prob, obs = get_input(ignore=ignore_input, defaults=defs)
            if plt_f:
                f.savefig(sv_folder+fldr+'/'+session+'.png')
                ax_traces.text(idx_max, 4.25, prob+': '+obs)
                ax_traces.set_ylim([-.1, 4.5])
            if plt_f and fldr == 'bad':
                print('Saving into issues pdf')
                pdf_issues.savefig(f.number)
            elif plt_f and fldr == 'good':
                print('Saving into selected pdf')
                pdf_selected.savefig(f.number)
            if plt_f:
                plt.close(f)
            print(observations[idx_ss])
            print(issue[idx_ss])
            color = ISSS_CLR[np.where(ISSUES == issue[idx_ss])[0]][0]
            ax_tmln.plot(days, i_r, '.', color=color)
            f_tmln.savefig(sv_folder+'/sessions_timeline.png')
            # SAVE DATA
            issue[idx_ss] = prob
            sess_classif[idx_ss] = fldr
            observations[idx_ss] = obs
            print(fldr)
            assert fldr != 'n.c.'
            extended_inv = get_extended_inv(inv, sess_classif, issue,
                                            observations)
            np.savez(main_folder+'/sess_inv_extended.npz', **extended_inv)
            if obs.endswith('EXIT'):
                pdf_issues.close()
                pdf_selected.close()
                import sys
                sys.exit()
    pdf_issues.close()
    pdf_selected.close()


if __name__ == '__main__':
    plt.close('all')
    redo = False  # whether to rewrite comments
    ignore_input = False  # whether to input comments (or just save the figures)
    plot_fig = True  # whether to plot the figures
    margin_psth = 2000
    num_ps = int(1e5)  # for traces plot
    home = 'molano'  # 'manuel'
    main_folder = '/home/'+home+'/fof_data/'
    if home == 'manuel':
        sv_folder = main_folder+'/ttl_psths/'
    elif home == 'molano':
        sv_folder = '/home/molano/Dropbox/project_Barna/FOF_project/ttl_psths_bis/'

    inv = np.load(main_folder+'/sess_inv_sbsFalse.npz', allow_pickle=1)
    # np.load('/home/'+home+'/fof_data/sess_inv_sbsTrue.npz', allow_pickle=1)
    inv_sbsmpld = None
    sel_rats = []  # ['LE113']  # 'LE101'
    sel_sess = []  # ['LE101_2021-05-31_12-34-48'] ['LE104_2021-03-31_14-14-20']

    batch_sessions(main_folder=main_folder, sv_folder=sv_folder, inv=inv,
                   redo=redo, sel_sess=sel_sess, sel_rats=sel_rats,
                   plot_fig=plot_fig, inv_sbsmpld=inv_sbsmpld,
                   margin_psth=margin_psth, num_ps=num_ps,
                   ignore_input=ignore_input)
