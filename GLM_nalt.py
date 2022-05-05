#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:58:28 2020

@author: molano
"""
import os
import sys
import numpy as np
from numpy import concatenate as conc
import itertools
from numpy import logical_and as and_
from numpy import logical_or as or_
import matplotlib.pyplot as plt
# import matplotlib as mpl
import glob
import helper_functions as hf
# from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression as LR
sys.path.append(os.path.expanduser("~/neurogym"))
import neurogym.utils.plotting as pl

outcome_tag = np.array(['-', '+'])
naranja = np.array((255, 127, 0))/255
negro = np.array((0, 0, 0))
colors = conc((negro.reshape((1, 3)), naranja.reshape((1, 3))))


def check_regressors(perf_, ch_, key, des_mat_keys, ev=None,
                     start=0, num_samples=1000, tr_temp=None, tr_out=None,
                     prev_perf=None, indx_cond=None, des_mat_real_idx=None):
    plt.figure()
    plt.plot(perf_[start:start+num_samples]-2, label='perf')
    plt.plot(ch_[start:start+num_samples], label='choice')
    if tr_temp is not None:
        plt.plot(ev[start:start+num_samples], label='ev')
    if tr_temp is not None:
        plt.plot(tr_temp[start:start+num_samples], label='tr_temp')
    if tr_out is not None:
        plt.plot(tr_out[start:start+num_samples], '--', label='tr_out')
    if prev_perf is not None:
        plt.plot(prev_perf[start:start+num_samples]-2, '--', label='prev_perf')
    if indx_cond is not None:
        plt.plot(indx_cond[start:start+num_samples]-5, '--', label='indx_cond')
    if des_mat_real_idx is not None:
        plt.plot(des_mat_real_idx[start:start+num_samples]-3, '--',
                 label=key)
    for ind in range(num_samples):
        plt.plot([ind, ind], [-5, 4], '--', color=(.7, .7, .7))
    plt.legend()


def shift(mat, n=1):
    if len(mat.shape) == 1:
        mat = mat.reshape((-1, 1))
    if n == 0:
        return mat
    else:
        mat = conc((np.full((n, mat.shape[1]), 0), mat[:-n, :]))
        return mat


def get_num_regressors(n_ch, num_tr_back, merge=False):
    if merge:
        # last part corresponds to the redundant terms L and T, and to the useless
        # regressors due to conditioning on prev. outcome
        num = n_ch+n_ch*2*num_tr_back+n_ch*4*num_tr_back - (2*n_ch+2*n_ch+n_ch)
    else:
        # not taking into account L_1 and T_1
        num = n_ch+n_ch*2*num_tr_back+n_ch*n_ch*4*(num_tr_back-1)
    return num


def get_regressors_names(n_ch, trs_back=4):
    """
    input df object, since it will calculate history*, it must contain
    consecutive trials returns preprocessed dataframe.
    """
    des_mat_keys = {'ev': (0, n_ch)}
    regr_count = n_ch
    # Lateral module
    for out in range(2):
        for ind_tr in range(1, trs_back+1):
            new_keys = {'L'+outcome_tag[out]+str(ch+1)+'_'+str(ind_tr+1):
                        (regr_count+ch, regr_count+ch+1) for ch in range(n_ch)}
            des_mat_keys.update(new_keys)
            regr_count += n_ch

    # transition
    transitions = itertools.product(np.arange(n_ch)+1, repeat=2)
    for trans in transitions:
        trans_str = '-'.join([str(x) for x in trans])
        outcomes = itertools.product(np.arange(2), repeat=2)
        for out in outcomes:
            out_str = ''.join([outcome_tag[x] for x in out])
            for ind_tr in range(1, trs_back):
                key = 'T'+out_str+trans_str+'_'+str(ind_tr+1)
                des_mat_keys[key] = (regr_count, regr_count+1)
                regr_count += 1
    return des_mat_keys


def get_GLM_nalt_regressors(data, n_ch, prev_out, trs_back=4, cond_nch_blck=None,
                            prev_ch_cond=1, period=300000, pred_gt=True,
                            chck_regr=False):
    """
    input df object, since it will calculate history*, it must contain
    consecutive trials returns preprocessed dataframe.
    """
    ev_ = data['stimulus'][-period:, 1: n_ch+1]
    if pred_gt:
        gt_ = data['gt'][-period:]
        ch_ = np.argmax(ev_, axis=1)+1
        perf_ = (ch_ == gt_)*1.
    else:
        perf_ = data['performance'][-period:].astype('float')
        ch_ = data['choice'][-period:].astype('float')
    # make nan all choices larger than cond_nch_blck
    ch_[ch_ > cond_nch_blck] = np.random.choice(cond_nch_blck,
                                                size=np.sum(ch_ > cond_nch_blck))+1
    nan_indx = np.logical_and(ch_ <= cond_nch_blck, ch_ != 0)[:, None]

    print('Performance: ', np.mean(perf_))
    print('Total number of trials: ', ch_.shape[0])
    prev_perf_ = shift(perf_)
    prev_ch_ = shift(ch_)

    indx_cond = and_.reduce((prev_perf_ == prev_out, prev_ch_ == prev_ch_cond,
                            nan_indx)).flatten()
    ev_ /= np.nanmax(ev_)
    num_regressors = get_num_regressors(n_ch, trs_back)
    design_mat = np.empty((np.sum(indx_cond), num_regressors))
    design_mat[:] = 0  # np.nan
    # evidence
    design_mat[:, 0:0+ev_.shape[1]] = ev_[indx_cond]
    des_mat_keys = {'ev': (0, ev_.shape[1])}
    regr_count = ev_.shape[1]
    # Lateral module
    for out in range(2):
        lat_ = np.zeros((ev_.shape[0], n_ch))
        indx_ch = (ch_-1).astype('int')
        lat_[np.arange(lat_.shape[0]), indx_ch] = 1
        lat_[perf_ != out, :] = 0
        for ind_tr in range(1, trs_back+1):
            lat_temp = shift(lat_, n=ind_tr+1)
            new_keys = {'L'+outcome_tag[out]+str(ch+1)+'_'+str(ind_tr+1):
                        (regr_count+ch, regr_count+ch+1) for ch in range(n_ch)}
            des_mat_keys.update(new_keys)
            design_mat[:, regr_count:regr_count+lat_temp.shape[1]] =\
                lat_temp[indx_cond, :]
            regr_count += lat_temp.shape[1]

    # transition
    transitions = itertools.product(np.arange(n_ch)+1, repeat=2)
    for trans in transitions:
        trans_str = '-'.join([str(x) for x in trans])
        tr_temp = np.zeros((ev_.shape[0],))
        indx_prev_ch = (prev_ch_ == trans[0]).flatten()
        indx_trns = and_(indx_prev_ch, ch_ == trans[1])
        tr_temp[indx_trns] = 1
        outcomes = itertools.product(np.arange(2), repeat=2)
        for out in outcomes:
            out_str = ''.join([outcome_tag[x] for x in out])
            tr_out = tr_temp.copy()
            indx_out = or_((prev_perf_ != out[0]).flatten(), perf_ != out[1])
            tr_out[indx_out] = 0
            for ind_tr in range(1, trs_back):
                key = 'T'+out_str+trans_str+'_'+str(ind_tr+1)
                tr_out_temp = shift(tr_out, n=ind_tr+1)
                des_mat_keys[key] =\
                    (regr_count, regr_count+1)
                design_mat[:, regr_count:regr_count+1] = tr_out_temp[indx_cond]
                regr_count += 1
                if key == 'T--2-1_1' and chck_regr:
                    start = 370
                    num_samples = 20
                    des_mat_real_idx = np.zeros_like(perf_)
                    des_mat_real_idx[indx_cond] =\
                        design_mat[:, des_mat_keys[key][0]]
                    indx = np.ones_like(indx_cond)  # indx_cond
                    check_regressors(perf_[indx], ch_[indx],
                                     key, des_mat_keys, start=start,
                                     tr_temp=tr_temp[indx], indx_cond=indx_cond,
                                     num_samples=num_samples, tr_out=tr_out[indx],
                                     prev_perf=prev_perf_[indx],
                                     des_mat_real_idx=des_mat_real_idx[indx])
                    sys.exit()
    # get target response
    if pred_gt:
        y = gt_.astype('float')[indx_cond]
    else:
        y = ch_[indx_cond]
    print('Design matrix shape: ', str(design_mat.shape))
    return y, design_mat, des_mat_keys, np.mean(perf_)


def fit_weights(design_mat, y):
    Lreg = LR(C=1, fit_intercept=True, penalty='l2', multi_class='auto',
              solver='saga', random_state=123, max_iter=10000000, n_jobs=-1)
    Lreg.fit(design_mat, y)
    return Lreg


def get_trans_weights(weights, keys, n_ch, n_tr_bck, outcomes=None, name='',
                      prev_ch=1, plot_props={'outcms': [(1, 1)], 'tr_bk': 2,
                                             'plt_ori': True}):
    prev_ch -= 1
    if outcomes is None:
        outcomes = list(itertools.product(np.arange(2), repeat=2))
    organized_weights = {}
    for out in outcomes:
        out_str = ''.join([outcome_tag[x] for x in out])
        for ind_tr in range(n_tr_bck):
            plt_flag = out in plot_props['outcms'] and ind_tr < plot_props['tr_bk']
            indx = [k[0] for x, k in keys.items() if x.find('T'+out_str) != -1 and
                    x.find('_'+str(ind_tr+1))+1+len(str(ind_tr+1)) == len(x)]
            # sel_keys = [x for x in keys if x.find('T'+out_str) != -1 and
            #             x.find('_'+str(ind_tr+1))+1+len(str(ind_tr+1)) == len(x)]
            # print(sel_keys)
            label = 'T'+out_str+'_'+str(ind_tr+1)
            if len(indx) != 0:
                ws = weights[:, indx]
                if plt_flag:
                    plt.figure()
                ch_order = np.arange(n_ch)
                ch_order[0] = prev_ch
                ch_order[prev_ch] = 0
                nxt_ch_ws = []
                max_ = -10000
                min_ = 10000
                plots = []
                for ind_next_ch, next_ch in enumerate(ch_order):
                    ws_reordered = ws[next_ch, :].copy()
                    ws_reordered = ws_reordered.reshape((n_ch, n_ch))
                    row_0 = ws_reordered[0, :].copy()
                    ws_reordered[0, :] = ws_reordered[prev_ch, :]
                    ws_reordered[prev_ch, :] = row_0
                    col_0 = ws_reordered[:, 0].copy()
                    ws_reordered[:, 0] = ws_reordered[:, prev_ch]
                    ws_reordered[:, prev_ch] = col_0
                    if ind_next_ch > 1:
                        row_1 = ws_reordered[1, :].copy()
                        ws_reordered[1, :] = ws_reordered[ind_next_ch, :]
                        ws_reordered[ind_next_ch, :] = row_1
                        col_1 = ws_reordered[:, 1].copy()
                        ws_reordered[:, 1] = ws_reordered[:, ind_next_ch]
                        ws_reordered[:, ind_next_ch] = col_1
                    nxt_ch_ws.append(ws_reordered)
                    if plt_flag:
                        rws_cls = int(np.ceil(np.sqrt(n_ch)))
                        plt.subplot(rws_cls, rws_cls, ind_next_ch+1)
                        im = plt.imshow(ws_reordered)
                        plots.append(im)
                        plt.title('REORDERED prev. outcm: '+name+'  '+out_str+'_' +
                                  str(ind_tr+1) + ' nxt-ch: '+str(next_ch+1))
                        plt.yticks(np.arange(n_ch),
                                   labels=[str(x+1) for x in ch_order])
                        plt.ylim([-0.5, n_ch-0.5])
                        plt.xticks(np.arange(n_ch),
                                   labels=[str(x+1) for x in ch_order])
                        plt.xlim([-0.5, n_ch-0.5])
                        max_ = max(max(ws_reordered.flatten()), max_)
                        min_ = min(min(ws_reordered.flatten()), min_)
                organized_weights[label] = nxt_ch_ws
                if plt_flag:
                    for im in plots:
                        im.set_clim(min_, max_)
                if plot_props['plt_ori'] and plt_flag:
                    plt.figure()
                    for ind_next_ch in range(n_ch):
                        plt.subplot(rws_cls, rws_cls, ind_next_ch+1)
                        plt.imshow(ws[ind_next_ch, :].reshape((n_ch, n_ch)))
                        plt.title('prev. outcm: '+name+'  '+out_str+'_' +
                                  str(ind_tr+1)+' nxt-ch: '+str(ind_next_ch+1))
                        plt.yticks(np.arange(n_ch),
                                   labels=[str(x+1) for x in np.arange(n_ch)])
                        plt.ylim([-0.5, n_ch-0.5])
                        plt.xticks(np.arange(n_ch),
                                   labels=[str(x+1) for x in np.arange(n_ch)])
                        plt.xlim([-0.5, n_ch-0.5])
    return organized_weights


def plot_merged_weights(all_ws, tag='T++', tr_back_plt=3, sv_folder='',
                        plot_indiv_mats=True, figs=None):
    trans_flag = tag.find('T') != -1
    max_ = -100000  # max_dict(all_ws)
    min_ = 100000  # min_dict(all_ws)
    plots = []
    n_ch = all_ws[0]['after-']['T++_2'][0].shape[0] if tag.find('T') != -1\
        else all_ws[0]['after-']['L+_2'].shape[0]
    ncols = 2 if n_ch > 2 else 1
    for pr_out in range(2):
        if figs is None:
            fig, ax = plt.subplots(nrows=tr_back_plt, ncols=ncols, figsize=(6, 8))
            sv_figs = True
        else:
            ax = figs[pr_out]
            sv_figs = False
        ax = ax.reshape(ax.shape[0], -1)
        ws_pr_out = [x['after'+outcome_tag[pr_out]] for x in all_ws]
        for i_tr in range(1, tr_back_plt+1):
            i_pl = i_tr-1
            if trans_flag:
                mean_all_ws = np.mean(np.array([x[tag+'_'+str(i_tr+1)]
                                                for x in ws_pr_out]), axis=0)
            else:
                lat_minus = np.mean(np.array([x[tag+'-_'+str(i_tr+1)]
                                              for x in ws_pr_out]), axis=0)
                lat_plus = np.mean(np.array([x[tag+'+_'+str(i_tr+1)]
                                             for x in ws_pr_out]), axis=0)
                mean_all_ws = [lat_minus, lat_plus]
            for ind_splt in range(ncols):
                if ind_splt == 0:
                    mean_aux = mean_all_ws[ind_splt]
                    aux = ax[i_pl, ind_splt].imshow(mean_aux,
                                                    cmap='viridis')
                else:
                    mean_aux = np.mean(mean_all_ws[ind_splt:], axis=0)
                    aux = ax[i_pl, ind_splt].imshow(mean_aux,
                                                    cmap='viridis')
                min_ = min(min(mean_aux.flatten()), min_)
                max_ = max(max(mean_aux.flatten()), max_)
                plots.append(aux)
                ax[i_pl, ind_splt].set_ylabel('Lag '+str(i_tr+1))
                ax[i_pl, ind_splt].set_ylim([-0.5, n_ch-0.5])
                ax[i_pl, ind_splt].set_xlim([-0.5, n_ch-0.5])
                if trans_flag and i_tr == 1:
                    if ind_splt == 0:
                        ax[i_pl, ind_splt].set_yticks([0])
                        ax[i_pl, ind_splt].set_yticklabels(['Previous \n choice'])
                        ax[i_pl, ind_splt].set_xticks([0])
                        ax[i_pl, ind_splt].set_xticklabels(['Next \n choice'])
                        title = 'After '+outcome_tag[pr_out]+' '+tag +\
                            ' (weights for prev. ch)'
                    else:
                        ax[i_pl, ind_splt].set_yticks([])
                        ax[i_pl, ind_splt].set_xticks([1])
                        ax[i_pl, ind_splt].set_xticklabels(['Next \n choice'])
                        title = 'After '+outcome_tag[pr_out]+' '+tag +\
                            ' (weights for other chs)'
                elif i_tr == 1:
                    if ind_splt == 0:
                        ax[i_pl, ind_splt].set_yticks([0])
                        ax[i_pl, ind_splt].set_yticklabels(['Previous \n choice'])
                        ax[i_pl, ind_splt].set_xticks([0])
                        ax[i_pl, ind_splt].set_xticklabels(['Lateral weight \n' +
                                                            ' prev. choice'])
                        title = 'After '+outcome_tag[pr_out]+' '+tag+'-'
                    else:
                        ax[i_pl, ind_splt].set_xticks([])
                        ax[i_pl, ind_splt].set_yticks([])
                        title = 'After '+outcome_tag[pr_out]+' '+tag+'+'
                else:
                    title = ''
                    ax[i_pl, ind_splt].set_xticks([])
                    ax[i_pl, ind_splt].set_yticks([])
                ax[i_pl, ind_splt].set_title(title)
            if trans_flag and plot_indiv_mats:
                plt.figure()
                rws_cls = int(np.ceil(np.sqrt(n_ch)))
                for ind_splt in range(n_ch):
                    plt.subplot(rws_cls, rws_cls, ind_splt+1)
                    plt.imshow(mean_all_ws[ind_splt], vmax=max_, vmin=min_)
                    if ind_splt == 0:
                        plt.title('Prev. outcm: '+str(pr_out)+' '+tag+'_' +
                                  str(i_tr+1))
                    plt.yticks(np.arange(n_ch),
                               labels=[str(x+1) for x in np.arange(n_ch)])
                    plt.ylim([-0.5, n_ch-0.5])
                    plt.xticks(np.arange(n_ch),
                               labels=[str(x+1) for x in np.arange(n_ch)])
                    plt.xlim([-0.5, n_ch-0.5])
                mean_all_ws = np.mean(np.array([x[tag+'_'+str(i_tr+1)]
                                                for x in ws_pr_out]), axis=0)
        if sv_folder != '' and sv_figs:
            for plot in plots:
                plot.set_clim((min_, max_))
            fig.savefig(sv_folder+'regrss_ws_'+tag+'_outcm_'+str(pr_out)+'.png',
                        dpi=400, bbox_inches='tight')
            # plt.close(fig)
    return min_, max_, plots


def get_lat_weights(weights, keys, n_ch, n_tr_bck, outcomes=None, name='',
                    pr_ch=1, plot_props={'outcms': [1], 'tr_bk': 2,
                                         'plt_ori': True}):
    organized_weights = {}
    pr_ch -= 1
    if outcomes is None:
        outcomes = [0, 1]
    for out in outcomes:
        out_str = outcome_tag[out]
        for ind_tr in range(n_tr_bck+1):
            plt_flag = out in plot_props['outcms'] and ind_tr < plot_props['tr_bk']

            indx = [k[0] for x, k in keys.items() if x.find('L'+out_str) != -1 and
                    x.find('_'+str(ind_tr+1)) != -1]
            labels = [x for x, k in keys.items() if x.find('L'+out_str) != -1 and
                      x.find('_'+str(ind_tr+1)) != -1]
            if len(indx) != 0:
                # organize matrix
                ch_order = np.arange(n_ch)
                ch_order[0] = pr_ch
                ch_order[pr_ch] = 0
                ws = weights[:, indx]
                ws_reordered = ws.copy()
                row_0 = ws_reordered[0, :].copy()
                ws_reordered[0, :] = ws_reordered[pr_ch, :]
                ws_reordered[pr_ch, :] = row_0
                col_0 = ws_reordered[:, 0].copy()
                ws_reordered[:, 0] = ws_reordered[:, pr_ch]
                ws_reordered[:, pr_ch] = col_0
                organized_weights['L'+out_str+'_'+str(ind_tr+1)] = ws_reordered
                if plt_flag:
                    plt.figure()
                    plt.imshow(ws_reordered, aspect='auto')
                    plt.title('SORTED Lateral prev. outcome: '+name+'  ' +
                              out_str+'_'+str(ind_tr+1)+' prev-ch: '+str(pr_ch+1))
                    plt.xticks(np.arange(len(indx)),
                               labels=np.array(labels)[ch_order])
                    plt.yticks(np.arange(n_ch),
                               labels=[str(x+1) for x in ch_order])
                    plt.ylim([-0.5, n_ch-0.5])
                if plot_props['plt_ori'] and plt_flag:
                    plt.figure()
                    plt.imshow(ws, aspect='auto')
                    plt.title('Lateral prev. outcome: '+name+'  '+out_str+'_' +
                              str(ind_tr+1)+' prev-ch: '+str(pr_ch+1))
                    plt.xticks(np.arange(len(indx)), labels=labels)
                    plt.yticks(np.arange(n_ch),
                               labels=[str(x+1) for x in np.arange(n_ch)])
                    plt.ylim([-0.5, n_ch-0.5])
    return organized_weights


def get_direct_transition_ws(weights, keys, n_ch, n_tr_bck, outcomes=None,
                             prev_ch=''):
    if outcomes is None:
        outcomes = list(itertools.product(np.arange(2), repeat=2))
    new_weights = {}
    for out in outcomes:
        out_str = ''.join([outcome_tag[x] for x in out])
        new_weights[out_str] = np.empty((n_tr_bck,))
        new_weights[out_str][:] = 0
        for ind_tr in range(n_tr_bck):
            for ch in range(n_ch):
                regr = 'T'+out_str+prev_ch+'-'+str(ch+1)+'_'+str(ind_tr+1)
                if regr in keys.keys():
                    indx = keys[regr][0]
                    ws = weights[ch, indx]
                else:
                    ws = np.nan
                new_weights[out_str][ind_tr] += ws
        new_weights[out_str] = np.array(new_weights[out_str])
    return new_weights


def get_all_transition_ws(weights, keys, n_ch, n_tr_bck, regr_type, outcomes=None):
    if outcomes is None:
        if regr_type == 'L':
            outcomes = [0, 1]
        elif regr_type == 'T':
            outcomes = list(itertools.product(np.arange(2), repeat=2))
    new_weights = {}
    for out in outcomes:
        out_str = ''.join([outcome_tag[x] for x in out])
        new_weights[out_str] = np.empty((n_tr_bck,))
        new_weights[out_str][:] = 0
        for ind_tr in range(n_tr_bck):
            indx = [k[0] for x, k in keys.items()
                    if x.find(regr_type+out_str) != -1 and
                    x.find('_'+str(ind_tr+1)) != -1]
            if len(indx) != 0:
                ws = weights[:, indx]
            else:
                ws = np.nan
            new_weights[out_str][ind_tr] = np.sum(np.abs(ws))
        new_weights[out_str] = np.array(new_weights[out_str])
    return new_weights


def get_lateral_weights(weights, keys, n_ch, n_tr_bck, outcomes=None, name=''):
    if outcomes is None:
        outcomes = [0, 1]
    new_weights = {}
    for out in outcomes:
        out_str = outcome_tag[out]
        new_weights[out_str] = np.empty((n_tr_bck+1,))
        new_weights[out_str][:] = np.nan
        for ind_tr in range(n_tr_bck+1):
            indx = [k[0] for x, k in keys.items() if x.find('L'+out_str) != -1 and
                    x.find('_'+str(ind_tr+1)) != -1]
            labels = [x for x, k in keys.items() if x.find('L'+out_str) != -1 and
                      x.find('_'+str(ind_tr+1)) != -1]
            if len(indx) != 0:
                ws = weights[:, indx]
                new_weights[out_str][ind_tr] =\
                    np.sum(ws[np.arange(ws.shape[0]), np.arange(len(indx))])
                if ind_tr < 3 and False:
                    plt.figure()
                    plt.imshow(ws, aspect='auto')
                    plt.title('Lateral prev. outcome: ' +
                              name+'  '+out_str+'_'+str(ind_tr+1))
                    plt.xticks(np.arange(len(indx)), labels=labels)
                    plt.yticks(np.arange(n_ch),
                               labels=[str(x+1) for x in np.arange(n_ch)])
                    plt.ylim([-0.5, n_ch-0.5])
            else:
                new_weights[out_str][ind_tr] = np.nan

        new_weights[out_str] = np.array(new_weights[out_str])
    return new_weights


def put_weights_together_merged_regressors(weights, keys, n_ch, n_tr_bck,
                                           outcomes=None):
    if outcomes is None:
        outcomes = list(itertools.product(np.arange(2), repeat=2))
    new_weights = {}
    for out in outcomes:
        out_str = ''.join([outcome_tag[x] for x in out])
        new_weights[out_str] = []
        for ind_ch in range(n_ch):
            indx = [k[0] for x, k in keys.items()
                    if x.find(out_str+str(ind_ch+1)) != -1]
            print([x for x, k in keys.items()
                   if x.find(out_str+str(ind_ch+1)) != -1])
            ws = weights[ind_ch, indx]
            if len(ws) == n_tr_bck - 1:
                ws = conc((np.array([np.nan]), ws))
            assert len(ws) == n_tr_bck
            new_weights[out_str].append(ws)
    return new_weights


def get_kernels(all_ws, tag, tr_back_plt):
    kernels = []
    for pr_out in range(2):
        kernel_out = []
        ws_pr_out = [x['after'+outcome_tag[pr_out]] for x in all_ws]
        for ind_tr in range(1, tr_back_plt):
            mean_all_ws = np.mean(np.array([x[tag+'_'+str(ind_tr+1)]
                                            for x in ws_pr_out]), axis=0)
            if tag.find('T') != -1:
                direct_regr = np.append(mean_all_ws[1:, 0, 1],
                                        mean_all_ws[0, 0, 0])
            else:
                direct_regr = np.mean(np.diagonal(mean_all_ws))
            kernel_out.append(np.mean(direct_regr))
        kernels.append(kernel_out)
    return kernels


def plot_kernels(all_ws_tr, all_ws_lat, trs_back, figs=None, norm=False, folder='',
                 **kwargs):
    plot_opts = {'lw': 1,  'label': '', 'alpha': 1, 'color_ac': naranja,
                 'color_ae': (0, 0, 0), 'lstyle_ac': '-',  'lstyle_ae': '-'}
    plot_opts.update(kwargs)
    opts_ac = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts_ac['color'] = plot_opts['color_ac']
    opts_ac['linestyle'] = plot_opts['lstyle_ac']
    opts_ae = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts_ae['color'] = plot_opts['color_ae']
    opts_ae['linestyle'] = plot_opts['lstyle_ae']

    if figs is None:
        f_tr, ax_tr = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True,
                                   figsize=(8, 6))
        f_lat, ax_lat = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True,
                                     figsize=(8, 4))

        save_figs = True
    else:
        ax_tr = figs[0]
        ax_lat = figs[1]
        save_figs = False
    ax_tr = ax_tr.flatten()
    for i_tg, tg in enumerate(['T++', 'T-+', 'T+-', 'T--']):
        ker = get_kernels(all_ws=all_ws_tr, tag=tg, tr_back_plt=trs_back)
        norm_factor = 1./np.abs(ker[1][0]) if norm else 1.
        ax_tr[i_tg].plot(np.array(ker[0])*norm_factor, **opts_ae)
        ax_tr[i_tg].plot(np.array(ker[1])*norm_factor, **opts_ac)
        ax_tr[i_tg].set_title(tg+' direct transitions')
        ax_tr[i_tg].set_xticks(np.arange(trs_back))
        ax_tr[i_tg].set_xticklabels(['lag -'+str(x)
                                     for x in np.arange(2, trs_back+2)])
        ax_tr[i_tg].set_xlabel('Trials back')
        ax_tr[i_tg].set_ylabel('Mean weight')
    ax_lat = ax_lat.flatten()
    for i_tg, tg in enumerate(['L+', 'L-']):
        ker = get_kernels(all_ws=all_ws_lat, tag=tg, tr_back_plt=trs_back)
        ax_lat[i_tg].plot(ker[0], **opts_ae)
        ax_lat[i_tg].plot(ker[1], **opts_ac)
        ax_lat[i_tg].set_title(tg+' direct regressors')
        ax_lat[i_tg].set_xticks(np.arange(trs_back))
        ax_lat[i_tg].set_xticklabels(['lag -'+str(x)
                                      for x in np.arange(2, trs_back+2)])
        ax_lat[i_tg].set_xlabel('Trials back')
        ax_lat[i_tg].set_ylabel('Mean weight')
    if save_figs:
        f_tr.savefig(folder+'/kernels_nGLM_trans.png', dpi=400,
                     bbox_inches='tight')
        f_lat.savefig(folder+'/kernels_nGLM_lat.png', dpi=400,
                      bbox_inches='tight')
        plt.close(f_tr)
        plt.close(f_lat)


def basic_agent(data, tr_window=40, lambd_hist=0.1, tau=20,
                period=1000000, reset=''):
    """
    Simulate a basic agent that uses passed transitions and stimulus to make
    a choice. If reset == '_reset', it will only do so after a correct trial

    Parameters
    ----------
    data : dict
        Experiment data including ground truth and stimulus
    tr_window : int, optional
        Number of trials back to compute bias. The default is 40.
    tau : float, optional
        Decay of weight for transitions. The default is 20.
    period : int, optional
        How many trials to consider for the simulation. The default is 1000000.
    lambd_hist: float, optional
        Weight of the trial history in the agent's choice. The default is 0.1.
    reset : str, optional
        If = '_reset', the agent will only use past history after a correct trial.
        The default is ''.

    Returns
    -------
    data : dict
       Dictionary containing the choices made by the simulated agent

    """
    reset = reset == '_reset'
    gt = data['gt'][-period:]
    n_ch = np.max(gt)
    stim = data['stimulus'][-period:, 1:n_ch+1]
    choices_hist = []
    choices = []
    prev_choice = -1
    for ind_trial in range(gt.shape[0]):
        if ind_trial % 100000 == 0:
            print(str(ind_trial)+' out of '+str(gt.shape[0]))
        stim_ = stim[ind_trial, :]
        indx_ch = np.where(choices_hist == prev_choice)[0]+1
        indx_ch = indx_ch[indx_ch < len(choices_hist)-1]
        tr_hist = np.zeros((n_ch,))
        if len(indx_ch) != 0:
            ch_cand = np.array(choices_hist)[indx_ch]-1
            indx_ch = indx_ch[ch_cand >= 0]
            ch_cand = ch_cand[ch_cand >= 0]
            if len(ch_cand) != 0:
                for ind_ch, ch_c in enumerate(ch_cand):
                    tr_hist[ch_c] += np.exp((indx_ch[ind_ch]-min(len(choices_hist),
                                                                 tr_window))/tau)
        logit = stim_+lambd_hist*tr_hist
        exps = np.exp(logit)
        sum_exps = np.sum(np.exp(logit))
        choice = np.argmax(exps/sum_exps) + 1
        outcome = choice == gt[ind_trial]
        if outcome == 1:
            choices_hist.append(choice)
            prev_choice = choice
        else:
            choices_hist.append(-1)
            prev_choice = 0 if reset else choice
        choices_hist = choices_hist[-tr_window:]
        choices.append(choice)
        # print(str(ind_trial)+' out of '+str(gt.shape[0]))
        # print('choice')
        # print(choice)
        # print('gt')
        # print(gt[ind_trial])
        # print('outcome')
        # print(outcome)
        # print('previous choices')
        # print(choices_hist)
        # print('previous choice')
        # print(prev_choice)
        # print('----------------------')

    # exps = np.exp(stim)
    # sum_exps = np.sum(np.exp(stim), axis=1)
    # ch_io = np.argmax(exps/sum_exps[:, None], axis=1)+1
    print('Performance subject:')
    print(np.mean(data['performance']))
    data['performance'] = gt == choices
    print('Performance basic agent:')
    print(np.mean(data['performance']))
    data['choice'] = np.array(choices)
    return data


def analyse_exp_cond_prev_ch(folder, outcomes_tr=[(1, 1), (0, 1), (1, 0), (0, 0)],
                             outcomes_lat=[0, 1], period=10000000, trs_back=8,
                             sim_data=False, reload=False, rerun=False, pr_ch=1,
                             tst_s=1000, pred_gt=False, plot_extra=False,
                             n_ch=None, **sim_props):
    sim = '_simulated_agent'+sim_props['reset'] if sim_data else ''
    pred_gt_str = 'predicting_gt' if pred_gt else ''
    if not os.path.exists(folder+'/bhvr_data_all.npz') or reload:
        data = pl.put_together_files(folder)
    else:
        data = hf.load_behavioral_data(folder+'bhvr_data_all.npz')
    if data:
        params = np.load(folder+'/params.npz', allow_pickle=1)
        n_ch = n_ch or params['task_kwargs'].item()['n_ch']

        if sim_data:
            agent_name = folder+sim+'.npz'
            if not os.path.exists(agent_name) or sim_props['rerun_sim']:
                del sim_props['rerun_sim']
                data = basic_agent(data, period=period, **sim_props)
                np.savez(agent_name, **data)
                rerun = True
            else:
                data = np.load(agent_name)
        all_ws_tr = {}
        all_ws_lat = {}
        scores = []
        for prev_out in range(2):
            fit_name = folder+'fit_'+outcome_tag[prev_out]+sim+'_' +\
                str(pr_ch)+'_'+pred_gt_str+'.npz'
            print(fit_name)
            if not os.path.exists(fit_name) or rerun:
                y,  design_mat, des_mat_keys, perf =\
                    get_GLM_nalt_regressors(data, prev_out=prev_out,
                                            trs_back=trs_back, n_ch=n_ch,
                                            cond_nch_blck=n_ch, pred_gt=pred_gt,
                                            prev_ch_cond=pr_ch, period=period)
                Lreg = fit_weights(design_mat=design_mat[:-tst_s], y=y[:-tst_s])
                fit = {'design_mat': design_mat, 'des_mat_keys': des_mat_keys,
                       'weights': Lreg.coef_, 'outcomes_tr': outcomes_tr,
                       'outcomes_lat': outcomes_lat, 'period': period, 'y': y,
                       'pr_ch': pr_ch, 'mrg_cmmn_or': False, 'perf': perf,
                       'score': Lreg.score(design_mat[-tst_s:], y[-tst_s:])}
                np.savez(fit_name, **fit)
            fit = np.load(fit_name, allow_pickle=1)
            des_mat_keys = fit['des_mat_keys'].item()
            if n_ch == 2:
                weights = conc((-fit['weights'], fit['weights']), axis=0)
            else:
                weights = fit['weights']
            if 'score' in fit.keys():
                score = fit['score']
            else:
                des_mat = fit['design_mat']
                y, _, _, perf =\
                    get_GLM_nalt_regressors(data, prev_out=prev_out,
                                            trs_back=trs_back, n_ch=n_ch,
                                            cond_nch_blck=n_ch, pred_gt=pred_gt,
                                            prev_ch_cond=pr_ch, period=period)
                score = np.mean((np.argmax(np.matmul(des_mat, weights.T),
                                           axis=1)+1) == y)/perf
            scores.append(score)
            # PLOTTING
            if plot_extra:
                plt.figure()
                plt.imshow(fit['design_mat'], aspect='auto')
                #
                plt.figure()
                plt.imshow(weights[:, n_ch:], aspect='auto')
                plt.xticks(np.arange(weights.shape[1]-n_ch),
                           labels=[x for x in des_mat_keys.keys() if x != 'ev'])
                plt.yticks(np.arange(n_ch),
                           labels=[str(x+1) for x in np.arange(n_ch)])
                plt.title('Previous outcome: '+outcome_tag[prev_out])
            ws_lat = get_lat_weights(weights=weights, keys=des_mat_keys,
                                     n_ch=n_ch, n_tr_bck=trs_back,
                                     name=str(prev_out), outcomes=outcomes_lat,
                                     pr_ch=pr_ch, plot_props={'outcms': [1],
                                                              'tr_bk': 0,
                                                              'plt_ori': False})
            all_ws_lat['after'+outcome_tag[prev_out]] = ws_lat
            ws_tr = get_trans_weights(weights=weights, keys=des_mat_keys,
                                      n_ch=n_ch, n_tr_bck=trs_back,
                                      outcomes=outcomes_tr, prev_ch=pr_ch,
                                      name=str(prev_out),
                                      plot_props={'outcms': [(1, 1)],
                                                  'tr_bk': 0,  # 2*(prev_out == 1),
                                                  'plt_ori': False})
            all_ws_tr['after'+outcome_tag[prev_out]] = ws_tr

        return all_ws_tr, all_ws_lat, scores
    else:
        return {}, {}, []


def max_dict(dict_):
    if isinstance(dict_, (float, int)):
        return dict_
    if isinstance(dict_, (list, np.ndarray)):
        return np.max([max_dict(x) for x in dict_])
    return np.max([max_dict(x) for x in dict_.values()])


def min_dict(dict_):
    if isinstance(dict_, (float, int)):
        return dict_
    if isinstance(dict_, (list, np.ndarray)):
        return np.min([min_dict(x) for x in dict_])
    return np.min([min_dict(x) for x in dict_.values()])


def analyse_full_exp(folder, rerun=False, sim_data=False, outcms_lat=[0, 1],
                     trs_back=8, n_ch=None, period=20000000, figs_krnls=None,
                     outcms_tr=[(1, 1), (0, 1), (1, 0), (0, 0)], figs_mats=None,
                     pred_gt=False, plt_knl_ops={}, plt_mats=True, norm_kls=False,
                     ax_scrs=None, reload=False, sim_prop={'tr_window': 8,
                                                           'tau': 2,  'reset': '',
                                                           'lambd_hist': 0.1,
                                                           'rerun_sim': False}):
    all_ws_tr = []
    all_ws_lat = []
    all_scores = []
    sim = '_simulated_agent'+sim_prop['reset'] if sim_data else ''
    pred_gt_str = 'predicting_gt' if pred_gt else ''
    fit_name = folder+sim+'_'+pred_gt_str+'_'
    n_ch = n_ch or 2
    for pr_ch in range(n_ch):
        ws_tr, ws_lat, scrs =\
            analyse_exp_cond_prev_ch(folder, outcomes_tr=outcms_tr, rerun=rerun,
                                     outcomes_lat=outcms_lat, period=period,
                                     pr_ch=pr_ch+1, sim_data=sim_data, n_ch=n_ch,
                                     plot_extra=False, trs_back=trs_back,
                                     pred_gt=pred_gt, reload=reload, **sim_prop)
        if len(ws_tr) > 0:
            all_ws_tr.append(ws_tr)
            all_ws_lat.append(ws_lat)
            all_scores.append(scrs)
    if len(all_ws_tr):
        plot_kernels(all_ws_tr, all_ws_lat, trs_back, figs=figs_krnls,
                     norm=norm_kls, folder=folder, **plt_knl_ops)
        if ax_scrs is not None:
            scrs = np.mean(np.array(all_scores), axis=0).reshape((-1, 2))
            scrs_std = np.std(np.array(all_scores), axis=0).reshape((-1, 2))
            ax_scrs['ax'].errorbar(ax_scrs['indx'], scrs[:, 1], scrs_std[:, 1],
                                   marker='+', color=naranja,
                                   label=ax_scrs['lbl'][0])
            ax_scrs['ax'].errorbar(ax_scrs['indx'], scrs[:, 0], scrs_std[:, 0],
                                   marker='+', color=(0, 0, 0),
                                   label=ax_scrs['lbl'][1])
        if plt_mats:
            if figs_mats is None:
                tr_back = 2
                f, ax = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))
                cax = f.add_axes([0.25, 0.03, 0.5, 0.02])
                abs_min = 1000000
                abs_max = -1000000
                all_plots = []
            for i_t, tr in enumerate(['T++', 'T-+', 'T+-', 'T--']):
                min_, max_, plots = plot_merged_weights(all_ws=all_ws_tr, tag=tr,
                                                        tr_back_plt=tr_back,
                                                        plot_indiv_mats=False,
                                                        sv_folder='',
                                                        figs=[ax[:2, i_t],
                                                              ax[2:, i_t]])
                abs_min = min(abs_min, min_)
                abs_max = max(abs_max, max_)
                all_plots.append(plots)
            all_plots = np.array(all_plots).flatten()
            for plts in all_plots:
                plts.set_clim((abs_min, abs_max))
            f.colorbar(plts, cax=cax, orientation='horizontal')
            f.suptitle(folder+' ('+str(abs_max)+', '+str(abs_min)+')')
            if figs_mats is None:
                f.savefig(fit_name+'regrss_ws.png')
            plot_merged_weights(all_ws=all_ws_lat, tag='L',
                                plot_indiv_mats=False, sv_folder=fit_name)
    return [all_ws_tr, all_ws_lat]


if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) == 1:
        # exp = 'unbalance_exps_4_test'
        # exp = 'unbalance_exps_4_2AFC_test'
        # exp = 'unbalance_exps_4'
        # exp = 'unbalance_exps_4_test'
        # exp = 'balance_exps_2_2AFC_test'
        exp = 'rats'
        # exp = 'diff_num_tr'
        # exp = 'var_n_ch_16'
        if exp == 'unbalance_exps_8':
            n_ch = 8
            trs_back = 5
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'larger_nets_longer_blks_bal_rand_mtrx/alg_ACER_seed_0_n_ch_8/'
        elif exp == 'old_exps_2':
            n_ch = 2
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'rand_trans_mtrx/alg_ACER_seed_4_n_ch_2/'
        elif exp == 'new_exps_2':
            n_ch = 2
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'rand_trans_mtrx_n_ch_8/alg_ACER_seed_5_n_ch_2/'
        elif exp == 'balance_exps_4_test':
            n_ch = 4
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'balanced_rand_mtrx_n_ch_4/alg_ACER_seed_6_n_ch_4/test/'
        elif exp == 'unbalance_exps_4':
            n_ch = 4
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'larger_nets_longer_blks_bal_rand_mtrx/alg_ACER_seed_2_n_ch_4/'
        elif exp == 'unbalance_exps_4_test':
            n_ch = 4
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'larger_nets_longer_blks_bal_rand_mtrx/alg_ACER_seed_2_n_ch_4/' +\
                'test/'
        elif exp == 'unbalance_exps_4_2AFC_test':
            n_ch = 4
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'larger_nets_longer_blks_bal_rand_mtrx/alg_ACER_seed_2_n_ch_4/' +\
                'test_2AFC/'
        elif exp == 'unbalance_exps_8_2AFC_test':
            n_ch = 8
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'larger_nets_longer_blks_bal_rand_mtrx/alg_ACER_seed_0_n_ch_8/' +\
                'test_2AFC/'
        elif exp == 'balance_exps_2_2AFC_test':
            n_ch = 2
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'balanced_rand_mtrx_n_ch_4/alg_ACER_seed_0_n_ch_2/' +\
                'test_2AFC/'
        elif exp == 'unbalance_exps_2_2AFC_test':
            n_ch = 2
            trs_back = 8
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'larger_nets_longer_blks_bal_rand_mtrx/alg_ACER_seed_0_n_ch_2/' +\
                'test_2AFC/'
        elif exp == 'rats':
            n_ch = 2
            trs_back = 12
            folder = '/home/molano/priors/rats/data_Ainhoa/Rat13Data20151113/'
        elif exp == '3_ctxts_balanced_exps_test':
            n_ch = 8
            trs_back = 6
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'ev_3blk_balanced_long_blocks/alg_ACER_seed_1_n_ch_8/test/'
        elif exp == 'diff_num_tr':
            n_ch = 2
            trs_back = 12
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'diff_num_units/alg_ACER_n_lstm_64_seed_3/test_2AFC/'
        elif exp == 'var_n_ch_16':
            n_ch = 2
            trs_back = 12
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'var_nch_predef_mats_larger_nets/' +\
                'alg_ACER_seed_6_n_ch_16_psych_curves_and_nGLM/test_2AFC/'
        elif exp == 'var_n_ch_2':
            n_ch = 2
            trs_back = 12
            folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
                'var_nch_predef_mats_larger_nets/' +\
                'alg_ACER_seed_6_n_ch_2_psych_curves_and_nGLM/test_2AFC/'
        analyse_full_exp(folder, rerun=True, sim_data=False, period=2000000,
                         outcms_lat=[0, 1], trs_back=trs_back, n_ch=n_ch,
                         outcms_tr=[(1, 1), (0, 1), (1, 0), (0, 0)], plt_mats=True,
                         pred_gt=False, sim_prop={'tr_window': 8, 'tau': 2,
                                                  'lambd_hist': 0.1, 'reset': '',
                                                  'rerun_sim': False})
    elif len(sys.argv) == 2:
        colors = hf.COLORES
        trs_back = 6
        # '/home/molano/priors/AnnaKarenina_experiments/balanced_rand_mtrx_n_ch_4/'
        main_folder = sys.argv[1]
        folders = glob.glob(main_folder+'/alg_*')
        max_num_ch =\
            np.max([int(os.path.basename(x)[os.path.basename(x).find('n_ch_')+5:])
                    for x in folders])
        for i_nk, norm_krnls in enumerate([False, True]):
            norm_krls_str = 'norm_krnls' if norm_krnls else ''
            f_nglm_tr, ax_tr = plt.subplots(nrows=2, ncols=2,
                                            figsize=(8, 6))
            axs_nglm_krnls = [ax_tr]
            f_nglm_lat, ax_lat = plt.subplots(nrows=1, ncols=2,
                                              figsize=(8, 4))
            axs_nglm_krnls.append(ax_lat)
            if i_nk == 0:
                f_scrs, ax_scrs = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
                rerun = True
            for i_f, f in enumerate(folders):
                print(f)
                name = os.path.basename(f)
                n_ch = int(name[name.find('n_ch_')+5:])
                if n_ch <= 30:
                    if i_nk == 0:
                        ax_scrs_info = {'ax': ax_scrs, 'indx': n_ch}
                        ax_scrs_info['lbl'] = ['After correct', 'After error']\
                            if i_f == 0 else ['', '']
                    else:
                        ax_scrs_info = None
                    alpha = n_ch/max_num_ch
                    plot_opts = {'lw': 1,  'label': name, 'alpha': alpha,
                                 'color_ac': colors[0], 'color_ae': colors[0],
                                 'lstyle_ac': '-',  'lstyle_ae': '--'}
                    analyse_full_exp(folder=f+'/test/', rerun=rerun,
                                     sim_data=False, period=10000000, n_ch=2,
                                     outcms_lat=[0, 1], trs_back=trs_back,
                                     figs_krnls=axs_nglm_krnls, pred_gt=True,
                                     outcms_tr=[(1, 1), (0, 1), (1, 0), (0, 0)],
                                     plt_knl_ops=plot_opts, plt_mats=False,
                                     norm_kls=norm_krnls, ax_scrs=ax_scrs_info)
            rerun = False
            ax_tr[1][1].legend()
            ax_lat[1].legend()
            f_nglm_tr.savefig(main_folder+'/nGLM_kernels_transition_' +
                              norm_krls_str+'.png', dpi=400, bbox_inches='tight')
            f_nglm_lat.savefig(main_folder+'/nGLM_kernels_lateral_' +
                               norm_krls_str+'.png', dpi=400, bbox_inches='tight')
            if i_nk == 0:
                ax_scrs.set_xlabel('Number of choices')
                ax_scrs.set_ylabel('Accuracy')
                ax_scrs.legend()
                f_scrs.savefig(main_folder+'/scores.png', dpi=400,
                               bbox_inches='tight')
