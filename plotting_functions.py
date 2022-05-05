#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
/*
* @Author: jorgedelpozolerida
* @Date: 2020-05-31 20:13:44
* @Last Modified by:   jorgedelpozolerida
* @Last Modified time: 2020-05-31 20:13:44  *
/
'''
from argparse import Namespace as nspc
import helper_functions as hf
import GLM_nalt as nglm
import numpy as np
import warnings
from numpy import logical_and as and_
import matplotlib as mat
from matplotlib.colors import ListedColormap
# from matplotlib import rcParams
import matplotlib.pyplot as plt
import os
import sys

# GLOBAL VARIABLES
XTICKS = np.array(['1', '2', '3', '4', '5', '6-10'])
model_cols = ['evidence',
              'L+1', 'L-1', 'L+2', 'L-2', 'L+3', 'L-3', 'L+4', 'L-4',
              'L+5', 'L-5', 'L+6-10', 'L-6-10',
              'T++1', 'T+-1', 'T-+1', 'T--1', 'T++2', 'T+-2', 'T-+2',
              'T--2', 'T++3', 'T+-3', 'T-+3', 'T--3', 'T++4', 'T+-4',
              'T-+4', 'T--4', 'T++5', 'T+-5', 'T-+5', 'T--5',
              'T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10', 'intercept']
afterc_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                  'T+-1', 'T--1']]
aftere_cols = [x for x in model_cols if x not in ['L+1', 'T++1',
                                                  'T-+1', 'L+2',
                                                  'L-2']]

rebound_tags = ['E' + 'X+'*i for i in range(20)]


# colors
rojo = np.array((228, 26, 28))/255
azul = np.array((55, 126, 184))/255
verde = np.array((77, 175, 74))/255
morado = np.array((152, 78, 163))/255
naranja = np.array((255, 127, 0))/255
marron = np.array((166, 86, 40))/255
amarillo = np.array((155, 155, 51))/255
rosa = np.array((247, 129, 191))/255
cyan = np.array((0, 1, 1))
gris = np.array((.5, .5, 0.5))
azul_2 = np.array([56, 108, 176])/255
rojo_2 = np.array([240, 2, 127])/255

COLORES = np.concatenate((azul.reshape((1, 3)), rojo.reshape((1, 3)),
                          verde.reshape((1, 3)), morado.reshape((1, 3)),
                          naranja.reshape((1, 3)), marron.reshape((1, 3)),
                          amarillo.reshape((1, 3)), rosa.reshape((1, 3))),
                         axis=0)
COLORES = np.concatenate((COLORES, 0.5*COLORES))
COLORS = [mat.cm.rainbow(x) for x in np.linspace(0, 1, 10)]
mat.rcParams['font.size'] = 8
mat.rcParams['lines.markersize'] = 3

### XXX: SECONDARY FUNCTIONS


def box_plot(data, ax, x, lw=.5, fliersize=4, color='k', widths=0.15):
    bp = ax.boxplot(data, positions=[x], widths=widths)
    for p in ['whiskers', 'caps', 'boxes', 'medians']:
        for bpp in bp[p]:
            bpp.set(color=color, linewidth=lw)
    bp['fliers'][0].set(markeredgecolor=color, markerfacecolor=color, alpha=0.5,
                        marker='x', markersize=fliersize)
    ax.set_xticks([])


def add_grad_colorbar(ax, color=np.array((0., 0., 0.)), yticks=[0, 1], n=100,
                      origin='lower'):
    """
    Plot colorbar with specify color and decreasing alpha.

    Parameters
    ----------
    ax : axis
        where to plot.
    color : array, optional
        color for the colorbar (hf.naranja)
    n : int, optional
        number of elements in color bar (100)
    Returns
    -------
    None.

    """
    newcolors = np.tile(np.append(color, 1), (n, 1))
    newcolors[:, 3] = newcolors[:, 3]*np.array(np.linspace(0.2, 1, n))
    newcolors = np.clip(newcolors, 0, 1)
    newcmp = ListedColormap(newcolors, name='diff_nch')
    ax.imshow(np.arange(n)[:, None], cmap=newcmp, origin=origin, aspect='auto')
    ax.set_yticks([0, n-1])
    ax.set_yticklabels(yticks)
    ax.tick_params(labelsize=6)
    ax.set_xticks([])
    ax.yaxis.tick_right()


def add_letters(fig, ax, letter, size=8, margin_x=0.06, margin_y=0.05):
    """
    Add leter to specific axis/panel.

    Parameters
    ----------
    ax : axis
        axis for which to add the letter.
    letter : str
        letter to add.

    Returns
    -------
    None.

    """
    plt.figure(fig.number)
    pos = ax.get_position()
    plt.text(x=pos.x0-margin_x, y=pos.y0+pos.height+margin_y, s=letter,
             transform=fig.transFigure, fontsize=size)


def sv_fig(f, name, sv_folder):
    """
    Save figure.

    Parameters
    ----------
    f : fig
        figure to save.
    name : str
        name to use to save the figure.

    Returns
    -------
    None.

    """
    f.savefig(sv_folder+'/'+name+'.svg', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/'+name+'.pdf', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/'+name+'.png', dpi=400, bbox_inches='tight')


def rm_top_right_lines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_dashed_lines(ax, minimo=-1, maximo=1, value=0.5):
    """
    Plot a dashline between minimo and maximo at the level indicated by value.

    Parameters
    ----------
    minimo : float
        min limit.
    maximo : float
        max limit.
    value : float, optional
        value level to plot line (0.5)

    Returns
    -------
    None.

    """
    ax.plot([0, 0], [0, 1], '--k', lw=0.2)
    ax.plot([minimo, maximo], [value, value], '--k', lw=0.2)


def define_title(figure, val, num_instances, name=''):
    """
    Define supertitle for figure.

    Parameters
    ----------
    figure : fig
        figure to which the supertitle will be added.
    val : float
        value to add to title.
    num_instances : int
        number of instances.
    name : str, optional
        additional name for the supertitle ('')

    Returns
    -------
    None.

    """
    figure.suptitle(name + ' ( Max N: ' +
                    r'$\bf{' + str(val) + '}$' + ', Instances: '
                    + r'$\bf{' + str(num_instances) + '}$' + ')')


def get_transition_probs_figure(trans_mat, annotate_mat,
                                col_bar='general', title='Ground truth',
                                end_title='', show_values=True,
                                displacement=0):
    '''
    Generates plot for transition matrix.

    Args:
        trans_mat: matrix contianing transition probabilities
        annotate_mat: matrix contianing values to annotate in each
        transition probability box
        choices: unique values of possible choices
        col_bar: str, determines whether to plot colorbar and how. If
        'general', colobar common for all plots is added. If 'specific',
        colorbar for each subplot is added. If None (default), no colorbar.
        blck_tr_hist_id: unique value for trialhistory blocks
        blck_n_ch_id: unique values for nch blocks
        title: 'Ground truth' or 'Choices' for example, to set as title.
        end_title: extra string to add at the end to title, default=''
        displacement: percentage of annotation displacement with respect
        to box size, thought for bif numbers
    '''
    xy_label_list = [str(i) for i in np.arange(1, trans_mat.shape[2]+1)]
    bin_ = 2 / trans_mat.shape[2]
    yticks = np.linspace(1 - bin_/2, -1 + bin_/2,  trans_mat.shape[2])
    xticks = np.linspace(-1 + bin_/2, 1 - bin_/2, trans_mat.shape[2])
    vmin, vmax = None, None
    if col_bar == 'general':
        # To normalize all plots and general colorbar.
        vmin = 0
        vmax = 1
    f, ax = plt.subplots(ncols=trans_mat.shape[1], nrows=trans_mat.shape[0],
                         figsize=(5, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    ax = ax.flatten()
    counter = 0

    for ind_trh in range(trans_mat.shape[0]):
        for ind_nch in range(trans_mat.shape[1]):
            im = ax[counter].imshow(
                trans_mat[ind_trh, ind_nch, :, :],
                cmap='viridis',  # We create a colormap for colorbar.
                extent=[-1, 1, -1, 1],  # To have known reference points.
                vmin=vmin, vmax=vmax)
            ax[counter].set_xlabel(title + ' at trial t+1')
            ax[counter].set_ylabel(title + ' at trial t')
            ax[counter].set_xticks(xticks)
            ax[counter].set_xticklabels(xy_label_list)
            ax[counter].xaxis.tick_top()
            ax[counter].set_yticks(yticks)
            ax[counter].set_yticklabels(xy_label_list)
            ax[counter].yaxis.set_visible(False)
            if ind_trh == 0:
                # set column names in superfigure
                ax[counter].set_title('Effective channels: ' + str(ind_nch),
                                      pad=40, fontsize=12)
                # TODO: set row names in superfigures
            if counter == 0 or counter % trans_mat.shape[1] == 0:
                # share y axis across a row.
                ax[counter].yaxis.set_visible(True)
            if show_values:
                # display value of counts on top
                bin_p = bin_ / 2
                if displacement != 0:
                    disp = ((displacement/100)*bin_p)
                else:
                    disp = 0
                for i in range(trans_mat.shape[2]):
                    for j in range(trans_mat.shape[2]):
                        init = bin_p/2 + i*bin_p - disp
                        end = bin_p/2 + j*bin_p - disp
                        index_r = trans_mat.shape[2] - j - 1
                        annotation = annotate_mat[ind_trh,
                                                  ind_nch, index_r, i]
                        ax[counter].annotate("{:.0f}".format(annotation),
                                             (init, end),
                                             textcoords='axes fraction',
                                             color="w")
            counter += 1
    if col_bar == 'general':
        # common colorbar for figure
        f.colorbar(im, ax=ax.ravel().tolist(), shrink=0.75)
    f.suptitle(title + ' probabilities transition matrix ' + '. N= '
               + str(trans_mat.shape[2]) + end_title,
               fontsize=14)
    return f, ax


def generate_GLM_fig(figsize=(6, 4)):
    """
    Generate figure for across-training GLM weights plots.

    Parameters
    ----------
    figsize : tuple, optional
        size of fig ((6, 4))

    Returns
    -------
    f : fig
        figure.
    ax : ax
        axes.

    """
    f, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True,
                         figsize=figsize)
    for ind_pl, subpl in enumerate(ax.flat):
        if ind_pl == 0 or ind_pl == 2:
            subpl.set(ylabel='GLM weights')
        if ind_pl == 2 or ind_pl == 3:
            subpl.set(xlabel='Steps')
    return f, ax


def generate_bias_fig(bias_type, figsize=(3, 4)):
    """
    Generate figure for across-training bias plots.

    Parameters
    ----------
    bias_type : str
        type of bias that will be plot.
    figsize : tuple, optional
        size of figure ((3, 4))

    Returns
    -------
    f : fig
        figure.
    ax : ax
        axes.

    """
    nrows = 3 if bias_type == 'psych' else 1
    f, ax = plt.subplots(nrows=nrows, ncols=1, sharey=True, sharex=True,
                         figsize=figsize)
    if nrows > 2:
        ax = ax.flatten()
        aux = ax[-1]
        ax[-1] = plt.axes([0.05, 0.05, 0.94, 0.25], sharex=ax[0])
        aux.remove()
        ax[-1].set_xlabel('Trials')
        ax[-1].set_ylabel('Repeating bias')
    else:
        ax.set_xlabel('Trials')
        ax.set_ylabel('Bias')
        ax = [ax]
    return f, ax


def get_reset_index_inset(num_indx=2, ax=None):
    """
    Create inset for reset index.

    Parameters
    ----------
    num_indx : int, optional
        number of reset indexes that will be plot (2)
    ax : ax, optional
        axis to modify. If None, an axis will be created. (None)

    Returns
    -------
    ax : ax
        inset.

    """
    font = {'family': 'normal', 'weight': 'bold', 'size': 6}
    mat.rc('font', **font)
    if num_indx == 2:
        ax = plt.axes((0.79, 0.15, 0.08, 0.1))
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['min-RI', 'full-RI'])
        # ax.set_xlim([0, 2])
    elif num_indx == 1:
        if ax is None:
            ax = plt.axes((0.79, 0.15, 0.12, 0.15))
        ax.set_xticks(np.arange(1, 9)*2)
        ax.set_ylabel('Reset Index')
        ax.set_xlabel('Number of choices')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([-0.05, 1.])
    return ax


def plot_zero_line(all_axs):
    """
    Plot horizontal dashed line at 0.

    Parameters
    ----------
    all_axs : array
        array containing axes.

    Returns
    -------
    None.

    """
    for ax in all_axs:
        try:
            for a in ax:
                xlim = a.get_xlim()
                a.plot(xlim, [0, 0], '--k', lw=0.5)
        except TypeError:
            xlim = ax.get_xlim()
            ax.plot(xlim, [0, 0], '--k', lw=0.5)


def customize_rebound_figure(a_seq_mat_cond, ax):
    """
    Tune axis of rebound figure.

    Parameters
    ----------
    a_seq_mat_cond : int
        number of items in the sequence.
    ax : axis
        ax to tune.

    Returns
    -------
    None.

    """
    ax.set_ylabel(' Repeating Bias b')
    xticks = rebound_tags[:a_seq_mat_cond.shape[0]-2]
    xticks.extend([xticks[-1] + 'E', xticks[-1] + 'EC'])
    ax.set_xticks([i for i in range(a_seq_mat_cond.shape[0])])
    ax.set_xticklabels(xticks)


# PRIMARY FUNCTIONS

def plot_choice_trans_probs(data, folder, after_c=False, col_bar='general',
                            sim_agent=False, prop_wrong=None, period=None,
                            plot_figs=True, show_values=False, sv_fig=True,
                            displacement=0):
    """
    Plots choice-transition probabilities once the agent has trained in
    an environment for those trials at the end of each block where there
    is no obvious higher stimulus and after a correct previous choice.
    (criteria: v=max(accumulated_stimuli) < percentile_10(v)).
    Casuistry: end of nch block (vertical axis) against end of repetition
    block (horizontal axis).
    Inside each subplot, for current choice (row) at trial t, probability
    of the other choices at trial t+1.
    Figure is saved in folder.

    Args:
        folder: path, folder from where to extarct data and save results.
        after_correct: bool, if True,  choice transition probabilities
        for trials after a correct trial are plotted.
        col_bar: str, determines whether to plot colorbar and how. If
        'general', colobar common for all plots is added. If 'specific',
        colorbar for each subplot is added. If None (default), no colorbar.
        sim_agent: if True, a simulated agent is generated
        prop_wrong: if stim_agent==True, proportion of wrong choices for it
        period: limits number of steps selected for calculation of bias
        plot_figs: if True, figures are plotted
        show_values: if True, total counts for each condition shown on
        top of normal plot
        sv_fig: if True, figure is saved in folder.

    """
    if period is None:
        period = [0, data['gt'].shape[0]]
    elif isinstance(period, int):
        period = [data['gt'].shape[0]-period, data['gt'].shape[0]]
    # We select necessary data
    ground_truth = data['gt'][period[0]:period[1]]
    choices = np.unique(ground_truth)
    # print('GT: \n', choices, np.unique(ground_truth, return_counts=True))
    if not sim_agent:
        choices_agent = data['choice'][period[0]:period[1]]
        # print('CHOICES: \n', choices_agent,
        #       np.unique(choices_agent, return_counts=True))
        performance = data['performance'][period[0]:period[1]]
    else:
        choices_agent = ground_truth.copy()
        size = (int(choices_agent.shape[0]*prop_wrong),)
        indx = np.random.choice(np.arange(choices_agent.shape[0],),
                                size=size,
                                replace=False)
        choices_agent[indx] = np.random.choice(choices, size=size)
        performance = choices_agent == ground_truth
    # Manage possible situation when active choices remain constant.
    try:
        block_n_ch = data['nch'][period[0]:period[1]]
    except KeyError:
        block_n_ch = np.full(ground_truth.shape, len(choices))
    block_tr_hist = data['tr_block'][period[0]:period[1]]
    # Percentile selection of highest stimuli.
    evidence = data['evidence'][period[0]:period[1]]
    evidence = np.append(evidence[1:], max(evidence))  # shift back evidence
    percetile_10 = np.percentile(evidence, 10)
    # Select between after error or after correct for bias discrimation.
    titles = ['after error', 'after correct']
    title = titles[after_c]
    extra_condition = and_(performance == after_c,
                           evidence <= percetile_10)
    # Matrix to fill for all causistry.
    trans_mat, counts_mat = hf.compute_transition_probs_mat(
        choices_agent, choices, block_n_ch,
        block_tr_hist, extra_condition=extra_condition)
    if plot_figs:
        f, ax = get_transition_probs_figure(trans_mat, counts_mat,
                                            col_bar='general',
                                            title=title + ' choice',
                                            end_title=', period= '
                                            + str(period),
                                            show_values=True,
                                            displacement=displacement)
        if sv_fig:
            f.savefig(folder + '/choice_transition_matrix_' + title + '.png')
    return trans_mat


def plot_glm_weights(weights_ac, weights_ae, tags_mat, step, per,
                     num_tr_back=3, axs=None, linewidth=0.5, nch=None,
                     **kwargs):
    '''
    Plotting function for GLM weights. For each list of tags in tags_mat, a figure
    is created. For eah tag in each list, a subplot is generated.

    weights_ac/ae rank = num-steps x 1 x num-regressors

    plot_opts = {'legend': False, 'lw': .5,  'label': '', 'alpha': 1,
                 'N': 0, 'compared_averages': False,
                 'num_tr_tm': None}
    '''
    plot_opts = {'legend': False, 'lw': .5,  'label': '', 'alpha': 1,
                 'N': 0, 'compared_averages': False,
                 'num_tr_tm': num_tr_back}  # TODO: num_tr_back is passed twice (?)
    plot_opts.update(kwargs)
    weights = [weights_ac, weights_ae]
    regressors = [afterc_cols, aftere_cols]
    l_styl = ['-', '--']
    titles = ['a_c', 'a_e']
    figs = []

    for ind_tag, tags in enumerate(tags_mat):
        if axs is None:
            f, ax = plt.subplots(nrows=len(tags)//2, ncols=2, sharex=True,
                                 sharey=True, figsize=(6, 4))
            figs.append(f)
            ax = ax.flatten()
        else:
            ax = axs[ind_tag]
        for ind_cond, cond in enumerate(zip(weights, regressors, l_styl)):
            l_st = cond[2]  # different line style for a_c and a_e
            weights_tmp = np.array(cond[0])[:, 0, :]  # weights
            rgr_tmp = np.array(cond[1])  # labels for data contained in weights
            for ind_t, tag in enumerate(tags):
                t_contr = and_(tag != 'evidence', tag != 'intercept')
                num_tr_tm = num_tr_back+1 if t_contr else 2
                if plot_opts['num_tr_tm']:
                    num_tr_tm = plot_opts['num_tr_tm']
                for ind_tr in range(1, num_tr_tm + 1):
                    t_tmp = tag+str(ind_tr) if t_contr else tag
                    if t_tmp in rgr_tmp:
                        # Plot tunning
                        # label given (for tunning legend) only when:
                        lbl = (plot_opts['label'] + 'trial lag ' + t_tmp[-1]
                               + ' ' + titles[ind_cond]) if ind_t == 0 and \
                               plot_opts['legend'] else ''
                        color = COLORES[ind_tr-1]
                        # for the case of only averages compared for nchs.
                        if plot_opts['compared_averages']:
                            color = COLORS[int(plot_opts['color_ind'])]
                            lbl = 'N=' + str(plot_opts['N']) \
                                  + ', lag= ' + str(ind_tr) + ' , ' \
                                  + titles[ind_cond]
                            lbl = lbl if ind_t == 0 else ''
                            alpha = 1-(1/(num_tr_tm+1))*(ind_tr-1)
                            plot_opts.update({'alpha': alpha})
                        if per is None or step is None:
                            assert weights_tmp.shape[0] == 1
                            xs = np.arange(weights_tmp.shape[0])
                            marker = '+'
                        else:
                            xs = np.arange(weights_tmp.shape[0])*step+per/2
                            marker = '.'
                        ax[ind_t].plot(xs, weights_tmp[:, rgr_tmp == t_tmp],
                                       color=color, linestyle=l_st,
                                       label=lbl, linewidth=plot_opts['lw'],
                                       alpha=plot_opts['alpha'], marker=marker)
                        if lbl != '':
                            ax[ind_t].legend()
                ax[ind_t].set_title(tag)
        # ax[0].legend()
    return figs


def plot_bias(bias, step, per, bias_type, ax=None, colors=None, **kwargs):
    """
    Plot bias across time.

    Parameters
    ----------
    bias : array
        time x perf x context (alt cont + alt, rep cont + rep, alt cont + rep,
                               rep cont + alt)

    step : int
        step used in sliding window process.
    per : int
        window used in sliding window process..
    bias_type : str
        'psych' or 'entropy', depending on the type of bias to be plotted.
    ax : ax, optional
        where to plot (None)
    **kwargs : dict
        plotting properties.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    if ax is None:
        f, ax = generate_bias_fig(bias_type)
    else:
        f = None

    plot_opts = {'legend': False, 'lw': .5,  'label': '', 'alpha': 1}
    plot_opts.update(kwargs)
    legend_on = plot_opts['legend']
    label = plot_opts['label']
    alpha = plot_opts['alpha']
    del plot_opts['legend'], plot_opts['label'], plot_opts['alpha']
    if step is None:
        assert bias.shape[0] == 1
        plot_opts['marker'] = '+'
        xs = [0.5]
    else:
        plot_opts['marker'] = ''
        xs = np.arange(bias.shape[0])*step+per/2
    if colors is not None:
        col1, col2 = colors, colors
        lw1, lw2 = '--', '-'
    else:
        col1, col2 = azul, rojo
        lw1, lw2 = '-', '-'
    err_factor = 0.2
    ax[0].plot(xs, bias[:, 1, 0], color=col1, linestyle=lw1, alpha=alpha,
               label='alt '+label, **plot_opts)
    ax[0].plot(xs, bias[:, 0, 0], color=col1, linestyle=lw1,
               alpha=err_factor*alpha, **plot_opts)
    if bias_type == 'psych':
        ax[0].plot(xs, bias[:, 1, 1], color=col2, alpha=alpha,  linestyle=lw2,
                   label='rep '+label, **plot_opts)
        ax[0].plot(xs, bias[:, 0, 1], color=col2, alpha=err_factor*alpha,
                   linestyle=lw2, **plot_opts)
        if len(ax) > 1:
            ax[1].plot(xs, bias[:, 1, 2], color=col1, alpha=alpha, linestyle=lw1,
                       label='alt '+label, **plot_opts)
            ax[1].plot(xs, bias[:, 0, 2], color=col1, alpha=err_factor*alpha,
                       linestyle=lw1, **plot_opts)
            ax[1].plot(xs, bias[:, 1, 3], color=col2, alpha=alpha,  linestyle=lw2,
                       label='rep '+label, **plot_opts)
            ax[1].plot(xs, bias[:, 0, 3], color=col2, alpha=err_factor*alpha,
                       linestyle=lw2, **plot_opts)
        if len(ax) > 2:
            reset_indx_psych_bias = compute_psycho_reset_index(bias)
            ax[2].plot(xs, reset_indx_psych_bias, color='k', **plot_opts)
            ax[2].set_ylim([-0.1, 1.1])
    for a in ax:
        a.set_xlabel('trials')
    if legend_on:
        ax[1].legend()
    return f


def plot_bias_2D(bias, ax=None, colors=None, **kwargs):
    """
    Plot bias across time.

    Parameters
    ----------
    bias : array
        time x perf x context (alt cont + alt, rep cont + rep, alt cont + rep,
                               rep cont + alt)
    ax : ax, optional
        where to plot (None)
    **kwargs : dict
        plotting properties.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=1)
    else:
        f = None

    plot_opts = {'legend': False, 'lw': .5,  'label': '', 'alpha': 1}
    plot_opts.update(kwargs)
    label = plot_opts['label']
    alpha = plot_opts['alpha']
    del plot_opts['legend'], plot_opts['label'], plot_opts['alpha']
    plot_opts['marker'] = ''
    if colors is not None:
        col1, col2 = colors, colors
        lw1, lw2 = '--', '-'
    else:
        col1, col2 = azul, rojo
        lw1, lw2 = '-', '-'
    ax.plot(bias[:, 0, 0], bias[:, 1, 0], color=col1, linestyle=lw1, alpha=alpha,
            label='alt '+label, **plot_opts)
    ax.plot(bias[:, 0, 1], bias[:, 1, 1], color=col2, alpha=alpha,  linestyle=lw2,
            label='rep '+label, **plot_opts)

    ax.plot(bias[:, 0, 2], bias[:, 1, 2], color=col1, linestyle=lw1,
            alpha=.5*alpha, label='alt '+label, **plot_opts)
    ax.plot(bias[:, 0, 3], bias[:, 1, 3], color=col2, alpha=.5*alpha,
            linestyle=lw2, label='rep '+label, **plot_opts)

    return f


def get_krnl(name, cols, weights, n_stps_ws):
    indx = np.array([np.where(np.array([x.startswith(name)
                                        for x in cols]))[0]])
    indx = np.array([x for x in indx if len(x) > 0])
    xtcks = np.array(cols)[indx][0]
    xs = [int(x[len(name):len(name)+1]) for x in xtcks]
    kernel = np.nanmean(weights[-n_stps_ws:, 0, indx], axis=0).flatten()
    return kernel, xs


def xtcks_krnls(xs, ax):
    xtcks = np.arange(1, max(xs)+1)
    ax.set_xticks(xtcks)
    ax.set_xlim([xtcks[0]-0.5, xtcks[-1]+0.5])
    xtcks_lbls = [str(x) for x in xtcks]
    xtcks_lbls[-1] = '6-10'
    ax.set_xticklabels(xtcks_lbls)


def get_opts_krnls(plot_opts, tag):
    opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts['color'] = plot_opts['color'+tag]
    opts['linestyle'] = plot_opts['lstyle'+tag]
    return opts


def plot_kernels(weights_ac, weights_ae, std_ac=None, std_ae=None, ac_cols=None,
                 ae_cols=None, ax=None, n_stps_ws=20, ax_inset=None, inset_xs=0.5,
                 regressors=['T++', 'T-+', 'T+-', 'T--'], **kwargs):
    plot_opts = {'lw': 1,  'label': '', 'alpha': 1, 'color_ac': naranja,
                 'fntsz': 7, 'color_ae': (0, 0, 0), 'lstyle_ac': '-',
                 'lstyle_ae': '-', 'marker': '.'}
    plot_opts.update(kwargs)
    fntsz = plot_opts['fntsz']
    del plot_opts['fntsz']
    ac_cols = afterc_cols if ac_cols is None else ac_cols
    ae_cols = aftere_cols if ae_cols is None else ae_cols
    if ax is None:
        n_regr = len(regressors)
        if n_regr > 2:
            ncols = int(np.sqrt(n_regr))
            nrows = int(np.sqrt(n_regr))
            figsize = (8, 5)
        else:
            ncols = n_regr
            nrows = 1
            figsize = (8, 3)
        f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize,
                             sharey=True)
        ax = ax.flatten()
        ax_inset = plt.axes((0.79, 0.15, 0.1, 0.15))
        for a in ax:
            a.invert_xaxis()
            a.axhline(y=0, linestyle='--', c='k', lw=0.5)
    else:
        f = None
    for j, name in enumerate(regressors):
        ax[j].set_ylabel('Weight (a.u.)', fontsize=fntsz)
        ax[j].set_xlabel('Trials back from decision', fontsize=fntsz)
        # after correct
        kernel_ac, xs_ac = get_krnl(name=name, cols=ac_cols, weights=weights_ac,
                                    n_stps_ws=n_stps_ws)
        if std_ac is not None:
            s_ac, _ = get_krnl(name=name, cols=ac_cols, weights=std_ac,
                               n_stps_ws=n_stps_ws)
        else:
            s_ac = np.zeros_like(kernel_ac)
        opts = get_opts_krnls(plot_opts=plot_opts, tag='_ac')
        ax[j].errorbar(xs_ac, kernel_ac, s_ac, **opts)

        # after error
        kernel_ae, xs_ae = get_krnl(name=name, cols=ae_cols, weights=weights_ae,
                                    n_stps_ws=n_stps_ws)
        if std_ae is not None:
            s_ae, _ = get_krnl(name=name, cols=ae_cols, weights=std_ae,
                               n_stps_ws=n_stps_ws)
        else:
            s_ae = np.zeros_like(kernel_ae)
        opts = get_opts_krnls(plot_opts=plot_opts, tag='_ae')
        ax[j].errorbar(xs_ae, kernel_ae, s_ae, **opts)

        # tune fig
        xtcks_krnls(xs=xs_ac, ax=ax[j])
    # PLOT RESET INDEX
    if ax_inset is not None:
        # compute reset
        ws_ac = np.nanmean(weights_ac[-n_stps_ws:, :, :], axis=0)
        ws_ac = np.expand_dims(ws_ac, 0)
        ws_ae = np.nanmean(weights_ae[-n_stps_ws:, :, :], axis=0)
        ws_ae = np.expand_dims(ws_ae, 0)
        xtcks = ['T++'+x for x in ['2', '3', '4', '5', '6-10']]
        reset, _, _ = compute_reset_index(ws_ac, ws_ae, xtcks=xtcks,
                                          full_reset_index=False)
        opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
        plot_reset_index(reset=reset, ax=ax_inset, xs=float(inset_xs), **opts)

    return f, kernel_ac, kernel_ae, xs_ac, xs_ae


def main_fig_kernels(weights_ac, weights_ae, ax=None, inset_xs=0.5, ax_inset=None,
                     n_stps_ws=20, regressors=['T++', 'T-+', 'T+-', 'T--'],
                     **kwargs):
    plot_opts = {'lw': 1,  'label': '', 'alpha': 1, 'color_ac': naranja,
                 'color_ae': (0, 0, 0), 'lstyle_ac': '-',  'lstyle_ae': '-'}
    plot_opts.update(kwargs)
    if ax is None:
        ncols = 1
        nrows = 1
        figsize = (8, 3)
        f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize,
                             sharey=True)
        ax_inset = plt.axes((0.79, 0.15, 0.1, 0.15))
    else:
        f = None
    ax.axhline(y=0, linestyle=':', c='k')
    ax.set_ylabel('Weight (a.u.)')
    ax.set_xlabel('Trials back from decision')
    # after correct
    name = 'T++'
    indx = np.array([np.where(np.array([x.startswith(name)
                                        for x in afterc_cols]))[0]])
    indx = np.array([x for x in indx if len(x) > 0])
    xtcks_ac = np.array(afterc_cols)[indx][0]
    xs = [int(x[len(name):len(name)+1]) for x in xtcks_ac]
    kernel_ac = np.nanmean(weights_ac[-n_stps_ws:, 0, indx], axis=0).flatten()
    opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts['color'] = plot_opts['color_ac']
    opts['linestyle'] = plot_opts['lstyle_ac']
    ax.plot(xs, kernel_ac, '-+', **opts)

    # after error
    indx = np.array([np.where(np.array([x.startswith(name)  # or x == 'T+-1'
                                        for x in aftere_cols]))[0]])
    indx = np.array([x for x in indx if len(x) > 0])
    xtcks_ae = np.array(afterc_cols)[indx][0]
    xs = [int(x[len(name):len(name)+1]) for x in xtcks_ae]
    kernel_ae = np.nanmean(weights_ae[-n_stps_ws:, 0, indx], axis=0).flatten()
    opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts['color'] = plot_opts['color_ae']
    opts['linestyle'] = plot_opts['lstyle_ae']
    ax.plot(xs, kernel_ae, '-+', **opts)
    xtcks = np.arange(1, max(xs)+1)
    ax.set_xticks(xtcks)
    xtcks_lbls = [name+str(x) for x in xtcks]
    xtcks_lbls[-1] = name+'6-10'
    ax.set_xticklabels(xtcks_lbls)
    # compute reset
    ws_ac = np.nanmean(weights_ac[-n_stps_ws:, :, :], axis=0)
    ws_ac = np.expand_dims(ws_ac, 0)
    ws_ae = np.nanmean(weights_ae[-n_stps_ws:, :, :], axis=0)
    ws_ae = np.expand_dims(ws_ae, 0)
    reset, _, _ = compute_reset_index(ws_ac, ws_ae, full_reset_index=False)
    opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts['color'] = 'k'
    plot_reset_index(reset=reset, ax=ax_inset, xs=float(inset_xs), **opts)

    return f


def compute_psycho_reset_index(bias):
    """
    Compute reset index from bias matrix.

    Parameters
    ----------
    bias : array
        steps x perf x context array containing bias values:
            2nd dim:
            0 --> after error
            1 --> after correct
            3rd dim:
            0 --> alt cont + alt
            1 --> rep cont + rep
            2 --> alt cont + rep
            3 --> rep cont + alt

    Returns
    -------
    reset_indx_psych_bias : array
        steps-element array with reset index.

    """
    reset_indx_psych_bias =\
        1-np.nanmean(np.abs(bias[:, 0, :]), axis=1) /\
        np.nanmean(np.abs(bias[:, 1, :]), axis=1)
    return reset_indx_psych_bias


def plot_psycho_reset_index(bias_mat, vals, per, step, save_folder=None,
                            ind_figs=True, figsize=None, n_steps_bck=5,
                            **plt_opts_ind_trcs):
    '''
    Given GLM matrices after correct and after error, calculates GLM
    results for val data in vals.

    Args:
        tags_mat: list of lists, each generating one figure. For each
        element in each list a subplot is generated.

    '''
    all_axs = []  # All axes saved here
    all_figs = []  # All figssaved here
    f_ms, ax_ms = plt.subplots(ncols=1, nrows=1, figsize=figsize, sharey=True)
    all_axs.append(ax_ms)
    all_figs.append(f_ms)
    f_main_fg, ax_main_fg = plt.subplots(ncols=1, nrows=1, figsize=(4, 3),
                                         sharey=True)
    ax_main_fg.spines['right'].set_visible(False)
    ax_main_fg.spines['top'].set_visible(False)
    all_axs.append([ax_main_fg])
    all_figs.append(f_main_fg)
    plot_something = False
    unq_vals = np.unique(vals)
    sfx = [float(x) for x in unq_vals]
    sorted_vals = [x for _, x in sorted(zip(sfx, unq_vals))]
    for ind_val, val in enumerate(sorted_vals):
        val_str = str(val)
        bias_mat_cond = bias_mat[vals == val_str]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in bias_mat_cond])
        if np.sum(lenghts) != 0:
            plot_something = True
            # Average
            res_indx_mat = []
            for ind_b, bias_mat_tmp in enumerate(bias_mat_cond):
                if len(bias_mat_tmp) != 0:
                    b_temp =\
                        np.mean(bias_mat_tmp[-n_steps_bck:, :, :], axis=0)[None]
                    reset_indx = compute_psycho_reset_index(b_temp)
                    res_indx_mat.append(reset_indx[0])
                    if ind_figs:
                        plot_opts = {'alpha': 0.5}
                        plot_opts.update(plt_opts_ind_trcs)
                        plot_reset_index(reset=reset_indx, ax=ax_ms, xs=val,
                                         **plot_opts)
            plot_opts = {'lw': 1, 'markersize': 6}
            plot_opts.update(plt_opts_ind_trcs)
            res_indx_mean = np.array([np.nanmean(res_indx_mat)])
            plot_reset_index(reset=res_indx_mean, ax=ax_ms, xs=val,
                             **plot_opts)
    if plot_something:
        if save_folder is not None:
            name = 'psycho_reset_index'
            plot_zero_line(all_axs)
            f_ms.savefig(save_folder+'/'+name+'.png', dpi=400, bbox_inches='tight')
    else:
        plt.close(f_ms)
        plt.close(f_main_fg)
    return all_figs, all_axs


def compute_reset_index(weights_ac, weights_ae, full_reset_index=False,
                        xtcks=None):
    """
    

    Parameters
    ----------
    weights_ac : array
        shape = n_steps x 1 x num-regressors.
    weights_ae : array
        shape = n_steps x 1 x num-regressors.
    full_reset_index : TYPE, optional
        DESCRIPTION. The default is False.
    xtcks : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    reset : TYPE
        DESCRIPTION.
    krnl_ac : TYPE
        DESCRIPTION.
    krnl_ae : TYPE
        DESCRIPTION.

    """
    if full_reset_index:
        # do not use T++_1 for fair comparison with after-error
        ws_corr = []
        ws_err = []
        for trans in ['++', '-+', '+-', '--']:
            # correct regressors (only T++ after correct)
            xtcks = ['T'+trans+x for x in XTICKS[1:]]
            indx = np.array([np.where(np.array(afterc_cols) == x)[0]
                             for x in xtcks])
            indx = np.array([x for x in indx if len(x) > 0])
            if trans == '++':
                ws_corr.append(np.abs(weights_ac[:, 0, indx]))
            else:
                ws_err.append(np.abs(weights_ac[:, 0, indx]))

            # error regressors (T+-, T-+ and T-- plus T++ after error )
            indx = np.array([np.where(np.array(aftere_cols) == x)[0]
                             for x in xtcks])
            indx = np.array([x for x in indx if len(x) > 0])
            ws_err.append(np.abs(weights_ae[:, 0, indx]))
        ws_corr = np.swapaxes(np.array(ws_corr)[:, :, :, 0], 0, 1)
        ws_corr = ws_corr.reshape(ws_corr.shape[0], -1)
        ws_err = np.swapaxes(np.array(ws_err)[:, :, :, 0], 0, 1)
        ws_err = ws_err.reshape(ws_err.shape[0], -1)
        ac_tr_contr = np.mean(ws_corr, axis=1)
        ae_tr_contr = np.mean(ws_err, axis=1)
    else:
        if xtcks is None:
            xtcks = ['T++'+x for x in XTICKS]+['T+-1']
        # after correct
        indx = np.array([np.where(np.array(afterc_cols) == x)[0]
                         for x in xtcks])
        indx = np.array([x for x in indx if len(x) > 0]).flatten()
        # assert len(indx) == 6
        krnl_ac = weights_ac[:, 0, indx]
        ac_tr_contr = np.abs(np.mean(krnl_ac))
        # after error
        indx = np.array([np.where(np.array(aftere_cols) == x)[0]
                         for x in xtcks])
        indx = np.array([x for x in indx if len(x) > 0]).flatten()
        # assert len(indx) == 6
        krnl_ae = weights_ae[:, 0, indx]
        ae_tr_contr = np.abs(np.mean(krnl_ae))
    reset = 1-(ae_tr_contr+1e-6)/(ac_tr_contr+1e-6)
    return reset, krnl_ac, krnl_ae


def plot_reset_index(reset, ax, step=None, per=None, xs=0.5, **plot_opts):
    if step is None:
        if 'marker' not in plot_opts.keys():
            plot_opts['marker'] = '+'
        if 'color' not in plot_opts.keys():
            plot_opts['color'] = 'b'
        xs = [xs]
    else:
        plot_opts['marker'] = ''
        xs = np.arange(reset.shape[0])*step+per/2

    ax.plot(xs, reset, **plot_opts)


def plot_figures_reset_index(glm_ac, glm_ae, vals, per, step,
                             save_folder=None, lw=0.5, full_reset_index=False):
    '''
    Given GLM matrices after correct and after error, calculates GLM
    results for val data in vals.

    Args:
        tags_mat: list of lists, each generating one figure. For each
        element in each list a subplot is generated.

    '''
    f_ms, ax_ms = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), sharey=True)
    plot_something = False
    for ind_val, val in enumerate(np.unique(vals)):
        glm_ac_cond = glm_ac[vals == str(val)]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in glm_ac_cond])
        if np.sum(lenghts) != 0:
            plot_something = True
            glm_ae_cond = glm_ae[vals == str(val)]
            num_instances = glm_ac_cond.shape[0]
            f, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), sharey=True)
            ri_mat = []
            for ind_glm, glm_ac_tmp in enumerate(glm_ac_cond):
                if len(glm_ac_tmp) != 0:
                    glm_ae_tmp = glm_ae_cond[ind_glm]
                    xtcks = ['T++'+x for x in ['2', '3', '4', '5', '6-10']]
                    reset, _, _ =\
                        compute_reset_index(glm_ac_tmp, glm_ae_tmp, xtcks=xtcks,
                                            full_reset_index=full_reset_index)
                    plot_opts = {'lw': lw,  'label': '',
                                 'color': COLORES[ind_val], 'alpha': 0.5}
                    plot_reset_index(reset, step=step, per=per, ax=ax,
                                     **plot_opts)
                    ri_mat.append(reset)
            # Average
            reset = hf.get_average(ri_mat)
            plot_opts = {'lw': 1,  'label': '', 'color': COLORES[ind_val]}
            plot_reset_index(reset, step=step, per=per, ax=ax, **plot_opts)
            name = 'reset_index_max_N_'+str(val)+'_fullRI_'+str(full_reset_index)
            define_title(f, val, num_instances, name)
            plot_zero_line([[ax]])
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')
                plt.close(f)
            # plot all mean traces in same figure (f_ms)
            plot_opts = {'lw': 1,  'label': 'N: '+val, 'color': COLORES[ind_val]}
            plot_reset_index(reset, step=step, per=per, ax=ax_ms, xs=float(val),
                             **plot_opts)
    if plot_something:
        f_ms.suptitle('Average Reset Index for different N', fontsize=13)
        ax_ms.legend()
        if save_folder is not None:
            name = 'compared_reset_index_averages_fullRI_'+str(full_reset_index)
            plot_zero_line([[ax_ms]])
            f_ms.savefig(save_folder + '/' + name + '.png', dpi=400,
                         bbox_inches='tight')
    else:
        plt.close(f_ms)


def plot_GLM_final_kernels(glm_ac, glm_ae, vals, per, step, save_folder=None,
                           lw=0.5, regressors=['T++', 'T-+', 'T+-', 'T--'],
                           ind_figs=True, figsize=None,
                           **plt_opts_ind_trcs):  # TODO: separate into diff. fns
    '''
    Given GLM matrices after correct and after error, calculates GLM
    results for val data in vals.

    Args:
        tags_mat: list of lists, each generating one figure. For each
        element in each list a subplot is generated.

    '''
    all_axs = []  # All axes saved here
    all_figs = []  # All figssaved here
    n_regr = len(regressors)
    if n_regr > 2:
        ncols = int(np.sqrt(n_regr))
        nrows = int(np.sqrt(n_regr))
        figsize = figsize or (8, 5)
    else:
        ncols = n_regr
        nrows = 1
        figsize = figsize or (4, 3)
    f_ms, ax_ms = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize,
                               sharey=True)
    ax_ms = ax_ms.flatten() if isinstance(ax_ms, np.ndarray) else [ax_ms]
    all_axs.append(ax_ms)
    all_figs.append(f_ms)
    f_ms_ri, ax_ms_ri = plt.subplots(ncols=1, nrows=1, figsize=(2, 2),
                                     sharey=True)
    ax_ms_ri = get_reset_index_inset(num_indx=1, ax=ax_ms_ri)
    all_axs.append([ax_ms_ri])
    all_figs.append(f_ms_ri)
    plot_something = False
    for ind_val, val in enumerate(np.unique(vals)):
        glm_ac_cond = glm_ac[vals == str(val)]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in glm_ac_cond])
        if np.sum(lenghts) != 0:
            plot_something = True
            glm_ae_cond = glm_ae[vals == str(val)]
            num_instances = glm_ac_cond.shape[0]
            name1, name2 = '', ''
            # Average
            a_glm_ac_cond = hf.get_average(glm_ac_cond)
            a_glm_ae_cond = hf.get_average(glm_ae_cond)
            if ind_figs:
                f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize,
                                     sharey=True)
                ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
                f_ins, ax_ins = plt.subplots(ncols=1, nrows=1, figsize=(0.25, 1))
                ax_ins = get_reset_index_inset(num_indx=1, ax=ax_ins)
                all_axs.append(ax)
                all_figs.append(f)
                all_axs.append(ax_ins)
                all_figs.append(f_ins)
                for ind_glm, glm_ac_tmp in enumerate(glm_ac_cond):
                    if len(glm_ac_tmp) != 0:
                        glm_ae_tmp = glm_ae_cond[ind_glm]
                        name1 = '_all_instances'
                        plot_opts = {'lw': lw, 'alpha': 0.5}
                        plot_opts.update(plt_opts_ind_trcs)
                        plot_kernels(glm_ac_tmp, glm_ae_tmp, ax=ax, inset_xs=val,
                                     ax_inset=ax_ins, regressors=regressors,
                                     **plot_opts)
                plot_opts = {'lw': 1, 'markersize': 6}
                plot_opts.update(plt_opts_ind_trcs)
                plot_kernels(a_glm_ac_cond, a_glm_ae_cond, ax=ax, ax_inset=ax_ins,
                             regressors=regressors, inset_xs=val, **plot_opts)
                name2 = '_averages'
                name = 'GLM_kernels_max_N_' + str(val) + name1 + name2
                define_title(f, val, num_instances, name)
                if save_folder is not None:
                    f.savefig(save_folder + '/' + name + '_'.join(regressors) +
                              '.png', dpi=400, bbox_inches='tight')
                    plt.close(f)
            # plot only means to compare
            plot_opts = {'lw': 1,  'label': 'N: '+val,
                         'alpha': float(val)/np.max([float(x) for x in vals])}
            warnings.warn("Computing reset index of average weights".upper() +
                          "instead of average of reset index".upper())
            plot_kernels(a_glm_ac_cond, a_glm_ae_cond, ax=ax_ms, inset_xs=val,
                         ax_inset=ax_ms_ri, regressors=regressors, **plot_opts)
    if plot_something:
        f_ms.suptitle('Average GLM for different N', fontsize=13)
        if save_folder is not None:
            name = 'compared_GLM_kernel_averages'
            plot_zero_line(all_axs)
            f_ms.savefig(save_folder + '/' + name + '_'.join(regressors)+'.png',
                         dpi=400, bbox_inches='tight')
            f_ms_ri.savefig(save_folder + '/reset_for_diff_nch.png',
                            dpi=400, bbox_inches='tight')
            f_ms_ri.savefig(save_folder + '/reset_for_diff_nch.svg',
                            dpi=400, bbox_inches='tight')
    else:
        plt.close(f_ms)
    return all_figs, all_axs


def plot_GLM_results(glm_ac, glm_ae, vals, tags_mat, per, step, num_tr_back=3,
                     save_folder=None, **kwargs):
    '''
    Given GLM matrices after correct and after error, calculates GLM
    results for val data in vals.

    Args:
        tags_mat: list of lists, each generating one figure. For each
        element in each list a subplot is generated.

    '''
    unq_vals = np.unique([float(v) for v in vals])
    all_axs = []  # All axes saved here
    f_ms, ax_ms = generate_GLM_fig()
    all_axs.append(ax_ms.flatten())
    ax_ms = [ax_ms.flatten()]
    plot_something = False
    for val in np.unique(unq_vals):
        val_str = str(val)
        plot_opts = {'legend': False, 'lw': 0.5, 'num_tr_tm': num_tr_back}
        plot_opts.update(kwargs)
        glm_ac_cond = glm_ac[vals == val_str]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in glm_ac_cond])
        if np.sum(lenghts) != 0:
            plot_something = True
            glm_ae_cond = glm_ae[vals == val_str]
            num_instances = glm_ac_cond.shape[0]
            name1, name2 = '', ''
            f, ax = generate_GLM_fig()
            all_axs.append(ax.flatten())
            ax = [ax.flatten()]
            plot_opts.update({'legend': False})
            for ind_glm, glm_ac_tmp in enumerate(glm_ac_cond):
                if len(glm_ac_tmp) != 0:
                    glm_ae_tmp = glm_ae_cond[ind_glm]
                    name1 = '_all_instances'
                    plot_glm_weights(glm_ac_tmp, glm_ae_tmp, tags_mat,
                                     num_tr_back=num_tr_back,
                                     axs=ax, step=step, per=per,
                                     **plot_opts)
            # Average
            a_glm_ac_cond = hf.get_average(glm_ac_cond)
            a_glm_ae_cond = hf.get_average(glm_ae_cond)
            plot_opts.update({'lw': 2, 'label': 'Average, ',
                              'legend': True})
            plot_glm_weights(a_glm_ac_cond, a_glm_ae_cond, tags_mat,
                             num_tr_back=num_tr_back, axs=ax, step=step,
                             per=per, **plot_opts)
            name2 = '_averages'
            plot_opts.update({'lw': 0.5, 'label': 'Average, '})
            plot_opts.update({'legend': True})
            name = 'GLM_acr_tr_max_N_' + val_str + name1 + name2 \
                   + '_num_tr_back_' + str(plot_opts['num_tr_tm'])
            define_title(f, val_str, num_instances, name)
            plot_zero_line(all_axs)
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')
                # plt.close(f)
            # plot means for comparison
            plot_opts.update({'lw': 2, 'N': val_str,
                              'color_ind': np.where(unq_vals == val)[0][0],
                              'compared_averages': True})  # Set to 2
            plot_glm_weights(a_glm_ac_cond, a_glm_ae_cond, tags_mat,
                             axs=ax_ms, step=step, per=per, **plot_opts)
    if plot_something:
        f_ms.suptitle('Average GLM for different N', fontsize=13)
        if save_folder is not None:
            name = 'compared_GLM_acr_tr_averages_num_tr_back_' \
                + str(plot_opts['num_tr_tm'])
            plot_zero_line(all_axs)
            f_ms.savefig(save_folder + '/' + name + '.png', dpi=400,
                         bbox_inches='tight')
    else:
        plt.close(f_ms)


def plot_bias_results(vals, bias_mat, step, per, save_folder=None,
                      bias_type='psych', figsize=None):
    '''
    Plots bias results of bias_mat for vals.
    '''
    unq_vals = np.unique([float(v) for v in vals])
    all_axs = []
    all_figs = []
    f_ms, ax_ms = generate_bias_fig(bias_type=bias_type, figsize=figsize)
    all_axs.append(ax_ms)
    all_figs.append(f_ms)
    plot_something = False
    for val in np.unique(unq_vals):
        val_str = str(val)
        bias_mat_cond = bias_mat[vals == val_str]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in bias_mat_cond])
        if np.sum(lenghts) != 0:
            plot_something = True
            num_instances = len(bias_mat_cond)
            f, ax = generate_bias_fig(bias_type=bias_type)
            all_axs.append(ax)
            all_figs.append(f)
            for ind_b, b_mat in enumerate(bias_mat_cond):
                if len(b_mat) != 0:
                    if bias_type == 'entropy':
                        b_mat = np.nanmean(b_mat, axis=2)
                        b_mat = np.expand_dims(b_mat, axis=2)
                    plot_bias(bias=b_mat, step=step, per=per, ax=ax,
                              bias_type=bias_type, **{'lw': 0.5})

            if bias_type == 'entropy':
                a_bias_mat_cond = [np.nanmean(x, axis=2)[:, :, None]
                                   for x in bias_mat_cond if len(x) > 0]
            else:
                a_bias_mat_cond = bias_mat_cond
            a_bias_mat_cond = hf.get_average(a_bias_mat_cond)
            plot_bias(bias=a_bias_mat_cond, step=step, per=per, ax=ax,
                      bias_type=bias_type, **{'lw': 2})
            ax[0].legend(['Alt. after correct', 'Alt. after error',
                          'Rep. after correct', 'Rep. after error'])
            name = 'bias_' + bias_type + '_max_N_' + val_str
            define_title(f, val, num_instances, name)
            plot_zero_line(all_axs)
            for a in ax:
                a.spines['right'].set_visible(False)
                a.spines['top'].set_visible(False)
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')
                # plt.close(f)
            colors = COLORS[np.where(unq_vals == val)[0][0]]

            # plot means
            if not (np.isnan(a_bias_mat_cond)).all():
                plot_bias(bias=a_bias_mat_cond, step=step, per=per, ax=ax_ms,
                          colors=colors, bias_type=bias_type,
                          **{'lw': 2, 'label': 'N:'+val_str})

    if plot_something:
        name = 'compared_average_bias_'+bias_type
        f_ms.suptitle('Average '+bias_type+' bias for different N', fontsize=13)
        tags = []
        for ind, tag in enumerate(np.repeat(np.unique(vals), 2)):
            if ind % 2 == 0:
                tags.append('N= ' + tag + ', Alt_cont')
            else:
                tags.append('N= ' + tag + ', Rep_cont')
        ax_ms[0].legend()
        for a in ax_ms:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)

        plot_zero_line(all_axs)
        if save_folder is not None:
            f_ms.savefig(save_folder + '/' + name + '.png', dpi=400,
                         bbox_inches='tight')
    else:
        plt.close(f_ms)
    return all_figs, all_axs


def plot_bias_seqs_results(vals, seq_mat, save_folder=None,
                           selected_vals=None):
    '''
    Plots seq results of seq_mat for vals.
    '''
    all_axs = []
    f_ms, ax_ms = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                               figsize=(14, 7))
    all_axs.append([ax_ms])
    if selected_vals is None:
        selected_vals = np.unique(vals)
    plot_something = False
    for ind_val, val in enumerate(selected_vals):
        val_str = str(val)
        seq_mat_cond = seq_mat[vals == val_str, :]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in seq_mat_cond])
        if np.sum(lenghts) != 0:
            plot_something = True
            num_instances = len(seq_mat_cond)
            name1, name2 = '', ''
            f, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                                 figsize=(14, 7))
            all_axs.append([ax])
            for ind_b, s_mat in enumerate(seq_mat_cond):
                if len(s_mat) != 0:
                    s_mat = np.array(s_mat)
                    ax.plot(s_mat[0], lw=0.5, color=azul)
                    ax.plot(s_mat[1], lw=0.5, color=rojo)
                    name1 = '_all_instances'
            a_seq_mat_cond_0 = hf.get_average(seq_mat_cond[:, 0])
            a_seq_mat_cond_1 = hf.get_average(seq_mat_cond[:, 1])
            ax.plot(a_seq_mat_cond_0, lw=2, color=azul)
            ax.plot(a_seq_mat_cond_1, lw=2, color=rojo)
            customize_rebound_figure(a_seq_mat_cond_0, ax)
            name2 = '_averages'
            name = 'seq_results' + name1 + name2 + 'max_N_ ' + str(val)
            define_title(f, val, num_instances, name)
            if save_folder is not None:
                ax.set_ylim([-40, 40])
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')
                plt.close(f)
            plot_opts = {'lw': 1,  'label': 'N: '+val,
                         'alpha': (ind_val+1)/np.max(len(selected_vals))}
            ax_ms.plot(a_seq_mat_cond_0, color=azul, **plot_opts)
            ax_ms.plot(a_seq_mat_cond_1, color=rojo, **plot_opts)
            customize_rebound_figure(a_seq_mat_cond_0, ax_ms)
    if plot_something:
        name = 'compared_average_seq'
        ax_ms.legend()
        plot_zero_line(all_axs)
        if save_folder is not None:
            f_ms.savefig(save_folder + '/' + name + '.png', dpi=400,
                         bbox_inches='tight')
    else:
        plt.close(f_ms)


def plot_performances(vals, perf_mat, step, per, spacing=10000, save_folder=None,
                      **plt_kwargs):
    '''
    Plots bias results of bias_mat for vals.
    '''
    plt_opts = {'figtype': '.png', 'figsize': (4, 3), 'lw': 0.5, 'plt_mean': True,
                'alpha': 0.4, 'colors': COLORS}
    plt_opts.update(plt_kwargs)
    plt_op = nspc(**plt_opts)
    unq_vals = np.unique([float(v) for v in vals])
    all_axs = []
    all_figs = []
    f_ms, ax_ms = plt.subplots(nrows=1, ncols=1, figsize=plt_op.figsize,
                               constrained_layout=True)
    all_axs.append([ax_ms])
    all_figs.append([f_ms])
    plot_something = False
    for val in np.unique(unq_vals):
        val_str = str(val)
        color = plt_opts['colors'][np.where(unq_vals == val)[0][0]]
        perf_mat_cond = perf_mat[vals == val_str]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in perf_mat_cond])
        if np.sum(lenghts) != 0:
            plot_something = True
            num_instances = len(perf_mat_cond)
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=plt_op.figsize,
                                 constrained_layout=True)
            all_axs.append([ax])
            all_figs.append([f])
            for ind_b, p_mat in enumerate(perf_mat_cond):
                if len(p_mat) != 0:
                    xs = np.arange(p_mat.shape[0])*spacing
                    ax.plot(xs, p_mat, lw=plt_op.lw, color=color,
                            alpha=plt_op.alpha)
            a_perf_mat_cond = hf.get_average(perf_mat_cond)
            xs = np.arange(a_perf_mat_cond.shape[0])*spacing
            if plt_op.plt_mean:
                ax.plot(xs, a_perf_mat_cond, color=color, lw=2, label='N:'+val_str)
                name = 'perf_' + '_max_N_' + val_str
                define_title(f, val, num_instances, name)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + plt_op.figtype, dpi=400,
                          bbox_inches='tight')
                plt.close(f)
            # plot means
            ax_ms.plot(xs, a_perf_mat_cond, color=color, lw=2,
                       label='N:'+val_str)

    if plot_something:
        name = 'compared_average_perf'
        f_ms.suptitle('Average performance for different N', fontsize=13)
        ax_ms.legend()
        ax_ms.spines['right'].set_visible(False)
        ax_ms.spines['top'].set_visible(False)
        plot_zero_line(all_axs)
        if save_folder is not None:
            f_ms.savefig(save_folder + '/' + name + plt_op.figtype, dpi=400,
                         bbox_inches='tight')
    else:
        plt.close(f_ms)
    return all_figs, all_axs


def plot_nGLM_results(glm_ac, glm_ae, vals, trs_back=8, save_folder=None,
                      **kwargs):
    '''
    Given GLM matrices after correct and after error, calculates GLM
    results for val data in vals.

    '''
    # plot_something = False
    for val in np.unique(vals):
        glm_ac_cond = glm_ac[vals == str(val)]
        lenghts = np.array([np.sum(~np.isnan(x)) for x in glm_ac_cond])
        if np.sum(lenghts) != 0:
            # plot_something = True
            glm_ae_cond = glm_ae[vals == str(val)]
            n_ch = float(val)
            names = nglm.get_regressors_names(n_ch=n_ch, trs_back=trs_back)
            # Average
            f_ms, ax_ms = plt.subplots(nrows=2, ncols=2, figsize=(12, 8),
                                       sharey=True, sharex=True)
            ax_ms = ax_ms.flatten()
            for ind_exp in range(glm_ac_cond.shape[0]):
                ws_ac = glm_ac_cond[ind_exp]
                if len(ws_ac) != 0:
                    ws_ae = glm_ae_cond[ind_exp]
                    ws_ac_sort =\
                        nglm.get_all_transition_ws(weights=ws_ac, keys=names,
                                                   n_ch=n_ch, n_tr_bck=8,
                                                   regr_type='T')
                    ax_ms[0].plot(ws_ac_sort['++'], color=naranja, alpha=0.2)
                    ax_ms[1].plot(ws_ac_sort['-+'], color=naranja, alpha=0.2)
                    ax_ms[2].plot(ws_ac_sort['+-'], color=naranja, alpha=0.2)
                    ax_ms[3].plot(ws_ac_sort['--'], color=naranja, alpha=0.2)
                    ws_ae_sort =\
                        nglm.get_all_transition_ws(weights=ws_ae, keys=names,
                                                   n_ch=n_ch, n_tr_bck=8,
                                                   regr_type='T')
                    ax_ms[0].plot(ws_ae_sort['++'], color=(0, 0, 0), alpha=0.2)
                    ax_ms[1].plot(ws_ae_sort['-+'], color=(0, 0, 0), alpha=0.2)
                    ax_ms[2].plot(ws_ae_sort['+-'], color=(0, 0, 0), alpha=0.2)
                    ax_ms[3].plot(ws_ae_sort['--'], color=(0, 0, 0), alpha=0.2)

                    # nglm.plot_trans_weights(weights=ws_ac, keys=names, n_ch=n_ch,
                    #                         n_tr_bck=2, outcomes=[(1, 1)],
                    #                         name='After correct '+str(ind_exp))
                    # nglm.plot_trans_weights(weights=ws_ae, keys=names, n_ch=n_ch,
                    #                         n_tr_bck=2, outcomes=[(1, 1)],
                    #                         name='After error '+str(ind_exp))
            glm_ac_cond_abs = [np.abs(x) for x in glm_ac_cond]
            glm_ae_cond_abs = [np.abs(x) for x in glm_ae_cond]
            a_glm_ac_cond_abs = hf.get_average(glm_ac_cond_abs)
            a_glm_ae_cond_abs = hf.get_average(glm_ae_cond_abs)
            ws_ac_sort =\
                nglm.get_all_transition_ws(weights=a_glm_ac_cond_abs, keys=names,
                                           n_ch=n_ch, n_tr_bck=8, regr_type='T')
            ax_ms[0].plot(ws_ac_sort['++'], color=naranja)
            ax_ms[1].plot(ws_ac_sort['-+'], color=naranja)
            ax_ms[2].plot(ws_ac_sort['+-'], color=naranja)
            ax_ms[3].plot(ws_ac_sort['--'], color=naranja)
            ws_ae_sort =\
                nglm.get_all_transition_ws(weights=a_glm_ae_cond_abs, keys=names,
                                           n_ch=n_ch, n_tr_bck=8, regr_type='T')
            ax_ms[0].plot(ws_ae_sort['++'], color=(0, 0, 0))
            ax_ms[1].plot(ws_ae_sort['-+'], color=(0, 0, 0))
            ax_ms[2].plot(ws_ae_sort['+-'], color=(0, 0, 0))
            ax_ms[3].plot(ws_ae_sort['--'], color=(0, 0, 0))
            a_glm_ac_cond = hf.get_average(glm_ac_cond)
            a_glm_ae_cond = hf.get_average(glm_ae_cond)

            nglm.plot_trans_weights(weights=a_glm_ac_cond, keys=names, n_ch=n_ch,
                                    n_tr_bck=2, outcomes=[(1, 1)],
                                    name='After correct')
            nglm.plot_trans_weights(weights=a_glm_ae_cond, keys=names, n_ch=n_ch,
                                    n_tr_bck=2, outcomes=[(1, 1)],
                                    name='After error')
            # f_ms, ax_ms = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            # ax_ms[0].imshow(a_glm_ac_cond[:, n_ch:], aspect='auto')
            # ax_ms[0].set_xticks(np.arange(a_glm_ac_cond.shape[1]-n_ch))
            # ax_ms[0].set_xticklabels([x for x in names.keys() if x != 'ev'])
            # ax_ms[0].set_yticks(np.arange(n_ch))
            # ax_ms[0].set_yticklabels([str(x+1) for x in np.arange(n_ch)])
            # ax_ms[0].set_title('Previous correct')
            # ax_ms[1].imshow(a_glm_ae_cond[:, n_ch:], aspect='auto')
            # ax_ms[1].set_xticks(np.arange(a_glm_ac_cond.shape[1]-n_ch))
            # ax_ms[1].set_xticklabels([x for x in names.keys() if x != 'ev'])
            # ax_ms[1].set_yticks(np.arange(n_ch))
            # ax_ms[1].set_yticklabels([str(x+1) for x in np.arange(n_ch)])
            # ax_ms[1].set_title('Previous error')
            # print(1)
            # plot means for compariosn
    # if plot_something:
    #     f_ms.suptitle('Average GLM for different N', fontsize=13)
    #     if save_folder is not None:
    #         name = 'compared_GLM_acr_tr_averages_num_tr_back'
    #         f_ms.savefig(save_folder + '/' + name + '.png', dpi=400,
    #                      bbox_inches='tight')
    # else:
    #     plt.close(f_ms)


def plot_trans_prob_mats_after_error(trans_mats, al_trans_mats, num_samples_mat,
                                     perf_mat, n_ch, name='', sv_folder='',
                                     axs=None, std_vals=None, plot_mats=True,
                                     **kwargs):
    plot_opts = {'color': 'k'}
    plot_opts.update(kwargs)
    if axs is None:
        f, ax = plt.subplots(nrows=2, ncols=len(trans_mats), figsize=(5, 4))
        pos = ax[0, 0].get_position()
        ax_perf = plt.axes([pos.x0, pos.y0+pos.height/2, 0.75, 0.1])
    else:
        ax = axs[0]
        ax_perf = axs[1]
        im = None
    ax_perf.plot(perf_mat, '+-', **plot_opts)
    if std_vals is not None:
        ax_perf.errorbar(np.arange(len(perf_mat)), perf_mat, std_vals['perf_mat'],
                         color='k', linestyle='')
    ax_perf.set_ylim([0, 1])
    rm_top_right_lines(ax=ax_perf)
    ax_perf.set_xticks([])
    ax_perf.set_xlim([-.5, len(trans_mats)-.7])
    for i_ax, (t_m, a_t_m, n_s) in enumerate(zip(trans_mats, al_trans_mats,
                                                 num_samples_mat)):
        if axs is None:
            pos = ax[0, i_ax].get_position()
            ax[0, i_ax].set_position([pos.x0, pos.y0-pos.height/3.8, pos.width,
                                      pos.height/2])
        ax[0, i_ax].scatter(np.arange(n_ch, 0, -1)+np.random.randn(n_ch)*0.1,
                            a_t_m, s=1, **plot_opts)
        if std_vals is not None:
            ax[0, i_ax].errorbar(np.arange(n_ch, 0, -1), a_t_m,
                                 std_vals['al_trans_mats'], marker='.',
                                 markersize=2, linestyle='', **plot_opts)
        if plot_mats:
            ax[0, i_ax].bar(np.arange(n_ch, 0, -1), height=a_t_m, fill=False)
            # plot specific bars with corresponding transition color
            vals = [-1, -2, 0] if n_ch > 2 else [-1, 0]
            clr = [hf.azul, hf.rojo_2, hf.rosa] if n_ch > 2 else [hf.azul, hf.rosa]
            for i_c, (o, c) in enumerate(zip(vals, clr)):
                ax[0, i_ax].bar(n_ch-(n_ch//2+o),
                                height=a_t_m[n_ch//2+o],
                                fill=False, edgecolor=c, zorder=100)
            rm_top_right_lines(ax=ax[0, i_ax])
            ax[0, i_ax].set_ylim([-.05, 1.05])
            ax[0, i_ax].set_xticks([])
            ax[0, i_ax].set_yticks([])
            im = ax[1, i_ax].imshow(t_m, cmap='gray', vmin=0, vmax=1)
            ax[1, i_ax].set_xticks([])
            ax[1, i_ax].set_yticks([])
            ax[1, i_ax].invert_yaxis()
            ax[0, i_ax].set_title('(N='+str(np.round(n_s))+')', fontsize=6)
    ax[0, 0].set_ylabel('Freq.', fontsize=7)
    ax[0, 0].set_yticks([0, 1])
    ax[0, 0].set_xlabel('Choice at t+1', fontsize=7)
    ax[1, 0].set_ylabel('Choice at t+1', fontsize=8)
    ax[1, 0].set_xlabel('Choice at t', fontsize=8)
    if axs is None:
        cbar_ax = f.add_axes([0.91, 0.18, 0.02, 0.21])
        f.colorbar(im, cax=cbar_ax)
        sv_fig(f=f, name=name, sv_folder=sv_folder)
    return im


def plot_alg(file, save_folder=None, perf_th=-1, sel_vals=None,
             **plt_kwargs):
    """
    Plot behavioral results.

    Parameters
    ----------
    file : str
        file containing behavioral data.
    save_folder : str, optional
        where to save figures (None)
    perf_th : float, optional
        performance therhold to filter results from underperforming networks.
        Note that performances are expressed wrt that of a perfect integrator (-1)
    sel_vals : list, optional
        selected values for which to plot results. Depending on the experiment,
        values can correspond to number of choices, number of units, etc (None)
    **plt_kwargs : dict
        dictionary containing another dictionary for each measure to plot,
        which at least must contain a 'go' key whose value indicates whether
        to plot the corresponding measure.

    Returns
    -------
    None.

    """
    # '''
    # Plots GLM results and Simple Bias for all values in file data.
    # Args:
    #     file: where all data is stored for a given algorithm

    # '''
    plt_opts = {'glm': {'do': True, 'num_tr_back': 3}, 'psycho_bias': {'do': True},
                'entropy_bias': {'do': True}, 'rebound': {'do': True},
                'n-GLM': {'do': True, 'num_tr_back': 3},
                'performances': {'do': True}, 'reset_index': {'do': True},
                'glm_kernels': {'do': True}, 'psycho_reset_index': {'do': True}}
    plt_opts.update(plt_kwargs)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    data = np.load(file, allow_pickle=1)
    per = data['acr_tr_per'].item()
    step = data['step'].item()
    if 'spacing' not in data.keys():
        spacing = 10000
    else:
        spacing = data['spacing']
    # transform vals to float and then back to str
    if sel_vals is None:
        sel_vals = np.unique(data['val_mat'])
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    vals = np.array([str(float(v)) for v, p in zip(data['val_mat'], mean_prf)
                     if p >= perf_th and v in sel_vals])
    ################################################################
    # PERFORMANCES
    ###############################################################
    if plt_opts['performances']['do']:
        perf_mat = np.array([g for g, v, p in zip(data['perf_mats'],
                                                  data['val_mat'],
                                                  mean_prf)
                             if p > perf_th and v in sel_vals])
        plot_performances(vals=vals, perf_mat=perf_mat, step=step, per=per,
                          save_folder=save_folder, spacing=spacing)
    ################################################################
    # GLM RESULTS
    ################################################################
    # GLM KERNELS
    # Loading data
    if plt_opts['glm_kernels']['do']:
        glm_ac = np.array([g for g, v, p in zip(data['glm_mats_ac'],
                                                data['val_mat'],
                                                mean_prf)
                           if p > perf_th and v in sel_vals])
        glm_ae = np.array([g for g, v, p in zip(data['glm_mats_ae'],
                                                data['val_mat'],
                                                mean_prf)
                           if p > perf_th and v in sel_vals])
        regressors = ['T++', 'T-+', 'T+-', 'T--']
        plot_GLM_final_kernels(glm_ac=glm_ac, glm_ae=glm_ae, vals=vals, per=per,
                               step=step, save_folder=save_folder,
                               regressors=regressors)
        regressors = ['L+', 'L-']
        plot_GLM_final_kernels(glm_ac=glm_ac, glm_ae=glm_ae, vals=vals, per=per,
                               step=step, save_folder=save_folder,
                               regressors=regressors)
    ###################
    # RESET INDEX
    if plt_opts['reset_index']['do']:
        glm_ac = np.array([g for g, v, p in zip(data['glm_mats_ac'],
                                                data['val_mat'],
                                                mean_prf)
                           if p > perf_th and v in sel_vals])
        glm_ae =\
            np.array([g for g, v, p in zip(data['glm_mats_ae'],
                                           data['val_mat'],
                                           mean_prf)
                      if p > perf_th and v in sel_vals])
        # RESET INDEX
        plot_figures_reset_index(glm_ac=glm_ac, glm_ae=glm_ae, vals=vals, per=per,
                                 step=step, save_folder=save_folder, lw=0.5,
                                 full_reset_index=False)
        plot_figures_reset_index(glm_ac=glm_ac, glm_ae=glm_ae, vals=vals, per=per,
                                 step=step, save_folder=save_folder, lw=0.5,
                                 full_reset_index=True)
    ###################
    # GLM WEIGHTS ACROSS TRAINING
    # Loading data
    if plt_opts['glm']['do']:
        tags_mat = [['T++', 'T-+', 'T+-', 'T--']]
        glm_ac =\
            np.array([g for g, v, p in zip(data['glm_mats_ac'],
                                           data['val_mat'],
                                           mean_prf)
                      if p > perf_th and v in sel_vals])
        glm_ae =\
            np.array([g for g, v, p in zip(data['glm_mats_ae'],
                                           data['val_mat'],
                                           mean_prf)
                      if p > perf_th and v in sel_vals])
        plot_GLM_results(glm_ac=glm_ac, glm_ae=glm_ae, vals=vals, per=per,
                         tags_mat=tags_mat, step=step,
                         num_tr_back=plt_opts['glm']['num_tr_back'],
                         save_folder=save_folder)
    ################################################################
    # N-GLM WEIGHTS ACROSS TRAINING
    ################################################################
    # Loading data
    if plt_opts['n-GLM']['do']:
        sys.exist(1)  # not tested!
        nglm_ac = data['nglm_mats_ac']
        nglm_ae = data['nglm_mats_ae']
        plot_nGLM_results(glm_ac=nglm_ac, glm_ae=nglm_ae, vals=vals,
                          save_folder=save_folder)
    ################################################################
    # BIAS RESULTS
    ################################################################
    # PSYCHO BIAS AND RESET INDEX ACROSS TRAINING
    # Loading data
    if plt_opts['psycho_bias']['do']:
        bias_mat = np.array([b for b, v, p in zip(data['bias_mats_psych'],
                                                  data['val_mat'],
                                                  mean_prf)
                             if p > perf_th and v in sel_vals])
        plot_bias_results(vals=vals, bias_mat=bias_mat, step=step, per=per,
                          save_folder=save_folder)
    ###################
    # GLM WEIGHTS ACROSS TRAINING
    if plt_opts['entropy_bias']['do']:
        bias_mat = np.array([b for b, v, p in zip(data['bias_mats_entr'],
                                                  data['val_mat'],
                                                  mean_prf)
                             if p > perf_th and v in sel_vals])
        plot_bias_results(vals=vals, bias_mat=bias_mat, step=step, per=per,
                          save_folder=save_folder, bias_type='entropy')

    ###################
    # PSYCHO RESET-INDEX
    if plt_opts['psycho_reset_index']['do']:
        plot_psycho_reset_index(vals=vals, bias_mat=bias_mat, step=step, per=per,
                                save_folder=save_folder)
    ################################################################
    # REBOUND RESULTS
    ################################################################
    # Loading data
    if plt_opts['rebound']['do']:
        seq_mat = np.array([b for b, v, p in zip(data['bias_sequence'],
                                                 data['val_mat'],
                                                 mean_prf)
                            if p > perf_th and v in sel_vals])
        plot_bias_seqs_results(vals=vals, seq_mat=seq_mat, save_folder=save_folder)


if __name__ == '__main__':
    #  PLOTTING AND SAVING GLM AND BIAS
    plt.close('all')
    balance = True
    if balance:
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'balanced_rand_mtrx_n_ch_4/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'ev_3blk_fix_n_ch_balanced/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'ev_3blk_balanced_long_blocks/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'ev_fix_2AFC_tr_prob_08/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'variable_nch_radom_ch_set/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'variable_nch_predef_tr_mats/'
        # main_folder = '/home/manuel/priors_analysis/annaK/' +\
        #     'variable_nch_predef_tr_mats/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'sims_21/'
        main_folder = '/home/manuel/priors_analysis/annaK/' +\
            'sims_21_rand_pretr/'
        # main_folder = '/home/manuel/priors_analysis/annaK/' +\
        #     'sims_21_diff_prob12/'
        # main_folder = '/home/manuel/priors_analysis/annaK/' +\
        #     'sims_21_diff_prob12/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'diff_num_units/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'bernstein_shorter_rollout/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'bernstein_diff_block_durs/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'sims_21/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'bernstein_LeftRight/'
        # main_folder = '/home/manuel/priors_analysis/annaK/' +\
        #     'diff_tr_probs/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'var_nch_predef_mats_larger_nets/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'history_block_cue/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'var_nch_predef_mats_long_stim/'
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'reaction_time_exps/'
    else:
        # main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        #     'larger_nets_longer_blks_bal_rand_mtrx/'
        main_folder =\
            '/home/molano/priors/AnnaKarenina_experiments/ev_3blk_fix_n_ch/'
    exp = 'train'
    exp = 'test_2AFC'
    # exp = 'rebound_9'
    # exp = 'rebound_2'
    # exp = 'training_2AFC'
    # exp = 'test_2AFC'
    if exp == 'train':
        file = main_folder + '/data_ACER__.npz'
        # file = main_folder + '/data_A2C__.npz'
        sv_folder = main_folder+'/figures/'
    elif exp == 'test_2AFC':
        file = main_folder + '/data_ACER_test_2AFC_.npz'
        sv_folder = main_folder+'/figures_test_2AFC/'
    elif exp == 'rebound_2':
        file = main_folder + '/data_ACER_test_2AFC__rebound_nch_2_.npz'
        sv_folder = main_folder+'/figures_test_2AFC_rebound/'
    elif exp == 'rebound_9':
        file = main_folder + '/data_ACER_test_2AFC__rebound_nch_9_.npz'
        sv_folder = main_folder+'/figures_test_2AFC_rebound/'
    elif exp == 'training_2AFC':
        file = main_folder + '/data_ACER___training_2AFC_.npz'
        sv_folder = main_folder+'/figures_training_2AFC/'
    elif exp == 'diff_nlstm':
        file = main_folder + '/data_ACER_test_2AFC_50000_tmstps.npz'
        sv_folder = main_folder+'/figures_test_2AFC/'
        file = main_folder + '/data_ACER__50000_tmstps.npz'
        sv_folder = main_folder+'/figures_500000_tmstps/'
    elif exp == 'tests':
        # file = main_folder + '/data_ctx_ch_prob_0.0125__.npz'
        # sv_folder = main_folder+'/figures_ctx_ch_prob_0.0125/'

        # file = main_folder + '/data_ctx_ch_prob_0.0125_test_2AFC_.npz'
        # sv_folder = main_folder+'/figures_test_ctx_ch_prob_0.0125/'

        # file = main_folder + '/data_ctx_ch_prob_0.0125__400K_per.npz'
        # sv_folder = main_folder+'/figures_ctx_ch_prob_0.0125_400K_per/'

        # file = main_folder + '/data_ctx_ch_prob_0.0125_test_2AFC_400K_per.npz'
        # sv_folder = main_folder+'/figures_test_ctx_ch_prob_0.0125_400K_per/'

        # file = main_folder + '/data_ctx_ch_prob_0.0125__1M_per.npz'
        # sv_folder = main_folder+'/figures_ctx_ch_prob_0.0125_1M_per/'

        # file = main_folder + '/data_ctx_ch_prob_0.0125_test_2AFC_1M_per.npz'
        # sv_folder = main_folder+'/figures_test_ctx_ch_prob_0.0125_1M_per/'

        file = main_folder + '/data_n_ch_2_ctx_ch_prob_0.0125__per_100K.npz'
        sv_folder = main_folder + '/figures_per_100K_stp_20K'
        file = main_folder + '/data_ACER_test_2AFC_all_model_168000000_steps_.npz'
        sv_folder = main_folder + '/test_model_168000000'

    vals = None  # ['2', '4', '8', '16']
    plt_opts = {'glm': {'do': True, 'num_tr_back': 3},
                'psycho_bias': {'do': False},
                'entropy_bias': {'do': False}, 'rebound': {'do': False},
                'n-GLM': {'do': False, 'num_tr_back': 3},
                'performances': {'do': True}, 'reset_index': {'do': True},
                'glm_kernels': {'do': True}, 'psycho_reset_index': {'do': False}}
    plot_alg(file, save_folder=sv_folder, sel_vals=vals, **plt_opts)

    # for nch in ['16', '32']:
    #     file = main_folder + '/data_n_ch_'+nch+'__.npz'
    #     sv_folder = main_folder+'/figures_n_ch_'+nch+'/'
    #     plot_alg(file, save_folder=sv_folder, num_tr_back=3, vals=vals,
    #              **plt_opts)
    #     file = main_folder + '/data_n_ch_'+nch+'_test_2AFC_.npz'
    #     sv_folder = main_folder+'/figures_n_ch_'+nch+'_test_2AFC/'
    #     plot_alg(file, save_folder=sv_folder, num_tr_back=3, vals=vals,
    #              **plt_opts)
    #     plt.close('all')
