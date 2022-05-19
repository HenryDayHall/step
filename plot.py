from matplotlib import pyplot as plt
import numpy as np


def hist_error(bin_edges, bin_values, errors, color, ax):
    bin_centers = (bin_edges[1:] + bin_edges[:-1])*0.5
    #import ipdb; ipdb.set_trace()
    lower = bin_values - errors
    upper = bin_values + errors
    ax.hist(bin_centers, weights=lower, bins=bin_edges,
            histtype='step', color=color)
    ax.hist(bin_centers, weights=upper, bins=bin_edges,
            histtype='step', color=color)


def plot_stat_syst_errors(bin_edges, bin_values, syst_error, stat_error, ax, n_syst_used=None):
    all_errors = syst_error + stat_error
    ylim_before = ax.get_ylim()
    hist_error(bin_edges, bin_values, syst_error, color='lawngreen', ax=ax)
    if n_syst_used is not None:
        ax.plot([], [], color='lawngreen', label="Systematic {} errors".format(n_syst_used))
    else:
        ax.plot([], [], color='lawngreen', label="Systematic errors")
    hist_error(bin_edges, bin_values, all_errors, color='green', ax=ax)
    ax.plot([], [], color='green', label="All errors")
    ax.set_ylim(*ylim_before)


def plot_histogram(bin_edges, bin_values, syst_error, stat_error, ax=None, n_syst_used=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    bin_centers = (bin_edges[1:] + bin_edges[:-1])*0.5
    #ax.plot(bin_centers, bin_values, c='k')
    #import ipdb; ipdb.set_trace()
    ax.hist(bin_centers, weights=bin_values, bins=bin_edges,
            histtype='stepfilled', color='dodgerblue', edgecolor='blue',
            label="nominal")
    plot_stat_syst_errors(bin_edges, bin_values, syst_error, stat_error, ax, n_syst_used)
    ax.set_xlabel("$p_T$")
    ax.set_ylabel("Cross section")


def plot_fit(polynomial, parameters, ax, label=None):
    n_points = 200
    xs = np.linspace(polynomial.min_val, polynomial.max_val, n_points)
    ys = polynomial(xs, parameters)
    ax.plot(xs, ys, c='red', label=label)


def plot_tension(polynomial, parameters, bin_edges, bin_values,
                 syst_error, stat_error,
                 abs_ax=None, ratio_ax=None, label=None):
    fitted_bin_values = polynomial.bin_heights(bin_edges, parameters)
    tension = fitted_bin_values - bin_values
    bin_centers = (bin_edges[1:] + bin_edges[:-1])*0.5
    if abs_ax is not None:
        #plot_stat_syst_errors(bin_edges, tension, syst_error, stat_error, abs_ax)
        #abs_ax.hist(bin_centers, weights=bin_values, bins=bin_edges,
        #            histtype='step', color='dodgerblue')
        #abs_ax.hist(bin_centers, weights=fitted_bin_values, bins=bin_edges,
        #            label=label, color='gray', histtype='step')
        abs_ax.hist(bin_centers, weights=tension, bins=bin_edges,
                    label=label, color='red', histtype='step')
        abs_ax.set_xlabel("$p_T$")
        abs_ax.set_ylabel("Fit - nominal")
    if ratio_ax is not None:
        ratio = fitted_bin_values/bin_values
        #plot_stat_syst_errors(bin_edges, ratio,
        #                      syst_error/bin_values, stat_error/bin_values,
        #                      ratio_ax)
        ratio_ax.hist(bin_centers, weights=ratio, bins=bin_edges,
                      label=label, color='red', histtype='step')
        ratio_ax.set_xlabel("$p_T$")
        ratio_ax.set_ylabel("Fit/nominal")
        ratio_ax.set_ylim(0.5, 1.8)
        ratio_ax.set_xlim(bin_centers[0]*0.9, bin_centers[-1]*1.1)


def comprehensive_plot(data, result_version='lowest_chi2_per_ndf', n_syst_used=None, root_file=None):
    n_etas = len(data.etas)
    if result_version is None:
        nrows = 1
    else:
        nrows = 3
    fig, ax_arr = plt.subplots(ncols=n_etas, nrows=nrows,
                               figsize=(2+2.*n_etas, 2*nrows),
                               sharex='col', squeeze=False)
    if root_file is not None and result_version is not None:
        import plot_root
        plot_root.plot_py(root_file, ax_arr[-1])
        ax_arr[0, -1].errorbar([], [], [], color='gray', fmt='.',
                               label="root's minimiser")
    for col, eta in enumerate(data.etas):
        ax = ax_arr[0, col]
        #ax.semilogy()
        ax.semilogx()
        syst_error = np.diag(data.syst_covarience[eta])
        stat_error = np.diag(data.stat_covarience[eta])
        plot_histogram(data.pt_bin_edges[eta], data.nominals[eta],
                       syst_error, stat_error, ax=ax,
                       n_syst_used=n_syst_used)
        #ylim = ax.get_ylim()
        title = "eta={}".format(eta)
        ax.set_title(title)
        if result_version is None or eta not in data.results:
            continue
        if isinstance(result_version, str):
            result_idx = getattr(data, result_version)[eta]
            name = result_version.replace('_', ' ')
        elif isinstance(result_version, int):
            result_idx = result_version
            name = "After {} releases".format(result_version)
        else:
            raise NotImplementedError
        result = data.results[eta][result_idx]
        print("eta={}, result={}".format(eta, result))
        if np.isnan(result.chi2):
            print("Problem in result")
            continue
        #title += ",chi2={:.2g},chi2/ndf={:.2g}".format(result.chi2, result.chi2_per_ndf)
        #ax.set_title(title)
        polynomial = data.polynomial[eta]
        plot_fit(polynomial, result.parameters, ax, label=name)
        #ax.set_ylim(*ylim)
        abs_tension_ax = ax_arr[1, col]
        ratio_tension_ax = ax_arr[2, col]
        plot_tension(polynomial, result.parameters, data.pt_bin_edges[eta],
                     data.nominals[eta], syst_error, stat_error,
                     abs_ax=abs_tension_ax, ratio_ax=ratio_tension_ax)
        # write the fit hyperparameters
        message = "chi2/ndf\n= {:.2g}/{}\n= {:.2g}".format(result.chi2, result.n_degrees_of_freedom, result.chi2_per_ndf)
        message += "\n n_coef={}".format(len(result.parameters))
        abs_tension_ax.text(0.5, 0.5, message, color='darkred',
                            verticalalignment='center', horizontalalignment='center',
                            transform=abs_tension_ax.transAxes)
    for ax in ax_arr[:, 1:].flatten():
        ax.set_ylabel(None)
    for ax in ax_arr[:-1].flatten():
        ax.set_xlabel(None)
        ax.tick_params(axis='x', which='both',
                       bottom=False, top=False,
                       labelbottom=False)
    #fig.set_tight_layout(True)
    fig.subplots_adjust(left=0.1, right=0.99,
                        top=0.93, bottom=0.1,
                        wspace=0.5, hspace=0.)
    leg = ax_arr[0, -1].legend()
    leg.set_draggable(True)
    return fig, ax_arr

