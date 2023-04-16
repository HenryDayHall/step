# run me with 
# python ATLAS_data.py --data HEPdata --binmeans --minimiser powell --allsyst --maxdegree 20 -noc

# if root is imported too early it overrides the help
# messages from argparse.
class __root_importer:
    def __getattr__(self, attr_name):
        global ROOT
        import ROOT
        ROOT.gROOT.SetBatch(True)
        # actually get the requested attribute
        return getattr(ROOT, attr_name)

# as soon at somthing tries to get an attribute,
# this will be replaced with the actual root package
ROOT = __root_importer()


import numpy as np
import os

def filter_files(dir_name, suffix):
    matching = [os.path.join(dir_name, name) for name in os.listdir(dir_name)
                if name.endswith(suffix)]
    return matching


def open_root_file(dir_name):
    root_files = filter_files(dir_name, ".root")
    if not root_files:
        raise RuntimeError("Didn't find a root file in " + args.data)
    if len(root_files) > 1:
        raise RuntimeError("Found more than one root file in " + args.data)
    root_file = root_files[0]
    tfile = ROOT.TFile.Open(root_file)
    return tfile


def get_eta(file_name):
    start_at_eta = file_name.lower().split("eta")[1]
    eta_str = next(start_at_eta[:i] for i, c in enumerate(start_at_eta)
                   if not c.isdigit())
    return int(eta_str)


def sort_by_eta(file_names):
    etas = [get_eta(name) for name in file_names]
    sorted_names = {eta: [] for eta in set(etas)}
    for eta, name in zip(etas, file_names):
        sorted_names[eta].append(name)
    return sorted_names


def read_pt_bins(file_name):
    with open(file_name, 'r') as csv_file:
        lines = csv_file.readlines()
    data_start = next(i for i, line in enumerate(lines)
                      if line[0].isdigit())
    bin_edges = []
    bin_values = []
    for line in lines[data_start:]:
        line = line.split()
        # values sometimes given in scientific notation
        # so must convert via float
        bin_edges.append(int(float(line[0])))
        bin_values.append(float(line[2]))
    bin_edges.append(int(float(lines[-1].split()[1])))
    # values sometimes given in scientific notation
    # so must convert via float
    return np.array(bin_edges), np.array(bin_values)


def get_replicas(file_names):
    all_values = []
    for name in file_names:
        assert os.path.isfile(name), "Couldn't find " + name
        bin_edges, bin_values = read_pt_bins(name)
        all_values.append(bin_values)
    return bin_edges, np.array(all_values)


def get_covarience_and_bins(file_names):
    bin_edges, all_values = get_replicas(file_names)
    n_obs, n_bins = all_values.shape
    means = np.mean(all_values, axis=0)
    all_variences = all_values - means
    # caluclate covarience
    covarience = np.zeros((n_bins, n_bins))
    for variences in all_variences:
        covarience += variences*variences.reshape((-1, 1))
    covarience /= n_obs
    return bin_edges, means, covarience


def correlation_matrix(covarience):
    diag_covar = np.diag(covarience)
    correlation = covarience/(diag_covar*diag_covar.reshape((-1, 1)))
    return correlation


def get_tfile_structure(tfile):
    list_of_keys = [key.GetName() for key in tfile.GetListOfKeys()]
    structure = {}
    for name in list_of_keys:
        part = tfile.Get(name)
        try:
            below = get_tfile_structure(part)
        except AttributeError:
            below = None
        structure[name] = below
    return structure


def get_syst_numbers(tfile, y):
    tfile_structure = get_tfile_structure(tfile)
    table_name = "Table {}".format(y)
    prefix = "Hist1D_y1_e"
    suffix = "plus"
    syst_numbers = []
    for name in tfile_structure[table_name]:
        if name.startswith(prefix) and name.endswith(suffix):
            num_string = name[len(prefix):-len(suffix)]
            syst_numbers.append(int(num_string))
    return np.array(syst_numbers)


def get_nominal(eta, tfile, epsilon=1e-20):
    nominal_hist = tfile.Get("Table {}/Hist1D_y1".format(eta))
    n_bins = nominal_hist.GetNbinsX()
    values = np.fromiter((nominal_hist[i + 1] for i in range(n_bins)), dtype=float)
    first_zero = next((i for i, v in enumerate(values)
                       if v <= epsilon), len(values))
    values = values[:first_zero]
    return values


def select_syst(eta, tfile, all_syst=True, include_syst=[], exclude_syst=[]):
    if all_syst:
        if include_syst:
            print( "-includesys has no effect as -allsyst also specified.")
        if exclude_syst:
            print("Using all systematics appart from the {} excluded values."\
                  .format(len(exclude_syst)))
        else:
            print("Using all systematics.")
        use_syst = get_syst_numbers(tfile, eta)
    else:
        if include_syst:
            print("Using {} requested systematics.".format(len(include_syst)))
            use_syst = include_syst
        else:
            print("No systematics.")
            use_syst = []
    if exclude_syst:
        for exclu in exclude_syst:
            use_syst.remove(exclu)
    print("Total systematics used {}".format(len(use_syst)))
    return np.array(use_syst)


def gather_syst(eta, tfile, use_syst, use_n_bins):
    syst = np.zeros((use_n_bins, use_n_bins))
    for syst_n in use_syst:
        for direction in ["plus", "minus"]:
            syst_part = tfile.Get("Table {}/Hist1D_y1_e{}{}".format(eta, syst_n, direction))
            syst_part = np.fromiter((syst_part[i] for i in range(use_n_bins)), dtype=float)
            syst += syst_part*syst_part.reshape((-1, 1))
    # symmetrise
    syst *= 0.5
    return syst


def process_eta(eta, tfile, eta_file_names, all_syst=True, include_syst=[], exclude_syst=[]):
    pt_bin_edges, means, stat_covar = get_covarience_and_bins(eta_file_names)
    use_syst = select_syst(eta, tfile, all_syst, include_syst, exclude_syst)
    n_syst = len(use_syst)
    nominal = get_nominal(eta, tfile)
    use_n_bins = len(nominal)
    syst_covar = gather_syst(eta, tfile, use_syst, use_n_bins)
    covarience = syst_covar + stat_covar
    return pt_bin_edges, nominal, covarience, syst_covar, stat_covar, n_syst


def _cmnd_args():
    import argparse
    from scipy.optimize._minimize import MINIMIZE_METHODS
    parser = argparse.ArgumentParser(description="Run a python version of the step library")
    parser.add_argument('-md', '--maxdegree', type=int, default=None,
                        help="Maximum degree of the chebyshev polynomial")
    parser.add_argument('-ns', '--nsigma', default=1., type=float)
    parser.add_argument('-as', '--autostop', action='store_true')
    parser.add_argument('-mi', '--maxiter', type=int, default=100,
                        help="Maxiterations per call to the optimiser")
    parser.add_argument('--minimiser', default='powell',
                        #choices=MINIMIZE_METHODS,
                        help="Minimiser to use")
    parser.add_argument('--allsyst', action='store_true', 
                        help="Use all systematic uncertanties found in the data")
    parser.add_argument('-xH', '--excludeHigh', default=0, type=int,
                        help="Number of bins at top pt end to exclude")
    parser.add_argument('-xL', '--excludeLow', default=0, type=int,
                        help="Number of bins at bottom pt end to exclude")
    parser.add_argument('-d', '--data', required=True,
                        help="Folder containing input data to use")
    parser.add_argument('--output',
                        help="Name to give the output")
    parser.add_argument('--excludesyst', default="",
                        help='Comma seperated list of systematics to exclude by index.')
    parser.add_argument('--includesyst', default="",
                        help='Comma seperated list of systematics to include by index.')
    parser.add_argument('--integrate', action='store_true',
                    help="Integrate the distribution over the bin, rather than using the bin center")
    parser.add_argument('--binmeans', action='store_true',
                    help="Evaluate the distribution at the mean of a fit, rather than using the bin center")
    parser.add_argument('-noc', '--nocurriculum', dest='curriculum_learning', action='store_false',
                        help="Should the optimiser forget the previous results"
                        "when it adds a dimension, and so not do curriculum learning?")
    parser.add_argument('--predefined', default="",
                        help='List of parameters (deliminated by ",") '
                        'for each eta value (etas deliminated by ";")')
    parser.add_argument('--rootcomparison', default=None,
                        help='Name of a root file to add to plots for comparison')
    args = parser.parse_args()
    arg_dict = vars(args)

    # I know this looks odd, but actually .split() behaves differently
    # to split with any other argument, and can return an empty list
    arg_dict["include_syst"] = [int(x) for x in args.includesyst.replace(',', ' ').split()]
    arg_dict["exclude_syst"] = [int(x) for x in args.excludesyst.replace(',', ' ').split()]

    return arg_dict


def get_defaults(**override):
    arg_dict = dict(data="HEPdata", output="ooooh", allsyst=True, 
                    exclude_syst=[], include_syst=[], 
                    excludeHigh=0, excludeLow=0, maxiter=100,
                    maxdegree=20, nsigma=1., autostop=False,
                    minimiser="powell", integrate=False, binmeans=True,
                    curriculum_learning=False)
    for name, value in override.items():
        arg_dict[name] = value
    return arg_dict


def trim(excludeLow, excludeHigh, *arrays):
    if excludeHigh == 0 and excludeLow == 0:
        return arrays
    new_arrays = []
    for array in arrays:
        assert len(set(array.shape)) == 1, "Array is not square"
        assert array.shape[0] > excludeHigh + excludeLow, \
            "Excluded more bins than avaliable"
        n_dims = len(array.shape)
        if excludeHigh:
            slices = slice(excludeLow, -excludeHigh)
        else:
            slices = slice(excludeLow, None)
        slices = tuple([slices]*n_dims)
        new_arrays.append(array[slices])
    return new_arrays


def run(data, output, allsyst, exclude_syst, include_syst,
        excludeHigh, excludeLow, maxiter,
        maxdegree, nsigma, autostop, minimiser, integrate,
        binmeans, curriculum_learning,
        predefined=False, to_improve=False,
        **unneeded):
    from alleta import AllEtasData
    import step

    predefined_params = []
    if predefined and isinstance(predefined, str):
        for part in predefined.split(';'):
            predefined_params.append([float(x) for x in part.split(',')])
    else:
        predefined_params = predefined

    root_file = open_root_file(data)

    txt_files = filter_files(data, ".txt")
    sorted_by_eta = sort_by_eta(txt_files)
    ROOT.TH1.SetDefaultSumw2(True)  # what does this do?
    
    data = AllEtasData()
    for eta in sorted_by_eta:
        pt_bin_edges, nominal, covarience, syst_covar, stat_covar, n_syst = \
            process_eta(eta, root_file, sorted_by_eta[eta], allsyst,
                        include_syst, exclude_syst)
        pt_bin_edges, nominal, covarience, syst_covar, stat_covar = \
            trim(excludeLow, excludeHigh, 
                 pt_bin_edges, nominal, covarience,
                 syst_covar, stat_covar)
        data.add_inputs(eta, pt_bin_edges, nominal, syst_covar, stat_covar)
        objective = step.get_objective_function(pt_bin_edges, nominal, covarience,
                                                integrate, binmeans)
        #import ipdb; ipdb.set_trace()
        if to_improve:
            results = step.improve_results(to_improve.results[eta],
                                           pt_bin_edges, nominal, covarience,
                                           maxdegree, nsigma, autostop,
                                           integrate, binmeans, minimiser, maxiter)
        elif predefined_params:
            results = step.from_predefined(predefined_params[eta-1],
                                           pt_bin_edges, nominal,
                                           covarience, integrate, binmeans)
        else:
            results = step.get_smooth_fit(pt_bin_edges, nominal, covarience,
                                          maxdegree, nsigma, autostop,
                                          integrate, binmeans, minimiser, maxiter,
                                          curriculum_learning)
        data.add_results(eta, results)
    return data, n_syst


def base_name(argdict, prefix=None):
    form = os.environ.get("OUTPUT_DIR", "outputs")
    if not form.endswith("/"):
        form += "/"
    if prefix is not None:
        form += prefix + "_"
    form += "allsyst_" if argdict["allsyst"] else "nosyst_"
    form += argdict["minimiser"]
    form += "_integrate" if argdict["integrate"] else ""
    form += "_binmean" if argdict["binmeans"] else ""
    form += "_maxiter{}_maxdeg{}".format(argdict["maxiter"],
                                         argdict["maxdegree"])
    form += "_13TeV" if "13TeV" in argdict["data"] else ""
    form += "_R4" if "4_" in argdict["data"] else ""
    excludeHigh = argdict["excludeHigh"]
    form += f"_excludeHigh{excludeHigh}" if excludeHigh else ""
    excludeLow = argdict["excludeLow"]
    form += f"_excludeLow{excludeLow}" if excludeLow else ""
    form += "_noc" if not argdict["curriculum_learning"] else ""
    return form


def display(data, n_syst, first_unitary_chi2_per_ndf=True,
            lowest_chi2_per_ndf=True, lowest_chi2=False,
            progress=False, root_file=None):
    print("Plotting ~~~~~~~~~~~~~~")
    from matplotlib import pyplot as plt
    import plot
    plt.ion()
    if lowest_chi2:
        plot.comprehensive_plot(data, 'lowest_chi2', n_syst, root_file)
        plt.show()
        input("Hit enter")
        plt.close()
    if lowest_chi2_per_ndf:
        plot.comprehensive_plot(data, 'lowest_chi2_per_ndf', n_syst, root_file)
        plt.show()
        input("Hit enter")
        plt.close()
    if first_unitary_chi2_per_ndf:
        plot.comprehensive_plot(data, 'first_unitary_chi2_per_ndf', n_syst, root_file)
        plt.show()
        input("Hit enter")
        plt.close()
    if progress:
        for i in range(10):
            plot.comprehensive_plot(data, i, n_syst)
            plt.show()
            if input("Hit enter"):
                break
            plt.close()


def save(data, argdict, n_syst, name_prefix=None, root_file=None):
    try:
        name_prefix = base_name(argdict, name_prefix)
    except KeyError:
        name_prefix = "results"
    from matplotlib import pyplot as plt
    import plot
    name_format = name_prefix + "_{}.txt"
    i = 0
    while os.path.exists(name_format.format(i)):
        i += 1
    name = name_format.format(i)
    print(name)
    # ensure the directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # write the data
    with open(name, 'w') as output:
        output.write(str(argdict))
        output.write(str(data))
    plt.ioff()
    plot.comprehensive_plot(data, 'first_unitary_chi2_per_ndf', n_syst, root_file)
    plt.savefig(name[:-4] + ".pdf")
    plt.savefig(name[:-4] + ".png")
    plt.close()
    return name


def test_minimisers(data, output, allsyst, exclude_syst, include_syst,
                    excludeHigh, excludeLow,
                    maxdegree, nsigma, autostop, integrate, binmeans,
                    **unneeded):
    import step
    import scipy.optimize
    from scipy.optimize._minimize import MINIMIZE_METHODS
    #MINIMIZE_METHODS = ["trust-constr", "slsqp", "bfgs", "powell"]
    #import humpday

    root_file = open_root_file(data)

    txt_files = filter_files(data, ".txt")
    sorted_by_eta = sort_by_eta(txt_files)
    ROOT.TH1.SetDefaultSumw2(True)  # what does this do?
    
    n_params = maxdegree
    optimised = np.zeros((len(MINIMIZE_METHODS),
                          len(sorted_by_eta),
                          n_params))
    chi2 = np.zeros((len(MINIMIZE_METHODS),
                     len(sorted_by_eta))) - 1
    bounded = np.zeros((len(MINIMIZE_METHODS),
                        len(sorted_by_eta)), dtype=bool)
    bounds = [(-30, 30)]*n_params
    working = np.ones(len(MINIMIZE_METHODS), dtype=bool)
    #objective = []
    for n, name in enumerate(MINIMIZE_METHODS):
        print(name)
        for e, eta in enumerate(sorted_by_eta):
            #import ipdb; ipdb.set_trace()
            pt_bin_edges, nominal, covarience, syst_covar, stat_covar, n_syst= \
                process_eta(eta, root_file, sorted_by_eta[eta], allsyst,
                            include_syst, exclude_syst)
            objective = step.get_objective_function(pt_bin_edges, nominal, covarience,
                                                    integrate, binmeans)
            inital_params = np.zeros(n_params)
            inital_params[:2] = step.get_initial_guess(nominal)
            max_iter = 1e5

            try:
                result = scipy.optimize.minimize(objective, inital_params, method=name,
                                                 options={'maxiter': max_iter},
                                                 jac=objective.Jacobian, hess=objective.Hessian)
            except Exception:
                # try again wiht bounds;
                try:
                    print("Trying bounded")
                    result = scipy.optimize.minimize(objective, inital_params, method=name,
                                                     options={'maxiter': max_iter},
                                                     jac=objective.Jacobian, hess=objective.Hessian,
                                                     bounds=bounds)
                    bounded[n, e] = True
                except Exception as e:
                    print("Cannot optimise")
                    print(e)
                    working[n] = False
                    break
            optimised[n, e] = result.x
            chi2[n, e] = objective(optimised[n, e])

            #recomendations = humpday.recommend(objective, 15, n_trials=130)
            #def wrap(inps, func=objective):
            #    return func(inps)
            #objectives.append(wrap)
            #print("eta={}, recomendations={}".format(eta, recomendations))
    #overall_points = humpday.points_race(objectives, n_dim=15)
    remove_unfilled = chi2
    remove_unfilled[remove_unfilled<=0] = np.inf
    min_for_eta = np.min(remove_unfilled, axis=0)
    chi2_above_min = chi2 - min_for_eta
    message = "Results ~~~~~~~\n"
    short_message = "Short Results ~~~~~~~\n"
    message += "minimiser, ".ljust(20) + "eta, ".rjust(7) + "chi2, ".rjust(7) +  "chi2 above min,".rjust(7) + "bounded\n"
    short_message += "minimiser, ".ljust(20) + "mean chi2\n"
    for n, name in enumerate(MINIMIZE_METHODS):
        if not working[n]:
            continue
        mean_chi2 = np.mean(chi2[n])
        short_message += "{:<18} {}\n".format(name, mean_chi2)
        for e, eta in enumerate(sorted_by_eta):
            message += "{:<18} {: >5}, {: 8.4f}, {: 8.4f}, {}\n".format(name, eta, chi2[n, e], chi2_above_min[n, e], bounded[n, e])
            name = ""
    print(message)
    print(short_message)
    with open("optimisers_compared.txt", 'w') as outfile:
        outfile.write(message + short_message)


def test_hybrid(data, output, allsyst, exclude_syst, include_syst,
                    excludeHigh, excludeLow,
                    maxdegree, integrate, binmeans,
                    **unneeded):
    import step
    import scipy.optimize
    MINIMIZE_METHODS = ["trust-constr", "slsqp", "bfgs", "powell"]
    n_methods = len(MINIMIZE_METHODS)
    #import humpday

    root_file = open_root_file(data)

    txt_files = filter_files(data, ".txt")
    sorted_by_eta = sort_by_eta(txt_files)
    ROOT.TH1.SetDefaultSumw2(True)  # what does this do?
    
    n_params = maxdegree
    optimised = np.zeros((len(sorted_by_eta),
                          n_params))
    chi2 = np.zeros(len(sorted_by_eta)) - 1
    improvements = {name : [] for name in MINIMIZE_METHODS}
    intial_method = "powell"
    inital_steps = 1e3
    batch_steps = 100
    import warnings
    warnings.filterwarnings("ignore")
    for e, eta in enumerate(sorted_by_eta):
        print("Eta = {}".format(eta))
        #import ipdb; ipdb.set_trace()
        pt_bin_edges, nominal, covarience, syst_covar, stat_covar = \
            process_eta(eta, root_file, sorted_by_eta[eta], allsyst,
                        include_syst, exclude_syst)
        objective = step.get_objective_function(pt_bin_edges, nominal, covarience,
                                                integrate, binmeans)
        inital_params = np.zeros(n_params)
        inital_params[:2] = step.get_initial_guess(nominal)

        result = scipy.optimize.minimize(objective, inital_params, method=intial_method,
                                         options={'maxiter': inital_steps},
                                         jac=objective.Jacobian, hess=objective.Hessian)
        new_params = result.x
        new_chi2 = objective(new_params)
        iteration = 0
        failed_iterations = 0
        while failed_iterations < n_methods:
            iteration += 1
            next_method = MINIMIZE_METHODS[iteration%n_methods]
            #print("Trying " + next_method, end='')
            improvement = 1
            count = 0
            while improvement > 0:
                print('.', end='')
                count += 1
                best_params = np.copy(new_params)
                best_chi2 = new_chi2
                result = scipy.optimize.minimize(objective, best_params, method=next_method,
                                                 options={'maxiter': batch_steps},
                                                 jac=objective.Jacobian, hess=objective.Hessian)
                new_params = result.x
                new_chi2 = objective(new_params)
                improvement = best_chi2 - new_chi2
                improvements[next_method].append(improvement)
            if count < 2:
                print('x', end='')
                failed_iterations += 1
            elif failed_iterations:
                print('^', end='')
                failed_iterations -= 1
            new_params = best_params
            new_chi2 = best_chi2
        print(best_chi2)
        chi2[e] = best_chi2
        optimised[e] = best_params
    #overall_points = humpday.points_race(objectives, n_dim=15)
    print("Results ~~~~~~~")
    print(chi2)
    print(optimised)
    for name in improvements:
        print("{} {: 8.6f}".format(name, np.mean(improvements[name])))


def test_switch_out(data, output, exclude_syst, include_syst,
                    excludeHigh, excludeLow, maxiter,
                    maxdegree, nsigma, autostop, minimiser, integrate,
                    binmeans, **unneeded):
    select_results, n_select = run(data, output, False, exclude_syst, include_syst, excludeHigh,
                                   excludeLow, maxiter, maxdegree, nsigma, autostop, minimiser,
                                   integrate, binmeans)
    allsyst_results, n_all = run(data, output, True, exclude_syst, include_syst, excludeHigh,
                                 excludeLow, maxiter, maxdegree, nsigma, autostop, minimiser,
                                 integrate, binmeans, to_improve=select_results)
    assert n_select < n_all
    display(select_results, n_select)
    display(allsyst_results, n_all)


if __name__ == "__main__":
    arg_dict = _cmnd_args()
    data, n_syst = run(**arg_dict)
    root_file = arg_dict['rootcomparison']
    save(data, arg_dict, n_syst, name_prefix='', root_file=root_file)
    #display(data, n_syst, arg_dict['rootcomparison'])
    #test_minimisers(**arg_dict)
    #test_hybrid(**arg_dict)
    #test_switch_out(**arg_dict)
    

