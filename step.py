# the underscore prevent the other package from entering the namespace
import numpy as _np
# optional debug printing
DEBUG=False
if DEBUG:
    log_print = print
else:
    def log_print(*args, **kwargs):
        pass

class Polynomial:
    """
    Class for custom function based on chebyshev polynomials.

    The function takes logs of the inputs,
    evaluates the chebyshev polynomials at these values
    then takes the exponent of the output.

    Properties
    ----------
    min_val : float
        The minimum value of the input domain.
        Must be strictly greater than 0.
    max_val : float
        The maximum value of the input domain.
        Must be strictly greater than min_val.
    """
    def __init__(self, min_val, max_val):
        """
        Parameters
        ----------
        min_val : float
            The minimum value of the input domain.
            Must be strictly greater than 0.
        max_val : float
            The maximum value of the input domain.
            Must be strictly greater than min_val.
        """
        assert min_val > 0, \
            "Polynomial uses assumptions based on a positived definite range, " \
            + f"given range ({min_val}, {max_val}), with min <= 0."
        assert max_val > min_val, \
            f"Problem with range ({min_val}, {max_val}), max_val < min_val"
        self.min_val = min_val
        self.max_val = max_val
        self._log_min_val = _np.log(min_val)
        self._delta_logs = _np.log(max_val) - self._log_min_val

    def T_series(self, x, n_terms):
        """
        Terms of the chebyshev polynomial series, evaluated at specified points.

        Parameters
        ----------
        x : arraylike of floats or float
            Points to evaluate at.  Can have any number of dimensions,
            the dimensions will be replicated in latter axis of the output.
        n_terms : int
            number of terms of the polynomial series to generate

        Returns
        -------
        terms : numpy array of (n_terms, [shape of x]) floats
            Each term of the chebyshev polynomial series, evaluated
            at the locations specified in x. 
            The first dimension corrisponds to the term number,
            latter dimensions replicate the dimenions of x.
            If x is a float, then the array returned is 1d.
        """
        terms = _np.zeros((n_terms, *_np.shape(x)), dtype=float)
        if n_terms > 0:
            terms[0] = 1
        if n_terms > 1:
            terms[1] = x
        if n_terms > 2:
            for i in range(2, n_terms):
                terms[i] = 2*x*terms[i-1]-terms[i-2]
        return terms

    def T(self, x, i):
        """
        A term of the chebyshev polynomial series, evaluated at specified points.

        Parameters
        ----------
        x : arraylike of floats or float
            Points to evaluate at.  Can have any number of dimensions,
            the dimensions will be replicated in shape of the output.
        i : int
            term of the polynomial series to generate

        Returns
        -------
        : numpy array of ([shape of x], ) floats
            Requested term of the chebyshev polynomial series, evaluated
            at the locations specified in x. 
            The shape replicates the dimenions of x.
            If x is a float, then a float is returned.
        """
        return self.T_series(x, i)[-1]

    def _rescale_input(self, x):
        nx = -1 + 2*(_np.log(x) - self._log_min_val)/self._delta_logs
        return _np.nan_to_num(nx)

    def __call__(self, x, parameters):
        """
        For given parameters, evaluate the function at chosen points.

        Paramters
        ---------
        x : arraylike of floats or float
            Points to evaluate at.  Can have any number of dimensions,
            the dimensions will be replicated in shape of the output.
        parameters : 1d arraylike of floats
            the coefficients of the chebyshev polynomial series.

        Returns
        -------
        : numpy array of ([shape of x],) floats
            the specified function, evaluated at x
            The shape replicates the dimenions of x.
            If x is a float, then a float is returned.
        """
        nx = self._rescale_input(x)
        #numpy_implementation = _np.polynomial.Chebyshev(parameters, domain=(self.min_val, self.max_val))
        T_series = self.T_series(nx, len(parameters))
        result = _np.sum(T_series.T*parameters, axis=-1).T
        return _np.exp(result)

    def integrate(self, min_val, max_val, parameters, n_points=10):
        """
        For given parameters, integrate the function between chosen points.

        Paramters
        ---------
        min_val : arraylike of floats or float
            Lower bounds to integrate from.  Can have any number of dimensions,
            the dimensions will be replicated in shape of the output.
        max_val : arraylike of floats or float
            Upper bounds to integrate to.  Must have the same shape as min_val.
        parameters : 1d arraylike of floats
            the coefficients of the chebyshev polynomial series.
        n_points : int (optional)
            The integraton is a numeric trapzoid integral,
            n_points controls the trade between accuracy and speed.
            Default=10

        Returns
        -------
        : numpy array of floats
            The integrals between min_val and max_val,
            with a shape reflecting min_val.
        """
        xs = _np.linspace(min_val, max_val, n_points)
        ys = self(xs, parameters)
        return _np.trapz(ys, xs)

    def bin_heights(self, bin_edges, parameters, n_points_per_bin=10):
        """
        For given parameters, calculate the bin heights for a histogram
        with chosen bin edges.

        Paramters
        ---------
        bin_edges : 1d arraylike of (n_bins+1,) floats
            edge values of the histogram bins,
            length is number of bins + 1 as both top and bottom
            edge are included
        parameters : 1d arraylike of floats
            the coefficients of the chebyshev polynomial series.
        n_points_per_bin : int (optional)
            The integraton is a numeric trapzoid integral,
            n_points_per_bin controls the trade between accuracy and speed.
            Default = 10

        Returns
        -------
        : numpy array of floats
            The highest that each bin should have in a histogram
            fitted by this function.
        """
        xs = _np.vstack([_np.linspace(a, b, n_points_per_bin)
                        for a, b in zip(bin_edges[:-1], bin_edges[1:])])
        ys = self(xs, parameters)
        integral_over_bins = _np.trapz(ys, xs)
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        return integral_over_bins/bin_widths

    def partial_derivatives(self, bin_edges, parameters,
                            double=False, n_points_per_bin=10):
        """
        For given parameters, calculate the partial deriatives of the
        bin heights for a histogram with chosen bin edges with respect to 
        the parameters.

        Paramters
        ---------
        bin_edges : 1d arraylike of (n_bins+1,) floats
            edge values of the histogram bins,
            length is number of bins + 1 as both top and bottom
            edge are included
        parameters : 1d arraylike of (n_params,) floats
            the coefficients of the chebyshev polynomial series.
        double : bool (optional)
            single partial derivatives or double partial derivatives?
            Default = False
        n_points_per_bin : int (optional)
            The integraton is a numeric trapzoid integral,
            n_points_per_bin controls the trade between accuracy and speed.
            Default = 10

        Returns
        -------
        : numpy array of (n_params, [n_params], n_bins) floats
            The partial deriatives of each bin with respect to the
            parameters.
            if double is False the array is (n_params, n_bins),
            if double if True the array is (n_params, n_params, n_bins)
        """
        #import ipdb; ipdb.set_trace()
        n_bins = len(bin_edges) - 1
        n_parameters = len(parameters)
        # n_bins x n_points_per_bin
        xs = _np.vstack([_np.linspace(a, b, n_points_per_bin)
                        for a, b in zip(bin_edges[:-1], bin_edges[1:])])
        nxs = self._rescale_input(xs)
        # n_parameters x n_bins x n_points_per_bin
        T_series = self.T_series(nxs, n_parameters)
        if double:
            # n_parameters**2 x n_bins x n_points_per_bin
            T_series = _np.tile(T_series, (n_parameters, 1, 1, 1))*T_series
        ys = T_series * self(xs, parameters)
        # n_parameters x n_bins or n_parameters**2 x n_bins
        integral_over_bins = _np.trapz(ys, xs, axis=-1)
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        partials = integral_over_bins/bin_widths
        #if double:
        #    assert partials.shape == (n_parameters, n_parameters, n_bins)
        #else:
        #    assert partials.shape == (n_parameters, n_bins)
        return partials


def check_covarience(covarience):
    """
    Verify that the covarience matrix is well formed.

    Paramters
    ---------
    covarience : 2d arrays of floats
        covariance matrix to be checked.
    """

    if not _np.all(_np.isfinite(covarience)):
        return False
    cov_diagonal = _np.diag(covarience)
    if _np.any(cov_diagonal) < 0:
        return False
    u, s, vh = _np.linalg.svd(covarience)
    if _np.any(s) < 0:
        return False
    return True


class Correlator:
    """
    Checks how different parameter choices change the chi2 on specified data.

    Parameters
    ----------
    covarience : array of (n_bins, n_bins) floats
    TODO finish docs

    """
    def __init__(self, bin_edges, bin_values, covarience, to_fit,
                 integrate=False, bin_means=False):
        self.bin_edges = bin_edges
        self.bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        self.bin_values = bin_values
        self.covarience = covarience
        self.inv_covarience = _np.linalg.inv(covarience)
        self.integrate = integrate
        self.to_fit = to_fit
        self.large_num = _np.nan_to_num(_np.inf)
        if bin_means:
            assert not integrate and bin_means,\
                "'integrate' and 'bin_means' are mutually exsclusive"
            import binmeans
            self.evaluate_at = binmeans.locate(bin_edges, bin_values)
        elif not integrate:
            self.evaluate_at = self.bin_centers

    def __call__(self, parameters):
        if self.integrate:
            fit_values = self.to_fit.bin_heights(self.bin_edges, parameters)
        else:
            fit_values = self.to_fit(self.evaluate_at, parameters)

            log_print("LOG; evaluation_points={}".format(self.evaluate_at.tolist()))
        log_print("LOG; fit_values={}".format(fit_values.tolist()))
        tension = self.bin_values - fit_values
        log_print("LOG; tension={}".format(tension.tolist()))
        transpose = tension.reshape((-1, 1))
        # we can get math overflow here
        with _np.errstate(over='raise', divide='raise'):
            try:
                result = _np.matmul(tension, _np.matmul(self.inv_covarience, transpose))
            except FloatingPointError:
                # consider this to be an arbitary high number
                return self.large_num
        return result[0]

    def Jacobian(self, parameters):
        fit_values = self.to_fit.bin_heights(self.bin_edges, parameters)
        tension = fit_values - self.bin_values
        transpose = tension.reshape((-1, 1))
        partial_derivatives = self.to_fit.partial_derivatives(
            self.bin_edges, parameters, double=False)
        result = 2*_np.matmul(partial_derivatives, 
                             _np.matmul(self.inv_covarience, transpose))
        result = result.flatten()
        #assert result.shape == (len(parameters),), result.shape
        return _np.nan_to_num(result)

    def Hessian(self, parameters):
        fit_values = self.to_fit.bin_heights(self.bin_edges, parameters)
        tension = fit_values - self.bin_values
        transpose = tension.reshape((-1, 1))
        partial_derivatives = self.to_fit.partial_derivatives(
            self.bin_edges, parameters, double=False)
        double_partial_derivatives = self.to_fit.partial_derivatives(
            self.bin_edges, parameters, double=True)
        term1 = 2*_np.matmul(double_partial_derivatives, 
                            _np.matmul(self.inv_covarience, transpose))
        term1 = term1[..., 0]  # remove len 1 axis
        term2 = 2*_np.matmul(partial_derivatives, 
                            _np.matmul(self.inv_covarience, partial_derivatives.T))
        result = 2*(_np.nan_to_num(term1) + _np.nan_to_num(term2))
        #assert result.shape == (len(parameters), len(parameters)), result.shape
        return result


def get_objective_function(bin_edges, bin_values, covarience, integrate=False, bin_means=False):
    to_fit = Polynomial(bin_edges[0], bin_edges[-1])
    to_minimise = Correlator(bin_edges, bin_values, covarience, to_fit, integrate, bin_means)
    return to_minimise


def get_minimiser(minimiser, n_params, maxiter):
    import nlopt
    import scipy.optimize
    maxiter *= n_params
    if (minimiser is None or 
            minimiser in scipy.optimize._minimize.MINIMIZE_METHODS):
        #bounds = [(-30, 30)]*n_params
        bounds = None
        def minimise(to_minimise, params):
            result = scipy.optimize.minimize(
                to_minimise, params, bounds=bounds, method=minimiser,
                options={'maxiter': maxiter},
                jac=to_minimise.Jacobian, hess=to_minimise.Hessian)
            if not result.success:
                print("Problem in optimiser")
                print(result.message)
            return result.x
    else:
        #opt = nlopt.opt("GN_DIRECT_L_RAND", n_params)
        opt = nlopt.opt(minimiser, n_params)
        def minimise(to_minimise, params):
            opt.set_min_objective(lambda x, _: to_minimise(x))
            opt.set_lower_bounds([-30]*n_params)
            opt.set_upper_bounds([30]*n_params)
            opt.set_maxeval(maxiter)
            try:
                return opt.optimize(params)
            except RuntimeError:
                print("Problems in optimiser")
                return [_np.nan]*n_params
    return minimise


def get_initial_guess(bin_values):
    log_first_value = _np.log(bin_values[0])
    log_last_value = _np.log(bin_values[-1])
    guess = [0.5*(log_last_value + log_first_value),
             0.5*(log_last_value - log_first_value)]
    return guess


def improve_params(inital_params, inital_chi2,
                   to_minimise, minimiser, maxiter, stable_minimiser=None):
    n_free = len(inital_params)
    minimise = get_minimiser(minimiser, n_free, maxiter)
    params = minimise(to_minimise, _np.copy(inital_params))
    chi2 = to_minimise(params)
    improvement = chi2 < inital_chi2
    if not improvement and stable_minimiser is not None:
        print("Poor result, trying again with " + stable_minimiser)
        # try again with a more stable method
        minimise = get_minimiser(stable_minimiser, n_free, maxiter)
        params = minimise(to_minimise, _np.copy(inital_params))
        chi2 = to_minimise(result)
        improvement =  chi2 < inital_chi2
    if improvement:
        return improvement, chi2, params
    return improvement, inital_chi2, inital_params


def improve_results(existing_results, bin_edges, bin_values, covarience, max_degree=None, n_sigma_stop=1,
                    autostop=True, integrate=False, bin_means=False, minimiser=None, maxiter=100):
    from result import Result
    print("Checking covarience")
    assert check_covarience(covarience), "Problem in covarience"
    print("Setting up optimisation problem")
    to_minimise = get_objective_function(bin_edges, bin_values, covarience, integrate, bin_means)
    new_results = []
    for result in existing_results:
        best_chi2 = result.chi2
        params = result.parameters
        print("Starting chi2 = {}".format(best_chi2))
        improvement, chi2, params = improve_params(params, best_chi2, to_minimise,
                                                   minimiser, maxiter)
        n_degrees_of_freedom = len(bin_values) - len(params)
        updated = Result(chi2, n_degrees_of_freedom, params)
        new_results.append(updated)
    return new_results


def get_smooth_fit(bin_edges, bin_values, covarience, max_degree=None, n_sigma_stop=1,
                   autostop=True, integrate=False, bin_means=False, minimiser=None, maxiter=100,
                   curriculum_learning=True):
    from result import Result
    if max_degree is None:
        max_degree = len(bin_values) - 2
    print("Checking covarience")
    assert check_covarience(covarience), "Problem in covarience"
    print("Setting up optimisation problem")
    to_minimise = get_objective_function(bin_edges, bin_values, covarience, integrate, bin_means)
    # intial guess
    inital_params = get_initial_guess(bin_values)
    start_free = len(inital_params)
    n_degrees_of_freedom = len(bin_values) - start_free
    use_params = _np.zeros(max_degree)
    use_params[:start_free] = inital_params
    chi2 = to_minimise(inital_params)
    best_chi2 = chi2
    print("Starting chi2 = {}".format(chi2))
    results = []
    results.append(Result(chi2, n_degrees_of_freedom, inital_params))
    stable_minimiser = 'nelder-mead'
    for n_free in range(start_free, max_degree + 1):
        print("Optimising with {} parameters".format(n_free))
        improvement, chi2, params = improve_params(use_params[:n_free], best_chi2, to_minimise,
                                                   minimiser, maxiter, stable_minimiser)
        if not improvement:
            print("Chi2 had increased")
            if autostop:
                break
        elif curriculum_learning:
            use_params[:n_free] = params
        print("\tChi2 = {}".format(chi2))
        n_degrees_of_freedom = len(bin_values) - n_free
        current = Result(chi2, n_degrees_of_freedom, params)
        results.append(current)
        if _np.isnan(chi2):
            print("Problem calculating chi2")
            break
        if abs(current.chi2_per_ndf - 1) <= n_sigma_stop*current.chi2_per_ndf_error:
            print("abs(Chi2/ndf - 1) <= nsigma_stop*err(ndf)")
        if n_degrees_of_freedom == 2:
            print("Only 2 d.o.f left. Stopping")
            break
    return results


def from_predefined(parameters, bin_edges, bin_values, covarience, integrate=False, bin_means=False):
    from result import Result
    log_print("LOG; parameters={}".format(parameters))
    max_degree = len(parameters)
    n_degrees_of_freedom = len(bin_values) - max_degree
    log_print("LOG; ndf={}".format(n_degrees_of_freedom))
    log_print("LOG; min_max=[{}, {}]".format(bin_edges[0], bin_edges[-1]))
    to_minimise = get_objective_function(bin_edges, bin_values, covarience, integrate, bin_means)
    chi2 = to_minimise(parameters)
    log_print("LOG; chi2={}".format(chi2))
    results = []
    results.append(Result(chi2, n_degrees_of_freedom, parameters))
    return results
    

