"""Module for the AllEtasData class, a data type for a collection of results across a range of eta"""
# double underscore keeps it out the namespace of the module
import numpy as _np
import step as _step

class AllEtasData:
    """
    Collection of results for each eta bin.

    Attributes
    ----------
    etas : list of ints
        etas contained in this AllEtasData,
        sorted in accending order
    pt_bin_edges : dict of arrays of floats
        keys of the dict are the eta values
        values of the dict are the bin edges
        for data of that eta
    nominals : dict of arrays of floats
        keys of the dict are the eta values
        values of the dict are the bin values
        for data of that eta
    syst_covarience : dict of 2d arrays of floats
        keys of the dict are the eta values
        values of the dict are the systematic covariance matrix
        for data of that eta
    stat_covarience : dict of 2d arrays of floats
        keys of the dict are the eta values
        values of the dict are the statistical covariance matrix
        for data of that eta
    results : dict of lists of Results
        keys of the dict are the eta values
        values of the dict are lists of fits (Results)
        for data of that eta
    lowest_chi2 : dict of ints
        keys of the dict are the eta values
        values of the dict are indices of the fit with lowest chi2
        for data of that eta
    lowest_chi2_per_ndf : dict of ints
        keys of the dict are the eta values
        values of the dict are indices of the fit with lowest chi2 per ndf
        for data of that eta
    first_unitary_chi2_per_ndf : dict of ints
        keys of the dict are the eta values
        values of the dict are indices of the fit of the first chi2 per ndf
        equal to 1 for data of that eta
    polynomial : dict of step.Polynomials
        keys of the dict are the eta values
        values of the dict are lists of functions for fits (Polynomials)
        for data of that eta

    Notes
    -----
    Call str(result) for a human readable description.
    """
    def __init__(self):
        self.pt_bin_edges = {}
        self.nominals = {}
        self.syst_covarience = {}
        self.stat_covarience = {}
        self.results = {}
        self.lowest_chi2 = {}
        self.lowest_chi2_per_ndf = {}
        self.first_unitary_chi2_per_ndf = {}
        self.polynomial = {}

    def __str__(self):
        output = ""
        for eta in self.lowest_chi2:
            output += "\neta = {}~~~~~~\n\n".format(eta)
            output += "Lowest chi2\n"
            output += str(self.results[eta][self.lowest_chi2[eta]])
            output += "\nLowest chi2 per ndf\n"
            output += str(self.results[eta][self.lowest_chi2_per_ndf[eta]])
            output += "\nFirst unitary chi2 per ndf\n"
            output += str(self.results[eta][self.first_unitary_chi2_per_ndf[eta]])
        return output

    @property
    def etas(self):
        return sorted(self.pt_bin_edges)

    def add_inputs(self, eta, pt_bin_edges, nominal,
                   syst_covarience, stat_covarience):
        """
        Add the input data corrisponding to one eta.

        Parameters
        ----------
        eta : int
            eta value for these inputs
        pt_bin_edges : array of floats
            the bin edges of pt bins for data of that eta
        nominal : arrays of floats
            bin cross section values for data of that eta
        syst_covarience : 2d arrays of floats
            the systematic covariance matrix for data of that eta
        stat_covarience : 2d arrays of floats
            the statistical covariance matrix for data of that eta
        """
        self.pt_bin_edges[eta] = pt_bin_edges
        self.nominals[eta] = nominal
        self.syst_covarience[eta] = syst_covarience
        self.stat_covarience[eta] = stat_covarience

    def add_results(self, eta, results):
        """
        Add the results corrisponding to one eta.

        Parameters
        ----------
        eta : int
            eta value for these inputs
        results : list of result.Result
            results for this eta
        """
        assert eta in self.pt_bin_edges, "Didn't add inputs for {}".format(eta)
        self.results[eta] = results
        self.lowest_chi2[eta] = min(range(len(results)),
                                    key=lambda i: _np.nan_to_num(results[i].chi2, nan=_np.inf))
        lowest = min(range(len(results)), key=lambda i: _np.nan_to_num(results[i].chi2_per_ndf, nan=_np.inf))
        self.lowest_chi2_per_ndf[eta] = lowest
        limit = 1 + 1e-3
        sorted_results = sorted(results, key = lambda r : -r.n_degrees_of_freedom)
        first = next((r for r in sorted_results if r.chi2_per_ndf < limit), results[lowest])
        self.first_unitary_chi2_per_ndf[eta] = results.index(first)
        # make the polynomial
        min_val = self.pt_bin_edges[eta][0]
        max_val = self.pt_bin_edges[eta][-1]
        polynomial = _step.Polynomial(min_val, max_val)
        self.polynomial[eta] = polynomial

