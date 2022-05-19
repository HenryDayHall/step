"""Module for the results class, a generic data type for a dingle result """
# double underscore keeps it out the namespace of the module
import numpy as _np

class Result:
    """
    Store a single parameter choice and the corrisponding
    metrics of suitability.

    Attributes
    ----------
    parameters : numpy array of floats
        parameters of the solution
    chi2 : float
        chi2 value for these parameters
    n_degrees_of_freedom : int
        number of degrees of freedom for the data and these paramters
    chi2_per_ndf : float
        chi2 divided by number of degrees of freedom
    chi2_per_ndf_error : float
        a possible measure of error on the chi2... 
        litrature seems split on how trustworthy this is

    Notes
    -----
    Call str(result) for a human readable description

    """

    def __init__(self, chi2, n_degrees_of_freedom, parameters):
        """
        Parameters
        ----------
        chi2 : float
            chi2 value for these parameters
        n_degrees_of_freedom : int
            number of degrees of freedom for the data and these paramters
        parameters : iterable of floats
            parameters of the solution
        """
        self.chi2 = chi2
        self.n_degrees_of_freedom = n_degrees_of_freedom
        self.parameters = _np.copy(parameters)

    @property
    def chi2_per_ndf(self):
        return self.chi2/self.n_degrees_of_freedom

    @property
    def chi2_per_ndf_error(self):
        return _np.sqrt(2./self.n_degrees_of_freedom)

    def __str__(self):
        message = "chi2/n_degrees_of_freedom = {}/{}".format(self.chi2, self.n_degrees_of_freedom)
        message += " = {} p/m {}.\n".format(self.chi2_per_ndf, self.chi2_per_ndf_error)
        message += "Parameters = {}".format(self.parameters)
        return message

