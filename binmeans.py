import numpy as np


def get_basic_fit(bin_edges, bin_values):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import array
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    # seems to be needed
    bin_edges = array.array('d', bin_edges)
    n_bins = len(bin_edges) - 1
    histogram = ROOT.TH1D("temp", "temp", n_bins, bin_edges)
    for v, c in zip(bin_values, bin_centers):
        histogram.Fill(c, v)
        
    basic_fit = ROOT.TF1("temp_fit", "[0]*pow(x/100,[1])*exp(-1*[2]*x)", bin_edges[0], bin_edges[-1])
    basic_fit.SetParameters(1, -0.01, 0.01)
        
    histogram.Fit(basic_fit, "RISAME")
    fit_params = np.array([basic_fit.GetParameter(i) for i in range(3)])
    histogram.Delete()
    return fit_params, basic_fit


def locate(bin_edges, bin_values):
    n_bins = len(bin_edges) - 1
    import ROOT
    fit_params, basic_fit = get_basic_fit(bin_edges, bin_values)
    bin_means = np.empty(n_bins)
    for bin_n in range(n_bins):
        upper = bin_edges[bin_n+1]
        lower = bin_edges[bin_n]
        mean = basic_fit.Integral(lower, upper)/(upper-lower)
        mean_loc = basic_fit.GetX(mean, lower, upper)
        bin_means[bin_n] = mean_loc
    return bin_means


if __name__ == "__main__":
    n_bins = 10
    bin_edges = np.sort(np.random.rand(n_bins+1)*10+1)
    bin_values = np.sort(np.random.normal(1, 2, n_bins))
    bin_means = locate(bin_edges, bin_values)
    from matplotlib import pyplot as plt
    plt.ion() 
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.scatter(bin_center, bin_means)
    input("Hit enter to close.")
    plt.close()
