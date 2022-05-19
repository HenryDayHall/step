def run():
    import numpy as np
    import binmeans
    n_bins = 10
    bin_edges = np.sort(np.random.rand(n_bins+1)*10+1)
    bin_values = np.sort(np.random.normal(1, 2, n_bins))
    bin_means = binmeans.locate(bin_edges, bin_values)
    print(bin_means)
    return bin_means

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    bin_means = run()
    plt.scatter(range(10), bin_means)
    input()
