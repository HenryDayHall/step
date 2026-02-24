 [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# A python library to implement Step

Inspired by the work at [https://arxiv.org/abs/2111.09968](https://arxiv.org/abs/2111.09968).


## Setup

This library needs python3, `pyroot`, `matplotlib`, `scipy` and `nlopt`.

To install these using conda;
```
> conda install --yes --file requirements.txt
```
It's not possible to install `pyroot` using pip at present, 
but if you have already installed a recent version of ROOT
then you likely have `pyroot` on your system already.
In that case, to install using pip;
```
> yes | pip install matplotlib scipy nlopt
```


Depending on what kind of data you would like to work on,
run one of the `download` scripts.

- For 8TeV deltaR=6 data, `cd HEPdata; source download`
- For 8TeV deltaR=4 data, `cd HEPdata4; source download`
- For 13TeV deltaR=4 data, `cd HEPdata4_13TeV; source download`

That should download a lot of text files, and one root file to
the corresponding directory.

## Running

The intended entry point is `ATLAS_data.py`.
This can fit data in from any of the folders `HEPdata`, `HEPdata4` or `HEPdata4_13TeV`.
It's easiest to run using command line arguments,
you can see a list of these by doing;

```bash
$ python ATLAS_data.py --help
```

A good example would be;

```bash
$ python ATLAS_data.py --data HEPdata --minimiser powell --maxdegree 20 --binmeans --nocurriculum --allsyst
```

- The `--data` flag must point to the location of a folder with downloaded data.
- The `--minimiser` flag must name one of scipy's or nlopt's minimisers. A list of scipy's minimisers is [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). A list of nlop's minimisers is [here](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/). The scipy minimiser `powell` is found to be quite reliable.
- The `--maxdegree` flag controls how many time the optimiser will try the optimisation again with another parameter. The fit will be tried with 1 to "maxdegree" terms of the Chebyshev series.
- The `--binmeans` flag changes the location that the fit is compared to the value of the histogram from the centre of the bin to the location of a fitted mean of the bin. See `binmeans.py` for exactly how that's done. There is also an `--integrate` flag that tried to integrate the function over the bin, but it's slow and unstable.
- The `--nocurriculum` flag tells the fitter not to start the optimisation of n+1 parameters with the results of n parameters. It's short for "no curriculum learning". This is generally found to increase stability.
- The `--allsyst` flag tells the fitter to include all systematic uncertainties. Without it, only statistical uncertainties are used.

There are more flags available than this, checkout `python ATLAS_data.py --help`.

You would expect this to make many image files in `outputs`.
You can change the location of the outputs by setting `$OUTPUT_DIR`.


## Structure

This program is divided into 7 modules;

1. `step.py`; this where the fitting happens. Also calculates the fit function, and it's Hessian and Jacobean. If you wanted to use this with different data, or in another script, this is likely the module you want to import.
2. `result.py`; data class used to store results from `step.py`.
3. `binmeans.py`; given a set of bin values, performs a crude fit and finds the location of the mean value in each bin according to that fit. `test_binmeans.py` is just a sanity check for this.
4. `plot.py`; responsible for plotting outputs.
5. `plot_root.py`; adds information from root files produced by the c++ equivalent of this code to the plots produced by `plot.py`. Good for comparisons.
6. `alleta.py`; data class that `plot.py` takes as input.
7. `ATLAS_data.py`; For reading data in the ATLAS format as supplied by HEPdata. Also the entry point, so it's accumulated a bit of experimental junk at the bottom.

Sorry about the lack of docstrings..... happy to respond to any questions :)
