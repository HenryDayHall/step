
def get_hists(filename):
    import ROOT
    # unless you retain the tfile object, the hists will disappear
    global tfile
    tfile = ROOT.TFile.Open(filename)
    ybins = [key.GetName() for key in tfile.GetListOfKeys()]
    bins = len(ybins)
    hist_names = ["{}/ratioExt".format(y) for y in ybins]
    hists = [tfile.Get(name) for name in hist_names]
    return ybins, hists


def reprocess_title(title):
    start, numbers = title.split('=')
    n_coeff = start.split('_')[1].split('/')[0]
    n_coeff = int(n_coeff.replace('{', '').replace('}', ''))
    denominator, numerator = numbers.split('/')
    denominator = float(denominator.strip())
    numerator = int(numerator.strip())
    message = "root; chi2/ndf\n= {:.2g}/{}\n= {:.2g}"\
        .format(denominator, numerator, denominator/numerator)
    message += "\n n_coef={}".format(n_coeff)
    return message


def plot_py(filename, ax_arr=None):
    from matplotlib import pyplot as plt
    # needed so that window management is given entirely to matpltolib
    import ROOT
    ROOT.gROOT.SetBatch(True)
    y_bins, hists = get_hists(filename)
    n_eta_bins = len(y_bins)
    if ax_arr is None:
        fig, ax_arr = plt.subplots(1, n_eta_bins, figsize=(12, 3))
    else:
        fig = None
    for hist, ax in zip(hists,ax_arr):
        try:
            n_bins = int(hist.GetEntries())
        except OverflowError:
            continue  # error in root data
        bin_values = [hist.GetBinContent(i) for i in range(n_bins)]
        peak = bin_values.index(max(bin_values))
        clip_point = next((peak + i for i, v in enumerate(bin_values[peak:])
                          if v==0), len(bin_values))
        bin_values = bin_values[:clip_point]
        bin_centers = [hist.GetBinCenter(i) for i in range(clip_point)]
        bin_errors = [[hist.GetBinErrorLow(i) for i in range(clip_point)],
                      [hist.GetBinErrorUp(i) for i in range(clip_point)]]
        ax.errorbar(bin_centers, bin_values, bin_errors, color='gray', fmt='.',
                    alpha=0.6, label="root's minimiser")
        ax.set_ylabel("Ratio to smooth fit")
        ax.set_xlabel("$p_T$")
        ax.semilogx()
        title = hist.GetTitle()
        message = reprocess_title(title)
        ax.text(0.5, 0.5, message, color='dimgrey',
                verticalalignment='center', horizontalalignment='center',
                transform=ax.transAxes)
        
    if fig is not None:
        fig.set_tight_layout(True)


def plot_test1(filename, zoom=False):
    import ROOT
    ybins, hists = get_hists(filename)
    n_eta_bins = len(ybins)
    canvas_name = filename.replace(".root", "_canvas")
    canvas = ROOT.TCanvas(canvas_name, canvas_name, 1400, 280)
    canvas.Divide(n_eta_bins)
    for i in range(1, n_eta_bins):
        canvas.cd(i)
        hist = hists[i-1]
        if hist:
            hist.Draw()
            if zoom:
                hist.GetYaxis().SetRangeUser(0.8, 1.1)
            hist.GetYaxis().SetTitle("Ratio to smooth fit")
            ROOT.gPad.SetLogx()
    save_types = [".png", ".pdf"]
    import os
    base_name = os.path.split(filename)[1].replace(".root", "")
    save_names = ["../outputs/cpp_py/" + base_name + stype
                  for stype in save_types]
    for save_name in save_names:
        canvas.SaveAs(save_name)


if __name__ == "__main__":
    import sys
    file_name = sys.argv[1]
    plot_test1(file_name)
