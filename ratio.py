import ROOT
import uproot
import pandas as pd
import numpy as np
import argparse
import yaml

def get_average_dn_deta(centmin, centmax):
    mapping = {
        (0, 1): 20.07,
        (1, 10): 15.94,
        (10, 30): 11.42,
        (30, 50): 7.93,
        (50, 70): 5.58,
        (70, 100): 3.52,
        (0, 100): 7.08}
    return mapping[(centmin, centmax)]


def get_raw_yields(config, cent_min, cent_max):
    """
    Retrieve raw yield histograms from a ROOT file based on centrality ranges.

    Parameters:
    - config (dict): Configuration dictionary containing input file paths.
    - cent_min (list): List of minimum centrality values.
    - cent_max (list): List of maximum centrality values.

    Returns:
    - tuple: A tuple containing two ROOT histograms:
        - h_raw_yields_ds: Histogram for raw yields of ds.
        - h_raw_yields_dplus: Histogram for raw yields of dplus.
    """
    with ROOT.TFile.Open(config['inputs']['raw_yields']) as in_file_raw_yields:
        if cent_min is None and cent_max is None:
            suffix = ""
        else:
            suffix = f"_{cent_min}_{cent_max}"

        h_raw_yields_ds = in_file_raw_yields.Get(f"h_raw_yields_ds{suffix}")
        h_raw_yields_ds.SetDirectory(0)
        h_raw_yields_dplus = in_file_raw_yields.Get(f"h_raw_yields_dplus{suffix}")
        h_raw_yields_dplus.SetDirectory(0)

    return h_raw_yields_ds, h_raw_yields_dplus


def get_efficiencies(config, cent_min, cent_max):
    """
    Retrieve efficiency histograms for prompt Ds and D+ from a ROOT file.

    Parameters:
    - config (dict): Configuration dictionary containing input file paths.
    - cent_min (list): List of minimum centrality values.
    - cent_max (list): List of maximum centrality values.

    Returns:
    - tuple: A tuple containing two ROOT histograms:
        - h_eff_prompt_ds: Efficiency histogram for prompt Ds.
        - h_eff_prompt_dplus: Efficiency histogram for prompt D+.
    """
    with ROOT.TFile.Open(config['inputs']['efficiency']) as eff_file:
        if cent_min is None and cent_max is None:
            suffix = ""
        else:
            suffix = f"_cent_{cent_min}_{cent_max}"

        h_eff_prompt_ds = eff_file.Get(f"eff_DsPrompt{suffix}")
        h_eff_prompt_ds.SetDirectory(0)
        h_eff_prompt_dplus = eff_file.Get(f"eff_DplusPrompt{suffix}")
        h_eff_prompt_dplus.SetDirectory(0)

    return h_eff_prompt_ds, h_eff_prompt_dplus


def get_prompt_fractions(config, cent_min, cent_max):
    """
    Retrieves the prompt fractions histograms for Ds and D+ from the specified ROOT files.

    Args:
    - config (dict): Configuration dictionary containing input file paths.

    Returns:
    - tuple: A tuple containing two ROOT histograms:
        - h_prompt_frac_ds: Prompt fraction for Ds.
        - h_prompt_frac_dplus: Prompt fraction for D+.
    """
    if config['mult_differential_frac']:
        if cent_min == 0 and cent_max == 100:
            file_name_ds = config['inputs']['prompt_frac_ds']
        else:
            file_name_ds = config['inputs']['prompt_frac_ds'].replace("MB", f"{cent_min}_{cent_max}")
    else:
        file_name_ds = config['inputs']['prompt_frac_ds']
    with ROOT.TFile.Open(file_name_ds) as in_file_prompt_frac_ds:
        h_prompt_frac_ds = in_file_prompt_frac_ds.Get(
            f"hRawFracPrompt_cent_{cent_min}_{cent_max}"
        )
        h_prompt_frac_ds.SetDirectory(0)

    if config['mult_differential_frac']:
        if cent_min == 0 and cent_max == 100:
            file_name_dplus = config['inputs']['prompt_frac_dplus']
        else:
            file_name_dplus = config['inputs']['prompt_frac_dplus'].replace("MB", f"{cent_min}_{cent_max}")
    else:
        file_name_dplus = config['inputs']['prompt_frac_dplus']
    with ROOT.TFile.Open(file_name_dplus) as in_file_prompt_frac_dplus:
        h_prompt_frac_dplus = in_file_prompt_frac_dplus.Get(
            f"hRawFracPrompt_cent_{cent_min}_{cent_max}"
        )
        h_prompt_frac_dplus.SetDirectory(0)
    #for i_pt in range(h_prompt_frac_ds.GetNbinsX()):
    #    h_prompt_frac_ds.SetBinContent(i_pt, 1)
    #    h_prompt_frac_ds.SetBinError(i_pt, 0)
    #    h_prompt_frac_dplus.SetBinContent(i_pt, 1)
    #    h_prompt_frac_dplus.SetBinError(i_pt, 0)
    return h_prompt_frac_ds, h_prompt_frac_dplus


def get_ratio_vs_pt(
    config, h_raw_yields_ds, h_raw_yields_dplus, h_eff_prompt_ds,
    h_eff_prompt_dplus, h_prompt_frac_ds, h_prompt_frac_dplus
    ):
    """
    Calculate the ratio of corrected yields of D_s^+ to D^+.

    Parameters:
    - config (dict): Configuration dictionary.
    - h_raw_yields_ds (ROOT.TH1): Histogram of raw yields for D_s^+.
    - h_raw_yields_dplus (ROOT.TH1): Histogram of raw yields for D^+.
    - h_eff_prompt_ds (ROOT.TH1): Histogram of prompt efficiency for D_s^+.
    - h_eff_prompt_dplus (ROOT.TH1): Histogram of prompt efficiency for D^+.
    - h_prompt_frac_ds (ROOT.TH1): Histogram of prompt fraction for D_s^+.
    - h_prompt_frac_dplus (ROOT.TH1): Histogram of prompt fraction for D^+.

    Returns:
        ROOT.TH1: Histogram of the ratio of corrected yields of D_s^+ to D^+.
    """
    h_corrected_yields_ds = h_raw_yields_ds.Clone("h_corrected_yields_ds")
    h_corrected_yields_ds.Divide(h_eff_prompt_ds)
    h_corrected_yields_ds.Scale(1 / config['br']['ds_to_phipi_to_kkpi'])
    h_corrected_yields_ds.Multiply(h_prompt_frac_ds)

    h_corrected_yields_dplus = h_raw_yields_dplus.Clone("h_corrected_yields_dplus")
    h_corrected_yields_dplus.Divide(h_eff_prompt_dplus)
    h_corrected_yields_dplus.Scale(1 / config['br']['dplus_to_phipi_to_kkpi'])
    h_corrected_yields_dplus.Multiply(h_prompt_frac_dplus)

    h_ratio = h_corrected_yields_ds.Clone("h_ratio")
    h_ratio.Divide(h_corrected_yields_dplus)
    h_ratio.SetTitle(";#it{p}_{T} (GeV/#it{c});D_{s}^{+}/D^{+} Ratio")

    return h_ratio


def get_ratios_vs_cent(h_ratios_vs_pt, cent_mins, cent_maxs):
    """
    Generate histograms of ratios versus centrality from input histograms of ratios versus pT.

    Parameters:
    - h_ratios_vs_pt (list of ROOT.TH1): List of histograms representing ratios
        versus pT for different centrality bins.
    - cent_mins (list of float): List of minimum centrality values for each bin.
    - cent_maxs (list of float): List of maximum centrality values for each bin.

    Returns:
    - histos (list of ROOT.TH1D): List of histograms representing ratios versus centrality.
    """
    h_ratios_vs_pt = h_ratios_vs_pt.copy()
    cent_tuple = list(zip(cent_mins, cent_maxs))
    if (0, 100) in cent_tuple:
        idx_zero_hundred = cent_tuple.index((0, 100))
        cent_mins.pop(idx_zero_hundred)
        cent_maxs.pop(idx_zero_hundred)
        h_ratios_vs_pt.pop(idx_zero_hundred)
    cent_edges = np.asarray(cent_mins + [cent_maxs[-1]], "d")
    histos = []
    for i in range(h_ratios_vs_pt[0].GetNbinsX()):
        suffix = f"{h_ratios_vs_pt[0].GetBinLowEdge(i+1)*10:.0f}_{h_ratios_vs_pt[0].GetBinLowEdge(i+2)*10:.0f}"
        histos.append(ROOT.TH1D(
            f"h_ratio_cent_{suffix}",
            ";Centrality percentile;D_{s}^{+}/D^{+} Ratio",
            len(cent_edges)-1, cent_edges
        ))
        histos[-1].SetDirectory(0)
        for i_cent in range(len(cent_mins)):
            histos[-1].SetBinContent(i_cent+1, h_ratios_vs_pt[i_cent].GetBinContent(i+1))
            histos[-1].SetBinError(i_cent+1, h_ratios_vs_pt[i_cent].GetBinError(i+1))

    return histos


def get_ratios_vs_dndeta(h_ratios_vs_cent, cent_mins, cent_maxs):
    cent_tuple = list(zip(cent_mins, cent_maxs))
    print(cent_tuple)
    if (0, 100) in cent_tuple:
        idx_zero_hundred = cent_tuple.index((0, 100))
        cent_mins.pop(idx_zero_hundred)
        cent_maxs.pop(idx_zero_hundred)
        h_ratios_vs_cent.pop(idx_zero_hundred)
    graphs = []
    for histo in h_ratios_vs_cent:
        y, y_err = [], []
        for i in range(histo.GetNbinsX()):
            y.append(histo.GetBinContent(i+1))
            y_err.append(histo.GetBinError(i+1))
        x = [get_average_dn_deta(cent_min, cent_max) for cent_min, cent_max in zip(cent_mins, cent_maxs)]
        x = np.asarray(x, "d")
        y = np.asarray(y, "d")
        y_err = np.asarray(y_err, "d")
        x_err = np.asarray([0]*len(x), "d")  
        suffix = histo.GetName().split("h_ratio_cent")[-1]
        graphs.append(ROOT.TGraphAsymmErrors(len(x), x, y, x_err, x_err, y_err, y_err))
        graphs[-1].SetName(f"g_ratio_dndeta{suffix}")
    return graphs

def evaluate_ratio(config_file_name):
    """
    Evaluate the Ds+/D+ ratio and save the histograms to a ROOT file.

    Parameters:
        config_file_name (str): Path to the configuration YAML file.
    """
    with open(config_file_name, 'r', encoding="utf8") as file:
        config = yaml.safe_load(file)
    
    with open(config['inputs']['cutset'], 'r', encoding="utf8") as file:
        cutset = yaml.safe_load(file)

    if "cent" in cutset:
        cent_mins, cent_maxs = cutset["cent"]["min"], cutset["cent"]["max"]
    else:
        cent_mins, cent_maxs = [None], [None]

    h_ratios_vs_pt = []
    with ROOT.TFile(config['output_file'], "RECREATE") as output_file:
        for cent_min, cent_max in zip(cent_mins, cent_maxs):
            h_rawy_ds, h_rawy_dplus = get_raw_yields(config, cent_min, cent_max)
            h_eff_ds_prompt, h_eff_dplus_prompt = get_efficiencies(config, cent_min, cent_max)
            h_fprompt_ds, h_fprompt_dplus = get_prompt_fractions(config, cent_min, cent_max)
            h_ratios_vs_pt.append(get_ratio_vs_pt(
                config, h_rawy_ds, h_rawy_dplus, h_eff_ds_prompt,
                h_eff_dplus_prompt, h_fprompt_ds, h_fprompt_dplus
            ))
            if cent_min is not None and cent_max is not None:
                output_file.mkdir(f"centrality_{cent_min}_{cent_max}")
                output_file.cd(f"centrality_{cent_min}_{cent_max}")
            
            h_rawy_ds.Write()
            h_rawy_dplus.Write()
            h_eff_ds_prompt.Write()
            h_eff_dplus_prompt.Write()
            h_fprompt_ds.Write()
            h_fprompt_dplus.Write()
            h_ratios_vs_pt[-1].Write()

        if cent_min is not None and cent_max is not None:
            h_ratios_vs_cent = get_ratios_vs_cent(h_ratios_vs_pt, cent_mins, cent_maxs)
            output_file.cd()
            for h in h_ratios_vs_cent:
                h.Write()
            g_ratios_vs_dndeta = get_ratios_vs_dndeta(h_ratios_vs_cent, cent_mins, cent_maxs)
            for g in g_ratios_vs_dndeta:
                g.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Ds+/D+ ratio')
    parser.add_argument('config_file', metavar='text', help='Path to the config file')
    args = parser.parse_args()

    evaluate_ratio(args.config_file)
