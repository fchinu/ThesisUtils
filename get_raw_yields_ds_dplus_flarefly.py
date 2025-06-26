"""
Script for fitting D+, D0 and Ds+ invariant-mass spectra.

run: python get_raw_yields_ds_dplus_flarefly.py config.yaml
"""

import argparse
import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import dataclasses  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import uproot  # noqa: E402
import yaml  # noqa: E402
import ROOT  # noqa: E402
import tensorflow as tf
import zfit
from flarefly.data_handler import DataHandler  # noqa: E402
from flarefly.fitter import F2MassFitter  # noqa: E402
from flarefly.utils import Logger  # noqa: E402
# pylint: disable=no-member


@dataclasses.dataclass
class BinsHelper:
    """
    Helper class for binning.

    Parameters:
    - mins (list or array-like): A list or array of minimum values for each bin.
    - maxs (list or array-like): A list or array of maximum values for each bin.

    Attributes:
    - mins (list or array-like): Stores the minimum values for each bin.
    - maxs (list or array-like): Stores the maximum values for each bin.
    - bins (list of tuples): A list of tuples where each tuple contains
        the minimum and maximum values for a bin.
    - edges (numpy.ndarray): An array of bin edges.
    - n_bins (int): The number of bins.
    """

    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs
        self.bins = [*zip(mins, maxs)]
        self.edges = self.mins + [self.maxs[-1]]
        self.edges = np.asarray(self.edges, 'd')
        self.n_bins = len(self.mins)


class HistHandler:  # pylint: disable=too-many-instance-attributes
    """
    Class designed to handle the creation, manipulation, and storage # noqa: D400, D205
    of histograms for various observables.

    Parameters:
    - pt_mins (list or array-like): Minimum values for pT bins.
    - pt_maxs (list or array-like): Maximum values for pT bins.
    - cent_mins (list or array-like, optional): Minimum values for centrality bins.
    Defaults to None.
    - cent_maxs (list or array-like, optional): Maximum values for centrality bins.
        Defaults to None.

    Attributes:
    - _pt_info (BinsHelper): Helper object for pT bins.
    - _cent_info (BinsHelper): Helper object for centrality bins.
    - _pt_axis_title (str): Title for the pT axis.
    - obs_common (list): List of common observables.
    - axes_titles_common (list): List of titles for common observables.
    - obs_not_common (list): List of non-common observables.
    - axes_titles_not_common (list): List of titles for non-common observables.
    - _n_ev (int or None): Number of events, initialized to None.
    """

    def __init__(self, pt_mins, pt_maxs, cent_mins=None, cent_maxs=None):
        if cent_mins is not None and cent_maxs is not None:
            self._cent_info = BinsHelper(cent_mins, cent_maxs)
        else:
            cent_mins, cent_maxs = None, None
            self._cent_info = None
        self._pt_info = BinsHelper(pt_mins, pt_maxs)
        self._pt_axis_title = '#it{p}_{T} (GeV/#it{c})'
        self.obs_common = [
            "raw_yields", "sigma", "sigma1", "sigma2", "frac1", "mean", "raw_yield_over_ev",
            "significance", "significance_over_sqrt_ev", "s_over_b", "signal", "background",
            "alphal", "alphar", "nl", "nr", "alpha", "n"
        ]
        self.axes_titles_common = [
            "Raw yields", "Width (GeV/#it{c}^{2})", "Width_{1} (GeV/#it{c}^{2})",
            "Width_{2} (GeV/#it{c}^{2})", "Gaussian fraction", "Mean (GeV/#it{c}^{2})",
            "Raw yields / N_{ev}", "Significance (3#sigma)", "Significance / #sqrt{N_{ev}}",
            "S/B (3#sigma)", "Signal (3#sigma)", "Background (3#sigma)", "#alpha_{l}",
            "#alpha_{r}", "n_{l}", "n_{r}", "#alpha", "n"
        ]
        self.obs_not_common = [
            "chi2", "sigma_ratio_second_first_peak", "corr_bkg_frac_over_dplus_0", "corr_bkg_frac_0"
        ]
        self.axes_titles_not_common = [
            "#chi^{2}/#it{ndf}", "Width second peak / width first peak",
            "Corr. bkg / D^{+} signal", "Corr. bkg fraction"
        ]
        self._n_ev = None

        self.__build_histos()

    def __create_histo(self, obs, y_title):
        """
        Create and append histograms to the `_histos` dictionary for a given observable.

        Parameters:
        - obs (str): The observable for which the histogram is being created.
        - y_title (str): The title for the y-axis of the histogram.
        """
        if self._cent_info is None:
            self._histos[obs].append(ROOT.TH1D(
                f'h_{obs}',
                f';{self._pt_axis_title};{y_title}', self._pt_info.n_bins, self._pt_info.edges
            ))
        else:
            for cent_min, cent_max in zip(self._cent_info.mins, self._cent_info.maxs):
                self._histos[obs].append(ROOT.TH1D(
                    f'h_{obs}_{cent_min:.0f}_{cent_max:.0f}',
                    f';{self._pt_axis_title};{y_title}', self._pt_info.n_bins, self._pt_info.edges
                ))
        for histos in self._histos.values():
            for hist in histos:
                hist.SetDirectory(0)

    def __build_histos(self):
        """Build histograms for the various observables."""
        self._histos = {f"{obs}_ds": [] for obs in self.obs_common}
        self._histos.update({f"{obs}_dplus": [] for obs in self.obs_common})
        self._histos.update({f"{obs}": [] for obs in self.obs_not_common})

        for obs, ax in zip(self.obs_common, self.axes_titles_common):
            self.__create_histo(f'{obs}_ds', ax)
            self.__create_histo(f'{obs}_dplus', ax)
        for obs, ax in zip(self.obs_not_common, self.axes_titles_not_common):
            self.__create_histo(f'{obs}', ax)

    def set_n_ev(self, n_ev):
        """
        Set the number of events.

        Parameters:
        - n_ev (int): Number of events.
        """
        self._n_ev = n_ev

    def set_histos(self, df):
        """
        Set histogram bin contents and errors based on the provided DataFrame.

        Parameters:
        - df (pandas.DataFrame): DataFrame containing the data to be used
            for setting histogram values.

        Raises:
        -------
        ValueError
            If the centrality bins are not consistent.
        """
        for _, row in df.iterrows():
            i_pt = row["i_pt"]
            if "cent_min" in row and "cent_max" in row:
                cent_min = row["cent_min"]
                cent_max = row["cent_max"]
                cent_edges = [*zip(self._cent_info.mins, self._cent_info.maxs)]
                i_cent = cent_edges.index((cent_min, cent_max))
            else:
                cent_min = None
                cent_max = None
                i_cent = 0
            common_df_cols = [
                "raw_yields", "mean",
                "significance", "signal", "background"
            ]
            if "sigma" in row:
                common_df_cols.append("sigma")
            else:
                common_df_cols.append("sigma1")
                common_df_cols.append("sigma2")
                common_df_cols.append("frac1")
            if "alphal" in row:
                common_df_cols.append("alphal")
                common_df_cols.append("alphar")
                common_df_cols.append("nl")
                common_df_cols.append("nr")
            if "alpha" in row:
                common_df_cols.append("alpha")
                common_df_cols.append("n")

            corr_bkg_cols = [col for col in list(df.columns) if "corr_bkg_frac" in col and "_cfg" not in col]
            not_common_df_cols = ["chi2"] + corr_bkg_cols
            for obs in common_df_cols:
                self._histos[f"{obs}_ds"][i_cent].SetBinContent(i_pt + 1, row[obs][0][0])
                self._histos[f"{obs}_ds"][i_cent].SetBinError(i_pt + 1, row[obs][0][1])
                if len(row[obs]) > 1:
                    self._histos[f"{obs}_dplus"][i_cent].SetBinContent(i_pt + 1, row[obs][1][0])
                    self._histos[f"{obs}_dplus"][i_cent].SetBinError(i_pt + 1, row[obs][1][1])
            for obs in not_common_df_cols:
                if not isinstance(row[obs], (tuple, list)):
                    self._histos[obs][i_cent].SetBinContent(i_pt + 1, row[obs])
                else:
                    self._histos[obs][i_cent].SetBinContent(i_pt + 1, row[obs][0])
                    self._histos[obs][i_cent].SetBinError(i_pt + 1, row[obs][1])
            self._histos["raw_yield_over_ev_ds"][i_cent].SetBinContent(
                i_pt + 1, row["raw_yields"][0][0] / self._n_ev
            )
            self._histos["raw_yield_over_ev_ds"][i_cent].SetBinError(
                i_pt + 1, row["raw_yields"][0][1] / self._n_ev
            )
            self._histos["significance_over_sqrt_ev_ds"][i_cent].SetBinContent(
                i_pt + 1, row["significance"][0][0] / np.sqrt(self._n_ev)
            )
            self._histos["significance_over_sqrt_ev_ds"][i_cent].SetBinError(
                i_pt + 1, row["significance"][0][1] / np.sqrt(self._n_ev)
            )
            if row["background"][0][0] != 0:
                self._histos["s_over_b_ds"][i_cent].SetBinContent(
                    i_pt + 1, row["signal"][0][0] / row["background"][0][0]
                )
                self._histos["s_over_b_ds"][i_cent].SetBinError(
                    i_pt + 1, row["signal"][0][1] / row["background"][0][0]
                )
            if len(row["raw_yields"]) > 1:
                self._histos["raw_yield_over_ev_dplus"][i_cent].SetBinContent(
                    i_pt + 1, row["raw_yields"][1][0] / self._n_ev
                )
                self._histos["raw_yield_over_ev_dplus"][i_cent].SetBinError(
                    i_pt + 1, row["raw_yields"][1][1] / self._n_ev
                )
                self._histos["significance_over_sqrt_ev_dplus"][i_cent].SetBinContent(
                    i_pt + 1, row["significance"][1][0] / np.sqrt(self._n_ev)
                )
                self._histos["significance_over_sqrt_ev_dplus"][i_cent].SetBinError(
                    i_pt + 1, row["significance"][1][1] / np.sqrt(self._n_ev)
                )
                self._histos["s_over_b_dplus"][i_cent].SetBinContent(
                    i_pt + 1, row["signal"][1][0] / row["background"][1][0]
                )
                self._histos["s_over_b_dplus"][i_cent].SetBinError(
                    i_pt + 1, row["signal"][1][1] / row["background"][1][0]
                )
                if "sigma" in row:
                    self._histos["sigma_ratio_second_first_peak"][i_cent].SetBinContent(
                        i_pt + 1, row["sigma"][1][0] / row["sigma"][0][0]
                    )

    def dump_to_root(self, output_file):
        """
        Dump histograms to a ROOT file.

        Parameters:
        - output_file (str): Path to the output ROOT file.
        """
        with ROOT.TFile(output_file, "RECREATE") as outfile:
            for key in self._histos:
                outfile.mkdir(key)
            for histos in self._histos.values():
                for hist in histos:
                    hist.Write()


def load_inputs(cfg, cut_set):
    """
    Load invariant mass histograms and event histogram from a ROOT file.

    Parameters:
    cfg (dict): Configuration dictionary containing the path to the input data file.
    cut_set (dict): Dictionary containing the cuts applied to the data.

    Returns:
    tuple: A tuple containing:
        - h_mass (list): List of invariant mass histograms.
        - h_ev (uproot.models.TH1): Event histogram.
    """
    pt_mins = cut_set["pt"]["min"]
    pt_maxs = cut_set["pt"]["max"]
    cent_mins, cent_maxs = None, None
    if "cent" in cut_set:
        cent_mins = cut_set["cent"]["min"] 
        cent_maxs = cut_set["cent"]["max"] 

    # load inv-mass histos
    h_mass = []
    with uproot.open(cfg["inputs"]["data"]) as f:
        h_ev = f["h_ev"]

        for pt_min, pt_max in zip(pt_mins, pt_maxs):
            if cent_mins is not None and cent_maxs is not None:
                for cent_min, cent_max in zip(cent_mins, cent_maxs):
                    suffix = f"{pt_min*10:.0f}_{pt_max*10:.0f}_cent_{cent_min:.0f}_{cent_max:.0f}"
                    h_mass.append(f[f'h_mass_{suffix}'])
            else:
                suffix = f"{pt_min*10:.0f}_{pt_max*10:.0f}"
                h_mass.append(f[f'h_mass_{suffix}'])

    return h_mass, h_ev


def create_fit_configs(cfg, cut_set):
    """
    Generate a list of fit configurations based on the provided configuration and cut sets.

    Args:
        cfg (dict): Configuration dictionary containing fit settings.
        cut_set (dict): Dictionary containing cut sets for pt and optionally cent.

    Returns:
        - fit_configs (list): A list of dictionaries (one per bin of pt-centrality),
            each representing a fit configuration with keys:
            - "cent_min" (optional): Minimum centrality cut.
            - "cent_max" (optional): Maximum centrality cut.
            - "pt_min": Minimum pt cut.
            - "pt_max": Maximum pt cut.
            - "mass_min": Minimum mass cut.
            - "mass_max": Maximum mass cut.
            - "rebin": Rebin factor.
            - "signal_func": Signal function.
            - "fix_ds_sigma": Fix Ds sigma.
            - "fix_dplus_sigma": Fix Dplus sigma.
            - "fix_sigma_dplus_to_ds": Fix sigma ratio of Dplus to Ds.
            - "bkg_func": Background function.
            - "use_bkg_templ": Use background template.
    """
    fit_configs = [(idx, *cfgs) for idx, cfgs in enumerate(zip(
        cut_set["pt"]["min"],
        cut_set["pt"]["max"],
        cfg["fit_configs"]["mass"]["mins"],
        cfg["fit_configs"]["mass"]["maxs"],
        cfg["fit_configs"]["rebin"],
        cfg["fit_configs"]["signal"]["signal_funcs"],
        cfg["fit_configs"]["signal"]["fix_ds_sigma"],
        cfg["fit_configs"]["signal"]["fix_dplus_sigma"],
        cfg["fit_configs"]["signal"]["fix_sigma_dplus_to_ds"],
        cfg["fit_configs"]["bkg"]["bkg_funcs"],
        cfg["fit_configs"]["bkg"]["use_bkg_templ"],
        *[sgn_func[par]["init"] for sgn_func in cfg["fit_configs"]["signal"]["par_init_limit"] for par in sgn_func],
        *[sgn_func[par]["min"] for sgn_func in cfg["fit_configs"]["signal"]["par_init_limit"] for par in sgn_func],
        *[sgn_func[par]["max"] for sgn_func in cfg["fit_configs"]["signal"]["par_init_limit"] for par in sgn_func],
        *[sgn_func[par]["fix"] for sgn_func in cfg["fit_configs"]["signal"]["par_init_limit"] for par in sgn_func]
    ))]
    if "cent" in cut_set:
        fit_configs = itertools.product(
            [*zip(cut_set["cent"]["min"], cut_set["cent"]["max"])],
            fit_configs
        )
        fit_configs = [(*cent, *config) for cent, config in fit_configs]
        config_keys = [
            "cent_min", "cent_max", "i_pt", "pt_min", "pt_max", "mass_min", "mass_max",
            "rebin", "signal_func", "fix_ds_sigma", "fix_dplus_sigma", "fix_sigma_dplus_to_ds",
            "bkg_func", "use_bkg_templ"
        ]
    else:
        config_keys = [
            "i_pt", "pt_min", "pt_max", "mass_min", "mass_max",
            "rebin", "signal_func", "fix_ds_sigma", "fix_dplus_sigma", "fix_sigma_dplus_to_ds",
            "bkg_func", "use_bkg_templ"
        ]
    config_keys += [
        *[f"{par}_init_{i_func}" for i_func, sgn_func in enumerate(cfg["fit_configs"]["signal"]["par_init_limit"]) for par in sgn_func],
        *[f"{par}_min_{i_func}" for i_func, sgn_func in enumerate(cfg["fit_configs"]["signal"]["par_init_limit"]) for par in sgn_func],
        *[f"{par}_max_{i_func}" for i_func, sgn_func in enumerate(cfg["fit_configs"]["signal"]["par_init_limit"]) for par in sgn_func],
        *[f"{par}_fix_{i_func}" for i_func, sgn_func in enumerate(cfg["fit_configs"]["signal"]["par_init_limit"]) for par in sgn_func]
    ]
    fit_configs = [dict(zip(config_keys, config)) for config in fit_configs]
    return fit_configs


def get_sigma_from_cfg(cfg, fit_config, particle_name):
    """
    Retrieve the sigma value for a given particle from the configuration.

    Parameters:
    - cfg (dict): Configuration dictionary containing input data and sigma settings.
    - fit_config (dict): Dictionary containing fit configuration parameters
        such as mass limits and rebinning.
    - particle_name (str): Name of the particle, either "ds" or "dplus".

    Returns:
    - float: The sigma value for the specified particle.

    Raises:
    - ValueError: If the particle_name is not "ds" or "dplus".
    """
    if particle_name not in ('ds', 'dplus'):
        raise ValueError(f"Invalid particle name: {particle_name}")
    if particle_name == "ds":
        idx_signal = 0
    else:
        idx_signal = 1
    cfg_sigma_fix = cfg["fit_configs"]["signal"][f"{particle_name}_sigma"]
    if not cfg_sigma_fix:  # fit MC
        pt_suffix = f"{fit_config['pt_min'] * 10:.0f}_{fit_config['pt_max'] * 10:.0f}"
        suffix = pt_suffix
        if "cent_min" in fit_config and "cent_max" in fit_config:
            suffix += f"_cent_{fit_config['cent_min']}_{fit_config['cent_max']}"
            data_hdl_mc = DataHandler(
                data=cfg["inputs"]["mc"][particle_name],
                histoname=f"h_mass_{suffix}",
                limits=[fit_config["mass_min"], fit_config["mass_max"]], rebin=fit_config["rebin"]
            )
        else:
            data_hdl_mc = DataHandler(
                data=cfg["inputs"]["mc"][particle_name],
                histoname=f"h_mass_{suffix}",
                limits=[fit_config["mass_min"], fit_config["mass_max"]], rebin=fit_config["rebin"]
            )
        fitter_mc = F2MassFitter(
            data_hdl_mc, name_signal_pdf=fit_config["signal_func"][idx_signal],
            name_background_pdf="nobkg",
            name=f"{particle_name}_pt{pt_suffix}_for_sigma", chi2_loss=True,
            verbosity=1, tol=1.e-1
        )
        fitter_mc.set_particle_mass(0, pdg_id=431 if particle_name == "ds" else 411)
        if fit_config["signal_func"][idx_signal] == "doublegaus":
            fitter_mc.set_signal_initpar(0, "sigma1", 0.01, limits=[0.001, 0.050])
            fitter_mc.set_signal_initpar(0, "sigma2", 0.02, limits=[0.001, 0.050])
            fitter_mc.set_signal_initpar(0, "frac1", 0.02, limits=[0., 1.])
        elif fit_config["signal_func"][idx_signal] == "gaussian":
            fitter_mc.set_signal_initpar(0, "sigma", 0.01, limits=[0.001, 0.050])
        elif fit_config["signal_func"][idx_signal] == "doublecb":
            fitter_mc.set_signal_initpar(0, "sigma", 0.01, limits=[0.001, 0.050])
            fitter_mc.set_signal_initpar(0, "alphal", 1.5, limits=[1., 3.])
            fitter_mc.set_signal_initpar(0, "alphar", 1.5, limits=[1., 3.])
            fitter_mc.set_signal_initpar(0, "nl", 50, limits=[30., 100.])
            fitter_mc.set_signal_initpar(0, "nr", 50, limits=[30., 100.])
        elif fit_config["signal_func"][idx_signal] == "doublecbsymm":
            fitter_mc.set_signal_initpar(0, "sigma", 0.01, limits=[0.001, 0.050])
            fitter_mc.set_signal_initpar(0, "alpha", 1.5, limits=[1., 3.])
            fitter_mc.set_signal_initpar(0, "n", 50, limits=[30., 100.])
        else:
            Logger("Signal function not supported", "FATAL")
        fitter_mc.mass_zfit()

        if fit_config["signal_func"][idx_signal] == "doublegaus":
            return (
                fitter_mc.get_signal_parameter(0, "sigma1")[0],
                fitter_mc.get_signal_parameter(0, "sigma2")[0],
                fitter_mc.get_signal_parameter(0, "frac1")[0]
            )
        elif fit_config["signal_func"][idx_signal] == "gaussian":
            return fitter_mc.get_sigma(0)[0]
        elif fit_config["signal_func"][idx_signal] == "doublecb":
            return (
                fitter_mc.get_signal_parameter(0, "sigma")[0],
                fitter_mc.get_signal_parameter(0, "alphar")[0],
                fitter_mc.get_signal_parameter(0, "alphal")[0],
                fitter_mc.get_signal_parameter(0, "nl")[0],
                fitter_mc.get_signal_parameter(0, "nr")[0]
            )
        elif fit_config["signal_func"][idx_signal] == "doublecbsymm":
            return (
                fitter_mc.get_signal_parameter(0, "sigma")[0],
                fitter_mc.get_signal_parameter(0, "alpha")[0],
                fitter_mc.get_signal_parameter(0, "n")[0]
            )


    if isinstance(cfg_sigma_fix, str):  # get from ROOT file
        with uproot.open(cfg_sigma_fix) as f:
            if "cent_min" in fit_config and "cent_max" in fit_config:
                if fit_config["signal_func"][idx_signal] == "doublegaus":
                    h_sigma1 = f[
                        f'h_sigma1_{particle_name}_{fit_config["cent_min"]}_{fit_config["cent_max"]}'
                    ]
                    h_sigma2 = f[
                        f'h_sigma2_{particle_name}_{fit_config["cent_min"]}_{fit_config["cent_max"]}'
                    ]
                    h_frac1 = f[
                        f'h_frac1_{particle_name}_{fit_config["cent_min"]}_{fit_config["cent_max"]}'
                    ]
                    return (
                        h_sigma1.values()[fit_config["i_pt"]],
                        h_sigma2.values()[fit_config["i_pt"]],
                        h_frac1.values()[fit_config["i_pt"]]
                    )
                elif fit_config["signal_func"][idx_signal] == "doublecbsymm":
                    h_sigma = f[f'h_sigma_{particle_name}_{fit_config["cent_min"]}_{fit_config["cent_max"]}']
                    h_alpha = f[f'h_alpha_{particle_name}_{fit_config["cent_min"]}_{fit_config["cent_max"]}']
                    h_n = f[f'h_n_{particle_name}_{fit_config["cent_min"]}_{fit_config["cent_max"]}']
                    return (
                        h_sigma.values()[fit_config["i_pt"]],
                        h_alpha.values()[fit_config["i_pt"]],
                        h_n.values()[fit_config["i_pt"]],
                    )
                else:
                    h_sigma = f[
                        f'h_sigma_{particle_name}_{fit_config["cent_min"]}_{fit_config["cent_max"]}'
                    ]
                    return h_sigma.values()[fit_config["i_pt"]]
            if fit_config["signal_func"][idx_signal] == "doublegaus":
                h_sigma1 = f[f'h_sigma1_{particle_name}']
                h_sigma2 = f[f'h_sigma2_{particle_name}']
                h_frac1 = f[f'h_frac1_{particle_name}']
                return (
                    h_sigma1.values()[fit_config["i_pt"]],
                    h_sigma2.values()[fit_config["i_pt"]],
                    h_frac1.values()[fit_config["i_pt"]]
                )
            elif fit_config["signal_func"][idx_signal] == "gaussian":
                h_sigma = f[f'h_sigma_{particle_name}']
                return h_sigma.values()[fit_config["i_pt"]]
            elif fit_config["signal_func"][idx_signal] == "doublecb":
                h_sigma = f[f'h_sigma_{particle_name}']
                h_alphar = f[f'h_alphar_{particle_name}']
                h_alphal = f[f'h_alphal_{particle_name}']
                h_nl = f[f'h_nl_{particle_name}']
                h_nr = f[f'h_nr_{particle_name}']
                return (
                    h_sigma.values()[fit_config["i_pt"]],
                    h_alphar.values()[fit_config["i_pt"]],
                    h_alphal.values()[fit_config["i_pt"]],
                    h_nl.values()[fit_config["i_pt"]],
                    h_nr.values()[fit_config["i_pt"]]
                )
            elif fit_config["signal_func"][idx_signal] == "doublecbsymm":
                h_sigma = f[f'h_sigma_{particle_name}']
                h_alpha= f[f'h_alpha_{particle_name}']
                h_n = f[f'h_n_{particle_name}']
                return (
                    h_sigma.values()[fit_config["i_pt"]],
                    h_alpha.values()[fit_config["i_pt"]],
                    h_n.values()[fit_config["i_pt"]],
                )

    if isinstance(cfg_sigma_fix, list):
        return cfg_sigma_fix[fit_config["i_pt"]]

    return cfg_sigma_fix  # if one value for all pt bins


def get_sigma_dplus_to_ds(cfg, fit_config, fitter):
    """
    Retrieve the sigma of the D+ peak using that of the Ds peak.

    Parameters:
    - cfg (dict): Configuration dictionary containing the ratio of sigma D+ to Ds.
    - fit_config (dict): Configuration dictionary for the fit.
    - fitter (flarefly.F2MassFitter): Fitter object used to perform the mass fit
        and extract sigma values.

    Returns:
    - float: The sigma value for the D+ peak.
    """
    # get sigma of ds peak
    fitter.mass_zfit()
    sigma = fitter.get_sigma(0)[0]
    cfg_sigma_ratio = cfg["fit_configs"]["signal"]["ratio_sigma_dplus_to_ds"]

    if not cfg_sigma_ratio:
        return sigma

    if isinstance(cfg_sigma_ratio, list):
        return sigma * cfg_sigma_ratio[fit_config["i_pt"]]

    if isinstance(cfg_sigma_ratio, str):
        with uproot.open(cfg_sigma_ratio) as f:
            h_sigma_ratio = f["h_sigma_ratio"]
            return sigma * h_sigma_ratio.values()[fit_config["i_pt"]]

    return sigma * cfg_sigma_ratio


def initialise_signal(fitter, fit_config, idx):
    """
    Initialise the signal parameters for the fitter based on the provided configuration.

    Parameters:
    - fitter (object): The fitter object that will be used to set the signal parameters.
    - fit_config (dict): A dictionary containing the configuration for the signal functions and their parameters.
    - idx (int): The index of the signal function to be initialised.
    """

    if f"mu_init_{idx}" in fit_config:
        fitter.set_signal_initpar(
            idx, "mu", fit_config.get(f"mu_init_{idx}", 1.9),
            limits=[fit_config.get(f"mu_min_{idx}", 0.), fit_config.get(f"mu_max_{idx}", 10.)],
            fix=fit_config.get(f"mu_fix_{idx}", False)
        )
    if fit_config["signal_func"][idx] == "gaussian":
        fitter.set_signal_initpar(
            idx, "sigma", fit_config.get(f"sigma_init_{idx}", 0.01),
            limits=[fit_config.get(f"sigma_min_{idx}", 0.001), fit_config.get(f"sigma_max_{idx}", 0.03)],
            fix=fit_config.get(f"sigma_fix_{idx}", False)
        )
    elif fit_config["signal_func"][idx] == "doublegaus": # double gauss
        fitter.set_signal_initpar(
            idx, "sigma1", fit_config.get(f"sigma1_init_{idx}", 0.01),
            limits=[fit_config.get(f"sigma1_min_{idx}", 0.001), fit_config.get(f"sigma1_max_{idx}", 0.03)],
            fix=fit_config.get(f"sigma1_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "sigma2", fit_config.get(f"sigma2_init_{idx}", 0.01),
            limits=[fit_config.get(f"sigma2_min_{idx}", 0.001), fit_config.get(f"sigma2_max_{idx}", 0.03)],
            fix=fit_config.get(f"sigma2_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "frac1", fit_config.get(f"frac1_init_{idx}", 0.01),
            limits=[fit_config.get(f"frac1_min_{idx}", 0.), fit_config.get(f"frac1_max_{idx}", 1.)],
            fix=fit_config.get(f"frac1_fix_{idx}", False)
        )
    elif fit_config["signal_func"][idx] == "doublecb":
        fitter.set_signal_initpar(
            idx, "sigma", fit_config.get(f"sigma_init_{idx}", 0.01),
            limits=[fit_config.get(f"sigma_min_{idx}", 0.001), fit_config.get(f"sigma_max_{idx}", 0.03)],
            fix=fit_config.get(f"sigma_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "alphar", fit_config.get(f"alphar_init_{idx}", 0.5),
            limits=[fit_config.get(f"alphar_min_{idx}", 0.), fit_config.get(f"alphar_max_{idx}", 10.)],
            fix=fit_config.get(f"alphar_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "alphal", fit_config.get(f"alphal_init_{idx}", 0.5),
            limits=[fit_config.get(f"alphal_min_{idx}", 0.), fit_config.get(f"alphal_max_{idx}", 10.)],
            fix=fit_config.get(f"alphal_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "nl", fit_config.get(f"nl_init_{idx}", 1.),
            limits=[fit_config.get(f"nl_min_{idx}", 0.), fit_config.get(f"nl_max_{idx}", 10.)],
            fix=fit_config.get(f"nl_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "nr", fit_config.get(f"nr_init_{idx}", 1.),
            limits=[fit_config.get(f"nr_min_{idx}", 0.), fit_config.get(f"nr_max_{idx}", 10.)],
            fix=fit_config.get(f"nr_fix_{idx}", False)
        )
    elif fit_config["signal_func"][idx] == "doublecbsymm":
        fitter.set_signal_initpar(
            idx, "sigma", fit_config.get(f"sigma_init_{idx}", 0.01),
            limits=[fit_config.get(f"sigma_min_{idx}", 0.001), fit_config.get(f"sigma_max_{idx}", 0.03)],
            fix=fit_config.get(f"sigma_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "alpha", fit_config.get(f"alpha_init_{idx}", 5.),
            limits=[fit_config.get(f"alpha_min_{idx}", 0.5), fit_config.get(f"alpha_max_{idx}", 10.)],
            fix=fit_config.get(f"alpha_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "n", fit_config.get(f"n_init_{idx}", 10.),
            limits=[fit_config.get(f"n_min_{idx}", 5.), fit_config.get(f"n_max_{idx}", 100.)],
            fix=fit_config.get(f"n_fix_{idx}", False)
        )
    elif fit_config["signal_func"][idx] == "genergausexptailsymm":
        fitter.set_signal_initpar(
            idx, "sigma", fit_config.get(f"sigma_init_{idx}", 0.01),
            limits=[fit_config.get(f"sigma_min_{idx}", 0.001), fit_config.get(f"sigma_max_{idx}", 0.03)],
            fix=fit_config.get(f"sigma_fix_{idx}", False)
        )
        fitter.set_signal_initpar(
            idx, "alpha", fit_config.get(f"alpha_init_{idx}", 3.),
            limits=[fit_config.get(f"alpha_min_{idx}", 0.), fit_config.get(f"alpha_max_{idx}", 10.)],
            fix=fit_config.get(f"alpha_fix_{idx}", False)
        )
    fitter.set_signal_initpar(idx, "frac", 0.1, limits=[0.0002, 1.])


def fix_signal_parameters(fitter, fit_config, idx):
    particle = "ds" if idx == 0 else "dplus"
    if fit_config["signal_func"][idx] == "gaussian":
        sigma_ds = get_sigma_from_cfg(cfg, fit_config, particle)
        fitter.set_signal_initpar(idx, "sigma", sigma_ds, fix=True)
    elif fit_config["signal_func"][idx] == "doublegaus":
        sigma_ds = get_sigma_from_cfg(cfg, fit_config, particle)
        fitter.set_signal_initpar(idx, "sigma1", sigma_ds[0], fix=True)
        fitter.set_signal_initpar(idx, "sigma2", sigma_ds[1], fix=True)
        fitter.set_signal_initpar(idx, "frac1", sigma_ds[2], fix=True)
    elif fit_config["signal_func"][idx] == "doublecb":
        sigma_ds = get_sigma_from_cfg(cfg, fit_config, particle)
        fitter.set_signal_initpar(idx, "sigma", sigma_ds, fix=True)
        fitter.set_signal_initpar(idx, "alphal", 1.5, limits=[1., 3.])
        fitter.set_signal_initpar(idx, "alphar", 1.5, limits=[1., 3.])
        fitter.set_signal_initpar(idx, "nl", 50, limits=[30., 100.])
        fitter.set_signal_initpar(idx, "nr", 50, limits=[30., 100.])

def do_fit(fit_config, cfg):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements # noqa: E501
    """Fit the invariant mass spectrum for a given configuration."""
    pt_min = fit_config["pt_min"]
    pt_max = fit_config["pt_max"]
    pt_cent_suffix = f"{pt_min*10:.0f}_{pt_max*10:.0f}"

    cent_min, cent_max = None, None
    if "cent_min" in fit_config and "cent_max" in fit_config:
        cent_min = fit_config["cent_min"]
        cent_max = fit_config["cent_max"]
        pt_cent_suffix += f"_cent_{cent_min:.0f}_{cent_max:.0f}"

        data_hdl = DataHandler(
            data=cfg["inputs"]["data"],
            histoname=f'h_mass_{pt_min*10:.0f}_{pt_max*10:.0f}_cent_{cent_min:.0f}_{cent_max:.0f}',
            limits=[fit_config["mass_min"], fit_config["mass_max"]], rebin=fit_config["rebin"]
        )
    else:
        data_hdl = DataHandler(
            data=cfg["inputs"]["data"],
            histoname=f'h_mass_{pt_min*10:.0f}_{pt_max*10:.0f}',
            limits=[fit_config["mass_min"], fit_config["mass_max"]], rebin=fit_config["rebin"]
        )

    bkg_funcs = fit_config["bkg_func"]
    label_signal_pdfs = [r"$\mathrm{D_{s}^{+}}$ signal", r"$\mathrm{D^{+}}$ signal"]
    label_bkg_pdfs = ["Combinatorial background"]
    if fit_config["use_bkg_templ"]:
        bkg_cfg = cfg["fit_configs"]["bkg"]["templ_norm"]["backgrounds"]
        data_corr_bkgs = []
        for i_bkg, bkg in enumerate(bkg_cfg):
            bkg_funcs.insert(i_bkg, "hist")
            label_bkg_pdfs.insert(
                i_bkg,
                bkg["name"] + "\ncorrelated background"
            )
            data_corr_bkgs.append(DataHandler(
                data=bkg["template_file"],
                histoname=bkg["template_hist_name"].format(
                    pt_min=f"{pt_min*10:.0f}", pt_max=f"{pt_max*10:.0f}",
                    cent_min=cent_min, cent_max=cent_max
                ),
                limits=[fit_config["mass_min"], fit_config["mass_max"]], rebin=fit_config["rebin"]
            ))

        fitter = F2MassFitter(
            data_hdl, name_signal_pdf=fit_config["signal_func"],
            name_background_pdf=bkg_funcs,
            name=f"ds_pt_{pt_cent_suffix}", chi2_loss=True,
            label_signal_pdf=label_signal_pdfs,
            label_bkg_pdf=label_bkg_pdfs,
            verbosity=1, tol=1.e-1
        )

        templ_norm_cfg = cfg["fit_configs"]["bkg"]["templ_norm"]
        if templ_norm_cfg["fix_with_br"][fit_config['i_pt']]:
            data_signal_reference = DataHandler(
                data=templ_norm_cfg["signal"]["file_norm"],
                histoname=templ_norm_cfg["signal"]["hist_name"].format(
                    pt_min=f"{pt_min*10:.0f}", pt_max=f"{pt_max*10:.0f}",
                    cent_min=cent_min, cent_max=cent_max
                ),
                limits=[fit_config["mass_min"], fit_config["mass_max"]], rebin=fit_config["rebin"]
            )

        for i_func, data_corr_bkg in enumerate(data_corr_bkgs):
            fitter.set_background_template(i_func, data_corr_bkg)
            fitter.set_background_initpar(i_func, "frac", 0.01, limits=[0., 1.])
            if templ_norm_cfg["fix_with_br"][fit_config['i_pt']]:
                data_corr_bkg_reference = DataHandler(
                    data=bkg_cfg[i_func]["file_norm"],
                    histoname=bkg_cfg[i_func]["norm_hist_name"].format(
                        pt_min=f"{pt_min*10:.0f}", pt_max=f"{pt_max*10:.0f}",
                        cent_min=cent_min, cent_max=cent_max
                    ),
                    limits=[fit_config["mass_min"], fit_config["mass_max"]], rebin=fit_config["rebin"]
                )
                fitter.fix_bkg_frac_to_signal_pdf(
                    i_func, 1, # correlated bkg to D+ signal
                    data_corr_bkg_reference.get_norm() *\
                        bkg_cfg[i_func]["br"]["pdg"] / bkg_cfg[i_func]["br"]["simulations"]
                    / (
                        data_signal_reference.get_norm() *\
                            templ_norm_cfg["signal"]["br"]["pdg"] /\
                                templ_norm_cfg["signal"]["br"]["simulations"]
                    )
                )
            elif f"corr_bkg_frac_over_dplus_{i_func}_cfg" in fit_config:
                fitter.fix_bkg_frac_to_signal_pdf(
                    i_func, 1, # correlated bkg to D+ signal
                    fit_config[f"corr_bkg_frac_over_dplus_{i_func}_cfg"]
                )
    else:
        fitter = F2MassFitter(
            data_hdl, name_signal_pdf=fit_config["signal_func"],
            name_background_pdf=fit_config["bkg_func"],
            name=f"ds_pt_{pt_cent_suffix}", chi2_loss=True,
            label_signal_pdf=label_signal_pdfs,
            label_bkg_pdf=label_bkg_pdfs,
            verbosity=1, tol=1.e-1
        )

    # signals initialisation
    fitter.set_particle_mass(0, pdg_id=431)
    initialise_signal(fitter, fit_config, 0)

    if len(fit_config["signal_func"]) > 1:
        fitter.set_particle_mass(1, pdg_id=411)
        initialise_signal(fitter, fit_config, 1)

    # bkg initialisation
    for i_func, bkg_func in enumerate(bkg_funcs):
        if bkg_func == "expo":
            fitter.set_background_initpar(i_func, "lam", -2)
        elif bkg_func == "chebpol2":
            fitter.set_background_initpar(i_func, "c0", 0.6)
            fitter.set_background_initpar(i_func, "c1", -0.2)
            fitter.set_background_initpar(i_func, "c2", 0.01)
        elif bkg_func == "chebpol3":
            fitter.set_background_initpar(i_func, "c0", 0.4)
            fitter.set_background_initpar(i_func, "c1", -0.2)
            fitter.set_background_initpar(i_func, "c2", -0.01)
            fitter.set_background_initpar(i_func, "c3", 0.01)

    if fit_config["fix_ds_sigma"]:
        if fit_config["signal_func"][0] == "gaussian":
            sigma_ds = get_sigma_from_cfg(cfg, fit_config, "ds")
            fitter.set_signal_initpar(0, "sigma", sigma_ds, fix=True)
        elif fit_config["signal_func"][0] == "doublegaus":
            sigma_ds = get_sigma_from_cfg(cfg, fit_config, "ds")
            fitter.set_signal_initpar(0, "sigma1", sigma_ds[0], fix=True)
            fitter.set_signal_initpar(0, "sigma2", sigma_ds[1], fix=True)
            fitter.set_signal_initpar(0, "frac1", sigma_ds[2], fix=True)
        elif fit_config["signal_func"][0] == "doublecbsymm":
            sigma_ds = get_sigma_from_cfg(cfg, fit_config, "ds")
            fitter.set_signal_initpar(0, "sigma", sigma_ds[0], fix=True)
            fitter.set_signal_initpar(0, "alpha", sigma_ds[1], fix=True)
            fitter.set_signal_initpar(0, "n", sigma_ds[2], fix=True)
    if fit_config["fix_dplus_sigma"]:
        if fit_config["signal_func"][1] == "gaussian":
            sigma_dplus = get_sigma_from_cfg(cfg, fit_config, "dplus")
            fitter.set_signal_initpar(1, "sigma", sigma_dplus, fix=True)
        elif fit_config["signal_func"][1] == "doublegaus":
            sigma_dplus = get_sigma_from_cfg(cfg, fit_config, "dplus")
            fitter.set_signal_initpar(1, "sigma1", sigma_dplus[0], fix=True)
            fitter.set_signal_initpar(1, "sigma2", sigma_dplus[1], fix=True)
            fitter.set_signal_initpar(1, "frac1", sigma_dplus[2], fix=True)
        elif fit_config["signal_func"][1] == "doublecbsymm":
            sigma_dplus = get_sigma_from_cfg(cfg, fit_config, "dplus")
            fitter.set_signal_initpar(1, "sigma", sigma_dplus[0], fix=True)
            fitter.set_signal_initpar(1, "alpha", sigma_dplus[1], fix=True)
            fitter.set_signal_initpar(1, "n", sigma_dplus[2], fix=True)
    if fit_config["fix_sigma_dplus_to_ds"]:
        sigma_dplus_ratio = get_sigma_dplus_to_ds(cfg, fit_config, fitter)
        fitter.set_signal_initpar(1, "sigma", sigma_dplus_ratio, fix=True)

    n_signal = len(fit_config["signal_func"])
    fit_result = fitter.mass_zfit()
    #if fit_result.converged:
    if cfg["outputs"]["save_all_fits"]:
        output_dir = os.path.join(
            os.path.expanduser(cfg["outputs"]["directory"]),
            "fits"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        loc = ["lower left", "upper left"]
        ax_title = r"$M(\mathrm{KK\pi})$ GeV$/c^2$"
        fig, _ = fitter.plot_mass_fit(
            style="ATLAS",
            show_extra_info = (bkg_funcs != ["nobkg"]),
            figsize=(8, 8), extra_info_loc=loc,
            legend_loc="upper right",
            axis_title=ax_title
        )
        figres = fitter.plot_raw_residuals(
            figsize=(8, 8), style="ATLAS",
            extra_info_loc=loc, axis_title=ax_title
        )
        for frmt in cfg["outputs"]["formats"]:
            if cent_min is not None and cent_max is not None:
                suffix = f"_{pt_min * 10:.0f}_{pt_max * 10:.0f}_cent_{cent_min:.0f}_{cent_max:.0f}_"  # pylint: disable=line-too-long # noqa: E501
            else:
                suffix = f"_{pt_min * 10:.0f}_{pt_max * 10:.0f}_"
            suffix += cfg["outputs"]["suffix"]
            if frmt == "root":
                fitter.dump_to_root(
                    f"{output_dir}/fits_{cfg['outputs']['suffix']}.{frmt}", 
                    option="update", suffix=suffix, num=5000
                )
            else:
                fig.savefig(f"{output_dir}/ds_mass_pt{suffix}.{frmt}")
                figres.savefig(f"{output_dir}/ds_massres_pt{suffix}.{frmt}")
                plt.close(fig)
                plt.close(figres)

        fracs = fitter._F2MassFitter__get_all_fracs() # pylint: disable=protected-access
        corr_bkg_frac_dict = {}
        corr_bkg_frac_over_dplus_dict = {}
        if len(fracs[1]) > 0:
            for i_corr_bkg, (corr_bkg_frac, corr_bkg_err) in enumerate(zip(fracs[1][:-1], fracs[4][:-1])):
                corr_bkg_frac_dict[f"corr_bkg_frac_{i_corr_bkg}"] = [corr_bkg_frac, corr_bkg_err]
                corr_bkg_frac_over_dplus_dict[f"corr_bkg_frac_over_dplus_{i_corr_bkg}"] = [corr_bkg_frac / fracs[0][1], corr_bkg_frac / fracs[0][1] * np.sqrt((corr_bkg_err / corr_bkg_frac)**2 + (fracs[3][1] / fracs[0][1])**2)]
        else:
            corr_bkg_frac_dict["corr_bkg_frac_0"] = [0, 0]
            corr_bkg_frac_over_dplus_dict["corr_bkg_frac_over_dplus_0"] = [0, 0]
        out_dict = {
            "raw_yields": [fitter.get_raw_yield(i) for i in range(n_signal)],
            "mean": [fitter.get_mass(i) for i in range(n_signal)],
            "chi2": float(fitter.get_chi2_ndf()),
            "significance": [fitter.get_significance(i, min=1.8, max=2.2) for i in range(n_signal)],
            "signal": [fitter.get_signal(i, min=1.8, max=2.) for i in range(n_signal)],
            "background": [fitter.get_background(i, min=1.8, max=2.) for i in range(n_signal)],
            **corr_bkg_frac_dict,
            "fracs": fracs,
            "converged": fit_result.converged, 
            **corr_bkg_frac_over_dplus_dict
        }

        if fit_config["signal_func"][0] == "doublegaus": #TODO: generalise in case different signal functions are used
            out_dict["sigma1"] = [fitter.get_signal_parameter(i, "sigma1") for i in range(n_signal)]
            out_dict["sigma2"] = [fitter.get_signal_parameter(i, "sigma2") for i in range(n_signal)]
            out_dict["frac1"] = [fitter.get_signal_parameter(i, "frac1") for i in range(n_signal)]
        elif fit_config["signal_func"][0] == "gaussian":
            out_dict["sigma"] = [fitter.get_sigma(i) for i in range(n_signal)]
        elif fit_config["signal_func"][0] == "doublecb":
            out_dict["sigma"] = [fitter.get_signal_parameter(i, "sigma") for i in range(n_signal)]
            out_dict["alphar"] = [fitter.get_signal_parameter(i, "alphar") for i in range(n_signal)]
            out_dict["alphal"] = [fitter.get_signal_parameter(i, "alphal") for i in range(n_signal)]
            out_dict["nl"] = [fitter.get_signal_parameter(i, "nl") for i in range(n_signal)]
            out_dict["nr"] = [fitter.get_signal_parameter(i, "nr") for i in range(n_signal)]
        elif fit_config["signal_func"][0] == "doublecbsymm":
            out_dict["sigma"] = [fitter.get_signal_parameter(i, "sigma") for i in range(n_signal)]
            out_dict["alpha"] = [fitter.get_signal_parameter(i, "alpha") for i in range(n_signal)]
            out_dict["n"] = [fitter.get_signal_parameter(i, "n") for i in range(n_signal)]
        elif fit_config["signal_func"][0] == "genergausexptailsymm":
            out_dict["sigma"] = [fitter.get_signal_parameter(i, "sigma") for i in range(n_signal)]
            out_dict["alpha"] = [fitter.get_signal_parameter(i, "alpha") for i in range(n_signal)]
    else:
        out_dict = {
            "raw_yields": [None] * n_signal,
            "sigma": [None] * n_signal,
            "mean": [None] * n_signal,
            "chi2": None,
            "significance": [None] * n_signal,
            "signal": [None] * n_signal,
            "background": [None] * n_signal,
            "fracs": None,
            "converged": False
        }
    return out_dict


def fit(config_file_name):
    """
    Perform fitting procedure based on the provided configuration file.

    Parameters:
        config_file_name (str): Path to the YAML configuration file.
    """
    # load config
    with open(config_file_name, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # load cut set
    with open(cfg["inputs"]["cutset"], "r", encoding="utf-8") as f:
        cut_set = yaml.safe_load(f)

    zfit.run.set_cpus_explicit(
        intra=cfg['zfit_cpus']['intra'],
        inter=cfg['zfit_cpus']['inter']
    )

    # load inputs
    _, h_ev = load_inputs(cfg, cut_set)
    n_ev = sum(h_ev.values())

    cent_mins, cent_maxs = None, None
    if "cent" in cut_set:
        cent_mins = cut_set["cent"]["min"]
        cent_maxs = cut_set["cent"]["max"]

    # create output handler
    h_handler = HistHandler(
        cut_set["pt"]["min"], cut_set["pt"]["max"],
        cent_mins, cent_maxs
    )
    h_handler.set_n_ev(n_ev)

    fit_configs = create_fit_configs(cfg, cut_set)

    # recreate root file if needed
    if "root" in cfg["outputs"]["formats"]:
        output_dir = os.path.join(
            os.path.expanduser(cfg["outputs"]["directory"]),
            "fits"
        )
        uproot.recreate(f"{output_dir}/fits_{cfg['outputs']['suffix']}.root")

    bkg_cfg = cfg["fit_configs"]["bkg"]
    if (any(bkg_cfg["use_bkg_templ"]) and any(bkg_cfg["templ_norm"]["fix_to_mb"])) or\
            any(cfg["fit_configs"]["signal"]["fix_sigma_to_mb"]):
        results_fracs = []
        with ProcessPoolExecutor(max_workers=cfg["max_workers"]) as executor:
            for fit_config in fit_configs:
                if "cent" in cut_set and fit_config["cent_min"] == 0 and fit_config["cent_max"] == 100:
                    results_fracs.append((executor.submit(do_fit, fit_config, cfg), fit_config))
        for result, fit_cfg_result in results_fracs:
            res = result.result()
            sig_fracs, bkg_fracs, _, _, _, _ = res["fracs"]
            for fit_config in fit_configs:
                if fit_config["pt_min"] == fit_cfg_result["pt_min"] and\
                        fit_config["pt_max"] == fit_cfg_result["pt_max"] and\
                            not (fit_config["cent_max"] == 100 and\
                                fit_config["cent_min"] == 0) and\
                                    bkg_cfg["templ_norm"]["fix_to_mb"][fit_config["i_pt"]]:
                    for i_corr_bkg, corr_bkg_frac in enumerate(bkg_fracs[:-1]):
                        fit_config[f"corr_bkg_frac_over_dplus_{i_corr_bkg}_cfg"] = corr_bkg_frac / sig_fracs[1]
                if fit_config["pt_min"] == fit_cfg_result["pt_min"] and\
                        fit_config["pt_max"] == fit_cfg_result["pt_max"] and\
                            not (fit_config["cent_max"] == 100 and\
                                fit_config["cent_min"] == 0) and\
                                    cfg["fit_configs"]["signal"]["fix_sigma_to_mb"][fit_config["i_pt"]]:
                    if fit_config["signal_func"][0] == 'gaussian':
                        fit_config["sigma_fix_0"] = True
                        fit_config["sigma_init_0"] = res["sigma"][0][0]
                        fit_config["sigma_min_0"] = res["sigma"][0][0] - 0.001
                        fit_config["sigma_max_0"] = res["sigma"][0][0] + 0.001
                    elif fit_config["signal_func"][0] == 'doublegaus':
                        fit_config["sigma1_fix_0"] = True
                        fit_config["sigma1_init_0"] = res["sigma1"][0][0]
                        fit_config["sigma1_min_0"] = res["sigma1"][0][0] - 0.001
                        fit_config["sigma1_max_0"] = res["sigma1"][0][0] + 0.001
                        fit_config["sigma2_fix_0"] = True
                        fit_config["sigma2_init_0"] = res["sigma2"][0][0]
                        fit_config["sigma2_min_0"] = res["sigma2"][0][0] - 0.001
                        fit_config["sigma2_max_0"] = res["sigma2"][0][0] + 0.001
                        fit_config["frac1_fix_0"] = True
                        fit_config["frac1_init_0"] = res["frac1"][0][0]
                        fit_config["frac1_min_0"] = res["frac1"][0][0] - 0.001
                        fit_config["frac1_max_0"] = res["frac1"][0][0] + 0.001
                    elif fit_config["signal_func"][0] == 'doublecbsymm':
                        fit_config["sigma_fix_0"] = True
                        fit_config["sigma_init_0"] = res["sigma"][0][0]
                        fit_config["sigma_min_0"] = res["sigma"][0][0] - 0.001
                        fit_config["sigma_max_0"] = res["sigma"][0][0] + 0.001
                        fit_config["alpha_fix_0"] = True
                        fit_config["alpha_init_0"] = res["alpha"][0][0]
                        fit_config["alpha_min_0"] = res["alpha"][0][0] - 0.001
                        fit_config["alpha_max_0"] = res["alpha"][0][0] + 0.001
                        fit_config["n_fix_0"] = True
                        fit_config["n_init_0"] = res["n"][0][0]
                        fit_config["n_min_0"] = res["n"][0][0] - 0.001
                        fit_config["n_max_0"] = res["n"][0][0] + 0.001
                    if fit_config["signal_func"][1] == 'gaussian':    
                        fit_config["sigma_fix_1"] = True
                        fit_config["sigma_init_1"] = res["sigma"][1][0]
                        fit_config["sigma_min_1"] = res["sigma"][1][0] - 0.001
                        fit_config["sigma_max_1"] = res["sigma"][1][0] + 0.001
                    elif fit_config["signal_func"][1] == 'doublegaus':
                        fit_config["sigma1_fix_1"] = True
                        fit_config["sigma1_init_1"] = res["sigma1"][1][0]
                        fit_config["sigma1_min_1"] = res["sigma1"][1][0] - 0.001
                        fit_config["sigma1_max_1"] = res["sigma1"][1][0] + 0.001
                        fit_config["sigma2_fix_1"] = True
                        fit_config["sigma2_init_1"] = res["sigma2"][1][0]
                        fit_config["sigma2_min_1"] = res["sigma2"][1][0] - 0.001
                        fit_config["sigma2_max_1"] = res["sigma2"][1][0] + 0.001
                        fit_config["frac1_fix_1"] = True
                        fit_config["frac1_init_1"] = res["frac1"][1][0]
                        fit_config["frac1_min_1"] = res["frac1"][1][0] - 0.001
                        fit_config["frac1_max_1"] = res["frac1"][1][0] + 0.001
                    elif fit_config["signal_func"][1] == 'doublecbsymm':
                        fit_config["sigma_fix_1"] = True
                        fit_config["sigma_init_1"] = res["sigma"][1][0]
                        fit_config["sigma_min_1"] = res["sigma"][1][0] - 0.001
                        fit_config["sigma_max_1"] = res["sigma"][1][0] + 0.001
                        fit_config["alpha_fix_1"] = True
                        fit_config["alpha_init_1"] = res["alpha"][1][0]
                        fit_config["alpha_min_1"] = res["alpha"][1][0] - 0.001
                        fit_config["alpha_max_1"] = res["alpha"][1][0] + 0.001
                        fit_config["n_fix_1"] = True
                        fit_config["n_init_1"] = res["n"][1][0]
                        fit_config["n_min_1"] = res["n"][1][0] - 0.001
                        fit_config["n_max_1"] = res["n"][1][0] + 0.001

    if bkg_cfg["templ_norm"]["fix_to_file_name"] is not None:
        with uproot.open(bkg_cfg["templ_norm"]["fix_to_file_name"]) as f:
            h_frac1 = f[bkg_cfg["templ_norm"]["hist_name"].format(
                #pt_min=f"{fit_cfg_result['pt_min']*10:.0f}", pt_max=f"{fit_cfg_result['pt_max']*10:.0f}",
                cent_min=0, cent_max=100 # TODO: generalise
            )]
            for fit_config in fit_configs:
                fit_config["corr_bkg_frac_over_dplus_0"] = h_frac1.values()[fit_config["i_pt"]]


    results = []
    with ProcessPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        for fit_config in fit_configs:
            results.append((executor.submit(do_fit, fit_config, cfg), fit_config))

    output_df = []
    for result, fit_config in results:
        out_dict = result.result()
        out_dict.update(fit_config)
        output_df.append(out_dict)
    output_df = pd.DataFrame(output_df)
    output_df.to_parquet(os.path.join(
        os.path.expanduser(cfg["outputs"]["directory"]),
        f'fit_results{cfg["outputs"]["suffix"]}.parquet'
    ), index=False)

    h_handler.set_histos(output_df)

    h_handler.dump_to_root(os.path.join(
        os.path.expanduser(cfg["outputs"]["directory"]),
        f'mass_fits{cfg["outputs"]["suffix"]}.root'
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('config_file_name', metavar='text', default='')
    args = parser.parse_args()

    fit(args.config_file_name)
