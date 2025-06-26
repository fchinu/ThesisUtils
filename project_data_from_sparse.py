"""
This script projects sparse data from a ROOT file based on the provided configuration.

Run:
    python project_data_from_sparse.py config.yaml
"""

import os
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import yaml
import ROOT
# pylint: disable=no-member


def load_sparse_from_task(config_input):
    """
    Load sparse data from a ROOT file based on the provided configuration.

    Args:
    - config_input (dict): Configuration dictionary containing input information.

    Returns:
    - sparses (list=: A list of THnSparse objects loaded from the ROOT file.
    """
    input_files_names = config_input["names"]
    if not isinstance(input_files_names, list):
        input_files_names = [input_files_names]
    sparse_names = config_input["sparses"]
    if not isinstance(sparse_names, list):
        sparse_names = [sparse_names]
    if len(sparse_names) != len(input_files_names):
        if len(sparse_names) == 1:
            sparse_names = sparse_names * len(input_files_names)
        elif len(input_files_names) == 1:
            input_files_names = input_files_names * len(sparse_names)
        else:
            raise ValueError("Number of input files and sparse names do not match.")
    
    sparses = []
    for input_file_name, sparse_name in zip(input_files_names, sparse_names):
        with ROOT.TFile.Open(input_file_name, "READ") as input_file:
            sparses.append(input_file.Get(sparse_name))
            print(input_file_name, sparses[-1])

    return sparses


def load_event_histogram_from_task(config_input):
    """
    Load event histogram from a ROOT file based on the provided configuration.

    Args:
    - config_input (dict): Configuration dictionary containing input information.

    Returns:
    - h_ev (TH1F): Event histogram loaded from the ROOT file.
    """
    input_files_names = config_input["names"]
    if not isinstance(input_files_names, list):
        input_files_names = [input_files_names]
    sparse_names = config_input["sparses"]
    if not isinstance(sparse_names, list):
        sparse_names = [sparse_names]
    h_evs = []
    for i_file, file_name in enumerate(input_files_names):
        with ROOT.TFile.Open(file_name, "READ") as input_file:
            h_evs.append(input_file.Get(config_input["event_histogram"]))
            h_evs[-1].SetDirectory(0)
    if len(input_files_names) == 1 and len(sparse_names) > 1:
        h_evs = h_evs * len(sparse_names)
    return h_evs


def get_cuts(cuts_config):
    """
    Generate a list of cuts based on the provided configuration.

    Parameters:
        - cuts_config (dict): A dictionary containing the cutset.

    Returns:
        - cuts (list): A list of cutsets, where each cutset is represented as a
            tuple of dictionaries.
            Each dictionary contains 'min', 'max', and 'axis' keys.
            If centrality selections are provided, they are included in the cuts.
    """
    if "cent" in cuts_config:
        centrality_sels = cuts_config.pop("cent")
        centrality_sels = [{
            'varname': 'cent',
            'axis': centrality_sels["axisnum"],
            'min': min_cent,
            'max': max_cent
        } for min_cent, max_cent in zip(centrality_sels["min"], centrality_sels["max"])]
    else:
        centrality_sels = None

    # Extract variable names and cuts from the config
    var_names = list(cuts_config.keys())
    var_names.remove('mass')

    cuts_list = [cuts_config[var] for var in var_names]
    axes = [cut['axisnum'] for cut in cuts_list]
    mins_list = [cut['min'] for cut in cuts_list]
    maxs_list = [cut['max'] for cut in cuts_list]

    cuts = []
    for (mins_var, maxs_var), axis, var_name in zip(zip(mins_list, maxs_list), axes, var_names):
        cuts.append([
            {'varname': var_name, 'min': min_var, 'max': max_var, 'axis': axis}
            for min_var, max_var in zip(mins_var, maxs_var)
        ])
    cuts = [*zip(*cuts)]

    if centrality_sels:
        cuts = [
            (*cut, centrality_sel)
            for cut, centrality_sel in product(cuts, centrality_sels)
        ]

    return cuts


def project_sparse_worker(sparse, cut):
    """
    Projects sparse data based on given cuts and returns histograms for mass and pt.

    Parameters:
        sparse (ROOT.THnSparse): The sparse data to be projected.
        cut (list of dict): List of dictionaries specifying the cuts. Each dictionary should have:
            - "varname" (str): The variable name (e.g., "Pt", "Cent").
            - "axis" (int): The axis index to apply the cut on.
            - "min" (float): The minimum value for the cut.
            - "max" (float): The maximum value for the cut.

    Returns:
        tuple: A tuple containing:
            - h_mass (ROOT.TH1): The projected mass histogram.
            - h_pt (ROOT.TH1): The projected pt histogram.
    """
    pt_min, pt_max = None, None
    cent_min, cent_max = None, None

    for cut_var in cut:
        sparse.GetAxis(cut_var["axis"]).SetRangeUser(cut_var["min"], cut_var["max"])
        if cut_var["varname"] == "pt":
            pt_min, pt_max = cut_var["min"], cut_var["max"]
        if cut_var["varname"] == "cent":
            cent_min, cent_max = cut_var["min"], cut_var["max"]

    # Project the sparse data
    h_mass = sparse.Projection(0)
    h_mass.SetDirectory(0)
    h_pt = sparse.Projection(1)
    h_pt.SetDirectory(0)

    if cent_min is not None and cent_max is not None:
        suffix = f'_{pt_min * 10:.0f}_{pt_max * 10:.0f}_cent_{cent_min:.0f}_{cent_max:.0f}'
    else:
        suffix = f'_{pt_min * 10:.0f}_{pt_max * 10:.0f}'
    h_mass.SetName(f'h_mass{suffix}')
    h_pt.SetName(f'h_pt{suffix}')

    return h_mass, h_pt


def project_sparse(config_file_name):  # pylint: disable=too-many-locals
    """
    Project THnSparse data based on the configuration provided in the YAML file.

    Parameters:
        config_file_name (str): Path to the configuration YAML file.
    """
    with open(config_file_name, 'r', encoding="utf8") as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    with open(config["cut_set_file_name"], 'r', encoding="utf8") as cuts_config_file:
        cuts_config = yaml.load(cuts_config_file, yaml.FullLoader)

    sparses = load_sparse_from_task(config["inputs"])
    h_evs = load_event_histogram_from_task(config["inputs"])

    cuts = get_cuts(cuts_config)
    output_labels = config["output"]["file_names"]
    if not isinstance(output_labels, list):
        output_labels = [output_labels]

    for sparse, h_ev, output_label in zip(sparses, h_evs, output_labels):
        results = []
        with ProcessPoolExecutor(max_workers=min(len(cuts), 64)) as executor:
            for cut in cuts:
                results.append(executor.submit(project_sparse_worker, sparse, cut))
        out_file_name = os.path.join(config["output"]["directory"], output_label)
        if not os.path.exists(config["output"]["directory"]):
            os.makedirs(config["output"]["directory"])
        with ROOT.TFile(out_file_name, "RECREATE") as _:
            for result in results:
                h_mass, h_pt = result.result()
                h_mass.Write()
                h_pt.Write()
            h_ev.Write("h_ev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('config_file_name', metavar='text', default='config.yaml',
                        help='configuration file name')
    args = parser.parse_args()

    project_sparse(args.config_file_name)
