'''
Script to evaluate the efficiency starting from the output THnSparse.
'''
from math import sqrt
import argparse
import itertools
import os
import numpy as np
import yaml
import ROOT
# pylint: disable=no-member
import sys # pylint: disable=wrong-import-order
sys.path.append("/home/fchinu/Run3/ThesisUtils")
from PlotUtils import get_discrete_matplotlib_palette # pylint: disable=wrong-import-position

PARTICLE_CLASSES = [("Ds", "Prompt"), ("Ds", "NonPrompt"),
    ("Dplus", "Prompt"), ("Dplus", "NonPrompt")]
ROOT.TH1.AddDirectory(False)
ROOT.TH2.AddDirectory(False)

def get_weights(cfg, cuts_cfg):
    """
    Get the weights from the input file.

    Parameters:
    - cfg (dict): Dictionary containing the configuration.

    Returns:
    - dict: Dictionary containing the weights.
    """
    
    histos_weights = {}
    particles = ["Ds", "Dplus"]
    origins = ["Prompt", "NonPrompt"]
    particle_origin = itertools.product(particles, origins)

    centrality_selections = cuts_cfg.copy().pop('cent')
    cent_mins = centrality_selections['min']
    cent_maxs = centrality_selections['max']

    with ROOT.TFile.Open(cfg["weights"]["file_name"]) as infile_weights:
        for particle, origin in particle_origin:
            for i_cent, (cent_min, cent_max) in enumerate(zip(cent_mins, cent_maxs)):
                print(cent_min, cent_max)
                histos_weights[f"{particle}{origin}_Cent_{cent_min}_{cent_max}"] = infile_weights.Get(
                    f"centrality_{cent_min}_{cent_max}/{particle}_{origin}/hNPvContribCands_weights_{particle}_{origin}")
                histos_weights[f"{particle}{origin}_Cent_{cent_min}_{cent_max}"].SetName(
                    f"Weight{particle}{origin}Cands_{cent_min}_{cent_max}")
                histos_weights[f"{particle}{origin}_Cent_{cent_min}_{cent_max}"].SetTitle(
                    f"Weight{particle}{origin}Cands_{cent_min}_{cent_max}")
    return histos_weights

def __create_histos(cfg, cuts_cfg):
    """
    Create histograms and canvases for efficiency evaluation.

    Parameters:
    - cfg (dict): Configuration dictionary containing various settings.
    - cuts_cfg (dict): Configuration dictionary containing cut variables.
    Returns:
    tuple: A tuple containing the following elements:
        - histos (list): 
            List of ROOT.TH1F histograms for efficiency.
        - histos_fine_bins (list): 
            List of ROOT.TH1F histograms for efficiency in fine bins.
        - histos_cent (list): 
            List of ROOT.TH1F histograms for efficiency with centrality selections.
        - histos_cent_fine_bins (list): 
            List of ROOT.TH1F histograms for efficiency in fine bins with centrality selections.
        - canvas_cent (list): 
            List of ROOT.TCanvas objects for centrality selections.
    """
    histos = []
    histos_fine_bins = []
    histos_cent = []
    histos_cent_fine_bins = []
    canvas_cent = []    

    particles = ["Ds", "Dplus"]
    origins = ["Prompt", "NonPrompt"]
    particle_origin = itertools.product(particles, origins)

    centrality_selections = cuts_cfg.copy().pop('cent')
    cent_mins = centrality_selections['min']
    cent_maxs = centrality_selections['max']

    pt_mins = cuts_cfg['pt']['min']
    pt_maxs = cuts_cfg['pt']['max']
    pt_edges = pt_mins + [pt_maxs[-1]]

    for particle, origin in particle_origin:

        histos.append(ROOT.TH1F(f"eff_{particle}{origin}",
            "Efficiency; #it{p}_{T} (GeV/c); Efficiency",
            len(pt_edges)-1, np.asarray(pt_edges, "d")))
        histos_fine_bins.append(ROOT.TH1F(f"eff_fine_bins_{particle}{origin}",
            "Efficiency in fine bins; #it{p}_{T} (GeV/c); Efficiency", 240, 0, 24))

        if cfg["weights"]["apply"]:

            for i_cent, (cent_min, cent_max) in enumerate(zip(cent_mins, cent_maxs)):
                histos_cent.append(ROOT.TH1F(f"eff_{particle}{origin}_cent_{cent_min}_{cent_max}",
                    f"Efficiency_Cent_{cent_min}_{cent_max}; #it{{p}}_{{T}} (GeV/c); Efficiency",
                    len(pt_edges)-1, np.asarray(pt_edges, "d")))
                histos_cent_fine_bins.append(ROOT.TH1F(
                    f"eff_fine_bins_{particle}{origin}_cent_{cent_min}_{cent_max}",
                    f"Efficiency in fine bins_Cent_{cent_min}_{cent_max};\
                    #it{{p}}_{{T}} (GeV/c); Efficiency", 240, 0, 24))

            canvas_cent.append(ROOT.TCanvas(
                f"canvas_cent_{particle}{origin}", f"canvas_cent_{particle}{origin}", 800, 600))
    
    return histos, histos_fine_bins, histos_cent, histos_cent_fine_bins, canvas_cent

def calculate_efficiencies_with_unc(sel: float, gen: float) -> tuple:
    """
    Calculate efficiencies with uncertainty.

    Parameters:
    - sel (float): The number of selected events.
    - gen (float): The number of generated events.

    Returns:
    - eff (float): The efficiency.
    - unc (float): The uncertainty of the efficiency.
    """
    eff = sel / gen # pylint: disable=redefined-outer-name
    return eff, sqrt(eff * (1 - eff) / gen)

def get_integral_of_projection(hist_2d, axis, minimum, maximum):
    """
    Get the integral of a 2D histogram projection along an axis.

    Parameters:
    - hist_2d (TH2F): The 2D histogram.
    - axis (str): "x" or "y".
    - minimum (float): The minimum value of the range.
    - maximum (float): The maximum value of the range.

    Returns:
    - integral (float): The integral of the projection.
    """
    if axis == "x":
        hist_1d = hist_2d.ProjectionX("projection", 0, -1, "EO")
    elif axis == "y":
        hist_1d = hist_2d.ProjectionY("projection", 0, -1, "EO")
    else:
        raise ValueError("Invalid axis. Choose 'x' or 'y'.")

    # Right edge of the bin is not included
    return hist_1d.Integral(hist_1d.FindBin(minimum), hist_1d.FindBin(maximum-0.001))

def get_eff(cfg, h_sparse_gen_particles, h_sparse_reco_ds, particle_class, cuts, # pylint: disable=too-many-arguments, disable=too-many-positional-arguments, redefined-outer-name, too-many-locals
    axes, pt_info, histos_weights = None, cent_info = (None, None)): # pylint: disable=redefined-outer-name
    """
    Calculate the efficiency of particle selection.

    Parameters:
    - h_sparse_gen_particles (THnSparse): THnSparse of generated particles.
    - h_sparse_reco_ds (THnSparse): THnSparse of reconstructed particles.
    - particle_class (tuple): Tuple containing the particle and its origin.
    - cuts (list): List of cuts for each axis.
    - axes (list): List of axis names.
    - pt_info (tuple): Tuple containing the index, minimum and maximum values of pT.
    - histos_weights (dict, optional): Dictionary of histograms used for weighting.
    - cent_info (tuple, optional): Tuple containing the minimum and maximum values of centrality.

    Returns:
    - float: Efficiency of particle selection.
    """
    particle, origin = particle_class # pylint: disable=redefined-outer-name
    i_pt, pt_min, pt_max = pt_info
    cent_min, cent_max = cent_info

    if histos_weights is not None:
        h_sparse_gen_sel = h_sparse_gen_particles.Clone(
            f"hSparseGenParticlesSel_{particle}{origin}_Cent_{cent_min}_{cent_max}_{i_pt}")
        h_sparse_reco_ds_sel = h_sparse_reco_ds.Clone(
            f"hSparseRecoDsSel_{particle}{origin}_Cent_{cent_min}_{cent_max}_{i_pt}")
    else:
        h_sparse_gen_sel = h_sparse_gen_particles.Clone(
            f"hSparseGenSel_{particle}{origin}")
        h_sparse_reco_ds_sel = h_sparse_reco_ds.Clone(
            f"hSparseRecoDsSel_{particle}{origin}")

    for i, axis in enumerate(axes):
        min_val = cuts[i][i_pt][0]
        max_val = cuts[i][i_pt][1]
        h_sparse_reco_ds_sel.GetAxis(axis).SetRangeUser(min_val, max_val)


    # WARNING: just like TH3::Project3D("yx") and TTree::Draw("y:x"), Projection(y,x)
    # uses the first argument to define the y-axis and the second for the x-axis!
    h_pt_y_projected_gen_particles = h_sparse_gen_sel.Projection(
        cfg['inputs']['sparse']['axis_npv_gen'], cfg['inputs']['sparse']['axis_pt_gen'], "EO")
    h_pt_y_projected_reco_ds = h_sparse_reco_ds_sel.Projection(
        cfg['inputs']['sparse']['axis_npv_reco'], cfg['inputs']['sparse']['axis_pt_reco'], "EO")

    if histos_weights is not None:
        for i in range(h_pt_y_projected_gen_particles.GetNbinsX()): # Loop on pT
            for j in range(h_pt_y_projected_gen_particles.GetNbinsY()): # Loop on NPV
                content = h_pt_y_projected_gen_particles.GetBinContent(i+1, j+1)
                content *= histos_weights[
                    f"{particle}{origin}_Cent_{cent_min}_{cent_max}"].GetBinContent(j+1)
                h_pt_y_projected_gen_particles.SetBinContent(i+1, j+1, content)

        for i in range(h_pt_y_projected_reco_ds.GetNbinsX()): # Loop on pT
            for j in range(h_pt_y_projected_reco_ds.GetNbinsY()): # Loop on NPV
                content = h_pt_y_projected_reco_ds.GetBinContent(i+1, j+1)
                content *= histos_weights[
                    f"{particle}{origin}_Cent_{cent_min}_{cent_max}"].GetBinContent(j+1)
                h_pt_y_projected_reco_ds.SetBinContent(i+1, j+1, content)

    gen_particles = get_integral_of_projection(
        h_pt_y_projected_gen_particles, "x", pt_min, pt_max)
    selected_particles = get_integral_of_projection(
        h_pt_y_projected_reco_ds, "x", pt_min, pt_max)

    return calculate_efficiencies_with_unc(selected_particles, gen_particles)

def main(config_file_name):
    # Load cuts configuration
    with open(config_file_name, 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(cfg['inputs']['cuts_file_name'], 'r', encoding="utf-8") as f:
        cuts_cfg = yaml.safe_load(f)

    particles = ["Ds", "Dplus"]
    origins = ["Prompt", "NonPrompt"]
    particle_origin = itertools.product(particles, origins)

    centrality_selections = None
    cent_mins = None
    cent_maxs = None
    if "cent" in cuts_cfg:
        centrality_selections = cuts_cfg.copy().pop('cent')
        cent_mins = centrality_selections['min']
        cent_maxs = centrality_selections['max']

    pt_mins = cuts_cfg['pt']['min']
    pt_maxs = cuts_cfg['pt']['max']
    pt_edges = pt_mins + [pt_maxs[-1]]

    # Extract variable names and cuts from the config
    var_names = list(cuts_cfg.keys())
    var_names.remove('mass')
    var_names.remove('pt')
    var_names.remove('cent')

    cuts = [list(zip(cuts_cfg[var]['min'],
        cuts_cfg[var]['max'])) for var in var_names]
    axes = [cuts_cfg[var]['axisnum'] for var in var_names]

    # Create output file
    out_file_name = os.path.join(
        cfg['output_dir'],
        f"efficiency_{cfg['suffix']}.root"
    )
    output_file = ROOT.TFile(out_file_name, "RECREATE")
    output_file.Close()

    if cfg['weights']['apply']:
        histos_weights = get_weights(cfg, cuts_cfg)

        # Save the weights in the output file
        with ROOT.TFile.Open(out_file_name, "UPDATE") as output_file:
            if not output_file.GetListOfKeys().Contains("Weights"):
                output_file.mkdir("Weights")
            output_file.cd("Weights")
            for particle, origin in particle_origin:
                for i_cent, (cent_min, cent_max) in enumerate(zip(cent_mins, cent_maxs)):
                    histos_weights[f"{particle}{origin}_Cent_{cent_min}_{cent_max}"].Write()

    histos, histos_fine_bins, histos_cent, histos_cent_fine_bins, canvas_cent = __create_histos(
        cfg, cuts_cfg
    )

    for idx, (particle, origin) in enumerate(PARTICLE_CLASSES):

        in_file = ROOT.TFile.Open(cfg["inputs"]["sparse"]["file_name"])
        h_sparse_gen_particles = in_file.Get(f"hf-task-ds/MC/{particle}/{origin}/hPtYNPvContribGen")
        h_sparse_reco_ds = in_file.Get(f"hf-task-ds/MC/{particle}/{origin}/hSparseMass")
        in_file.Close()

        for i_pt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
            if cfg["verbose"]:
                print(f"Processing pT bin {pt_min} - {pt_max} for {particle} {origin}")

            if not cfg ["weights"]["apply"]:
                eff, unc = get_eff(cfg, h_sparse_gen_particles, h_sparse_reco_ds,
                    (particle, origin), cuts, axes, (i_pt, pt_min, pt_max))

                histos[idx].SetBinContent(i_pt+1, eff)
                histos[idx].SetBinError(i_pt+1, unc)
                #histos_fine_bins[idx].Divide(hPtProjectedRecoDs,
                #   hPtProjectedGenParticles, 1, 1, "b")

            if cfg ["weights"]["apply"]:
                for i_cent, (cent_min, cent_max) in enumerate(zip(cent_mins, cent_maxs)):
                    eff, unc = get_eff(cfg, h_sparse_gen_particles, h_sparse_reco_ds,
                        (particle, origin), cuts, axes, (i_pt, pt_min, pt_max),
                        histos_weights, (cent_min, cent_max))

                    i_histos = idx * len(cent_mins) + i_cent

                    histos_cent[i_histos].SetBinContent(i_pt+1, eff)
                    histos_cent[i_histos].SetBinError(i_pt+1, unc)
                    #histos_cent_fine_bins[i_histos].Divide(hPtProjectedRecoDs,
                    #   hPtProjectedGenParticles, 1, 1, "b")

        del h_sparse_gen_particles, h_sparse_reco_ds

    colors, _ = get_discrete_matplotlib_palette("tab20")

    if cfg ["weights"]["apply"]:
        legs = []
        for idx, (particle, origin) in enumerate(PARTICLE_CLASSES):
            legs.append(ROOT.TLegend(0.6, 0.2, 0.9, 0.8))
            legs[-1].SetBorderSize(0)
            legs[-1].SetFillStyle(0)
            legs[-1].SetTextSize(0.035)
            canvas_cent[idx].cd().SetLogy()
            histos[idx].SetMarkerColor(colors[0])
            histos[idx].SetLineColor(colors[0])
            histos[idx].Draw()
            legs[idx].AddEntry(histos[idx], "Min. bias", "l")
            for i_cent, (cent_min, cent_max) in enumerate(zip(cent_mins, cent_maxs)):
                histos_cent[idx*len(cent_mins)+i_cent].SetMarkerColor(colors[i_cent+1])
                histos_cent[idx*len(cent_mins)+i_cent].SetLineColor(colors[i_cent+1])
                histos_cent[idx*len(cent_mins)+i_cent].Draw("same")
                legs[idx].AddEntry(histos_cent[idx*len(cent_mins)+i_cent],
                    f"{cent_min}#font[122]{{-}}{cent_max}%", "l")
            legs[idx].Draw()

    output_file = ROOT.TFile.Open(out_file_name, "update")
    for histo, histo_fine_bins in zip(histos, histos_fine_bins):
        histo.Write()
        histo_fine_bins.Write()

    if cfg ["weights"]["apply"]:
        for histo_cent, histo_cent_fine_bins in zip(histos_cent, histos_cent_fine_bins):
            histo_cent.Write()
            histo_cent_fine_bins.Write()
        for canvas in canvas_cent:
            canvas.Write()
    output_file.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate efficiencies for ML selection.')
    parser.add_argument('config_file', type=str, help='Path to the configuration file.')
    args = parser.parse_args()

    main(args.config_file)
