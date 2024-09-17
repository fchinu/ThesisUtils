'''
Script to evaluate the efficiency starting from the output THnSparse.
'''
from math import sqrt
import argparse
import numpy as np
import yaml
import ROOT
# pylint: disable=no-member
import sys # pylint: disable=wrong-import-order
sys.path.append("/home/fchinu/Run3/ThesisUtils")
from PlotUtils import get_discrete_matplotlib_palette # pylint: disable=wrong-import-position

PARTICLE_CLASSES = [("Ds", "Prompt"), ("Ds", "NonPrompt"),
    ("Dplus", "Prompt"), ("Dplus", "NonPrompt")]
AXISPT_GEN = 0
AXISNPV_GEN = 2
AXISPT_RECO = 1
AXISNPV_RECO = 6
ROOT.TH1.AddDirectory(False)
ROOT.TH2.AddDirectory(False)

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

def get_eff(h_sparse_gen_particles, h_sparse_reco_ds, particle_class, cuts, # pylint: disable=too-many-arguments, redefined-outer-name, too-many-locals
    axes, pt_info, histos_weights = None, cent_info = (None, None)): # pylint: disable=redefined-outer-name
    """
    Calculate the efficiency of particle selection.

    Parameters:
    - h_sparse_gen_particles (THnSparse): THnSparse histogram of generated particles.
    - h_sparse_reco_ds (THnSparse): THnSparse histogram of reconstructed particles.
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
        AXISNPV_GEN, AXISPT_GEN, "EO")
    h_pt_y_projected_reco_ds = h_sparse_reco_ds_sel.Projection(
        AXISNPV_RECO, AXISPT_RECO, "EO")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate efficiencies for ML selection.')
    parser.add_argument('sparseName', metavar='text', help='Path to the input file')
    parser.add_argument('-s', '--cuts', type=str,
        help='Path to the YAML cuts file', required=True)
    parser.add_argument('-w', '--weights', type=str,
        help='Path to the weights file', required=False)
    parser.add_argument('-o', '--output', type=str,
        help='Path to the output file', required=True)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Print debug information')
    args = parser.parse_args()

    # Load cuts configuration
    with open(args.cuts, 'r', encoding="utf-8") as f:
        cuts_cfg = yaml.safe_load(f)

    centrality_selections = None # pylint: disable=invalid-name
    cent_mins = None # pylint: disable=invalid-name
    cent_maxs = None # pylint: disable=invalid-name
    if "Cent" in cuts_cfg['cutvars']:
        centrality_selections = cuts_cfg['cutvars'].pop('Cent')
        cent_mins = centrality_selections['min']
        cent_maxs = centrality_selections['max']

    ptmins = cuts_cfg['cutvars']['Pt']['min']
    ptmaxs = cuts_cfg['cutvars']['Pt']['max']
    ptedges = ptmins + [ptmaxs[-1]]

    # Extract variable names and cuts from the config
    var_names = list(cuts_cfg['cutvars'].keys())
    var_names.remove('InvMass')
    var_names.remove('Pt')

    cuts = [list(zip(cuts_cfg['cutvars'][var]['min'],
        cuts_cfg['cutvars'][var]['max'])) for var in var_names]
    axes = [cuts_cfg['cutvars'][var]['axisnum'] for var in var_names]

    # Create output file
    output_file = ROOT.TFile(args.output, "recreate")
    output_file.Close()
    histos = []
    histosFineBins = []
    histosCent = []
    histosCentFineBins = []
    canvasCent = []

    if args.weights:
        histosWeights = {}

    for particle, origin in PARTICLE_CLASSES:

        histos.append(ROOT.TH1F(f"Eff_{particle}{origin}",
            "Efficiency; #it{p}_{T} (GeV/c); Efficiency",
            len(ptedges)-1, np.asarray(ptedges, "d")))
        histosFineBins.append(ROOT.TH1F(f"EffFineBins_{particle}{origin}",
            "Efficiency in fine bins; #it{p}_{T} (GeV/c); Efficiency", 240, 0, 24))

        if args.weights:
            infileWeights = ROOT.TFile.Open(args.weights)
            for iCent, (centmin, centmax) in enumerate(zip(cent_mins, cent_maxs)):
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"] = infileWeights.Get(
                    f"hWeight{particle}{origin}Cands_{centmin}_{centmax}")
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].SetName(
                    f"Weight{particle}{origin}Cands_{centmin}_{centmax}")
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].SetTitle(
                    f"Weight{particle}{origin}Cands_{centmin}_{centmax}")
            infileWeights.Close()

            output_file = ROOT.TFile(args.output, "update")
            if not output_file.GetListOfKeys().Contains("Weights"):
                output_file.mkdir("Weights")
            output_file.cd("Weights")
            for iCent, (centmin, centmax) in enumerate(zip(cent_mins, cent_maxs)):
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].Write()
            output_file.Close()

            for iCent, (centmin, centmax) in enumerate(zip(cent_mins, cent_maxs)):
                histosCent.append(ROOT.TH1F(f"Eff_{particle}{origin}_Cent_{centmin}_{centmax}",
                    f"Efficiency_Cent_{centmin}_{centmax}; #it{{p}}_{{T}} (GeV/c); Efficiency",
                    len(ptedges)-1, np.asarray(ptedges, "d")))
                histosCentFineBins.append(ROOT.TH1F(
                    f"EffFineBins_{particle}{origin}_Cent_{centmin}_{centmax}",
                    f"Efficiency in fine bins_Cent_{centmin}_{centmax};\
                    #it{{p}}_{{T}} (GeV/c); Efficiency", 240, 0, 24))

            canvasCent.append(ROOT.TCanvas(
                f"canvasCent_{particle}{origin}", f"canvasCent_{particle}{origin}", 800, 600))

    for idx, (particle, origin) in enumerate(PARTICLE_CLASSES):

        inFile = ROOT.TFile.Open(args.sparseName)
        hSparseGenParticles = inFile.Get(f"hf-task-ds/MC/{particle}/{origin}/hPtYNPvContribGen")
        hSparseRecoDs = inFile.Get(f"hf-task-ds/MC/{particle}/{origin}/hSparseMass")
        inFile.Close()

        for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs)):
            if args.verbose:
                print(f"Processing pT bin {ptmin} - {ptmax} for {particle} {origin}")

            if not args.weights:
                eff, unc = get_eff(hSparseGenParticles, hSparseRecoDs,
                    (particle, origin), cuts, axes, (iPt, ptmin, ptmax))

                histos[idx].SetBinContent(iPt+1, eff)
                histos[idx].SetBinError(iPt+1, unc)
                #histosFineBins[idx].Divide(hPtProjectedRecoDs,
                #   hPtProjectedGenParticles, 1, 1, "b")

            if args.weights:
                for iCent, (centmin, centmax) in enumerate(zip(cent_mins, cent_maxs)):
                    eff, unc = get_eff(hSparseGenParticles, hSparseRecoDs,
                        (particle, origin), cuts, axes, (iPt, ptmin, ptmax),
                        histosWeights, (centmin, centmax))

                    iHistos = idx * len(cent_mins) + iCent

                    histosCent[iHistos].SetBinContent(iPt+1, eff)
                    histosCent[iHistos].SetBinError(iPt+1, unc)
                    #histosCentFineBins[iHistos].Divide(hPtProjectedRecoDs,
                    #   hPtProjectedGenParticles, 1, 1, "b")

        del hSparseGenParticles, hSparseRecoDs

    colors, _ = get_discrete_matplotlib_palette("tab20")

    if args.weights:
        legs = []
        for idx, (particle, origin) in enumerate(PARTICLE_CLASSES):
            legs.append(ROOT.TLegend(0.6, 0.2, 0.9, 0.8))
            legs[-1].SetBorderSize(0)
            legs[-1].SetFillStyle(0)
            legs[-1].SetTextSize(0.035)
            canvasCent[idx].cd().SetLogy()
            histos[idx].SetMarkerColor(colors[0])
            histos[idx].SetLineColor(colors[0])
            histos[idx].Draw()
            legs[idx].AddEntry(histos[idx], "Min. bias", "l")
            for iCent, (centmin, centmax) in enumerate(zip(cent_mins, cent_maxs)):
                histosCent[idx*len(cent_mins)+iCent].SetMarkerColor(colors[iCent+1])
                histosCent[idx*len(cent_mins)+iCent].SetLineColor(colors[iCent+1])
                histosCent[idx*len(cent_mins)+iCent].Draw("same")
                legs[idx].AddEntry(histosCent[idx*len(cent_mins)+iCent],
                    f"{centmin}#font[122]{{-}}{centmax}%", "l")
            legs[idx].Draw()

    output_file = ROOT.TFile.Open(args.output, "update")
    for histo, histoFineBins in zip(histos, histosFineBins):
        histo.Write()
        histoFineBins.Write()

    if args.weights:
        for histoCent, histoCentFineBins in zip(histosCent, histosCentFineBins):
            histoCent.Write()
            histoCentFineBins.Write()
        for canvas in canvasCent:
            canvas.Write()
    output_file.Close()
