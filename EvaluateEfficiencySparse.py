import pandas as pd
import numpy as np
from math import sqrt
import uproot
from DfUtils import read_parquet_in_batches
import ROOT
#from hipe4ml.model_handler import ModelHandler
import sys
sys.path.append("/home/fchinu/Run3/ThesisUtils")
from PlotUtils import get_discrete_matplotlib_palette

import argparse
import yaml

def calculate_efficiencies_with_unc(df, selToApply):
    eff = len(df.query(selToApply)) / len(df)
    return eff, sqrt(eff * (1 - eff) / len(df))

PARTICLE_CLASSES = [("Ds", "Prompt"), ("Ds", "NonPrompt"), ("Dplus", "Prompt"), ("Dplus", "NonPrompt")]
AXISPT_GEN = 0
AXISNPV_GEN = 2
AXISPT_RECO = 1
AXISNPV_RECO = 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate efficiencies for ML selection.')
    parser.add_argument('sparseName', metavar='text', help='Path to the input file')
    parser.add_argument('-s', '--cuts', type=str, help='Path to the YAML cuts file', required=True)
    parser.add_argument('-w', '--weights', type=str, help='Path to the weights file', required=False)
    parser.add_argument('-o', '--output', type=str, help='Path to the output file', required=True)
    args = parser.parse_args()

    # Load cuts configuration
    with open(args.cuts, 'r') as f:
        cuts_cfg = yaml.safe_load(f)

    centralitySelections = None
    if "Cent" in cuts_cfg['cutvars']:
        centralitySelections = cuts_cfg['cutvars'].pop('Cent')
        centmins = centralitySelections['min']
        centmaxs = centralitySelections['max']
        centmins.insert(0, 0)
        centmaxs.insert(0, 100)
    
    ptmins = cuts_cfg['cutvars']['Pt']['min']
    ptmaxs = cuts_cfg['cutvars']['Pt']['max']
    ptedges = ptmins + [ptmaxs[-1]]

    # Extract variable names and cuts from the config
    var_names = list(cuts_cfg['cutvars'].keys())
    var_names.remove('InvMass')
    var_names.remove('Pt')

    cuts = [list(zip(cuts_cfg['cutvars'][var]['min'], cuts_cfg['cutvars'][var]['max'])) for var in var_names]
    axes = [cuts_cfg['cutvars'][var]['axisnum'] for var in var_names]

    # Create output file
    output_file = ROOT.TFile(args.output, "recreate")
    output_file.Close()
    histos = []
    histosFineBins = []
    histosReco = []
    histosBDT = []
    histosCent = []
    histosCentReco = []
    histosCentBDT = []
    histosCentFineBins = []
    canvasCent = []

    if args.weights:
        histosWeights = {}


    for particle, origin in PARTICLE_CLASSES:
        
        histos.append(ROOT.TH1F(f"Eff_{particle}{origin}", f"Efficiency; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
        histosReco.append(ROOT.TH1F(f"RecoEff_{particle}{origin}", f"Reconstruction Efficiency; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
        histosBDT.append(ROOT.TH1F(f"BDTEff_{particle}{origin}", f"BDT Efficiency; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
        histosFineBins.append(ROOT.TH1F(f"EffFineBins_{particle}{origin}", f"Efficiency in fine bins; #it{{p}}_{{T}} (GeV/c); Efficiency", len(np.linspace(0,24,241))-1, np.asarray(np.linspace(0,24,241), "d")))
                    
        if args.weights:
            infileWeights = ROOT.TFile.Open(args.weights)
            for iCent, (centmin, centmax) in enumerate(zip(centmins, centmaxs)):
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"] = infileWeights.Get(f"hWeight{particle}{origin}Cands_{centmin}_{centmax}")
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].SetDirectory(0)
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].SetName(f"Weight{particle}{origin}Cands_{centmin}_{centmax}")
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].SetTitle(f"Weight{particle}{origin}Cands_{centmin}_{centmax}")
            infileWeights.Close()

            output_file = ROOT.TFile(args.output, "update")
            output_file.mkdir("Weights")
            output_file.cd("Weights")
            for iCent, (centmin, centmax) in enumerate(zip(centmins, centmaxs)):
                histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].Write()
            output_file.Close()

            for iCent, (centmin, centmax) in enumerate(zip(centmins, centmaxs)):
                histosCent.append(ROOT.TH1F(f"Eff_{particle}{origin}_Cent_{centmin}_{centmax}", f"Efficiency_Cent_{centmin}_{centmax}; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
                histosCentReco.append(ROOT.TH1F(f"RecoEff_{particle}{origin}_Cent_{centmin}_{centmax}", f"Reconstruction Efficiency_Cent_{centmin}_{centmax}; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
                histosCentBDT.append(ROOT.TH1F(f"BDTEff_{particle}{origin}_Cent_{centmin}_{centmax}", f"BDT Efficiency_Cent_{centmin}_{centmax}; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
                histosCentFineBins.append(ROOT.TH1F(f"EffFineBins_{particle}{origin}_Cent_{centmin}_{centmax}", f"Efficiency in fine bins_Cent_{centmin}_{centmax}; #it{{p}}_{{T}} (GeV/c); Efficiency", len(np.linspace(0,24,241))-1, np.asarray(np.linspace(0,24,241), "d")))
            
            canvasCent.append(ROOT.TCanvas(f"canvasCent_{particle}{origin}", f"canvasCent_{particle}{origin}", 800, 600))


    for idx, (particle, origin) in enumerate(PARTICLE_CLASSES):

        inFile = ROOT.TFile.Open(args.sparseName)
        hSparseGenParticles = inFile.Get(f"hf-task-ds/MC/{particle}/{origin}/hPtYNPvContribGen")
        hSparseRecoDs = inFile.Get(f"hf-task-ds/MC/{particle}/{origin}/hSparseMass")
        inFile.Close()

        for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs)):

            print(f"Processing pT bin {ptmin} - {ptmax} for {particle} {origin}")

            hSparseGenSel = hSparseGenParticles.Clone(f"hSparseGenSel_{particle}{origin}")
            hSparseRecoDsSel = hSparseRecoDs.Clone(f"hSparseRecoDsSel_{particle}{origin}")
            
            hPtProjectedGenParticles = hSparseGenParticles.Projection(AXISPT_GEN, "EO")       
            genParticles = hPtProjectedGenParticles.Integral(hPtProjectedGenParticles.FindBin(ptmin), hPtProjectedGenParticles.FindBin(ptmax)-1)  # right edge of the bin is not included
            hPtProjectedRecoDs = hSparseRecoDs.Projection(AXISPT_RECO, "EO")
            recoParticles = hPtProjectedRecoDs.Integral(hPtProjectedRecoDs.FindBin(ptmin), hPtProjectedRecoDs.FindBin(ptmax)-1)  # right edge of the bin is not included

            for i, axis in enumerate(axes):
                min = cuts[i][iPt][0]
                max = cuts[i][iPt][1]
                hSparseRecoDsSel.GetAxis(axis).SetRangeUser(min, max)

            hPtProjectedRecoDs = hSparseRecoDsSel.Projection(AXISPT_RECO, "EO")
            selectedParticles = hPtProjectedRecoDs.Integral(hPtProjectedRecoDs.FindBin(ptmin), hPtProjectedRecoDs.FindBin(ptmax)-1)  # right edge of the bin is not included

            recoEff = recoParticles/genParticles
            recoEffUnc = sqrt(recoEff * (1 - recoEff) / recoParticles)
            BDTEff = selectedParticles/recoParticles
            recoBDTUnc = sqrt(BDTEff * (1 - BDTEff) / recoParticles)
            eff = selectedParticles / genParticles
            unc = sqrt(eff * (1 - eff) / genParticles)

            histos[idx].SetBinContent(iPt+1, eff)
            histos[idx].SetBinError(iPt+1, unc)
            histosReco[idx].SetBinContent(iPt+1, recoEff)
            histosReco[idx].SetBinError(iPt+1, recoEffUnc)
            histosBDT[idx].SetBinContent(iPt+1, BDTEff)
            histosBDT[idx].SetBinError(iPt+1, recoBDTUnc)
            histosFineBins[idx].Divide(hPtProjectedRecoDs, hPtProjectedGenParticles, 1, 1, "b")

            if args.weights:
                for iCent, (centmin, centmax) in enumerate(zip(centmins, centmaxs)):
                    hSparseGenParticlesSel = hSparseGenParticles.Clone(f"hSparseGenParticlesSel_{particle}{origin}_Cent_{centmin}_{centmax}")
                    hSparseRecoDsSel = hSparseRecoDs.Clone(f"hSparseRecoDsSel_{particle}{origin}_Cent_{centmin}_{centmax}")
                    
                    n_dims_gen = hSparseGenParticlesSel.GetNdimensions()
                    n_dims_reco = hSparseRecoDsSel.GetNdimensions()
                    bin_indices_gen = np.zeros(n_dims_gen, dtype=np.intc)
                    bin_indices_reco = np.zeros(n_dims_reco, dtype=np.intc)

                    for i in range(hSparseGenParticlesSel.GetNbins()):
                        content = hSparseGenParticlesSel.GetBinContent(i, bin_indices_gen) # Fills bin_indices_gen with bin indices of bin i for the axes
                        content *= histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].GetBinContent(int(bin_indices_gen[AXISNPV_GEN]))
                        hSparseGenParticlesSel.SetBinContent(bin_indices_gen, content) 

                    for i in range(hSparseRecoDsSel.GetNbins()):
                        content = hSparseRecoDsSel.GetBinContent(i, bin_indices_reco) # Fills bin_indices_reco with bin indices of bin i for the axes
                        content *= histosWeights[f"{particle}{origin}_Cent_{centmin}_{centmax}"].GetBinContent(int(bin_indices_reco[AXISNPV_RECO]))
                        hSparseRecoDsSel.SetBinContent(bin_indices_reco, content) 
                    
                    hPtProjectedGenParticles = hSparseGenParticlesSel.Projection(AXISPT_GEN, "EO")       
                    genParticles = hPtProjectedGenParticles.Integral(hPtProjectedGenParticles.FindBin(ptmin), hPtProjectedGenParticles.FindBin(ptmax)-1)  # right edge of the bin is not included
                    
                    hPtProjectedRecoDs = hSparseRecoDsSel.Projection(AXISPT_RECO, "EO")
                    recoParticles = hPtProjectedRecoDs.Integral(hPtProjectedRecoDs.FindBin(ptmin), hPtProjectedRecoDs.FindBin(ptmax)-1)  # right edge of the bin is not included

                    for i, axis in enumerate(axes):
                        min = cuts[i][iPt][0]
                        max = cuts[i][iPt][1]
                        hSparseRecoDsSel.GetAxis(axis).SetRangeUser(min, max)

                    hPtProjectedRecoDs = hSparseRecoDsSel.Projection(AXISPT_RECO, "EO")
                    selectedParticles = hPtProjectedRecoDs.Integral(hPtProjectedRecoDs.FindBin(ptmin), hPtProjectedRecoDs.FindBin(ptmax)-1)  # right edge of the bin is not included

                    recoEff = recoParticles/genParticles
                    recoEffUnc = sqrt(recoEff * (1 - recoEff) / recoParticles)
                    BDTEff = selectedParticles/recoParticles
                    recoBDTUnc = sqrt(BDTEff * (1 - BDTEff) / recoParticles)
                    eff = selectedParticles / genParticles

                    iHistos = idx * len(centmins) + iCent

                    histosCent[iHistos].SetBinContent(iPt+1, eff)
                    histosCent[iHistos].SetBinError(iPt+1, unc)
                    histosCentReco[iHistos].SetBinContent(iPt+1, recoEff)
                    histosCentReco[iHistos].SetBinError(iPt+1, recoEffUnc)
                    histosCentBDT[iHistos].SetBinContent(iPt+1, BDTEff)
                    histosCentBDT[iHistos].SetBinError(iPt+1, recoBDTUnc)
                    histosCentFineBins[iHistos].Divide(hPtProjectedRecoDs, hPtProjectedGenParticles, 1, 1, "b")
                    
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
            for iCent, (centmin, centmax) in enumerate(zip(centmins, centmaxs)):
                histosCent[idx*len(centmins)+iCent].SetMarkerColor(colors[iCent+1])
                histosCent[idx*len(centmins)+iCent].SetLineColor(colors[iCent+1])
                histosCent[idx*len(centmins)+iCent].Draw("same")
                legs[idx].AddEntry(histosCent[idx*len(centmins)+iCent], f"{centmin}#font[122]{{-}}{centmax}%", "l")
            legs[idx].Draw()
            

    output_file = ROOT.TFile.Open(args.output, "update")
    for histo, histoBDT, histoReco, histoFineBins in zip(histos, histosBDT, histosReco, histosFineBins):
        histo.Write()
        histoBDT.Write()
        histoReco.Write()
        histoFineBins.Write()

    if args.weights:
        for histoCent, histoCentBDT, histoCentReco, histoCentFineBins in zip(histosCent, histosCentBDT, histosCentReco, histosCentFineBins):
            histoCent.Write()
            histoCentBDT.Write()
            histoCentReco.Write()
            histoCentFineBins.Write()
        for canvas in canvasCent:
            canvas.Write()
    output_file.Close()
