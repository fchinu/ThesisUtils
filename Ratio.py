from ROOT import TFile, TCanvas, TH1F
import pandas as pd
import numpy as np
import argparse

def ConvertGraphToHist(graph):
    n = graph.GetN()
    x = graph.GetX()
    xbins = np.zeros(n+1)
    for i in range(n):
        xbins[i] = x[i]-graph.GetEXlow()[i]
    xbins[n] = x[n-1]+graph.GetEXlow()[n-1]
    y = graph.GetY()
    ex = graph.GetEXlow()
    ey = graph.GetEYlow()

    hist = TH1F(graph.GetName(), graph.GetTitle(), n, np.asarray(xbins, "d"))
    for i in range(n):
        hist.SetBinContent(i+1, y[i])
        hist.SetBinError(i+1, ey[i])
    return hist

parser = argparse.ArgumentParser(description='Calculate Ds+/D+ ratio')
parser.add_argument('-r', '--raw_yields_file', metavar='raw_yields_file', type=str, help='Path to the raw yields file', default="/home/fchinu/Run3/Ds_pp_13TeV/Projections_RawYields/Data/RawYields_Data.root")
parser.add_argument('-e', '--efficiency_file', type=str, help='Path to the efficiency file', default="/home/fchinu/Run3/Ds_pp_13TeV/Efficiency/Efficiencies.root")
parser.add_argument('-o', '--output_file', type=str, help='Path to the output file', default="/home/fchinu/Run3/Ds_pp_13TeV/Ratios/DsOverDplus.root")
parser.add_argument('-fds', '--ds_frac', type=str, help='Ds+ prompt fraction', required=False)
parser.add_argument('-fdp', '--dplus_frac', type=str, help='D+ prompt fraction', required=False)
args = parser.parse_args()

RawYieldsFileName = args.raw_yields_file
EffFileName = args.efficiency_file
outputFileName = args.output_file
DsFracFileName = args.ds_frac
DplusFracFileName = args.dplus_frac

RawYieldsFile = TFile.Open(RawYieldsFileName)
EffFile = TFile.Open(EffFileName)
outputFile = TFile.Open(outputFileName, "RECREATE")

DsRawYields = RawYieldsFile.Get("hRawYields")
DsEff = EffFile.Get("Eff_DsPrompt")
DplusRawYields = RawYieldsFile.Get("hRawYieldsSecondPeak")
DplusEff = EffFile.Get("Eff_DplusPrompt")

CorrectedDsYields = DsRawYields.Clone("hCorrectedDsYields")
CorrectedDsYields.Divide(DsEff)
CorrectedDsYields.Scale(1/2.21e-2)  #BR https://pdg.lbl.gov/2023/listings/rpp2023-list-Ds-plus-minus.pdf

CorrectedDplusYields = DplusRawYields.Clone("hCorrectedDplusYields")
CorrectedDplusYields.Divide(DplusEff)
CorrectedDplusYields.Scale(1/2.69e-3)  #BR https://pdg.lbl.gov/2023/listings/rpp2023-list-D-plus-minus.pdf


UncorrectedRatioEffFD = DsRawYields.Clone("hUncorrectedRatioForEfficiencyFD")
UncorrectedRatioEffFD.SetTitle(";p_{T} (GeV/c);D_{s}^{+}/D^{+} Uncorrected ratio")
UncorrectedRatioEffFD.Divide(DplusRawYields)
UncorrectedRatioEffFD.Scale(2.69e-3/2.21e-2)

UncorrectedRatioEff = DsRawYields.Clone("hUncorrectedRatioForFD")
UncorrectedRatioEff.SetTitle(";p_{T} (GeV/c);D_{s}^{+}/D^{+} Uncorrected ratio")
UncorrectedRatioEff.Divide(DplusRawYields)
UncorrectedRatioEff.Multiply(DplusEff)
UncorrectedRatioEff.Divide(DsEff)
UncorrectedRatioEff.Scale(2.69e-3/2.21e-2)

if DsFracFileName is not None and DplusFracFileName is not None:
    try:
        DsFracFile = TFile.Open(DsFracFileName)
        gDsFrac = DsFracFile.Get("gfraction")
        DsFracFile.Close()
        gDsFrac.GetN()
        hDsFrac = ConvertGraphToHist(gDsFrac)
        CorrectedDsYields.Multiply(hDsFrac)

        DplusFracFile = TFile.Open(DplusFracFileName)
        gDplusFrac = DplusFracFile.Get("gfraction")
        DplusFracFile.Close()
        hDplusFrac = ConvertGraphToHist(gDplusFrac)
        CorrectedDplusYields.Multiply(hDplusFrac)
    except:
        DsFracFile = TFile.Open(DsFracFileName)
        hDsFrac = DsFracFile.Get("hRawFracPrompt")
        hDsFrac.SetDirectory(0)
        DsFracFile.Close()
        CorrectedDsYields.Multiply(hDsFrac)

        DplusFracFile = TFile.Open(DplusFracFileName)
        hDplusFrac = DplusFracFile.Get("hRawFracPrompt")
        hDplusFrac.SetDirectory(0)
        DplusFracFile.Close()
        CorrectedDplusYields.Multiply(hDplusFrac)

    PromptFracRatio = hDsFrac.Clone("hPromptFracRatio")
    PromptFracRatio.SetTitle(";p_{T} (GeV/c);D_{s}^{+}/D^{+} Prompt fraction ratio")
    PromptFracRatio.Divide(hDplusFrac)
    

    UncorrectedRatioFD = DsRawYields.Clone("hUncorrectedRatioForEfficiency")
    UncorrectedRatioFD.SetTitle(";p_{T} (GeV/c);D_{s}^{+}/D^{+} Uncorrected ratio")
    UncorrectedRatioFD.Divide(DplusRawYields)
    UncorrectedRatioFD.Multiply(hDsFrac)
    UncorrectedRatioFD.Divide(hDplusFrac)
    UncorrectedRatioFD.Scale(2.69e-3/2.21e-2)

Ratio = CorrectedDsYields.Clone("hRatio")
Ratio.SetTitle(";p_{T} (GeV/c);D_{s}^{+}/D^{+} Ratio")
Ratio.Divide(CorrectedDplusYields)

hEffRatio = DsEff.Clone("hEffRatio")
hEffRatio.SetTitle(";p_{T} (GeV/c);D_{s}^{+}/D^{+} Efficiency ratio")
hEffRatio.Divide(DplusEff)

outputFile.cd()
DsRawYields.Write()
DplusRawYields.Write()
DsEff.Write()
DplusEff.Write()
CorrectedDsYields.Write()
CorrectedDplusYields.Write()
Ratio.Write()
UncorrectedRatioEffFD.Write()
hEffRatio.Write()
if DsFracFileName is not None and DplusFracFileName is not None:
    UncorrectedRatioEff.Write()
    UncorrectedRatioFD.Write()
    hDsFrac.Write()
    hDplusFrac.Write()
    PromptFracRatio.Write()
outputFile.Close()