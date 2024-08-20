'''
Script for fitting D+, D0 and Ds+ invariant-mass spectra
run: python GetRawYieldsDsDplus.py fitConfigFileName.yml inputFileName.root outFileName.root
            [--save][--batch][-j N]
'''

import sys
import argparse
import ctypes
import numpy as np
import yaml
import zfit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from particle import Particle
sys.path.append("DmesonAnalysis")
from ROOT import TFile, TCanvas, TH1D, TH1F, TF1, TDatabasePDG, TDirectoryFile # pylint: disable=import-error,no-name-in-module
from ROOT import gROOT, gPad, kBlack, kRed, kFullCircle, kFullSquare # pylint: disable=import-error,no-name-in-module
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle, DivideCanvas
from utils.FitUtils import SingleGaus, DoubleGaus, DoublePeakSingleGaus, DoublePeakDoubleGaus
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import matplotlib as mpl
import matplotlib.pyplot as plt
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
zfit.run.set_cpus_explicit(intra=30, inter=30)


def fit(input_file, input_file_bkgtempl, output_dir, config, **kwargs):
    """
    Method for fitting
    """

    cent_min = config["cent_min"]
    cent_max = config["cent_max"]
    pt_min = config["pt_min"]
    pt_max = config["pt_max"]
    mass_min = config["mass_min"]
    mass_max = config["mass_max"]
    rebin = config["rebin"]
    fixSigma = config["FixSigma"]
    fixSigmaFile = config["SigmaFile"]
    bkg_func = config["bkg_func"]
    fixSigmaSecPeak = config["fixSigmaSecPeak"]
    sigmaMultFactorSecPeak = config["SigmaMultFactorSecPeak"]

    if cent_min is not None and cent_max is not None:

        data_hdl = DataHandler(data=input_file, var_name="fM",
                           histoname=f'hMass_{pt_min*10:.0f}_{pt_max*10:.0f}_Cent_{cent_min:.0f}_{cent_max:.0f}',
                           limits=[mass_min,mass_max], rebin=rebin)
    else:
        data_hdl = DataHandler(data=input_file, var_name="fM",
                           histoname=f'hMass_{pt_min*10:.0f}_{pt_max*10:.0f}',
                           limits=[mass_min,mass_max], rebin=rebin)

    if input_file_bkgtempl is None:
        fitter = F2MassFitter(data_hdl, name_signal_pdf=["gaussian", "gaussian"],
                            name_background_pdf=[bkg_func],
                            name=f"ds_pt{pt_min*10:.0f}_{pt_max*10:.0f}", chi2_loss=False,
                            verbosity=1, tol=1.e-1)

        # bkg initialisation
        if bkg_func == "expo":
            fitter.set_background_initpar(0, "lam", -2)
        elif bkg_func == "chebpol2":
            fitter.set_background_initpar(0, "c0", 0.6)
            fitter.set_background_initpar(0, "c1", -0.2)
            fitter.set_background_initpar(0, "c2", 0.01)
        elif bkg_func == "chebpol3":
            fitter.set_background_initpar(0, "c0", 0.4)
            fitter.set_background_initpar(0, "c1", -0.2)
            fitter.set_background_initpar(0, "c2", -0.01)
            fitter.set_background_initpar(0, "c3", 0.01)


    else:
        data_corr_bkg = DataHandler(data=input_file_bkgtempl, var_name="fM",
                                    histoname=f'hDplusTemplate_{pt_min*10:.0f}_{pt_max*10:.0f}',
                                    limits=[mass_min,mass_max], rebin=rebin)
        fitter = F2MassFitter(data_hdl, name_signal_pdf=["gaussian", "gaussian"],
                            name_background_pdf=["hist", bkg_func],
                            name=f"ds_pt{pt_min*10:.0f}_{pt_max*10:.0f}", chi2_loss=False,
                            verbosity=1, tol=1.e-1)

        # bkg initialisation
        if bkg_func == "expo":
            fitter.set_background_initpar(1, "lam", -2)
        elif bkg_func == "chebpol2":
            fitter.set_background_initpar(1, "c0", 0.6)
            fitter.set_background_initpar(1, "c1", -0.2)
            fitter.set_background_initpar(1, "c2", 0.01)
        elif bkg_func == "chebpol3":
            fitter.set_background_initpar(1, "c0", 0.4)
            fitter.set_background_initpar(1, "c1", -0.2)
            fitter.set_background_initpar(1, "c2", -0.01)
            fitter.set_background_initpar(1, "c3", 0.01)

        fitter.set_background_template(0, data_corr_bkg)
        fitter.set_background_initpar(0, "frac", 0.01, limits=[0., 1.])



    # signals initialisation
    fitter.set_particle_mass(0, pdg_id=431, limits=[Particle.from_pdgid(431).mass*0.99e-3,
                                                    Particle.from_pdgid(431).mass*1.01e-3])
    fitter.set_signal_initpar(0, "sigma", 0.008, limits=[0.002, 0.030])
    fitter.set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])
    fitter.set_particle_mass(1, pdg_id=411,
                             limits=[Particle.from_pdgid(411).mass*0.99e-3,
                                     Particle.from_pdgid(411).mass*1.01e-3])
    fitter.set_signal_initpar(1, "sigma", 0.006, limits=[0.002, 0.030])
    fitter.set_signal_initpar(1, "frac", 0.1, limits=[0., 1.])

    if fixSigma:
        print(fixSigmaFile)
        if fixSigmaFile:
            sigmaFile = TFile.Open(fixSigmaFile)
            sigmaHist = sigmaFile.Get(f'hRawYieldsSigma')
            sigmaSecPeakHist = sigmaFile.Get(f'hRawYieldsSigmaSecondPeak')
            fitter.set_signal_initpar(0, "sigma", sigmaHist.GetBinContent(sigmaHist.FindBin(pt_min)), fix=True)
            fitter.set_signal_initpar(1, "sigma", sigmaSecPeakHist.GetBinContent(sigmaSecPeakHist.FindBin(pt_min)), fix=True)
            sigmaFile.Close()
        else:
            print("ERROR: No sigma factor provided, please check your config file")
            sys.exit()

    if fixSigmaSecPeak:
        if sigmaMultFactorSecPeak:
            fit_result = fitter.mass_zfit() # First fit to get the sigma
            fitter.set_signal_initpar(1, "sigma", fitter.get_sigma(0)[0]*sigmaMultFactorSecPeak, fix=True)
        else:
            print("ERROR: No sigma factor provided, please check your config file")
            sys.exit()
    if True:   
    #try:
        fit_result = fitter.mass_zfit()
        if fit_result.converged and kwargs.get("save", False):
            loc = ["lower left", "upper left"]
            ax_title = r"$M(\mathrm{KK\pi})$ GeV$/c^2$"
            fig = fitter.plot_mass_fit(style="ATLAS", show_extra_info=True,
                                    figsize=(8, 8), extra_info_loc=loc,
                                    axis_title=ax_title)
            figres = fitter.plot_raw_residuals(figsize=(8, 8), style="ATLAS",
                                            extra_info_loc=loc, axis_title=ax_title)
            if cent_min is not None and cent_max is not None:
                fig.savefig(f"{output_dir}/ds_mass_pt{pt_min:.1f}_{pt_max:.1f}_Cent_{cent_min:.0f}_{cent_max:.0f}.png")
                figres.savefig(f"{output_dir}/ds_massres_pt{pt_min:.1f}_{pt_max:.1f}_Cent_{cent_min:.0f}_{cent_max:.0f}.png")
                fitter.dump_to_root(f"{output_dir}/fits.root",option="update",suffix=f"{pt_min:.1f}_{pt_max:.1f}_Cent_{cent_min:.0f}_{cent_max:.0f}",num=5000)
            else:
                fig.savefig(f"{output_dir}/ds_mass_pt{pt_min:.1f}_{pt_max:.1f}.png")
                figres.savefig(f"{output_dir}/ds_massres_pt{pt_min:.1f}_{pt_max:.1f}.png")
                fitter.dump_to_root(f"{output_dir}/fits.root",option="update",suffix=f"{pt_min:.1f}_{pt_max:.1f}",num=5000)

        outDict = {"rawyields": [fitter.get_raw_yield(i) for i in range(2)],
                "rawyields_bincounting": [[fitter.get_raw_yield_bincounting(i, nsigma=j) for j in kwargs.get("nsigma", [3, 5])] for i in range(2)],
                "sigma": [fitter.get_sigma(i) for i in range(2)],
                "mean": [fitter.get_mass(i) for i in range(2)],
                "chi2": fitter.get_chi2_ndf(),
                "significance": [fitter.get_significance(i) for i in range(2)],
                "signal": [fitter.get_signal(i) for i in range(2)],
                "background": [fitter.get_background(i) for i in range(2)],
                "fracs": fitter._F2MassFitter__get_all_fracs(),
                "converged": fit_result.converged}

    #except Exception as e:
    #    print("An error occurred during mass fitting:", e)
    #    outDict = {"rawyields": -999.,
    #            "rawyields_bincounting": -999.,
    #            "sigma": -999.,
    #            "mean": -999.,
    #            "chi2": -999.,
    #            "significance": -999.,
    #            "signal": -999.,
    #            "background": -999.,
    #            "converged": False } 

    return outDict

def fillHisto(result, iPt):

    redchi2 = result.result()["chi2"]
        
    rawyield, rawyielderr = result.result()["rawyields"][0]
    sigma, sigmaerr= result.result()["sigma"][0]

    mean,meanerr  = result.result()["mean"][0]
    signif, signiferr = result.result()["significance"][0]
    sgn, sgnerr = result.result()["signal"][0]
    bkg, bkgerr = result.result()["background"][0]

    hRawYields.SetBinContent(iPt+1, rawyield)
    hRawYields.SetBinError(iPt+1, rawyielderr)
    hRawYieldsSigma.SetBinContent(iPt+1, sigma)
    hRawYieldsSigma.SetBinError(iPt+1, sigmaerr)
    hRawYieldsMean.SetBinContent(iPt+1, mean)
    hRawYieldsMean.SetBinError(iPt+1, meanerr)
    hRawYieldsOverEv.SetBinContent(iPt+1, rawyield/nEv)
    hRawYieldsOverEv.SetBinError(iPt+1, rawyielderr/nEv)
    hRawYieldsSignificance.SetBinContent(iPt+1, signif)
    hRawYieldsSignificance.SetBinError(iPt+1, signiferr)
    hRawYieldsSignifOverSqrtEv.SetBinContent(iPt+1, signif/np.sqrt(nEv))
    hRawYieldsSignifOverSqrtEv.SetBinError(iPt+1, signiferr/np.sqrt(nEv))
    hRawYieldsSoverB.SetBinContent(iPt+1, sgn/bkg)
    hRawYieldsSoverB.SetBinError(iPt+1, sgn/bkg*np.sqrt(
        sgnerr**2/sgn**2+bkgerr**2/bkg**2))
    hRawYieldsSignal.SetBinContent(iPt+1, sgn)
    hRawYieldsSignal.SetBinError(iPt+1, sgnerr)
    hRawYieldsBkg.SetBinContent(iPt+1, bkg)
    hRawYieldsBkg.SetBinError(iPt+1, bkgerr)
    hRawYieldsChiSquare.SetBinContent(iPt+1, redchi2)
    hRawYieldsChiSquare.SetBinError(iPt+1, 1.e-20)

    rawyieldSecPeak, rawyieldSecPeakerr =  result.result()["rawyields"][1]
    meanSecPeak, meanSecPeakerr =  result.result()["mean"][1]
    sigmaSecPeak, sigmaSecPeakerr =  result.result()["sigma"][1]

    bkgSecPeak, bkgSecPeakerr =  result.result()["background"][1]
    signalSecPeak, signalSecPeakerr =  result.result()["signal"][1]
    signifSecPeak, signifSecPeakerr =  result.result()["significance"][1]

    hRawYieldsSecPeak.SetBinContent(iPt+1, rawyieldSecPeak)
    hRawYieldsSecPeak.SetBinError(iPt+1, rawyieldSecPeakerr)
    hRawYieldsMeanSecPeak.SetBinContent(iPt+1, meanSecPeak)
    hRawYieldsMeanSecPeak.SetBinError(iPt+1, meanSecPeakerr)
    hRawYieldsSigmaSecPeak.SetBinContent(iPt+1, sigmaSecPeak)
    hRawYieldsSigmaSecPeak.SetBinError(iPt+1, sigmaSecPeakerr)
    hRawYieldsOverEvSecPeak.SetBinContent(iPt+1, rawyieldSecPeak/nEv)
    hRawYieldsOverEvSecPeak.SetBinError(iPt+1, rawyieldSecPeakerr/nEv)
    hRawYieldsSignificanceSecPeak.SetBinContent(iPt+1, signifSecPeak)
    hRawYieldsSignificanceSecPeak.SetBinError(iPt+1, signifSecPeakerr)
    hRawYieldsSignificanceSecPeakOverSqrtEv.SetBinContent(iPt+1, signifSecPeak/np.sqrt(nEv))
    hRawYieldsSignificanceSecPeakOverSqrtEv.SetBinError(iPt+1, signifSecPeakerr/np.sqrt(nEv))
    hRawYieldsSigmaRatioSecondFirstPeak.SetBinContent(iPt+1, sigmaSecPeak/sigma)
    hRawYieldsSigmaRatioSecondFirstPeak.SetBinError(iPt+1, \
        np.sqrt(sigmaerr**2/sigma**2+sigmaSecPeakerr**2/sigmaSecPeak**2)*sigmaSecPeak/sigma)
    hRawYieldsSoverBSecPeak.SetBinContent(iPt+1, signalSecPeak/bkgSecPeak)
    hRawYieldsSoverBSecPeak.SetBinError(iPt+1, \
        signalSecPeak/bkgSecPeak*np.sqrt(signalSecPeakerr**2/signalSecPeak**2+bkgSecPeakerr**2/bkgSecPeak**2))
    hRawYieldsSignalSecPeak.SetBinContent(iPt+1, signalSecPeak)
    hRawYieldsSignalSecPeak.SetBinError(iPt+1, signalSecPeakerr)
    hRawYieldsBkgSecPeak.SetBinContent(iPt+1, bkgSecPeak)
    hRawYieldsBkgSecPeak.SetBinError(iPt+1, bkgSecPeakerr)

    counter = 0
    for idx, type in enumerate(result.result()["fracs"][:3]):
        for i, frac in enumerate(type):
            hFracs[counter].SetBinContent(iPt+1, frac)
            hFracs[counter].SetBinError(iPt+1, result.result()["fracs"][idx+3][i])
            counter += 1


parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('inFileName', metavar='text', default='')
parser.add_argument('fitConfigFileName', metavar='text', default='config_Ds_Fit.yml')
parser.add_argument('outFileName', metavar='text', default='')
parser.add_argument('--batch', help='suppress video output', action='store_true')
parser.add_argument('--save', help='save fits', action='store_true')
parser.add_argument('-j', '--jobs', type=int, default=30)
args = parser.parse_args()

cent = 'pp13TeVPrompt'

with open(args.fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
    fitConfig = yaml.load(ymlfitConfigFile, yaml.FullLoader)

gROOT.SetBatch(args.batch)
SetGlobalStyle(padleftmargin=0.14, padbottommargin=0.12, padtopmargin=0.12, opttitle=1)

centMins = fitConfig[cent]['CentMin']
centMaxs = fitConfig[cent]['CentMax']
ptMins = fitConfig[cent]['PtMin']
ptMaxs = fitConfig[cent]['PtMax']
massMins = fitConfig[cent]['MassMin']
massMaxs = fitConfig[cent]['MassMax']
fixSigma = fitConfig[cent]['FixSigma']
fixSigmaFile = fitConfig[cent]['SigmaFile']
FixSigmaSecPeak = fitConfig[cent]['FixSigmaSecPeak']
SigmaMultFactorSecPeak = fitConfig[cent]["SigmaMultFactorSecPeak"]
fixMean = fitConfig[cent]['FixMean']
Rebins = fitConfig[cent]['Rebin']
bkg_funcs = fitConfig[cent]['BkgFunc']
ptLims = list(ptMins)
nPtBins = len(ptMins)
ptLims.append(ptMaxs[-1])
particleName = fitConfig[cent]['Particle']
inclSecPeak = fitConfig[cent]['InclSecPeak']

# load inv-mass histos
infile = TFile.Open(args.inFileName)
if not infile or not infile.IsOpen():
    print(f'ERROR: file "{args.inFileName}" cannot be opened! Exit!')
    sys.exit()

hEv = infile.Get("hEv")
hEv.SetDirectory(0)
nEv = hEv.GetEntries()

hMass = []
for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
    if centMins and centMaxs:
        for centMin, centMax in zip(centMins, centMaxs):
            hMass.append(infile.Get(f'hMass_{ptMin*10:.0f}_{ptMax*10:.0f}_Cent_{centMin:.0f}_{centMax:.0f}'))
            hMass[-1].SetDirectory(0)
    else:
        hMass.append(infile.Get(f'hMass_{ptMin*10:.0f}_{ptMax*10:.0f}'))
        hMass[-1].SetDirectory(0)
infile.Close()

if args.save:   # Recreate file
    f = TFile(f"{os.path.dirname(args.outFileName)}/fits/fits.root","RECREATE")
    f.Close()

ptBinsArr = np.asarray(ptLims, 'd')
ptTit = '#it{p}_{T} (GeV/#it{c})'
hRawYields = TH1D('hRawYields', f';{ptTit};raw yield', nPtBins, ptBinsArr)
hRawYieldsSigma = TH1D('hRawYieldsSigma', f';{ptTit};width (GeV/#it{{c}}^{{2}})', nPtBins, ptBinsArr)
hRawYieldsMean = TH1D('hRawYieldsMean', f';{ptTit};mean (GeV/#it{{c}}^{{2}})', nPtBins, ptBinsArr)
hRawYieldsOverEv = TH1D('hRawYieldsOverEv', f';{ptTit};raw yield / N_{{ev}}', nPtBins, ptBinsArr)
hRawYieldsSignificance = TH1D('hRawYieldsSignificance', f';{ptTit};significance (3#sigma)', nPtBins, ptBinsArr)
hRawYieldsSignifOverSqrtEv = TH1D('hRawYieldsSignifOverSqrtEv', f';{ptTit};significance / #sqrt{{N_{{ev}}}}', nPtBins, ptBinsArr)
hRawYieldsSoverB = TH1D('hRawYieldsSoverB', f';{ptTit};S/B (3#sigma)', nPtBins, ptBinsArr)
hRawYieldsSignal = TH1D('hRawYieldsSignal', f';{ptTit};Signal (3#sigma)', nPtBins, ptBinsArr)
hRawYieldsBkg = TH1D('hRawYieldsBkg', f';{ptTit};Background (3#sigma)', nPtBins, ptBinsArr)
hRawYieldsChiSquare = TH1D('hRawYieldsChiSquare', f';{ptTit};#chi^{{2}}/#it{{ndf}}', nPtBins, ptBinsArr)
hRawYieldsSecPeak = TH1D('hRawYieldsSecondPeak', f';{ptTit};raw yield second peak', nPtBins, ptBinsArr)
hRawYieldsMeanSecPeak = TH1D('hRawYieldsMeanSecondPeak', f';{ptTit};mean second peak (GeV/#it{{c}}^{{2}})',
                             nPtBins, ptBinsArr)
hRawYieldsSigmaSecPeak = TH1D('hRawYieldsSigmaSecondPeak', f';{ptTit};width second peak (GeV/#it{{c}}^{{2}})',
                              nPtBins, ptBinsArr)
hRawYieldsOverEvSecPeak = TH1D('hRawYieldsOverEvSecondPeak', f';{ptTit};raw yield second peak / N_{{ev}}', nPtBins, ptBinsArr)
hRawYieldsSignificanceSecPeak = TH1D('hRawYieldsSignificanceSecondPeak',
                                     f';{ptTit};signficance second peak (3#sigma)', nPtBins, ptBinsArr)
hRawYieldsSignificanceSecPeakOverSqrtEv = TH1D('hRawYieldsSignificanceSecondPeakOverSqrtEv', f';{ptTit};significance / #sqrt{{N_{{ev}}}}', nPtBins, ptBinsArr)
hRawYieldsSigmaRatioSecondFirstPeak = TH1D('hRawYieldsSigmaRatioSecondFirstPeak',
                                           f';{ptTit};width second peak / width first peak', nPtBins, ptBinsArr)
hRawYieldsSoverBSecPeak = TH1D('hRawYieldsSoverBSecondPeak', f';{ptTit};S/B second peak (3#sigma)',
                               nPtBins, ptBinsArr)
hRawYieldsSignalSecPeak = TH1D('hRawYieldsSignalSecondPeak', f';{ptTit};Signal second peak (3#sigma)',
                               nPtBins, ptBinsArr)
hRawYieldsBkgSecPeak = TH1D('hRawYieldsBkgSecondPeak', f';{ptTit};Background second peak (3#sigma)',
                            nPtBins, ptBinsArr)
hRawYieldsTrue = TH1D('hRawYieldsTrue', f';{ptTit};true signal', nPtBins, ptBinsArr)
hRawYieldsSecPeakTrue = TH1D('hRawYieldsSecondPeakTrue', f';{ptTit};true signal second peak',
                             nPtBins, ptBinsArr)
hRelDiffRawYieldsFitTrue = TH1D('hRelDiffRawYieldsFitTrue', f';{ptTit}; (Y_{{fit}} - Y_{{true}}) / Y_{{true}}',
                                nPtBins, ptBinsArr)
hRelDiffRawYieldsSecPeakFitTrue = TH1D('hRelDiffRawYieldsSecondPeakFitTrue',
                                       f';{ptTit};(Y_{{fit}} - Y_{{true}}) / Y_{{true}} second peak',
                                       nPtBins, ptBinsArr)
hFracs = []

SetObjectStyle(hRawYields, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsSigma, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsMean, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsOverEv, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsSignificance, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsSignifOverSqrtEv, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsSoverB, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsSignal, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsBkg, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsChiSquare, color=kBlack, markerstyle=kFullCircle)
SetObjectStyle(hRawYieldsSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsMeanSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsSigmaSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsOverEvSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsSignificanceSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsSignificanceSecPeakOverSqrtEv, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsSigmaRatioSecondFirstPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsSoverBSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsSignalSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsBkgSecPeak, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsTrue, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRawYieldsSecPeakTrue, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRelDiffRawYieldsFitTrue, color=kRed, markerstyle=kFullSquare)
SetObjectStyle(hRelDiffRawYieldsSecPeakFitTrue, color=kRed, markerstyle=kFullSquare)

results = []
with ProcessPoolExecutor(max_workers=args.jobs) as executor:
    if centMins and centMaxs:
        for iCent, (centmin, centmax) in enumerate(zip(centMins, centMaxs)):
            for idx, (ptmin, ptmax, massmin, massmax, rebin, bkg_func, fixSigmaSecPeak) in enumerate(zip(ptMins, ptMaxs, massMins, massMaxs, Rebins, bkg_funcs, FixSigmaSecPeak)):
                config = {'mass_min': massmin, 'mass_max': massmax, 'cent_min': centmin, 'cent_max': centmax, 'pt_min': ptmin, 'pt_max': ptmax, 'rebin': rebin, \
                        'FixSigma': fixSigma, 'fixSigmaSecPeak': fixSigmaSecPeak, 'SigmaFile': fixSigmaFile, 'bkg_func': bkg_func, 'SigmaMultFactorSecPeak': SigmaMultFactorSecPeak}
                results.append((executor.submit(fit, args.inFileName, fitConfig[cent]['TemplateFile'],
                    os.path.dirname(args.outFileName) + "/fits", config, save=args.save), idx))
                
            names_Fracs = ["signal", "background", "reflections", "signal_err", "background_err", "reflections_err"]
            for idx, type in enumerate(results[0][0].result()["fracs"][:3]):
                for i, frac in enumerate(type):
                    hFracs.append(TH1D(f'hFracs_{names_Fracs[idx]}{i}_{ptmin*10:.0f}_{ptmax*10:.0f}_Cent_{centmin:.0f}_{centmax:.0f}', f';{ptTit};fraction', nPtBins, ptBinsArr))

            for result, iPt in results:
                fillHisto(result, iPt)
            outFile = TFile(args.outFileName.replace('.root',f'_Cent_{centmin}_{centmax}.root'), 'recreate')

            for hist in hMass:
                if f"Cent_{centmin}_{centmax}" in hist.GetName():
                    hist.Write()
                    del hist

            hRawYields.Write()
            hRawYieldsSigma.Write()
            hRawYieldsMean.Write()
            hRawYieldsOverEv.Write()
            hRawYieldsSignificance.Write()
            hRawYieldsSignifOverSqrtEv.Write()
            hRawYieldsSoverB.Write()
            hRawYieldsSignal.Write()
            hRawYieldsBkg.Write()
            hRawYieldsChiSquare.Write()
            hRawYieldsSecPeak.Write()
            hRawYieldsMeanSecPeak.Write()
            hRawYieldsSigmaSecPeak.Write()
            hRawYieldsOverEvSecPeak.Write()
            hRawYieldsSignificanceSecPeak.Write()
            hRawYieldsSignificanceSecPeakOverSqrtEv.Write()
            hRawYieldsSigmaRatioSecondFirstPeak.Write()
            hRawYieldsSoverBSecPeak.Write()
            hRawYieldsSignalSecPeak.Write()
            hRawYieldsBkgSecPeak.Write()
            hRawYieldsTrue.Write()
            hRawYieldsSecPeakTrue.Write()
            hRelDiffRawYieldsFitTrue.Write()
            hRelDiffRawYieldsSecPeakFitTrue.Write()
            for hist in hFracs:
                if f"Cent_{centmin}_{centmax}" in hist.GetName():
                    hist.Write()
                    del hist
            hEv.Write("hEv")
            outFile.Close()

    for idx, (ptmin, ptmax, massmin, massmax, rebin, bkg_func, fixSigmaSecPeak) in enumerate(zip(ptMins, ptMaxs, massMins, massMaxs, Rebins, bkg_funcs, FixSigmaSecPeak)):
        config = {'mass_min': massmin, 'mass_max': massmax, 'cent_min': None, 'cent_max': None, 'pt_min': ptmin, 'pt_max': ptmax, 'rebin': rebin, \
                    'FixSigma': fixSigma, 'fixSigmaSecPeak': fixSigmaSecPeak, 'SigmaFile': fixSigmaFile, 'bkg_func': bkg_func, 'SigmaMultFactorSecPeak': SigmaMultFactorSecPeak}
        results.append((executor.submit(fit, args.inFileName, fitConfig[cent]['TemplateFile'],
            os.path.dirname(args.outFileName) + "/fits", config, save=args.save), idx))
        names_Fracs = ["signal", "background", "reflections", "signal_err", "background_err", "reflections_err"]
        for idx, type in enumerate(results[0][0].result()["fracs"][:3]):
            for i, frac in enumerate(type):
                hFracs.append(TH1D(f'hFracs_{names_Fracs[idx]}{i}', f';{ptTit};fraction', nPtBins, ptBinsArr))

        for result, iPt in results:
            fillHisto(result, iPt)

        outFile = TFile(args.outFileName, 'recreate')

        for hist in hMass:
            hist.Write()
            del hist

        hRawYields.Write()
        hRawYieldsSigma.Write()
        hRawYieldsMean.Write()
        hRawYieldsOverEv.Write()
        hRawYieldsSignificance.Write()
        hRawYieldsSignifOverSqrtEv.Write()
        hRawYieldsSoverB.Write()
        hRawYieldsSignal.Write()
        hRawYieldsBkg.Write()
        hRawYieldsChiSquare.Write()
        hRawYieldsSecPeak.Write()
        hRawYieldsMeanSecPeak.Write()
        hRawYieldsSigmaSecPeak.Write()
        hRawYieldsOverEvSecPeak.Write()
        hRawYieldsSignificanceSecPeak.Write()
        hRawYieldsSignificanceSecPeakOverSqrtEv.Write()
        hRawYieldsSigmaRatioSecondFirstPeak.Write()
        hRawYieldsSoverBSecPeak.Write()
        hRawYieldsSignalSecPeak.Write()
        hRawYieldsBkgSecPeak.Write()
        hRawYieldsTrue.Write()
        hRawYieldsSecPeakTrue.Write()
        hRelDiffRawYieldsFitTrue.Write()
        hRelDiffRawYieldsSecPeakFitTrue.Write()
        for hist in hFracs:
            hist.Write()
            del hist
        hEv.Write("hEv")
        outFile.Close()


del hRawYields, hRawYieldsSigma, hRawYieldsMean, hRawYieldsOverEv, hRawYieldsSignificance, hRawYieldsSignifOverSqrtEv, hRawYieldsSoverB, \
    hRawYieldsSignal, hRawYieldsBkg, hRawYieldsChiSquare, hRawYieldsSecPeak, hRawYieldsMeanSecPeak, hRawYieldsSigmaSecPeak, \
    hRawYieldsOverEvSecPeak, hRawYieldsSignificanceSecPeak, hRawYieldsSignificanceSecPeakOverSqrtEv, hRawYieldsSigmaRatioSecondFirstPeak, \
    hRawYieldsSoverBSecPeak, hRawYieldsSignalSecPeak, hRawYieldsBkgSecPeak, hRawYieldsTrue, hRawYieldsSecPeakTrue, \
    hRelDiffRawYieldsFitTrue, hRelDiffRawYieldsSecPeakFitTrue, hEv