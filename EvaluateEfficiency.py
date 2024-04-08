import pandas as pd
import numpy as np
from math import sqrt
from DfUtils import apply_model_in_batches
from ROOT import TH1F, TFile
from hipe4ml.model_handler import ModelHandler

import argparse
import yaml

def calculate_efficiencies_with_unc(df, selToApply):
    eff = len(df.query(selToApply)) / len(df)
    return eff, sqrt(eff * (1 - eff) / len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate efficiencies for ML selection.')
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML configuration file', required=True)
    parser.add_argument('-s', '--cuts', type=str, help='Path to the YAML cuts file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to the output file', required=True)
    args = parser.parse_args()

    # Load the YAML config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Load cuts configuration
    with open(args.cuts, 'r') as f:
        cuts_cfg = yaml.safe_load(f)

    ptmins = config['pt_bins']['mins']
    ptmaxs = config['pt_bins']['maxs']
    ptedges = ptmins + [ptmaxs[-1]]
    
    # Create output file
    output_file = TFile(args.output, "recreate")
    histos = []
    histosReco = []
    histosBDT = []

    for idx, ParticleClass in enumerate(config['input_files']):
        if 'Ds' in ParticleClass:
            particle = 'Ds'
        elif 'Dplus' in ParticleClass:
            particle = 'Dplus'
        if 'Prompt' in ParticleClass:
            origin = 'Prompt'
        elif 'FD' in ParticleClass:
            origin = 'FD'
        
        histos.append(TH1F(f"Eff_{particle}{origin}", f"Efficiency; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
        histosReco.append(TH1F(f"RecoEff_{particle}{origin}", f"Reconstruction Efficiency; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))
        histosBDT.append(TH1F(f"BDTEff_{particle}{origin}", f"BDT Efficiency; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))


    for iPt, (ptmin, ptmax) in enumerate(zip(config['pt_bins']['mins'], config['pt_bins']['maxs'])):
        
        # Load the model and apply it to the data
        ModelHandl = ModelHandler()
        ModelHandl.load_model_handler(config['model_handlers'][iPt])
        cols_to_keep = ["fM", "fPt"]
        
        for idx, ParticleClass in enumerate(config['input_files']):
            print(f"Processing pT bin {ptmin} - {ptmax} for {ParticleClass}")

            preselection = f"and {config['preselections'][iPt]}" if config["preselections"][iPt] else ""
            
            # Load the data and apply preselection
            df = apply_model_in_batches(ModelHandl, cols_to_keep, config['input_files'][ParticleClass], f"fPt > {ptmin} and fPt < {ptmax} {preselection}")

            selToApply = ""
            for varName in cuts_cfg['cutvars']:
                if varName == 'InvMass' or varName == 'Pt':
                    continue
                if selToApply != '':
                    selToApply += ' & '
                selToApply += f"({cuts_cfg['cutvars'][varName]['min'][iPt]}<{cuts_cfg['cutvars'][varName]['name']}<{cuts_cfg['cutvars'][varName]['max'][iPt]})"

            if 'Ds' in ParticleClass:
                particle = 'Ds'
            elif 'Dplus' in ParticleClass:
                particle = 'Dplus'
            if 'Prompt' in ParticleClass:
                origin = 'Prompt'
            elif 'FD' in ParticleClass:
                origin = 'FD'

            if origin == 'Prompt':
                analysisResult = TFile.Open(config['analysis_result_file_prompt'])
                hGenParticles = analysisResult.Get(f"hf-task-ds/hPtGen{particle}Prompt")                
                genParticles = hGenParticles.Integral(hGenParticles.FindBin(ptmin), hGenParticles.FindBin(ptmax)-1)  # right edge of the bin is not included
                hRecoDs = analysisResult.Get(f"hf-task-ds/hPtRecSig{particle}Prompt")
                recoParticles = hRecoDs.Integral(hRecoDs.FindBin(ptmin), hRecoDs.FindBin(ptmax)-1)  # right edge of the bin is not included
                recoEff = recoParticles/genParticles
                recoEffUnc = sqrt(recoEff * (1 - recoEff) / recoParticles)
                BDTEff = len(df.query(selToApply))/len(df)
                recoBDTUnc = sqrt(BDTEff * (1 - BDTEff) / len(df))
                eff = len(df.query(selToApply)) / genParticles / config['dataset_eff_frac'][idx]
                unc = sqrt(eff * (1 - eff) / genParticles / config['dataset_eff_frac'][idx]) # in case only part of the MC is used for efficiency calculation
            else:
                analysisResult = TFile.Open(config['analysis_result_file_FD'])
                hGenParticles = analysisResult.Get(f"hf-task-ds/hPtGen{particle}NonPrompt")                
                genParticles = hGenParticles.Integral(hGenParticles.FindBin(ptmin), hGenParticles.FindBin(ptmax)-1)  # right edge of the bin is not included
                hRecoDs = analysisResult.Get(f"hf-task-ds/hPtRecSig{particle}NonPrompt")
                recoParticles = hRecoDs.Integral(hRecoDs.FindBin(ptmin), hRecoDs.FindBin(ptmax)-1)  # right edge of the bin is not included
                recoEff = recoParticles/genParticles
                recoEffUnc = sqrt(recoEff * (1 - recoEff) / recoParticles)
                BDTEff = len(df.query(selToApply))/len(df)
                recoBDTUnc = sqrt(BDTEff * (1 - BDTEff) / len(df))
                eff = len(df.query(selToApply)) / genParticles / config['dataset_eff_frac'][idx]
                unc = sqrt(eff * (1 - eff) / genParticles / config['dataset_eff_frac'][idx])

            histos[idx].SetBinContent(iPt+1, eff)
            histos[idx].SetBinError(iPt+1, unc)
            histosReco[idx].SetBinContent(iPt+1, recoEff)
            histosReco[idx].SetBinError(iPt+1, recoEffUnc)
            histosBDT[idx].SetBinContent(iPt+1, BDTEff)
            histosBDT[idx].SetBinError(iPt+1, BDTEff)
        

    output_file.cd()
    for histo, histoBDT, histoReco in zip(histos, histosBDT, histosReco):
        histo.Write()
        histoBDT.Write()
        histoReco.Write()
    output_file.Close()
