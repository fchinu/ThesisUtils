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
    args = parser.parse_args()

    # Load the YAML config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    ptmins = config['pt_bins']['mins']
    ptmaxs = config['pt_bins']['maxs']
    ptedges = ptmins + [ptmaxs[-1]]
    labels = config['labels']

    # Define the histograms
    histos = []
    for label in labels:
        histos.append(TH1F(f"Eff_{label}", f"Efficiency; #it{{p}}_{{T}} (GeV/c); Efficiency", len(ptedges)-1, np.asarray(ptedges, "d")))

    # Load cuts configuration
    with open(args.cuts, 'r') as f:
        cuts_cfg = yaml.safe_load(f)

    # Create output file
    output_file = TFile(config['output_file'], "recreate")

    for iPt, (ptmin, ptmax) in enumerate(zip(config['pt_bins']['mins'], config['pt_bins']['maxs'])):
        print(f"Processing pT bin {ptmin} - {ptmax}")
        
        # Load the model and apply it to the data
        ModelHandl = ModelHandler()
        ModelHandl.load_model_handler(config['model_handlers'][iPt])
        cols_to_keep = ["fM", "fPt"]
        
        for idx, (input_file, label) in enumerate(zip(config['input_files'], config['labels'])):
            preselection = f"and {config['preselections'][iPt]}" if config["preselections"][iPt] else ""
            
            # Load the data and apply preselection
            df = apply_model_in_batches(ModelHandl, cols_to_keep, input_file, f"fPt > {ptmin} and fPt < {ptmax} {preselection}")

            selToApply = ""
            for varName in cuts_cfg['cutvars']:
                if varName == 'InvMass' or varName == 'Pt':
                    continue
                if selToApply != '':
                    selToApply += ' & '
                selToApply += f"({cuts_cfg['cutvars'][varName]['min'][iPt]}<{cuts_cfg['cutvars'][varName]['name']}<{cuts_cfg['cutvars'][varName]['max'][iPt]})"

            eff, unc = calculate_efficiencies_with_unc(df, selToApply)
            histos[idx].SetBinContent(iPt+1, eff)
            histos[idx].SetBinError(iPt+1, unc)

    for histo in histos:
        histo.Write()
    output_file.Close()
