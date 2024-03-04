import pandas as pd
import numpy as np
import sys
import yaml
import argparse
from ROOT import TH1F, TFile
from DfUtils import read_parquet_in_batches
import itertools
import concurrent.futures
import time

def ApplySelection(df, cutsname, cut, cuts_limits, cutidx):
    """
    Apply cuts to a DataFrame and create a histogram.

    Parameters:
    - df: DataFrame
    - cuts: dict
    - cutidx: int

    Returns:
    - TH1F histogram
    """
    dfquery = ""
    for idx, (name, CutValue, cutlimit) in enumerate(zip(cutsname, cut, cuts_limits)):
        if name == "NSigma":
            dfquery += f"((fIsSelDsToKKPi > 3 and abs(fNSigTpcTofKa0) < {CutValue} and abs(fNSigTpcTofKa1) < {CutValue} and abs(fNSigTpcTofPi2) < {CutValue}) or (fIsSelDsToPiKK > 3 and abs(fNSigTpcTofPi0) < {CutValue} and abs(fNSigTpcTofKa1) < {CutValue} and abs(fNSigTpcTofKa2) < {CutValue})) and "
        else:
            if cutlimit == "upper":
                dfquery += f"{name} < {CutValue} and "
            elif cutlimit == "lower":
                dfquery += f"{name} > {CutValue} and "
            else:
                print("Error: cutlimit not specified")
                exit(1)
    dfquery = dfquery[:-5]  # Remove the last "and"
    dfSel = df.query(dfquery)
    MassMin = 1.7
    MassMax = 2.1
    bins = 400
    massBins = np.arange(MassMin, MassMax, 0.001)

    histo = TH1F(f"histo{cutidx}_Pt{ptmin}_{ptmax}", f"{dfquery};Mass (GeV/c^{2});Counts", bins, MassMin, MassMax)
    for mass in dfSel["fM"]:
        histo.Fill(mass)
    del dfSel, df
    return histo
    

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Apply cuts and create histograms.')
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML configuration file', required=True)
    args = parser.parse_args()

    # Run the main function with the specified config file
    config_file = args.config
    # Load configuration from YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters from config
    outfilename = config["outfilename"]
    infile = config["infile"]
    PtBins = config["PtBins"]
    cuts_config = config["cuts"]

    # Create the output file
    outfile = TFile(outfilename, "recreate")
    outfile.Close()

    # Loop over Pt bins
    for binidx, (ptmin, ptmax) in enumerate(zip(PtBins[:-1], PtBins[1:])):
        cutsname = []
        cuts_steps = []
        cuts_limits = []
        for cut in config["cuts"]:
            cuts_min = cut["min"][binidx]
            cuts_max = cut["max"][binidx]
            cuts_step = cut["Nsteps"]
            cuts_steps.append(np.linspace(cuts_min, cuts_max, cuts_step))
            cutsname.append(cut["name"])
            cuts_limits.append(cut["limit"])

        cuts = list(itertools.product(*cuts_steps))

        # Load the data
        print(f"Loading data for Pt {ptmin} - {ptmax}", end="\r")
        start_time = time.time()
        df = read_parquet_in_batches(infile, f"fPt > {ptmin} and fPt < {ptmax}", batch_size=1000000)
        print(f"Data loaded for Pt {ptmin} - {ptmax} in {time.time() - start_time} seconds")

        histos = []
        futures = []

        # Apply cuts concurrently
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            for cutidx, cut in enumerate(cuts):
                futures.append(executor.submit(ApplySelection, df, cutsname, cut, cuts_limits, cutidx))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                histos.append(future.result())
        print(f"Selections done in {time.time() - start_time} seconds")
        del df

        # Save the histograms
        outfile = TFile(outfilename, "update")
        subDir = outfile.mkdir(f"Pt{ptmin}_{ptmax}")
        subDir.cd()
        for histo in histos:
            histo.Write()
            del histo
        outfile.Close()
