import pandas as pd
import ROOT
import argparse
import yaml
import os
import uproot
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/fchinu/Run3/ThesisUtils/')
from DfUtils import split_and_dump_parquet

def download_filenames_from_grid(grid, input_filename, output_filename):
    with open(input_filename, "r") as f:
        input_files = f.readlines()
        input_files = [x.strip() for x in input_files]

    file_list = []

    for file in input_files:
        grid_result = grid.Ls(file)
        file_info_list = grid_result.GetFileInfoList()

        for i in range(file_info_list.GetSize()):
            directory_name = grid_result.GetFileName(i)

            if "aod_collection" not in directory_name and "analysis" not in directory_name and "full_config" not in directory_name:
                file_list.append(file + '/' + directory_name)

    with open(output_filename, "w") as f:
        for file in file_list:
            f.write(file + "\n")
        
    return len(file_list)

def download_analysis_results(input_list_filename, output_directory, n_files_max):

    merge_analysis_results = ROOT.TFileMerger(False)
    count = 0

    with open(input_list_filename, "r") as file:
        for line in file:
            count += 1
            if count > n_files_max:
                break

            file_path = line.strip()
            merge_analysis_results.AddFile(f"alien://{file_path}/AnalysisResults.root")

    analysis_output_filename = output_directory + "_AnalysisResults.root"

    merge_analysis_results.OutputFile(analysis_output_filename, "RECREATE")
    merge_analysis_results.Merge()

def download_aod(input_list_filename, output_directory, n_files_max):
    
    merge_aod = ROOT.TFileMerger(False)
    count = 0

    with open(input_list_filename, "r") as file:
        for line in file:
            count += 1
            if count > n_files_max:
                break

            file_path = line.strip()
            merge_aod.AddFile(f"alien://{file_path}/AO2D.root")

    analysis_output_filename = output_directory + "_AO2D.root"

    merge_aod.OutputFile(analysis_output_filename, "RECREATE")
    merge_aod.Merge()

def convert_aod_to_parquet(input_filename, output_filename, treename, nThreads = 3, selections = None, train_frac = 0.8, isMC = False):
    
    InputFile = ROOT.TFile(input_filename,'READ')
    DirIter = InputFile.GetListOfKeys()
    DirNames = [i.GetName() for i in DirIter if "parentFiles" not in i.GetName()]
    FileName = [input_filename] * len(DirNames)
    TreeName = treename
    TreeName = [TreeName] * len(DirNames)
    InputFile.Close()

    if os.path.exists(output_filename):
        print("File already exists. Removing it...")
        os.remove(output_filename)

    inputs = [f'{inFile}:{inDir}/{inTree}' for (inFile, inDir, inTree) in zip(FileName, DirNames, TreeName)]
    executor = ThreadPoolExecutor(nThreads)
    iterator = uproot.iterate(inputs, library='pd', decompression_executor=executor,
                            interpretation_executor=executor)

    print("Converting to parquet...", end="\r")
    for data in iterator:
        if selections:
            data = data.query(selections)
        if os.path.exists(output_filename):
            data.to_parquet(output_filename, engine='fastparquet', append=True)
        else:
            data.to_parquet(output_filename, engine='fastparquet')
    del data, iterator, executor
    print("Converting to parquet... Done!")
    
    if isMC:
        print("Splitting MC dataframe...", end="\r")
        df = pd.read_parquet(output_filename)
        df = df.query('abs(fFlagMcMatchRec) == 4')                                      # Only matched candidates
        
        if train_frac < 1:
            suffixes = ["_Train.parquet", "_Eff.parquet"]
            PromptDs = df.query('fOriginMcRec == 1 and fFlagMcDecayChanRec == 1')
            split_and_dump_parquet(PromptDs, output_filename.replace('.parquet', '_PromptDs'), suffixes, train_frac)
            del PromptDs
            NonPromptDs = df.query('fOriginMcRec == 2 and fFlagMcDecayChanRec == 1')
            split_and_dump_parquet(NonPromptDs, output_filename.replace('.parquet', '_NonPromptDs'), suffixes, train_frac)
            del NonPromptDs
            PromptDplus = df.query('fOriginMcRec == 1 and fFlagMcDecayChanRec == 3')
            split_and_dump_parquet(PromptDplus, output_filename.replace('.parquet', '_PromptDplus'), suffixes, train_frac)
            del PromptDplus
            NonPromptDplus = df.query('fOriginMcRec == 2 and fFlagMcDecayChanRec == 3')
            split_and_dump_parquet(NonPromptDplus, output_filename.replace('.parquet', '_NonPromptDplus'), suffixes, train_frac)
            del NonPromptDplus, df
        else:
            PromptDs = df.query('fOriginMcRec == 1 and fFlagMcDecayChanRec == 1')
            PromptDs.to_parquet(output_filename.replace('.parquet', '_PromptDs.parquet'), engine='fastparquet')
            del PromptDs
            NonPromptDs = df.query('fOriginMcRec == 2 and fFlagMcDecayChanRec == 1')
            NonPromptDs.to_parquet(output_filename.replace('.parquet', '_NonPromptDs.parquet'), engine='fastparquet')
            del NonPromptDs
            PromptDplus = df.query('fOriginMcRec == 1 and fFlagMcDecayChanRec == 3')
            PromptDplus.to_parquet(output_filename.replace('.parquet', '_PromptDplus.parquet'), engine='fastparquet')
            del PromptDplus
            NonPromptDplus = df.query('fOriginMcRec == 2 and fFlagMcDecayChanRec == 3')
            NonPromptDplus.to_parquet(output_filename.replace('.parquet', '_NonPromptDplus.parquet'), engine='fastparquet')
            del NonPromptDplus, df
        print("Splitting MC dataframe... Done!")
    else:
        if train_frac < 1:
            print("Splitting dataframe...", end="\r")
            df = pd.read_parquet(output_filename)
            suffixes = ["_Train.parquet", "_Eff.parquet"]
            split_and_dump_parquet(df, output_filename, suffixes, train_frac)
            del df            
            print("Splitting dataframe... Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create list of files from grid and merge files.')
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML configuration file', required=True)
    parser.add_argument('--aod', action='store_true', default=False, help='Run only the AOD download and merge')
    parser.add_argument('--analysis', action='store_true', default=False, help='Run only the analysis results download and merge')
    parser.add_argument('--parquet', action='store_true', default=False, help='Run only the conversion to Parquet')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)


    names_config = config['download']
    merge_config = config['merge']
    parquet_config = config['convert_to_parquet']


    if args.analysis:
        grid = ROOT.TGrid.Connect("alien://")
        if not grid:
            print("No grid connection available. Exiting.")
            exit(1)
        nFiles = download_filenames_from_grid(grid, names_config['input'], names_config['output'])
        download_analysis_results(merge_config['input'], merge_config['output'], merge_config['max_files'])
    elif args.aod:
        grid = ROOT.TGrid.Connect("alien://")
        if not grid:
            print("No grid connection available. Exiting.")
            exit(1)
        nFiles = download_filenames_from_grid(grid, names_config['input'], names_config['output'])
        download_aod(merge_config['input'], merge_config['output'], merge_config['max_files'])
    elif args.parquet:
        convert_aod_to_parquet(parquet_config['input'], parquet_config['output'], \
                                parquet_config['treename'], parquet_config['nThreads'], \
                                parquet_config['selections'], parquet_config['train_fraction'], parquet_config['isMC'])
    else:
        grid = ROOT.TGrid.Connect("alien://")
        if not grid:
            print("No grid connection available. Exiting.")
            exit(1)
        nFiles = download_filenames_from_grid(grid, names_config['input'], names_config['output'])
        
        if merge_config['input'] is None:
            merge_config['input'] = names_config['output']
        if parquet_config['input'] is None:
            parquet_config['input'] = merge_config['output'] + '_AO2D.root'

        print(f"{min(nFiles, merge_config['max_files'])} files contained in the {merge_config['input']} will be merged. Press enter to continue.")
        input()

        download_analysis_results(merge_config['input'], merge_config['output'], merge_config['max_files'])
        download_aod(merge_config['input'], merge_config['output'], merge_config['max_files'])
        convert_aod_to_parquet(parquet_config['input'], parquet_config['output'], \
                                parquet_config['treename'], parquet_config['nThreads'], \
                                parquet_config['selections'], parquet_config['train_fraction'], parquet_config['isMC'])