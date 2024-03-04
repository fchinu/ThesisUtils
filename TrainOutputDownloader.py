import ROOT
import argparse
import yaml
import os
import uproot
from concurrent.futures import ThreadPoolExecutor

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

def convert_aod_to_parquet(input_filename, output_filename, treename, nThreads = 3, selections = None):
    
    InputFile = ROOT.TFile(input_filename,'READ')
    DirIter = InputFile.GetListOfKeys()
    DirNames = [i.GetName() for i in DirIter if "parentFiles" not in i.GetName()]
    FileName = [input_filename] * len(DirNames)
    TreeName = treename
    TreeName = [TreeName] * len(DirNames)
    InputFile.Close()

    if os.path.exists(output_filename + ".parquet"):
        print("File already exists. Removing it...")
        os.remove(output_filename + ".parquet")

    inputs = [f'{inFile}:{inDir}/{inTree}' for (inFile, inDir, inTree) in zip(FileName, DirNames, TreeName)]
    executor = ThreadPoolExecutor(nThreads)
    iterator = uproot.iterate(inputs, library='pd', decompression_executor=executor,
                            interpretation_executor=executor)
    for data in iterator:
        if selections:
            data = data.query(selections)
        if os.path.exists(output_filename + ".parquet"):
            data.to_parquet(output_filename + ".parquet", engine='fastparquet', append=True)
        else:
            data.to_parquet(output_filename + ".parquet", engine='fastparquet')
    del data, iterator, executor
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create list of files from grid and merge files.')
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML configuration file', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    grid = ROOT.TGrid.Connect("alien://")
    if not grid:
        print("No grid connection available. Exiting.")
        exit(1)

    nFiles = download_filenames_from_grid(grid, config['download']['input'], config['download']['output'])

    if config['merge']['input'] is None:
        config['merge']['input'] = config['download']['output']
    if config['convert_to_parquet']['input'] is None:
        config['convert_to_parquet']['input'] = config['merge']['output']

    print(f"{min(nFiles, config['merge']['max_files'])} files contained in the {config['merge']['input']} will be merged. Press enter to continue.")
    input()

    #download_analysis_results(config['merge']['input'], config['merge']['output'], config['merge']['max_files'])
    #download_aod(config['merge']['input'], config['merge']['output'], config['merge']['max_files'])
    convert_aod_to_parquet(config['convert_to_parquet']['input'] + '_AO2D.root', config['convert_to_parquet']['output'], config['convert_to_parquet']['treename'], config['convert_to_parquet']['nThreads'], config['convert_to_parquet']['selections'])