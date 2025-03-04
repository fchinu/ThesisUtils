'''
This script is used to download the output files from the grid,
merge them and convert them to parquet format.
Usage:
    python train_output_downloader.py <config_file> [--aod] [--analysis] [--parquet]
'''

import os
import json
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import ROOT
import uproot
from sklearn.model_selection import train_test_split
# pylint: disable=no-member


def split_dataset(df, config):
    """
    Split the dataset based on given selections and save the resulting dataframes as parquet files.

    Args:
        df (pandas.DataFrame): The input dataframe.
        config (dict): Configuration parameters.

    Returns:
        None
    """

    folder = "MC" if config["isMC"] else "Data"
    output_filename_no_suffix = config["output_directory"] + "/" + \
        folder + f"/Train{config['train_run']}/{config['suffix']}"

    for sel_name, selection in config["mc_selections"].items():
        df_sel = df.query(selection)
        df_sel.to_parquet(output_filename_no_suffix + f"_{sel_name}.parquet")
        if config["train_fraction"] < 1:
            df_train, df_eff = train_test_split(
                df_sel,
                train_size=config["train_fraction"],
                random_state=42
            )
            df_train.to_parquet(output_filename_no_suffix + f"_{sel_name}_Train.parquet")
            df_eff.to_parquet(output_filename_no_suffix + f"_{sel_name}_Eff.parquet")
            del df_train, df_eff
        del df_sel


def get_files_to_download(grid, config):
    """
    Retrieves a list of files to download based on the given grid and configuration.
    Args:
        grid (ROOT.TJAlien): The grid object used for file operations.
        config (dict): The configuration object containing input information.
    Returns:
        A list of files to download.
    """
    with open(config["input"], "r", encoding="utf8") as f:
        input_files = f.readlines()
        input_files = [x.strip() for x in input_files]  # Remove leading/trailing whitespaces

    file_list = []
    if config["is_slim"]:
        file_list = input_files
    else:
        for file in input_files:
            grid_result = grid.Ls(file)
            if not grid_result:
                print(f"\033[93mWARINING\033[0m: File {file} not found on the grid.")
                continue
            file_info_list = grid_result.GetFileInfoList()

            for i in range(file_info_list.GetSize()):
                directory_name = grid_result.GetFileName(i)

                if "aod_collection" not in directory_name and "analysis" \
                        not in directory_name and "full_config" not in directory_name:
                    file_list.append(file + '/' + directory_name)

    return file_list


def alien_copy_files(input_list_filename, output_directory):
    """
    Copies files from the grid to a local directory.

    Args:
        input_list_filename (list): List of input file names.
        output_directory (str): The output directory to copy the files to.

    Returns:
        None
    """
    for input_file in input_list_filename:
        i_file, file_name = input_file
        file_path = file_name.strip()  # Remove leading/trailing whitespaces
        os.system(f"alien_cp {file_path} file:{output_directory}/file_{i_file}.root")


def merge_save_analysis_results(input_list_filename, output_directory, config, i_job):
    """
    Merges multiple ROOT files into a single file and saves it to the specified output directory.

    Args:
        input_list_filename (list): List of filenames to be merged.
        output_directory (str): Directory where the merged file will be saved.
        config (dict): Configuration dictionary containing the suffix
            to be added to the output filename.
        i_job (int): Job index to be included in the output filename.

    Returns:
        None
    """
    str_input_files = ' '.join(str(i) for i in input_list_filename)
    os.system(
        f"hadd -f \
            {output_directory}/AnalysisResults_{config['suffix']}_{i_job}.root\
                {str_input_files}"
    )


def download_analysis_results(input_list_filename, config):
    """
    Downloads analysis results from input file names and merges them into a single output file.

    Args:
        input_list_filename (list): List of input file names.
        config (dict): Configuration dictionary containing parameters.

    Returns:
        None
    """
    # Create folder for output
    folder = "MC" if config["isMC"] else "Data"
    output_directory = config["output_directory"] + "/" + folder + f"/Train{config['train_run']}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    input_filenames_jobs = []
    for ifile, file in enumerate(input_list_filename):
        if ifile > config["max_files_to_download"]:
            break
        if ifile % config['n_jobs'] == 0:
            input_filenames_jobs.append([])
        input_filenames_jobs[-1].append((ifile, file + '/AnalysisResults.root'))

    analysis_results_partial_folder = os.path.join(output_directory, "hy")
    if not os.path.exists(analysis_results_partial_folder):
        os.makedirs(analysis_results_partial_folder)

    with ProcessPoolExecutor(max_workers=config['n_jobs']) as executor:
        for i_job, job_file_names in enumerate(input_filenames_jobs):
            executor.submit(alien_copy_files, job_file_names, analysis_results_partial_folder)

    input_filenames_jobs = []
    for ifile, file in enumerate(os.listdir(analysis_results_partial_folder)):
        if ifile % config['n_files_for_merge'] == 0:
            input_filenames_jobs.append([])
        input_filenames_jobs[-1].append(os.path.join(analysis_results_partial_folder, file))

    n_workers_merge = min(config['n_jobs'], len(input_filenames_jobs))
    with ProcessPoolExecutor(max_workers=n_workers_merge) as executor:
        for i_job, job_file_names in enumerate(input_filenames_jobs):
            executor.submit(
                merge_save_analysis_results,
                job_file_names, output_directory, config, i_job
            )
    executor.shutdown(wait=True)
    os.system(f"rm -rf {analysis_results_partial_folder}")


def download_aod(input_list_filename, config):
    """
    Downloads AOD files from a list of input file names and merges them into a single output file.
    Args:
        input_list_filename (str): The path to the file containing the list of input file names.
        config (dict): A dictionary containing configuration parameters.
    Returns:
        None
    """

    merge_aod = ROOT.TFileMerger(False)
    count = 0
    total_size = 0

    for file_name in input_list_filename:
        count += 1
        if count < config["start_file"]:
            continue
        if count > config["max_files_to_download"] + config["start_file"]:
            break

        file_path = file_name.strip()  # Remove leading/trailing whitespaces
        file = ROOT.TFile.Open(f"alien://{file_path}/AO2D.root")
        total_size += file.GetSize()
        merge_aod.AddAdoptFile(file)

    total_size = total_size / 1024 / 1024
    print(f"Total size of files to download: {total_size:.2f} MB")
    if (total_size / 1024) > 3:
        print("\033[93mWARINING\033[0m: Total size of files to download "
              "is greater than 3 GB. Are you sure you want to continue? (y/n)")
        answer = input()
        if answer.lower() != 'y':
            print("Exiting.")
            sys.exit()

    # Create folder for output
    folder = "MC" if config["isMC"] else "Data"
    output_directory = config["output_directory"] + "/" + folder + f"/Train{config['train_run']}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    analysis_output_filename = output_directory + f"/AO2D_{config['suffix']}.root"

    merge_aod.OutputFile(analysis_output_filename, "RECREATE")
    merge_aod.Merge()


def convert_aod_to_parquet(config):  # pylint: disable=too-many-locals, too-many-branches
    """
    Converts AOD (Analysis Object Data) file to Parquet format.
    Args:
        config (dict): Configuration parameters for the conversion process.
    Returns:
        None
    """

    folder = "MC" if config["isMC"] else "Data"
    input_filename = config["output_directory"] + "/" + folder + \
        f"/Train{config['train_run']}/AO2D_{config['suffix']}.root"
    output_filename = config["output_directory"] + "/" + folder + \
        f"/Train{config['train_run']}/{config['suffix']}.parquet"

    if os.path.exists(output_filename):
        print("File already exists. Removing it...")
        os.remove(output_filename)

    print("Converting to parquet...", end="\r")
    dfs = []
    with uproot.open(input_filename) as f:
        for folder_name, run_folder in f.items(recursive=False):  # Loop over the run folders
            if "DF" in folder_name:
                dfs_folder = []
                if "*" in config["tree_name"]:  # Loop over all trees in the folder
                    for obj_name, class_name in run_folder.classnames().items():
                        if "TTree" in class_name:
                            dfs_folder.append(run_folder[obj_name].arrays(library="pd"))
                else:
                    if not isinstance(config["tree_name"], list):
                        config["tree_name"] = [config["tree_name"]]
                    for tree_name in config["tree_name"]:
                        dfs_folder.append(run_folder[tree_name].arrays(library="pd"))
                dfs.append(pd.concat(dfs_folder, axis=1))
                del dfs_folder
                if config["selections"]:
                    dfs[-1] = dfs[-1].query(config["selections"])

    df = pd.concat(dfs)
    df.to_parquet(output_filename)
    print("Converting to parquet... Done!")

    if config["isMC"]:
        if config["mc_selections"]:
            split_dataset(df, config)

        if config["train_fraction"] < 1:
            df_train, df_eff = train_test_split(
                df,
                train_size=config["train_fraction"],
                random_state=42
            )
            df_train.to_parquet(output_filename.replace(".parquet", "_Train.parquet"))
            df_eff.to_parquet(output_filename.replace(".parquet", "_Eff.parquet"))
            del df


def download_files_from_grid(config, aod=False, analysis=False, parquet=False):
    """
    Downloads files from the grid based on the provided configuration.

    Args:
        config (object): The configuration object.
        aod (bool, optional): Flag indicating whether to download AOD files.
        analysis (bool, optional): Flag indicating whether to download analysis results.
        parquet (bool, optional): Flag indicating whether to convert AOD files to Parquet format.
    Note:
        If no flags are provided, all operations are performed.
    """

    if not aod and not analysis and not parquet:
        aod = True
        analysis = True
        parquet = True

    if aod or analysis:
        # Get files from grid
        grid = ROOT.TGrid.Connect("alien://")
        if not grid:
            print("No grid connection available. Exiting.")
            sys.exit()
        files_to_download = get_files_to_download(grid, config)

        if aod:
            download_aod(files_to_download, config)
        if analysis:
            download_analysis_results(files_to_download, config)
    if parquet:
        convert_aod_to_parquet(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create list of files from grid and merge files.')
    parser.add_argument('config_file', type=str,
                        help='Path to the JSON configuration file')
    parser.add_argument('--aod', action='store_true', default=False,
                        help='Run only the AOD download and merge')
    parser.add_argument('--analysis', action='store_true', default=False,
                        help='Run only the analysis results download and merge')
    parser.add_argument('--parquet', action='store_true', default=False,
                        help='Run only the conversion to Parquet')
    args = parser.parse_args()

    with open(args.config_file, encoding="utf8") as cfg_file:
        cfg = json.load(cfg_file)

    download_files_from_grid(cfg, aod=args.aod, analysis=args.analysis, parquet=args.parquet)
