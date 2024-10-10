import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import yaml
from flarefly.fitter import F2MassFitter
from flarefly.data_handler import DataHandler
import uproot
import ROOT

def draw_selection_scan(df, cfg, i_pt):
    pt_min = cfg['pt_mins'][i_pt]
    pt_max = cfg['pt_maxs'][i_pt]

    fig, ax = plt.subplots(4, 2, figsize=(16, 16))

    cols_to_plot = [
        "efficiencies_ds_prompt", "efficiencies_ds_fd",
        "expected_signals_ds_prompt", "expected_bkgs_ds_prompt",
        "expected_significances_ds_prompt", "expected_significances_dplus_prompt",
        "prompt_fracs_ds", "non_prompt_fracs_ds"
    ]

    cols_names = [
        r"Efficiency $\mathrm{D_s^+}$ prompt", r"Efficiency $\mathrm{D_s^+}$ non-prompt",
        r"Expected signal $\mathrm{D_s^+}$ prompt", r"Expected background $\mathrm{D_s^+}$ prompt",
        r"Expected significance $\mathrm{D_s^+}$ prompt", r"Expected significance $\mathrm{D^+}$ prompt",
        r"Prompt fraction $\mathrm{D_s^+}$", r"Non-prompt fraction $\mathrm{D_s^+}$"
    ]

    x_axis = "ML_output_Bkg"
    x_axis_name = "BDT background score <"
    y_axis = "ML_output_Prompt"
    y_axis_name = "BDT prompt score >"

    for i, (col, col_name) in enumerate(zip(cols_to_plot, cols_names)):
        df_pivot = df.pivot(index=y_axis, columns=x_axis, values=col)
        sns.heatmap(
            df_pivot, ax=ax[i // 2, i % 2], cmap='coolwarm', cbar=True,
            norm=matplotlib.colors.LogNorm() if 'efficiencies' in col else None)
        # Get the actual x and y labels
        x_labels = ['' for _ in df_pivot.columns]
        y_labels = ['' for _ in df_pivot.index]
        x_labels[0] = f"{df_pivot.columns[0]:.2f}"
        x_labels[-1] = f"{df_pivot.columns[-1]:.2f}"
        y_labels[0] = f"{df_pivot.index[0]:.2f}"
        y_labels[-1] = f"{df_pivot.index[-1]:.2f}"

        # Set the x and y tick labels
        ax[i // 2, i % 2].invert_yaxis()
        ax[i // 2, i % 2].set_xticks(np.arange(len(x_labels)) + 0.5)  # Set x ticks to center the labels
        ax[i // 2, i % 2].set_xticklabels(x_labels)
        ax[i // 2, i % 2].set_yticks(np.arange(len(y_labels)) + 0.5)  # Set y ticks to center the labels
        ax[i // 2, i % 2].set_yticklabels(y_labels)
        
        ax[i // 2, i % 2].set_title(col_name, fontsize=25)
        ax[i // 2, i % 2].set_xlabel(x_axis_name, fontsize=20)
        ax[i // 2, i % 2].set_ylabel(y_axis_name, fontsize=20)
        ax[i // 2, i % 2].tick_params(axis='both', which='both', labelsize=20)
        ax[i // 2, i % 2].tick_params(axis='x', which='both')
        cbar = ax[i // 2, i % 2].collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        if cfg["working_points"] is not None:
            x_on_plot = (cfg["working_points"][x_axis][i_pt] - df_pivot.columns.min()) / (df_pivot.columns.max() - df_pivot.columns.min())
            x_on_plot *= len(df_pivot.columns)
            y_on_plot = (cfg["working_points"][y_axis][i_pt] - df_pivot.index.min()) / (df_pivot.index.max() - df_pivot.index.min())
            y_on_plot *= len(df_pivot.index)
            ax[i // 2, i % 2].scatter(
                x_on_plot,
                y_on_plot,
                color='black', 
                marker='X', s=100
            )


    fig.tight_layout()
    fig.savefig(
        os.path.join(
            cfg['output_dir'],
            f"selection_scan_pt{pt_min * 10:.0f}_{pt_max * 10:.0f}.pdf"
        )
    )
    plt.close(fig)

def dump_results(results, cfg, pt_min, pt_max):

    out_dicts = []
    for result in results:
        results_dict = result[0].result()
        out_dict = {}
        for var in results_dict:
            for particle_origin in results_dict[var]:
                out_dict[f"{var}_{particle_origin}"] = results_dict[var][particle_origin]
    
        selection = result[1]
        selections = selection.split(" and ")
        for sel in selections:        
            out_dict[sel.split(" ")[0]] = float(sel.split(" ")[2])

        out_dicts.append(out_dict)
        
    out_df = pd.DataFrame(out_dicts)
    out_file = os.path.join(
        cfg['output_dir'],
        f"selection_scan_pt{pt_min * 10:.0f}_{pt_max * 10:.0f}.parquet"
    )
    out_df.to_parquet(out_file)

    return out_df

def load_data(cfg_input):
    data_files = cfg_input['background']['filename']
    mc_ds_prompt_files = cfg_input['signal']['prompt']['filename']
    mc_ds_fd_files = cfg_input['signal']['feeddown']['filename']
    mc_dplus_prompt_files = cfg_input['secpeak']['prompt']['filename']
    mc_dplus_fd_files = cfg_input['secpeak']['feeddown']['filename']

    df_data = pd.concat([pd.read_parquet(f) for f in data_files])
    df_mc_ds_prompt = pd.concat([pd.read_parquet(f) for f in mc_ds_prompt_files])
    df_mc_ds_fd = pd.concat([pd.read_parquet(f) for f in mc_ds_fd_files])
    df_mc_dplus_prompt = pd.concat([pd.read_parquet(f) for f in mc_dplus_prompt_files])
    df_mc_dplus_fd = pd.concat([pd.read_parquet(f) for f in mc_dplus_fd_files])

    return {
        'data': df_data, 'mc_ds_prompt': df_mc_ds_prompt, 'mc_ds_fd': df_mc_ds_fd,
        'mc_dplus_prompt': df_mc_dplus_prompt, 'mc_dplus_fd': df_mc_dplus_fd
    }


def load_predictions(cfg_predictions):
    """
    Load predictions from a ROOT file.

    Parameters:
    - cfg_predictions (dict): Configuration dictionary.

    Returns:
    - dict: A dictionary with the following keys:
        - 'ds_prompt': The differential cross-section 
            uproot.Model_TH1D_v3 histogram for Ds prompt.
        - 'ds_feeddown': The differential cross-section 
            uproot.Model_TH1D_v3 histogram for Ds feeddown.
        - 'dplus_prompt': The differential cross-section 
            uproot.Model_TH1D_v3 histogram for D+ prompt.
        - 'dplus_feeddown': The differential cross-section 
            uproot.Model_TH1D_v3 histogram for D+ feeddown.
    """
    with uproot.open(cfg_predictions['crosssec']['filename']) as f:
        prediction_ds_prompt = f[cfg_predictions['crosssec']['histonames']['ds']['prompt']]
        prediction_ds_fd = f[cfg_predictions['crosssec']['histonames']['ds']['feeddown']]
        prediction_dplus_prompt = f[cfg_predictions['crosssec']['histonames']['dplus']['prompt']]
        prediction_dplus_fd = f[cfg_predictions['crosssec']['histonames']['dplus']['feeddown']]

    return {
        'ds_prompt': prediction_ds_prompt,
        'ds_feeddown': prediction_ds_fd,
        'dplus_prompt': prediction_dplus_prompt,
        'dplus_feeddown': prediction_dplus_fd
    }

def __get_prediction(predictions_hist, particle_origin, pt_min, pt_max, br_corr):
    particle, origin = particle_origin
    pt_bins = predictions_hist[f'{particle}_{origin}'].axis().edges()
    pt_min_idx = np.nonzero(np.isclose(pt_bins, pt_min))[0][0]
    pt_max_idx = np.nonzero(np.isclose(pt_bins, pt_max))[0][0]

    dsigma_dpt = 0.
    for i_pt in range(pt_min_idx, pt_max_idx):
        d_pt = pt_bins[i_pt + 1] - pt_bins[i_pt]
        dsigma_dpt += predictions_hist[f'{particle}_{origin}'].values()[i_pt] * d_pt

    return {f"{particle}_{origin}": dsigma_dpt / (pt_max - pt_min) * br_corr}

def get_pt_predictions(predictions_histos, pt_min, pt_max, cfg):
    """
    Calculate the differential cross-section (dsigma/dpt x BR) for prompt and feeddown
    components of specified particles within a specified transverse momentum (pt) range.

    Parameters:
    - predictions_histos (dict): A dictionary containing 'prompt' and 'feeddown' keys, 
        each associated with a uproot.Model_TH1D_v3 object containing the predictions.
    - pt_min (float): The minimum pt value of the range.
    - pt_max (float): The maximum pt value of the range.

    Returns:
        dict: A dictionary containing pT predictions for the specified particles and origins.
    """
    particles_origin = itertools.product(['ds', 'dplus'], ['prompt', 'feeddown'])

    predictions_pt = {}
    for particle_origin in particles_origin:
        br_corr = cfg['predictions']['crosssec']['histonames'][particle_origin[0]]['br_corr']
        predictions_pt.update(__get_prediction(predictions_histos, particle_origin, pt_min, pt_max, br_corr))

    return predictions_pt

def get_acceptance(cfg_acceptance):
    """
    Load the acceptance from a ROOT file.

    Parameters:
    - cfg_acceptance (dict): Configuration dictionary.
    """
    particles_origin = itertools.product(['Ds', 'Dplus'], ['Prompt', 'NonPrompt'])
    out_names = itertools.product(['ds', 'dplus'], ['prompt', 'fd'])
    h_gen = {}
    h_reco = {}
    with ROOT.TFile.Open(cfg_acceptance['filename']) as in_file:
        for particle_origin, out_name in zip(particles_origin, out_names):
            particle, origin = particle_origin
            sparse_gen = in_file.Get(f"hf-task-ds/MC/{particle}/{origin}/hPtYNPvContribGen")
            sparse_reco = in_file.Get(f"hf-task-ds/MC/{particle}/{origin}/hSparseMass")

            particle, origin = out_name
            h_gen[f"{particle}_{origin}"] = sparse_gen.Projection(cfg_acceptance['axispt_gen'], "EO")
            h_gen[f"{particle}_{origin}"].SetDirectory(0)
            h_reco[f"{particle}_{origin}"] = sparse_reco.Projection(cfg_acceptance['axispt_reco'], "EO")
            h_reco[f"{particle}_{origin}"].SetDirectory(0)

    return h_gen, h_reco

def get_pt_acceptance(acceptance_histos, pt_min, pt_max, cfg):
    h_gen, h_reco = acceptance_histos
    particles_origin = itertools.product(['ds', 'dplus'], ['prompt', 'fd'])
    acceptance = {}

    for particle_origin in particles_origin:
        particle, origin = particle_origin
        n_gen = h_gen[f"{particle}_{origin}"].Integral(
            h_gen[f"{particle}_{origin}"].FindBin(pt_min),
            h_gen[f"{particle}_{origin}"].FindBin(pt_max-0.001)
        )
        n_reco = h_reco[f"{particle}_{origin}"].Integral(
            h_reco[f"{particle}_{origin}"].FindBin(pt_min),
            h_reco[f"{particle}_{origin}"].FindBin(pt_max-0.001)
        )
        acceptance[f"{particle}_{origin}"] = n_reco / n_gen

    return acceptance

def get_selections(cut_vars, i_pt):
    selections = []
    var_selections = []
    for var in cut_vars:
        var_selections.append(np.linspace(cut_vars[var]['min'][i_pt], cut_vars[var]['max'][i_pt], cut_vars[var]['steps'][i_pt]).tolist())
    combined_selections = itertools.product(*var_selections)
    for selection in combined_selections:
        sel = ""
        for i, var in enumerate(cut_vars):
            if cut_vars[var]['upper_lower_cut'] == 'upper':
                sign = "<"
            else:
                sign = ">"
            sel += f'{var} {sign} {selection[i]} and '
        selections.append(sel[:-4])
    return selections


def get_trial(dfs, predictions, acceptance, selection, cfg, i_pt):
    return {
        'dfs': dfs,
        'predictions': predictions,
        'acceptance': acceptance,
        'selection': selection,
        'min_mass': cfg['min_mass'][i_pt],
        'max_mass': cfg['max_mass'][i_pt],
        'fraction_to_keep': cfg['infiles']['background']['fraction_to_keep'][i_pt],
        'i_pt': i_pt,
        'pt_min': cfg['pt_mins'][i_pt],
        'pt_max': cfg['pt_maxs'][i_pt]
    }


def load_data_handlers(trial):
    data_handlers = {}
    for df in trial['dfs']:
        if df == 'data':
            data_handlers[df] = DataHandler(
                trial['dfs'][df].sample(frac=trial['fraction_to_keep'], random_state=42),
                var_name='fM', limits=[trial['min_mass'], trial['max_mass']]
            )
        else:
            data_handlers[df] = DataHandler(
                trial['dfs'][df], var_name='fM',
                limits=[trial['min_mass'], trial['max_mass']]
            )
    return data_handlers


def load_mc_fitters(trial, data_handlers, cfg):
    """
    Load and configure fitters for different datasets based on the provided trial, data handlers, and configuration.

    Parameters:
    - trial (dict): A dictionary containing trial information.
    - data_handlers (dict): A dictionary of data handlers for different datasets.
    - cfg (dict): Configuration dictionary containing information about the background fit function.

    Returns:
    dict: A dictionary of configured fitters for each dataset.
    """
    fitters = {}
    for df in trial['dfs']:
        if df == 'data':
            continue # we will set it after the MC fits
        else:
            fitters[df] = F2MassFitter(
                data_handlers[df], name_signal_pdf=['gaussian'],
                name_background_pdf=["nobkg"],
                name=f"fit_{df}_pt{trial['pt_min'] * 10:.0f}_{trial['pt_max'] * 10:.0f}",
                verbosity=1, tol=1.e-1
            )

    fitters['mc_ds_prompt'].set_particle_mass(0, pdg_id=431)
    fitters['mc_ds_prompt'].set_signal_initpar(0, "sigma", 0.008, limits=[0., 0.1])
    fitters['mc_ds_prompt'].set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])

    fitters['mc_ds_fd'].set_particle_mass(0, pdg_id=431)
    fitters['mc_ds_fd'].set_signal_initpar(0, "sigma", 0.008, limits=[0., 0.1])
    fitters['mc_ds_fd'].set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])

    fitters['mc_dplus_prompt'].set_particle_mass(0, pdg_id=411)
    fitters['mc_dplus_prompt'].set_signal_initpar(0, "sigma", 0.008, limits=[0., 0.1])
    fitters['mc_dplus_prompt'].set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])

    fitters['mc_dplus_fd'].set_particle_mass(0, pdg_id=411)
    fitters['mc_dplus_fd'].set_signal_initpar(0, "sigma", 0.008, limits=[0., 0.1])
    fitters['mc_dplus_fd'].set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])

    return fitters


def get_efficiencies(trial):
    """
    Calculate the efficiencies.

    Parameters:
    - trial (dict): The considerd trial.

    Returns:
    - efficiencies (dict): A dictionary with the calculated efficiencies.
    """
    efficiencies = {}
    for df in trial['dfs']:
        if "mc" in df:
            sel_cands = len(trial['dfs'][df].query(trial['selection']))
            tot_cands = len(trial['dfs'][df])
            particle_origin = df.replace("mc_", "")
            efficiencies[particle_origin] = sel_cands / tot_cands
            efficiencies[particle_origin] *= trial['acceptance'][particle_origin]

    return efficiencies


def get_expected_signals(trial, cfg, efficiencies):
    """
    Calculate the expected signals.

    Parameters:
    - trial (dict): The considered trial.
    - cfg (dict): The configuration dictionary.
    - efficiencies (dict): The calculated efficiencies.

    Returns:
    - expected_signals (dict): A dictionary with the calculated expected signals.
    """
    expected_signals = {}
    int_lumi = cfg['n_expected_events'] / cfg['sigma_mb']
    d_pt = trial['pt_max'] - trial['pt_min']
    corr_factors = 2 * int_lumi * d_pt
    expected_signals['ds_prompt'] = efficiencies['ds_prompt'] * trial['predictions']['ds_prompt'] * corr_factors
    expected_signals['ds_fd'] = efficiencies['ds_fd'] * trial['predictions']['ds_feeddown'] * corr_factors
    expected_signals['dplus_prompt'] = efficiencies['dplus_prompt'] * trial['predictions']['dplus_prompt'] * corr_factors
    expected_signals['dplus_fd'] = efficiencies['dplus_fd'] * trial['predictions']['dplus_feeddown'] * corr_factors

    return expected_signals

def get_expected_bkgs(trial, cfg):
    for df in trial['dfs']:
        trial['dfs'][df] = trial['dfs'][df].copy()
        trial['dfs'][df] = trial['dfs'][df].query(trial['selection'])

    data_handlers = load_data_handlers(trial)
    fitters = load_mc_fitters(trial, data_handlers, cfg)
    # First, we fit the mc signal to get means and sigmas
    fitters['mc_ds_prompt'].mass_zfit()
    fitters['mc_dplus_prompt'].mass_zfit()
    ds_mean = fitters['mc_ds_prompt'].get_mass(0)[0]
    ds_sigma = fitters['mc_ds_prompt'].get_sigma(0)[0]
    dplus_mean = fitters['mc_dplus_prompt'].get_mass(0)[0]
    dplus_sigma = fitters['mc_dplus_prompt'].get_sigma(0)[0]
    n_sigma = cfg['infiles']['background']['n_sigma']
    limits = [
        [trial['min_mass'], dplus_mean - n_sigma * dplus_sigma],
        [ds_mean + n_sigma * ds_sigma, trial['max_mass']]
    ]

    fitters['data'] = F2MassFitter(
        data_handlers['data'], name_signal_pdf=['nosignal'],
        name_background_pdf=[cfg['infiles']['background']['fit_func']],
        name=f"fit_data_pt{trial['pt_min'] * 10:.0f}_{trial['pt_max'] * 10:.0f}_{trial['selection']}",
        limits=limits,
        verbosity=1, tol=1.e-1
    )
    if cfg['infiles']['background']['fit_func'] == "expo":
        fitters['data'].set_background_initpar(0, "lam", -4)
    elif bkg_func == "chebpol2":
        fitters['data'].set_background_initpar(0, "c0", 0.6)
        fitters['data'].set_background_initpar(0, "c1", -0.2)
        fitters['data'].set_background_initpar(0, "c2", 0.01)

    fitters['data'].mass_zfit()
    #fig, ax = fitters['data'].plot_mass_fit(figsize=(8, 8))
    #fig.savefig(f"/home/fchinu/Run3/Ds_pp_13TeV/Optimization/fits/mass_fit_pt{trial['pt_min'] * 10:.0f}_{trial['pt_max'] * 10:.0f}_{trial['selection']}.pdf")
    #plt.close(fig)
    scale_factor = cfg['n_expected_events'] / cfg['infiles']['background']['n_events'] / trial['fraction_to_keep']

    ds_bkg = fitters['data'].get_background(
        min=ds_mean - n_sigma * ds_sigma,
        max=ds_mean + n_sigma * ds_sigma
    )[0] * scale_factor

    dplus_bkg = fitters['data'].get_background(
        min=dplus_mean - n_sigma * dplus_sigma,
        max=dplus_mean + n_sigma * dplus_sigma
    )[0] * scale_factor

    bkg = {
        'ds_prompt': ds_bkg,
        'ds_fd': ds_bkg,
        'dplus_prompt': dplus_bkg,
        'dplus_fd': dplus_bkg
    }

    return bkg

def __get_signif(sgn, bkg):
    """
    Calculate the significance of a signal over background.

    Parameters:
    - sgn (float): The number of signal events.
    - bkg (float): The number of background events.

    Returns:
    - float: The significance of the signal.
    """
    return sgn / np.sqrt(sgn + bkg)

def get_expected_significances(expected_signals, expected_bkg):
    """
    Calculate the expected significances for different signal and background combinations.

    Parameters:
    - expected_signals (dict): A dictionary containing the expected signal counts.
    - expected_bkg (dict): A dictionary containing the expected background counts.

    Returns:
    - dict: A dictionary with the calculated significances for each signal type.
    """

    # No difference between expected_bkg for prompt and feeddown
    return {
        'ds_prompt': __get_signif(expected_signals['ds_prompt'], expected_bkg['ds_prompt']),
        'ds_fd': __get_signif(expected_signals['ds_fd'], expected_bkg['ds_fd']),
        'dplus_prompt': __get_signif(expected_signals['dplus_prompt'], expected_bkg['dplus_prompt']),
        'dplus_fd': __get_signif(expected_signals['dplus_fd'], expected_bkg['dplus_fd'])
    }

def get_fracs(expected_signals, trial):
    """
    Calculate the prompt and non-prompt fractions.

    Parameters:
    - expected_signals (dict): A dictionary containing the expected signal counts.
    - trial (dict): The considered trial.

    Returns:
    - tuple: A tuple containing the prompt and non-prompt fractions.
    """
    prompt_fracs = {}
    non_prompt_fracs = {}
    prompt_fracs['ds'] = expected_signals['ds_prompt'] / (expected_signals['ds_prompt'] + expected_signals['ds_fd'])
    non_prompt_fracs['ds'] = expected_signals['ds_fd'] / (expected_signals['ds_prompt'] + expected_signals['ds_fd'])
    prompt_fracs['dplus'] = expected_signals['dplus_prompt'] / (expected_signals['dplus_prompt'] + expected_signals['dplus_fd'])
    non_prompt_fracs['dplus'] = expected_signals['dplus_fd'] / (expected_signals['dplus_prompt'] + expected_signals['dplus_fd'])

    return prompt_fracs, non_prompt_fracs
    

def run_selection(trial, cfg):

    efficiencies = get_efficiencies(trial)
    expected_signals = get_expected_signals(trial, cfg, efficiencies)
    expected_bkgs = get_expected_bkgs(trial, cfg)
    expected_significances = get_expected_significances(expected_signals, expected_bkgs)
    prompt_fracs, non_prompt_fracs = get_fracs(expected_signals, trial)

    return {
        'efficiencies': efficiencies,
        'expected_signals': expected_signals,
        'expected_bkgs': expected_bkgs,
        'expected_significances': expected_significances,
        'prompt_fracs': prompt_fracs,
        'non_prompt_fracs': non_prompt_fracs
    }

def run_selection_scan(config_file_name, draw=False):
    with open(config_file_name, 'r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)

    pt_mins = cfg['pt_mins']
    pt_maxs = cfg['pt_maxs']

    dfs = load_data(cfg['infiles'])
    predictions = load_predictions(cfg['predictions'])
    acceptance_histos = get_acceptance(cfg['infiles']['acceptance'])

    for i_pt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        dfs_pt = {}
        if not draw:
            for df in dfs:
                dfs_pt[df] = dfs[df].query(f'{pt_min} < fPt < {pt_max}')
            predictions_pt = get_pt_predictions(predictions, pt_min, pt_max, cfg)
            acceptance_pt = get_pt_acceptance(acceptance_histos, pt_min, pt_max, cfg)
            selections = get_selections(cfg['cut_vars'], i_pt)

            results = []
            with ThreadPoolExecutor(max_workers=cfg['n_workers']) as executor:
                for selection in selections:
                    trial = get_trial(dfs_pt.copy(), predictions_pt, acceptance_pt, selection, cfg, i_pt)
                    results.append((executor.submit(run_selection, trial.copy(), cfg), selection))
            
            out_df = dump_results(results, cfg, pt_min, pt_max)
        else:
            infile = os.path.join(
                cfg['output_dir'],
                f"selection_scan_pt{pt_min * 10:.0f}_{pt_max * 10:.0f}.parquet"
            )
            out_df = pd.read_parquet(infile)
        draw_selection_scan(out_df, cfg, i_pt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a BDT selection scan')
    parser.add_argument('config_file', type=str, help='Configuration file')
    parser.add_argument('--draw', action='store_true', help='Just draw the selection scan')
    args = parser.parse_args()

    run_selection_scan(args.config_file, args.draw)