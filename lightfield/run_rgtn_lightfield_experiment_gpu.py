import argparse
import logging
import os
import time
import torch
import scipy.io
import tracemalloc
import copy
import hashlib
import numpy as np
import random
from tensor_network import GraphTN, RGTN, initialize_model_from_data

def analyze_and_log_history(search_history, logging):
    """Analyzes search history to find best models for different RE tiers."""
    logging.info("\n" + "="*80)
    logging.info("CROSS-TIER PERFORMANCE ANALYSIS")
    logging.info("="*80)

    if not search_history:
        logging.warning("Search history is empty. No analysis to perform.")
        return

    tiers = {
        "0.05-0.08 level (0.05 <= RE < 0.08)": {"re_min": 0.05, "re_max": 0.08,  "best_cr": float('inf'), "result": None},
        "0.07-0.19 level (0.07 <= RE < 0.19)":  {"re_min": 0.07,  "re_max": 0.19,  "best_cr": float('inf'), "result": None},
    }

    for entry in search_history:
        re = entry['re']
        cr = entry['cr']
        
        for tier_name, tier_info in tiers.items():
            if tier_info['re_min'] <= re < tier_info['re_max']:
                if cr < tier_info['best_cr']:
                    tier_info['best_cr'] = cr
                    tier_info['result'] = entry

    for tier_name, tier_info in tiers.items():
        logging.info(f"\n--- Best Result for {tier_name} ---")
        if tier_info['result']:
            result = tier_info['result']
            logging.info(f"  - RE:     {result['re']:.6f}")
            logging.info(f"  - CR:     {result['cr']:.6f}")
            logging.info(f"  - Params: {result['params']:,}")
            logging.info(f"  - Config: {result['config']}")
        else:
            logging.info("  - No model found within this RE threshold.")
            
    logging.info("\n" + "="*80)

def setup_logging(run_name):
    """Initializes logging to file and console."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{run_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def run_experiment(args):
    """Main experiment function."""
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"lightfield_rgtn_multiscale_{timestamp}"
    log_file = setup_logging(run_name)
    
    logging.info(f"--- Starting RGTN Lightfield MULTI-SCALE Experiment ---")
    logging.info(f"--- Log file: {os.path.basename(log_file)} ---")
    
    tracemalloc.start()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}, Args: {args}")

    # --- Load Data ---
    print(f"\n--- Loading Data from {args.data_path} ---")
    try:
        mat_data = scipy.io.loadmat(args.data_path)
        raw_tensor = torch.from_numpy(mat_data['X']).float()
        
        if raw_tensor.ndim == 5:
            target_tensor = raw_tensor.permute(2, 3, 4, 0, 1)
        else:
            target_tensor = raw_tensor
    except Exception as e:
        logging.error(f"Failed to load or process data: {e}")
        return

    target_tensor = target_tensor.to(device)
    logging.info(f"Target tensor loaded successfully. Shape: {target_tensor.shape}")

    # --- Create Initial Model using Data-Driven Initialization ---
    initial_model = initialize_model_from_data(
        target_data=target_tensor,
        num_nodes=len(target_tensor.shape),  # One node per dimension initially
        svd_rank_threshold=args.svd_threshold,
        initial_bond_dim=args.initial_bond_dim
    )
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    initial_topo_path = os.path.join(results_dir, f"topology_initial_{run_name}.png")
    initial_model.plot_topology(save_path=initial_topo_path, title=f"Data-Driven Initial Topology for {run_name}")
    logging.info(f"Saved initial topology to {initial_topo_path}")

    # --- Run RGTN Multi-Scale Search ---
    rgtn_search = RGTN(initial_model=initial_model, target_data=target_tensor)
    
    logging.info("\n" + "="*60)
    logging.info("Starting RGTN Multi-Scale Search")
    logging.info("="*60)

    final_model = rgtn_search.search(
        max_scales=args.max_scales,
        expansion_steps=args.expansion_steps,
        compression_steps=args.compression_steps,
        re_target=args.re_target,
        cr_target=args.cr_target,
        lambda_re=args.lambda_re,
        epochs_per_proposal=args.epochs_per_proposal,
        timeout_per_proposal=args.timeout
    )

    # --- Log Results and Save Final Model ---
    final_solution = rgtn_search.best_solution
    logging.info("\n" + "="*80)
    logging.info("RGTN Multi-Scale Search Finished")
    logging.info("="*80)
    logging.info(f"Best Solution Found:")
    logging.info(f"  - Final RE:     {final_solution['re']:.6f}")
    logging.info(f"  - Final CR:     {final_solution['cr']:.6%}")
    logging.info(f"  - Final Loss:   {final_solution['loss']:.6f}")
    
    if final_model:
        final_topo_path = os.path.join(results_dir, f"topology_final_{run_name}.png")
        final_model.plot_topology(save_path=final_topo_path, title=f"Final Optimized Topology for {run_name}")
        logging.info(f"Saved final topology to {final_topo_path}")
    else:
        logging.warning("Search did not yield a final model.")

    analyze_and_log_history(rgtn_search.search_history, logging)

    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logging.info(f"Memory Usage: Current {mem_current / 1e6:.2f}MB, Peak {mem_peak / 1e6:.2f}MB")


def main():
    parser = argparse.ArgumentParser(description="RGTN Multi-Scale Experiment for Lightfield Data")
    
    # --- Data and Setup ---
    parser.add_argument('--data_path', type=str, default='data/Bunny.mat', help='Path to the .mat file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    # --- RGTN Search Parameters ---
    parser.add_argument('--max_scales', type=int, default=4, help='Number of coarse-graining scales.')
    parser.add_argument('--expansion_steps', type=int, default=15, help='Number of expansion steps per scale.')
    parser.add_argument('--compression_steps', type=int, default=15, help='Number of compression steps per scale.')
    parser.add_argument('--re_target', type=float, default=0.08, help='Target reconstruction error for the final model.')
    parser.add_argument('--cr_target', type=float, default=0.1, help='Target compression ratio for the final model.')
    parser.add_argument('--lambda_re', type=float, default=2.0, help='Penalty factor for RE in the loss function.')

    # --- Model and Optimization Parameters ---
    parser.add_argument('--initial_bond_dim', type=int, default=4, help='Initial bond dimension for data-driven model.')
    parser.add_argument('--svd_threshold', type=float, default=1e-3, help='SVD threshold for data-driven initialization.')
    parser.add_argument('--epochs_per_proposal', type=int, default=150, help='Number of optimization epochs for each proposal.')
    parser.add_argument('--timeout', type=int, default=90, help='Timeout in seconds for a single proposal optimization.')

    args = parser.parse_args()
    run_experiment(args)

if __name__ == '__main__':
    main() 