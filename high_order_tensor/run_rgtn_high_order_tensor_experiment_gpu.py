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
from datetime import datetime

def analyze_and_log_history(search_history, logging):
    """Analyzes search history to find the best model within the specified RE range."""
    logging.info("\n" + "="*80)
    logging.info("CUSTOM RE-BOUNDED PERFORMANCE ANALYSIS")
    logging.info("="*80)

    if not search_history:
        logging.warning("Search history is empty. No analysis to perform.")
        return

    # User-specified RE range
    re_min_bound = 0.002
    re_max_bound = 0.03
    
    best_result_in_range = {
        "re": float('inf'),
        "cr": float('inf'),
        "result": None
    }

    for entry in search_history:
        re = entry['re']
        cr = entry['cr']
        
        # Check if the result is within the desired RE range
        if re_min_bound <= re < re_max_bound:
            # If it is, check if it offers a better (lower) CR
            if cr < best_result_in_range['cr']:
                best_result_in_range['cr'] = cr
                best_result_in_range['re'] = re
                best_result_in_range['result'] = entry

    logging.info(f"\n--- Best Result within RE range [{re_min_bound}, {re_max_bound}) ---")
    if best_result_in_range['result']:
        result = best_result_in_range['result']
        logging.info(f"  - Found Best CR: {result['cr']:.6f} (or {result['cr']:.4%})")
        logging.info(f"  - With RE:       {result['re']:.6f}")
        logging.info(f"  - Params:        {result['params']:,}")
        logging.info(f"  - Config:        {result['config']}")
    else:
        logging.info(f"  - No model found that satisfies the RE requirement {re_min_bound} <= RE < {re_max_bound}.")
            
    logging.info("\n" + "="*80)

def setup_logging(order, run_name_prefix="rgtn_multiscale"):
    """Initializes logging to file and console, incorporating the tensor order."""
    log_dir = "../logs"  # Relative to the script's location
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # New log filename format: order_X_prefix_timestamp.log
    log_filename = f"order_{order}_{run_name_prefix}_{current_time}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    
    # Use basicConfig to set up handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ],
        force=True  # Force re-configuration, useful in interactive environments
    )
    
    logging.info(f"--- Starting RGTN High-Order Tensor Experiment (Order: {order}) ---")
    logging.info(f"--- Log file: {log_filename} ---")

def run_experiment(args):
    """Main experiment function."""
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    tracemalloc.start()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}, Args: {args}")

    # --- Load Data ---
    print(f"\n--- Loading Data from {args.data_path} ---")
    try:
        mat_data = scipy.io.loadmat(args.data_path)
        raw_tensor = torch.from_numpy(mat_data['tensor']).float()
        
        # We assume the tensor in the .mat file has the correct dimension order.
        # The specific permutation for lightfield data has been removed for generality.
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
    initial_topo_path = os.path.join(results_dir, "initial_topology.png")
    initial_model.plot_topology(save_path=initial_topo_path, title="Data-Driven Initial Topology")
    logging.info(f"Saved initial topology to {initial_topo_path}")

    # --- Run RGTN Multi-Scale Search ---
    rgtn_search = RGTN(initial_model=initial_model, target_data=target_tensor)
    
    logging.info("\n" + "="*60)
    logging.info("Starting RGTN Multi-Scale Search")
    logging.info("="*60)

    start_time = time.time()
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
    end_time = time.time()
    total_time_seconds = end_time - start_time

    # --- Log Results and Save Final Model ---
    final_solution = rgtn_search.best_solution
    logging.info("\n" + "="*80)
    logging.info("RGTN Multi-Scale Search Finished")
    logging.info("="*80)
    logging.info(f"Total Experiment Time: {total_time_seconds:.2f} seconds ({total_time_seconds / 60:.2f} minutes)")
    logging.info(f"Best Solution Found:")
    logging.info(f"  - Final RE:     {final_solution['re']:.6f}")
    logging.info(f"  - Final CR:     {final_solution['cr']:.6%}")
    logging.info(f"  - Final Loss:   {final_solution['loss']:.6f}")
    
    if final_model:
        final_topo_path = os.path.join(results_dir, "final_topology.png")
        final_model.plot_topology(save_path=final_topo_path, title="Final Optimized Topology")
        logging.info(f"Saved final topology to {final_topo_path}")
    else:
        logging.warning("Search did not yield a final model.")

    analyze_and_log_history(rgtn_search.search_history, logging)

    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logging.info(f"Memory Usage: Current {mem_current / 1e6:.2f}MB, Peak {mem_peak / 1e6:.2f}MB")


def main():
    parser = argparse.ArgumentParser(description="RGTN Multi-Scale Experiment for High-Order Tensors")
    
    # --- Data and Setup ---
    parser.add_argument('--data_path', type=str, default='data/structured_6th_order_tensor.mat', help='Path to the .mat file containing the tensor.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--order', type=int, default=6, help='Order of the tensor (e.g., 6 for 6th-order tensor).')
    
    # --- RGTN Search Parameters ---
    parser.add_argument('--max_scales', type=int, default=4, help='Number of coarse-graining scales.')
    parser.add_argument('--expansion_steps', type=int, default=20, help='Number of expansion steps per scale.')
    parser.add_argument('--compression_steps', type=int, default=20, help='Number of compression steps per scale.')
    parser.add_argument('--re_target', type=float, default=0.01, help='Target relative error, matching the paper.')
    parser.add_argument('--cr_target', type=float, default=0.0, help='Target compression ratio (set to 0 as we primarily target RE).')
    parser.add_argument('--lambda_re', type=float, default=10.0, help='Increased penalty for RE to focus on achieving the target.')

    # --- Model and Optimization Parameters ---
    parser.add_argument('--initial_bond_dim', type=int, default=2, help='Initial bond dimension for data-driven model.')
    parser.add_argument('--svd_threshold', type=float, default=1e-3, help='SVD threshold for data-driven initialization.')
    parser.add_argument('--epochs_per_proposal', type=int, default=200, help='Number of optimization epochs for each proposal.')
    parser.add_argument('--timeout', type=int, default=90, help='Timeout in seconds for a single proposal optimization.')

    args = parser.parse_args()
    
    # Setup logging right after parsing args
    setup_logging(order=args.order)
    
    run_experiment(args)

if __name__ == '__main__':
    main() 