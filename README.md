# Paper Code Repository

This directory contains the cleaned and organized code for the RGTN paper submission. All Chinese comments and emoji symbols have been converted to English for academic publication standards.

## Directory Structure

```
paper_code/
├── lightfield/                    # Light field data compression experiment
│   ├── run_rgtn_lightfield_experiment_gpu.py
│   └── tensor_network.py
├── high_order_tensor/             # High-order tensor decomposition experiment
│   ├── run_rgtn_high_order_tensor_experiment_gpu.py
│   ├── tensor_network.py
│   └── generate_data.py
└── video/                         # Video completion experiment
    ├── run_intelligent_rgtn_experiment.py
    ├── tensor_network.py
    ├── utils.py
    └── gpu_optimized_rgtn.py
```

## Experiment Descriptions

### Light Field Experiment
- **Purpose**: Multi-scale compression of light field data using RGTN
- **Dataset**: Truck light field dataset (5D tensor: [3, 9, 9, 40, 60])
- **Key Metrics**: Reconstruction Error (RE) and Compression Ratio (CR)
- **Main File**: `run_rgtn_lightfield_experiment_gpu.py`

### High-Order Tensor Experiment
- **Purpose**: Scalability testing with synthetic high-order tensors
- **Data**: 6th, 8th, and 10th-order synthetic tensors
- **Key Metrics**: Reconstruction Error (RE) and computational efficiency
- **Main Files**: 
  - `run_rgtn_high_order_tensor_experiment_gpu.py`
  - `generate_data.py` (for synthetic tensor generation)

### Video Completion Experiment
- **Purpose**: Missing data recovery in video sequences
- **Dataset**: Standard video sequences with 90% missing entries
- **Key Metrics**: PSNR and MPSNR (Mean Peak Signal-to-Noise Ratio)
- **Main Files**:
  - `run_intelligent_rgtn_experiment.py`
  - `gpu_optimized_rgtn.py` (optimized RGTN implementation)
  - `utils.py` (utility functions)

## Code Standards

- All comments and print statements are in English
- No emoji symbols or non-ASCII characters
- Consistent code formatting and documentation
- Academic publication ready

## Usage

Each experiment directory contains the necessary files to reproduce the results. Follow the individual README files in each subdirectory for specific instructions.

## Dependencies

- PyTorch
- NumPy
- SciPy
- NetworkX
- Matplotlib
- CUDA (for GPU acceleration)

## Citation

When using this code, please cite the corresponding paper. 