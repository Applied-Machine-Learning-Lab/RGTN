import torch
import numpy as np
import json
import logging
import os
from datetime import datetime
import sys
import tracemalloc
import subprocess
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

def setup_logging(experiment_name):
    """Sets up the logging for an experiment."""
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    log_filename = os.path.join(logs_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def get_relative_error(tensor1, tensor2):
    """Computes the relative error between two tensors."""
    return torch.norm(tensor1 - tensor2) / (torch.norm(tensor2) + 1e-9)

def psnr(img1, img2, max_val=1.0):
    """Computes the Peak Signal-to-Noise Ratio between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def load_video_as_tensor(video_path, output_dir="temp_frames"):
    """
    Loads a video file, extracts its frames into a temporary directory,
    and converts them into a PyTorch tensor of shape (F, C, H, W).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Use ffmpeg to extract frames
    # The command `-y` overwrites output files without asking.
    # `%04d.png` saves frames as 0001.png, 0002.png, etc.
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        os.path.join(output_dir, '%04d.png')
    ]
    
    print(f"Running ffmpeg to extract frames from {video_path}...")
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Frame extraction complete.")

    # Load frames into a tensor
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    if not frame_files:
        raise ValueError("No frames were extracted. Check the video file or ffmpeg command.")
        
    frames = []
    to_tensor_transform = ToTensor()
    for frame_file in frame_files:
        img_path = os.path.join(output_dir, frame_file)
        img = Image.open(img_path).convert('RGB')
        frames.append(to_tensor_transform(img))
    
    # Stack frames along a new dimension (F, C, H, W)
    video_tensor = torch.stack(frames)
    
    # For this experiment, we'll work with (F, H, W, C) for easier dimension mapping
    # to tensor network nodes. So we permute C to the end.
    video_tensor = video_tensor.permute(0, 2, 3, 1)

    print(f"Video loaded as tensor with shape: {video_tensor.shape}")
    return video_tensor


def create_random_mask(tensor, missing_fraction=0.5):
    """
    Creates a binary mask of the same shape as the input tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to create a mask for.
        missing_fraction (float): The fraction of elements to be marked as missing (0).

    Returns:
        torch.Tensor: A binary mask with the same shape as the input tensor.
    """
    # Must generate mask on CPU, then transfer back to original device
    mask = torch.ones(tensor.shape, dtype=torch.float32, device='cpu')
    num_missing = int(mask.numel() * missing_fraction)
    
    # Use a separate random state to avoid interference with global seed
    rng = torch.Generator()
    rng.manual_seed(torch.randint(0, 1000000, (1,)).item())
    
    # Get a random permutation of indices to set to zero
    indices = torch.randperm(mask.numel(), generator=rng)[:num_missing]
    
    # Flatten the mask, set the missing indices to 0, and then unflatten
    mask_flat = mask.view(-1)
    mask_flat[indices] = 0
    
    # Transfer back to original tensor device
    return mask.to(tensor.device)

def save_tensor_frame_as_image(tensor_frame, file_path):
    """
    Saves a single frame from a tensor (H, W, C) as a PNG image.
    Assumes tensor values are in the range [0, 1].
    """
    if tensor_frame.dim() != 3 or tensor_frame.shape[2] not in [1, 3, 4]:
        raise ValueError(f"Input must be a 3D tensor with shape (H, W, C), but got {tensor_frame.shape}")

    # ToPILImage expects (C, H, W), so we permute the channels.
    # Also clone to avoid modifying the original tensor's memory layout.
    img_tensor = tensor_frame.permute(2, 0, 1).clone()

    # Convert to PIL Image and save
    pil_img = ToPILImage()(img_tensor)
    pil_img.save(file_path)

def compare_structures(true_ranks, learned_ranks, threshold=1e-3):
    """
    Compares the true network structure with the learned one.
    Returns the accuracy of identifying zero vs. non-zero bonds.
    """
    correct_predictions = 0
    total_bonds = len(true_ranks)
    
    for bond_name, true_rank in true_ranks.items():
        # A bond is considered 'present' if its rank > 0
        true_bond_present = (true_rank > 0)
        
        # A learned bond is considered 'present' if its effective rank > 0
        learned_rank = learned_ranks.get(bond_name, 0)
        learned_bond_present = (learned_rank > 0)
        
        if true_bond_present == learned_bond_present:
            correct_predictions += 1
            
    return correct_predictions / total_bonds if total_bonds > 0 else 1.0

def save_structure(ranks, filepath):
    """Saves the network structure (ranks) to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(ranks, f, indent=4)

def load_structure(filepath):
    """Loads a network structure (ranks) from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f) 

def get_memory_usage():
    """Gets the peak memory usage in MB."""
    if not tracemalloc.is_tracing():
        return 0.0
    _, peak = tracemalloc.get_traced_memory()
    return peak / 10**6

class TimeoutException(Exception):
    pass 

def auto_align_video_tensor(tensor, target_shape=(50, 144, 176, 3)):
    """
    Align input tensor shape using scaling instead of cropping to preserve complete information.
    """
    F, H, W, C = tensor.shape
    tgt_F, tgt_H, tgt_W, tgt_C = target_shape
    assert C == tgt_C, f"Channel count mismatch: {C} vs {tgt_C}"
    
    # Frame count alignment
    if F < tgt_F:
        reps = (tgt_F + F - 1) // F
        tensor = tensor.repeat(reps, 1, 1, 1)[:tgt_F]
    elif F > tgt_F:
        tensor = tensor[:tgt_F]
    
    # Use scaling instead of cropping to preserve complete information
    if H != tgt_H or W != tgt_W:
        # Convert to (C, H, W) format for F.interpolate
        tensor = tensor.permute(0, 3, 1, 2)  # (F, C, H, W)
        
        # Use bilinear interpolation to scale to target size
        tensor = torch.nn.functional.interpolate(
            tensor, 
            size=(tgt_H, tgt_W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert back to (F, H, W, C) format
        tensor = tensor.permute(0, 2, 3, 1)
    
    return tensor 