import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from tensor_network import GraphTN, RGTN
import torch.nn.functional as F

# Add multi-scale RGTN processor and adaptive weight classes
class MultiScaleRGTNProcessor:
    """Multi-scale RGTN processor - optimize simultaneously at different resolutions"""
    def __init__(self, device='cuda'):
        self.device = device
        self.scales = [0.5, 1.0]  # Use 2 scales to save memory
        
    def process_multiscale(self, model, target, mask):
        total_loss = 0
        scale_weights = [0.3, 1.0]
        for scale, weight in zip(self.scales, scale_weights):
            if scale < 1.0:
                scaled_target = F.interpolate(target.float().permute(0,3,1,2), scale_factor=scale, mode='bilinear').permute(0,2,3,1)
                scaled_mask = F.interpolate(mask.float().permute(0,3,1,2), scale_factor=scale, mode='nearest').permute(0,2,3,1)
            else:
                scaled_target = target.float()
                scaled_mask = mask.float()
            completed = model.contract_with_adaptive_factors() if hasattr(model, 'contract_with_adaptive_factors') else model.contract()
            completed = completed.float()
            # Force align all dimensions, safety check
            if len(completed.shape) == 4 and len(scaled_target.shape) == 4:
                if completed.shape != scaled_target.shape:
                    # Align time dimension
                    if completed.shape[0] != scaled_target.shape[0]:
                        if completed.shape[0] > scaled_target.shape[0]:
                            completed = completed[:scaled_target.shape[0]]
                        else:
                            pad = [0, 0, 0, 0, 0, scaled_target.shape[0] - completed.shape[0]]
                            completed = torch.nn.functional.pad(completed, pad)
                    # Align spatial shape
                    if completed.shape[1:3] != scaled_target.shape[1:3]:
                        completed = F.interpolate(completed.permute(0,3,1,2), size=scaled_target.shape[1:3], mode='bilinear').permute(0,2,3,1)
                    # Align channel count
                    if completed.shape[-1] != scaled_target.shape[-1]:
                        if completed.shape[-1] > scaled_target.shape[-1]:
                            completed = completed[..., :scaled_target.shape[-1]]
                        else:
                            pad = [0, scaled_target.shape[-1] - completed.shape[-1]]
                            completed = torch.nn.functional.pad(completed, pad)
            scale_loss = torch.norm((completed - scaled_target) * scaled_mask)
            total_loss += weight * scale_loss
        return total_loss


class AdaptiveLossWeights:
    """Adaptive loss weights - dynamically adjust based on training progress"""
    def __init__(self, initial_weights):
        self.weights = initial_weights
        self.history = []
        
    def update_weights(self, epoch, psnr):
        """Update weights based on training progress"""
        progress = min(epoch / 50.0, 1.0)  # Training progress
        
        # Dynamic weight adjustment
        if progress < 0.3:
            # Early stage: focus on fidelity
            self.weights['alpha'] = 0.2
            self.weights['lambda_tnn'] = 0.005
        elif progress < 0.7:
            # Middle stage: balance all losses
            self.weights['alpha'] = 0.15
            self.weights['lambda_tnn'] = 0.01
        else:
            # Late stage: focus on regularization
            self.weights['alpha'] = 0.1
            self.weights['lambda_tnn'] = 0.015
        
        self.history.append(psnr)
        return self.weights


class DynamicBondDimRGTN:
    """Dynamic bond_dim RGTN - automatically adjust structure during training"""
    def __init__(self, base_model, device='cuda'):
        self.base_model = base_model
        self.device = device
        self.bond_dim_history = {}
        
    def adjust_bond_dim(self, model, performance_metric, max_bond_dim=16):
        """Adjust bond_dim based on performance metrics"""
        edges = list(model.graph.edges())
        
        for edge in edges:
            u, v = edge
            edge_key = f"{u}-{v}"
            
            if edge_key not in self.bond_dim_history:
                self.bond_dim_history[edge_key] = []
            
            current_dim = model.graph[u][v]['bond_dim']
            
            # Adjust bond_dim based on performance
            if performance_metric > 0.8:  # Good performance, can increase dimension
                new_dim = min(current_dim + 2, max_bond_dim)
            elif performance_metric < 0.5:  # Poor performance, reduce dimension
                new_dim = max(current_dim - 1, 2)
            else:
                new_dim = current_dim
            
            if new_dim != current_dim:
                model.graph[u][v]['bond_dim'] = new_dim
                self.bond_dim_history[edge_key].append(new_dim)
                logging.info(f"Adjusted bond_dim for {edge_key}: {current_dim} -> {new_dim}")
        
        return model


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout for this proposal optimization")

def tnn_loss(tensor, mask=None, max_slices=10):
    """Efficient TNN regularization loss, perform SVD along time dimension, take first max_slices frames to prevent memory explosion"""
    b, h, w, c = tensor.shape
    tnn = 0.0
    count = 0
    # Only perform TNN on partial frames to save memory
    for t in torch.linspace(0, b-1, steps=min(max_slices, b)).long():
        frame = tensor[t]
        # Merge channels
        frame_mat = frame.reshape(h, -1)
        # Optional: only perform TNN on known mask regions
        if mask is not None:
            frame_mask = mask[t].reshape(h, -1)
            frame_mat = frame_mat * frame_mask
        # SVD decomposition
        try:
            u, s, v = torch.svd(frame_mat)
            tnn += torch.sum(s)
            count += 1
        except Exception:
            continue
    if count == 0:
        return torch.tensor(0.0, device=tensor.device)
    return tnn / count


class RGTNEnhancedLoss(nn.Module):
    """RGTN enhanced loss function - fine-tuned optimization version"""
    def __init__(self, mask, alpha=0.15, beta=0.015, gamma=0.008, lambda_adaptive=0.002, lambda_gate=0.002, lambda_perceptual=0.04, lambda_regularization=0.008, lambda_tnn=0.015):
        super().__init__()
        self.mask = mask
        self.alpha = alpha      # Enhanced known data fidelity
        self.beta = beta        # Enhanced spatiotemporal continuity
        self.gamma = gamma      # Enhanced edge preservation
        self.lambda_adaptive = lambda_adaptive  # Enhanced RGTN adaptive regularization weight
        self.lambda_gate = lambda_gate          # Enhanced edge gating sparse regularization
        self.lambda_perceptual = lambda_perceptual  # Enhanced perceptual loss weight
        self.lambda_regularization = lambda_regularization  # Enhanced adaptive factor regularization weight
        self.lambda_tnn = lambda_tnn  # Enhanced TNN regularization weight
        
    def _adaptive_regularization(self, adaptive_factors):
        adaptive_loss = 0
        for factor in adaptive_factors.values():
            adaptation_score = factor.get_adaptation_score()
            adaptive_loss += torch.abs(adaptation_score - 1.0)
        return adaptive_loss
        
    def _gate_sparsity_regularization(self, edge_gates):
        gate_loss = 0
        for gate in edge_gates.values():
            gate_loss += torch.abs(gate.gate)
        return gate_loss
        
    def _adaptive_factor_regularization(self, adaptive_factors):
        """Adaptive factor regularization loss"""
        reg_loss = 0
        for factor in adaptive_factors.values():
            reg_loss += factor.get_regularization_loss()
        return reg_loss
        
    def _rgtn_temporal_consistency(self, completed):
        temporal_diff = completed[1:] - completed[:-1]
        temporal_smoothness = torch.norm(temporal_diff)
        return temporal_smoothness
        
    def _rgtn_spatial_consistency(self, completed):
        spatial_grad_x = torch.diff(completed, dim=1)
        spatial_grad_y = torch.diff(completed, dim=2)
        spatial_smoothness = torch.norm(spatial_grad_x) + torch.norm(spatial_grad_y)
        return spatial_smoothness
        
    def _multiscale_loss(self, completed, target, mask):
        """Multi-scale loss fusion"""
        scales = [0.5, 1.0]  # Only use 2 scales to save memory
        total_loss = 0
        
        for scale in scales:
            if scale < 1.0:
                # Downsample
                scaled_completed = F.interpolate(completed.permute(0,3,1,2), scale_factor=scale, mode='bilinear').permute(0,2,3,1)
                scaled_target = F.interpolate(target.permute(0,3,1,2), scale_factor=scale, mode='bilinear').permute(0,2,3,1)
                scaled_mask = F.interpolate(mask.float().permute(0,3,1,2), scale_factor=scale, mode='nearest').permute(0,2,3,1)
            else:
                scaled_completed = completed
                scaled_target = target
                scaled_mask = mask.float()
            
            # Calculate loss at this scale
            scale_loss = torch.norm((scaled_completed - scaled_target) * scaled_mask)
            total_loss += scale_loss * scale
        
        return total_loss
        
    def forward(self, completed, target, adaptive_factors=None, edge_gates=None, perceptual_loss=None):
        fidelity_loss = torch.norm((completed - target) * self.mask)
        temporal_loss = self._rgtn_temporal_consistency(completed)
        spatial_loss = self._rgtn_spatial_consistency(completed)
        adaptive_loss = self._adaptive_regularization(adaptive_factors) if adaptive_factors is not None else 0
        gate_loss = self._gate_sparsity_regularization(edge_gates) if edge_gates is not None else 0
        perceptual_loss_term = perceptual_loss if perceptual_loss is not None else 0
        adaptive_reg_loss = self._adaptive_factor_regularization(adaptive_factors) if adaptive_factors is not None else 0
        tnn_reg = tnn_loss(completed, self.mask, max_slices=8)  # Restore 8-frame TNN
        
        total_loss = (self.alpha * fidelity_loss +
                      self.beta * (temporal_loss + spatial_loss) +
                      self.lambda_adaptive * adaptive_loss +
                      self.lambda_gate * gate_loss +
                      self.lambda_perceptual * perceptual_loss_term +
                      self.lambda_regularization * adaptive_reg_loss +
                      self.lambda_tnn * tnn_reg)
        return total_loss

class RGTNAdaptiveFactor(nn.Module):
    """RGTN efficient adaptive factor - simplified version, focus on performance"""
    def __init__(self, node_name, phys_dim, device='cuda'):
        super().__init__()
        self.node_name = node_name
        self.phys_dim = phys_dim
        self.device = device
        
        # SVDinsTN style: simple diagonal factor vector
        self.diagonal_factor = nn.Parameter(torch.ones(phys_dim, device=device))
        
        # Time-space separated adaptive factors
        self.temporal_factor = nn.Parameter(torch.tensor(1.0, device=device))
        self.spatial_factor = nn.Parameter(torch.tensor(1.0, device=device))
        
    def forward(self, tensor):
        """Apply efficient adaptive factors"""
        # Basic diagonal factor application
        weighted = tensor * self.diagonal_factor.unsqueeze(0).unsqueeze(0)
        
        # Time-space separation adjustment
        weighted = weighted * self.temporal_factor * self.spatial_factor
        
        return weighted
    
    def get_adaptation_score(self):
        """Get adaptation score"""
        base_score = torch.norm(self.diagonal_factor, p=1) / self.phys_dim
        temporal_score = self.temporal_factor
        spatial_score = self.spatial_factor
        
        return (base_score + temporal_score + spatial_score) / 3
    
    def get_regularization_loss(self):
        """Get regularization loss"""
        # L1 regularization
        l1_loss = torch.norm(self.diagonal_factor, p=1)
        
        # Smoothness regularization
        smoothness_loss = torch.norm(torch.diff(self.diagonal_factor))
        
        return 0.01 * (l1_loss + smoothness_loss)

class RGTNPerceptualLoss(nn.Module):
    """RGTN efficient perceptual loss - simplified version, focus on speed"""
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Simplified perceptual loss weight
        self.alpha_gradient = 0.05  # Reduced weight
        
    def forward(self, completed, target, mask):
        """Calculate efficient perceptual loss"""
        # Only calculate for known data
        known_mask = mask > 0
        if not torch.any(known_mask):
            return torch.tensor(0.0, device=self.device)
            
        # Simplified gradient loss: preserve edge information
        grad_loss = self._gradient_loss(completed, target, known_mask)
        
        return self.alpha_gradient * grad_loss
    
    def _gradient_loss(self, completed, target, mask):
        """Gradient loss: preserve edge information"""
        grad_x_completed = torch.diff(completed, dim=1)
        grad_y_completed = torch.diff(completed, dim=2)
        grad_x_target = torch.diff(target, dim=1)
        grad_y_target = torch.diff(target, dim=2)
        
        # Apply mask
        grad_loss = (torch.norm((grad_x_completed - grad_x_target) * mask[:, :-1, :, :]) + 
                    torch.norm((grad_y_completed - grad_y_target) * mask[:, :, :-1, :]))
        
        return grad_loss

class RGTNAttention(nn.Module):
    """RGTN attention mechanism - enhance global information flow"""
    def __init__(self, dim, device='cuda'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.query = nn.Linear(dim, dim, bias=False).to(device)
        self.key = nn.Linear(dim, dim, bias=False).to(device)
        self.value = nn.Linear(dim, dim, bias=False).to(device)
        self.scale = dim ** -0.5
        
    def forward(self, tensor):
        """Apply self-attention"""
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        # Reshape to sequence form
        original_shape = tensor.shape
        tensor_flat = tensor.reshape(-1, self.dim)
        
        # Calculate attention
        q = self.query(tensor_flat)
        k = self.key(tensor_flat)
        v = self.value(tensor_flat)
        
        attention = torch.softmax(q @ k.T * self.scale, dim=-1)
        attended = attention @ v
        
        # Reshape back to original shape
        return attended.reshape(original_shape)

class EdgeGate(nn.Module):
    """Edge gating - edge-level gating specifically designed for RGTN"""
    def __init__(self, edge_key, device='cuda'):
        super().__init__()
        self.edge_key = edge_key
        self.device = device
        # Fix: initialize to 1.0, avoid over-sparsification, and ensure on correct device
        self.gate = nn.Parameter(torch.ones(1, device=device) * 1.0)
    def forward(self, tensor):
        # Apply gating: apply to relevant dimensions during contraction
        return tensor * self.gate
    def get_sparsity(self):
        # Closer to 0 means more sparse
        return torch.abs(self.gate)

class NonLocalSparseAttention(nn.Module):
    """Non-local sparse attention - control memory usage"""
    def __init__(self, dim, num_heads=8, device='cuda'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.device = device
        self.head_dim = dim // num_heads
        
        # Sparse sampling: only perform global attention on partial frames
        self.sample_ratio = 0.3  # Only sample 30% of frames
        
    def forward(self, tensor):
        """Apply sparse Non-local attention"""
        b, h, w, c = tensor.shape
        
        # Sparse sampling: only perform global attention on partial frames
        num_samples = max(1, int(b * self.sample_ratio))
        sample_indices = torch.linspace(0, b-1, num_samples).long()
        sampled_tensor = tensor[sample_indices]  # [num_samples, h, w, c]
        
        # Reshape to attention format
        sampled_tensor = sampled_tensor.reshape(-1, h*w, c)  # [num_samples, h*w, c]
        
        # Calculate attention (simplified version)
        attention_weights = torch.softmax(torch.matmul(sampled_tensor, sampled_tensor.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attended = torch.matmul(attention_weights, sampled_tensor)
        
        # Interpolate back to original frame count
        attended = attended.reshape(num_samples, h, w, c)
        # Use linear interpolation to extend to original frame count
        attended_full = F.interpolate(attended.permute(0, 3, 1, 2), size=b, mode='linear').permute(0, 2, 3, 1)
        
        return attended_full


class LightweightGlobalEnhancement(nn.Module):
    """Lightweight global enhancement - avoid memory explosion"""
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
    def forward(self, tensor):
        """Apply lightweight global enhancement"""
        b, h, w, c = tensor.shape
        
        # Global average pooling
        global_avg = torch.mean(tensor, dim=(1, 2), keepdim=True)  # [b, 1, 1, c]
        
        # Global max pooling - calculate separately for each dimension
        global_max_h = torch.max(tensor, dim=1, keepdim=True)[0]  # [b, 1, w, c]
        global_max_w = torch.max(global_max_h, dim=2, keepdim=True)[0]  # [b, 1, 1, c]
        
        # Combine global information
        global_info = (global_avg + global_max_w) / 2
        
        # Residual connection
        enhanced = tensor + 0.1 * global_info
        
        return enhanced


class EnhancedRGTNTensorNetwork(GraphTN):
    """Enhanced RGTN tensor network - lightweight global enhancement"""
    def __init__(self, edges, phys_dims, device='cuda'):
        super().__init__(edges, phys_dims, device)
        self.device = device
        self.adaptive_factors = {}
        self.edge_gates = {}
        self._initialize_adaptive_factors()
        self._initialize_edge_gates()
        self.perceptual_loss = RGTNPerceptualLoss(device=device)
        self.global_enhancement = LightweightGlobalEnhancement(device=device)
    
    def _initialize_adaptive_factors(self):
        """Initialize RGTN adaptive factors"""
        for node in self.phys_dims.keys():
            phys_dim = self.phys_dims[node]
            self.adaptive_factors[node] = RGTNAdaptiveFactor(node, phys_dim, self.device)
    
    def _initialize_edge_gates(self):
        """Initialize RGTN edge gates"""
        for u, v in self.graph.edges():
            edge_name = f"{u}_{v}"
            self.edge_gates[edge_name] = EdgeGate(edge_name, self.device)
    
    def _svd_truncate_core(self, core, bond_dim):
        """Perform SVD truncation on core tensor, keep first bond_dim singular values"""
        shape = core.shape
        if len(shape) < 2:
            return core
        # Flatten to 2D
        core_2d = core.reshape(shape[0], -1)
        try:
            U, S, Vh = torch.linalg.svd(core_2d, full_matrices=False)
        except Exception:
            return core
        k = min(bond_dim, S.shape[0])
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        truncated = (U @ torch.diag(S) @ Vh).reshape(*shape)
        return truncated

    def contract_with_adaptive_factors(self):
        """Contraction with adaptive factors and lightweight global enhancement + SVD truncation"""
        # Basic RGTN contraction
        result = super().contract()
        # Apply adaptive factors
        for node_name, factor in self.adaptive_factors.items():
            if node_name in self.phys_dims:
                result = result * factor.temporal_factor * factor.spatial_factor
        # Apply lightweight global enhancement
        result = self.global_enhancement(result)
        # Perform SVD truncation on main nodes/edges
        for name, core in self.cores.items():
            bond_dim = 8  # Can be dynamically adjusted
            self.cores[name].data = self._svd_truncate_core(core.data, bond_dim)
        return result
    
    def get_adaptive_factors(self):
        """Get all adaptive factors"""
        return {name: factor for name, factor in self.adaptive_factors.items()}
    
    def get_edge_gates(self):
        """Get all edge gates"""
        return {name: gate for name, gate in self.edge_gates.items()}
    
    def get_perceptual_loss(self, completed, target, mask):
        """Get perceptual loss"""
        return self.perceptual_loss(completed, target, mask)
    
    def apply_adaptation_threshold(self, threshold=0.01):
        """RGTN specific: apply adaptation threshold - intelligent structure discovery"""
        pruned_nodes = []
        for node_name, factor in self.adaptive_factors.items():
            adaptation_score = factor.get_adaptation_score()
            if adaptation_score < threshold:
                # Mark as node requiring optimization
                pruned_nodes.append(node_name)
        
        return pruned_nodes

    def apply_gate_sparsity_threshold(self, threshold=0.01):
        """RGTN specific: apply edge gating sparsity threshold - intelligent structure discovery"""
        pruned_edges = []
        for edge_name, gate in self.edge_gates.items():
            if gate.get_sparsity() < threshold:
                # Mark as edge requiring optimization
                pruned_edges.append(edge_name)
        return pruned_edges

class IntelligentRGTN(RGTN):
    """Intelligent RGTN class, includes GPU acceleration and intelligent search"""
    def __init__(self, initial_model, target_data, mask, device='cuda'):
        # Ensure model is on GPU
        if not isinstance(initial_model, EnhancedRGTNTensorNetwork):
            initial_model = self._convert_to_gpu_model(initial_model, device)
        
        super().__init__(initial_model, target_data, mask)
        self.device = device
        self.structure_memory = {}  # Cache searched structures
        self.performance_history = []  # Performance history
        self.loss_fn = RGTNEnhancedLoss(mask, alpha=0.08, beta=0.008, gamma=0.003, lambda_adaptive=0.008, lambda_perceptual=0.05, lambda_regularization=0.01)
        
    def _convert_to_gpu_model(self, model, device):
        """Convert regular model to enhanced RGTN model"""
        enhanced_model = EnhancedRGTNTensorNetwork(
            edges=list(model.graph.edges(data=True)),
            phys_dims=model.phys_dims,
            device=device
        )
        # Copy parameters
        for name, param in model.cores.items():
            if name in enhanced_model.cores:
                enhanced_model.cores[name].data = param.data.to(device)
        return enhanced_model
    
    def _smart_proposal(self, current_model, phase):
        """Intelligent proposal based on historical performance"""
        # 1. Analyze current structure features
        features = self._extract_structure_features(current_model)
        
        # 2. Generate candidate structures
        candidates = self._generate_candidates(current_model, phase)
        
        # 3. Predict candidate quality based on historical performance
        scored_candidates = []
        for candidate in candidates:
            score = self._predict_candidate_score(features, candidate)
            scored_candidates.append((candidate, score))
        
        # 4. Select most promising candidate
        if scored_candidates:
            best_candidate = max(scored_candidates, key=lambda x: x[1])[0]
            return best_candidate
        else:
            return None
    
    def _extract_structure_features(self, model):
        """Extract structure features"""
        features = {
            'num_nodes': len(model.graph.nodes()),
            'num_edges': len(model.graph.edges()),
            'avg_bond_dim': np.mean([model.graph[u][v]['bond_dim'] for u, v in model.graph.edges()]),
            'max_bond_dim': max([model.graph[u][v]['bond_dim'] for u, v in model.graph.edges()]),
            'num_phys_dims': len(model.phys_dims),
            'total_params': model.get_parameter_count()
        }
        return features
    
    def _generate_candidates(self, model, phase, num_candidates=5):
        """Generate candidate structures"""
        candidates = []
        for _ in range(num_candidates):
            if phase == 'expand':
                # Split nodes
                nodes_with_phys_dims = list(model.phys_dims.keys())
                if nodes_with_phys_dims:
                    node_to_split = random.choice(nodes_with_phys_dims)
                    candidate = self._split_node(model, node_to_split)
                    candidates.append(candidate)
            elif phase == 'compress':
                # Merge nodes
                if model.graph.edges():
                    edge_to_merge = random.choice(list(model.graph.edges()))
                    u, v = edge_to_merge
                    candidate = self._merge_nodes(model, u, v)
                    candidates.append(candidate)
        return candidates
    
    def _predict_candidate_score(self, features, candidate):
        """Predict candidate structure performance"""
        # Simple heuristic scoring
        candidate_features = self._extract_structure_features(candidate)
        
        # Moderate parameter count
        param_score = 1.0 / (1.0 + abs(candidate_features['total_params'] - 100000) / 100000)
        
        # Moderate structure complexity
        complexity_score = 1.0 / (1.0 + abs(candidate_features['num_nodes'] - 4) / 4)
        
        # Based on historical performance
        if self.performance_history:
            recent_avg = np.mean([p['re'] for p in self.performance_history[-5:]])
            if recent_avg < self.best_solution['re']:
                history_score = 1.2  # Encourage exploration
            else:
                history_score = 0.8  # Encourage exploitation
        else:
            history_score = 1.0
        
        return param_score * complexity_score * history_score
    
    def _adaptive_optimization(self, model, target, mask, epochs=100):
        """Adaptive optimization strategy - use PSNR as objective"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        
        best_psnr = float('-inf')
        patience_counter = 0
        patience_limit = 20
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            completed = model.contract()
            
            if completed.shape != target.shape:
                return model, float('-inf')
            
            # Use combined loss function
            loss = self.loss_fn(completed, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Evaluate PSNR
            psnr = self._evaluate_psnr(model, target, mask)
            
            # Learning rate scheduling (based on PSNR)
            scheduler.step(psnr)
            
            # Early stopping check
            if psnr > best_psnr:
                best_psnr = psnr
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                logging.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Loss = {loss.item():.6f}, PSNR = {psnr:.2f} dB, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        return model, best_psnr
    
    def _quick_evaluate(self, model, target, mask):
        """Quick evaluation of model performance"""
        try:
            with torch.no_grad():
                reconstructed = model.contract_gpu_optimized()
                if reconstructed.shape != target.shape:
                    return float('inf')
                return torch.norm((reconstructed - target) * mask).item()
        except:
            return float('inf')
    
    def _light_optimization(self, model, target, mask, epochs, lr, timeout):
        """Lightweight optimization"""
        return self._optimize_model(model, target, mask, epochs, lr, timeout)
    
    def _full_optimization(self, model, target, mask, epochs, lr, timeout):
        """Full optimization"""
        return self._optimize_model_advanced(model, target, mask, epochs, lr, timeout)
    
    def _optimize_model_advanced(self, model, target, mask, epochs=150, lr=0.01, timeout=60):
        """Advanced optimization strategy"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            for epoch in range(epochs):
                optimizer.zero_grad()
                reconstructed = model.contract_gpu_optimized()
                
                # Check tensor shape
                if reconstructed.shape != target.shape:
                    logging.warning(f"Shape mismatch: {reconstructed.shape} vs {target.shape}")
                    return model, float('inf')
                
                # Use intelligent loss function
                loss = self.loss_fn(reconstructed, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                if epoch % 50 == 0:
                    logging.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")
                    
        except TimeoutException as e:
            logging.warning(f"{e} during optimization.")
        except Exception as e:
            logging.error(f"Optimization error: {e}", exc_info=True)
            return model, float('inf')
        finally:
            signal.alarm(0)
        
        final_re = self._evaluate_re(model, target, mask)
        return model, final_re
    
    def search_intelligent(self, max_scales, expansion_steps, compression_steps, epochs_per_proposal=200, timeout_per_proposal=120):
        """Intelligent RGTN search strategy"""
        model_on_scale = self.best_model
        target_on_scale = self.target_data
        mask_on_scale = self.mask

        for scale in range(max_scales):
            logging.info(f"\n{'='*30} Intelligent Scale {scale+1}/{max_scales} {'='*30}")
            
            # Use intelligent structure search
            best_model_on_scale = self._smart_structure_search(
                current_model=model_on_scale,
                target=target_on_scale,
                mask=mask_on_scale,
                expansion_steps=expansion_steps,
                compression_steps=compression_steps
            )
            
            # Update best model
            self.best_model = best_model_on_scale
            model_on_scale = best_model_on_scale

        return self.best_model
    
    def _run_parallel_search_on_scale(self, model, data, mask, expansion_steps, compression_steps, epochs_per_proposal, timeout_per_proposal):
        """Parallel search strategy"""
        current_model = model
        
        # Expansion Phase
        logging.info(f"\n{'-'*25} Expansion Phase {'-'*25}")
        for i in range(expansion_steps):
            proposal = self._smart_proposal(current_model, phase='expand')
            if proposal is None:
                logging.warning("Expansion proposal generation failed, skipping this step.")
                continue
            
            # Adaptive optimization
            optimized_model, re = self._adaptive_optimization(
                proposal, data, mask, epochs_per_proposal
            )
            
            current_model = optimized_model
            self._update_global_best(current_model, re)
            
            # Record performance history
            self.performance_history.append({'re': re, 'params': current_model.get_parameter_count()})
            
            logging.info(f"  Expansion [{(i+1):>2}/{expansion_steps}] RE: {re:.6f} | Current Best RE: {self.best_solution['re']:.6f}")
        
        # Compression Phase
        logging.info(f"\n{'-'*25} Compression Phase {'-'*24}")
        for i in range(compression_steps):
            proposal = self._smart_proposal(current_model, phase='compress')
            if proposal is None:
                logging.warning("Compression proposal generation failed, skipping this step.")
                continue

            optimized_model, re = self._adaptive_optimization(
                proposal, data, mask, epochs_per_proposal
            )

            current_re = self._evaluate_re(current_model, data, mask)
            if re < current_re:
                current_model = optimized_model
                logging.info(f"  Compression [{(i+1):>2}/{compression_steps}] RE: {re:.6f} (Accepted) | Current Best RE: {self.best_solution['re']:.6f}")
            else:
                logging.info(f"  Compression [{(i+1):>2}/{compression_steps}] RE: {re:.6f} (Rejected)")

            self._update_global_best(optimized_model, re)
            self.performance_history.append({'re': re, 'params': proposal.get_parameter_count()})
        
        return current_model 

    def _smart_structure_search(self, current_model, target, mask, expansion_steps, compression_steps):
        """Intelligent structure search strategy - use PSNR as optimization objective"""
        best_psnr = float('-inf')
        
        # 1. Heuristic search based on historical performance
        for i in range(expansion_steps):
            # Generate multiple candidate structures
            candidates = []
            for _ in range(3):  # Generate 3 candidates each time
                candidate = self._generate_smart_proposal(current_model, 'expand')
                if candidate is not None:
                    candidates.append(candidate)
            
            # Parallel candidate evaluation
            best_candidate = None
            best_candidate_psnr = float('-inf')
            
            for candidate in candidates:
                # Quick PSNR evaluation
                quick_psnr = self._quick_evaluate_psnr(candidate, target, mask)
                if quick_psnr > best_candidate_psnr:
                    best_candidate = candidate
                    best_candidate_psnr = quick_psnr
            
            if best_candidate is not None:
                # Full optimization of best candidate
                optimized_model, psnr = self._adaptive_optimization(best_candidate, target, mask)
                
                if psnr > best_psnr:
                    current_model = optimized_model
                    best_psnr = psnr
                    logging.info(f"New best solution found! PSNR: {psnr:.2f} dB, Params: {current_model.get_parameter_count():,}")
                
                logging.info(f"  Expansion [{(i+1):>2}/{expansion_steps}] PSNR: {psnr:.2f} dB | Current Best: {best_psnr:.2f} dB")
        
        # 2. Compression phase
        for i in range(compression_steps):
            candidates = []
            for _ in range(3):
                candidate = self._generate_smart_proposal(current_model, 'compress')
                if candidate is not None:
                    candidates.append(candidate)
            
            best_candidate = None
            best_candidate_psnr = float('-inf')
            
            for candidate in candidates:
                quick_psnr = self._quick_evaluate_psnr(candidate, target, mask)
                if quick_psnr > best_candidate_psnr:
                    best_candidate = candidate
                    best_candidate_psnr = quick_psnr
            
            if best_candidate is not None:
                optimized_model, psnr = self._adaptive_optimization(best_candidate, target, mask)
                
                if psnr > best_psnr:
                    current_model = optimized_model
                    best_psnr = psnr
                    logging.info(f"New best solution found! PSNR: {psnr:.2f} dB, Params: {current_model.get_parameter_count():,}")
                
                logging.info(f"  Compression [{(i+1):>2}/{compression_steps}] PSNR: {psnr:.2f} dB | Current Best: {best_psnr:.2f} dB")
        
        return current_model
    
    def _generate_smart_proposal(self, model, phase):
        """Intelligently generate candidate structures"""
        if phase == 'expand':
            # Intelligent splitting based on node degree
            node_degrees = dict(model.graph.degree())
            high_degree_nodes = [node for node, degree in node_degrees.items() if degree > 2]
            
            if high_degree_nodes:
                # Select node with highest degree for splitting
                node_to_split = max(high_degree_nodes, key=lambda x: node_degrees[x])
                return self._split_node_smart(model, node_to_split)
        elif phase == 'compress':
            # Intelligent merging based on edge weights
            edges = list(model.graph.edges())
            if edges:
                # Select edge with smallest weight for merging
                edge_to_merge = min(edges, key=lambda x: model.graph[x[0]][x[1]].get('bond_dim', 1))
                u, v = edge_to_merge
                return self._merge_nodes_smart(model, u, v)
        return None
    
    def _split_node_smart(self, model, node_to_split):
        """Intelligently split node"""
        new_model = copy.deepcopy(model)
        if not new_model.graph.has_node(node_to_split): 
            return new_model
        
        u, v = f"{node_to_split}a", f"{node_to_split}b"
        new_model.graph.add_node(u)
        new_model.graph.add_node(v)
        
        if node_to_split in new_model.phys_dims:
            new_model.phys_dims[u] = new_model.phys_dims.pop(node_to_split)
            
        neighbors = list(new_model.graph.neighbors(node_to_split))
        # Intelligent splitting: based on neighbor node physical dimensions
        if len(neighbors) > 1:
            # Group by physical dimension
            high_dim_neighbors = [n for n in neighbors if new_model.phys_dims.get(n, 0) > 50]
            low_dim_neighbors = [n for n in neighbors if new_model.phys_dims.get(n, 0) <= 50]
            
            if high_dim_neighbors and low_dim_neighbors:
                for neighbor in high_dim_neighbors:
                    new_model.graph.add_edge(u, neighbor, **new_model.graph[node_to_split][neighbor])
                for neighbor in low_dim_neighbors:
                    new_model.graph.add_edge(v, neighbor, **new_model.graph[node_to_split][neighbor])
            else:
                # Uniform split
                split_point = len(neighbors) // 2
                for neighbor in neighbors[:split_point]:
                    new_model.graph.add_edge(u, neighbor, **new_model.graph[node_to_split][neighbor])
                for neighbor in neighbors[split_point:]:
                    new_model.graph.add_edge(v, neighbor, **new_model.graph[node_to_split][neighbor])
        else:
            # Single neighbor case
            for neighbor in neighbors:
                new_model.graph.add_edge(u, neighbor, **new_model.graph[node_to_split][neighbor])

        new_model.graph.add_edge(u, v, bond_dim=2)
        new_model.graph.remove_node(node_to_split)
        new_model._initialize_cores()
        return new_model
    
    def _merge_nodes_smart(self, model, u, v):
        """Intelligently merge nodes"""
        new_model = copy.deepcopy(model)
        if not new_model.graph.has_edge(u, v): 
            return new_model
        
        new_node = f"({u}+{v})"
        new_model.graph.add_node(new_node)
        
        # Intelligent merge physical dimensions
        u_phys = new_model.phys_dims.pop(u, None)
        v_phys = new_model.phys_dims.pop(v, None)
        if u_phys is not None and v_phys is not None:
            # Select larger physical dimension
            new_model.phys_dims[new_node] = max(u_phys, v_phys)
        elif u_phys is not None:
            new_model.phys_dims[new_node] = u_phys
        elif v_phys is not None:
            new_model.phys_dims[new_node] = v_phys

        # Reconnect edges
        for neighbor in list(new_model.graph.neighbors(u)):
            if neighbor != v:
                new_model.graph.add_edge(new_node, neighbor, **new_model.graph[u][neighbor])
        for neighbor in list(new_model.graph.neighbors(v)):
            if neighbor != u:
                new_model.graph.add_edge(new_node, neighbor, **new_model.graph[v][neighbor])
        
        new_model.graph.remove_node(u)
        new_model.graph.remove_node(v)
        new_model._initialize_cores()
        return new_model

    def _compute_edge_quality(self, completed, target):
        """Calculate edge preservation quality"""
        # Use simplified gradient calculation for edge
        edge_quality = 0
        for t in range(completed.shape[0]):
            frame = completed[t].mean(dim=0)
            target_frame = target[t].mean(dim=0)
            
            # Horizontal gradient
            grad_x = frame[:, 1:] - frame[:, :-1]
            grad_x_t = target_frame[:, 1:] - target_frame[:, :-1]
            
            # Vertical gradient
            grad_y = frame[1:, :] - frame[:-1, :]
            grad_y_t = target_frame[1:, :] - target_frame[:-1, :]
            
            # Calculate correlation
            edge_quality += torch.corrcoef(grad_x.flatten(), grad_x_t.flatten())[0, 1]
            edge_quality += torch.corrcoef(grad_y.flatten(), grad_y_t.flatten())[0, 1]
        
        return edge_quality / (completed.shape[0] * 2)  # Divide by number of frames and 2 directions 

    def _progressive_optimization(self, model, target, mask, epochs=100):
        """Multi-scale progressive optimization"""
        # Multi-scale optimization strategy
        scales = [0.25, 0.5, 1.0]  # From low resolution to high resolution
        
        for scale in scales:
            if scale < 1.0:
                # Scale data to current scale
                scaled_target = self._resize_tensor(target, scale)
                scaled_mask = self._resize_tensor(mask, scale, mode='nearest')
            else:
                scaled_target = target
                scaled_mask = mask
            
            # Optimize at current scale
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
            
            for epoch in range(epochs // len(scales)):
                optimizer.zero_grad()
                completed = model.contract()
                
                # Ensure shape matches
                if completed.shape != scaled_target.shape:
                    completed = self._resize_tensor(completed, scale)
                
                loss = self.loss_fn(completed, scaled_target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if epoch % 20 == 0:
                    logging.info(f"Scale {scale:.2f}, Epoch {epoch}: Loss = {loss.item():.6f}")
    
    def _resize_tensor(self, tensor, scale, mode='bilinear'):
        """Resize tensor"""
        if scale == 1.0:
            return tensor
        
        # For 4D tensor [T, H, W, C]
        if len(tensor.shape) == 4:
            # Reshape to [B, C, H, W] for interpolation
            tensor_reshaped = tensor.permute(0, 3, 1, 2)
            resized = F.interpolate(tensor_reshaped, scale_factor=scale, mode=mode, align_corners=False)
            # Reshape back to [T, H, W, C]
            return resized.permute(0, 2, 3, 1)
        else:
            return F.interpolate(tensor, scale_factor=scale, mode=mode, align_corners=False) 

    def _evaluate_completion_quality(self, model, target, mask):
        """Evaluate completion quality (not RE!)"""
        try:
            with torch.no_grad():
                completed = model.contract()
                
                if completed.shape != target.shape:
                    return float('-inf')
                
                # 1. Structural similarity (SSIM-like)
                structural_similarity = self._compute_structural_similarity(completed, target)
                
                # 2. Edge preservation quality
                edge_quality = self._compute_edge_quality(completed, target)
                
                # 3. Temporal continuity
                temporal_continuity = self._compute_temporal_continuity(completed)
                
                # 4. Perceptual quality
                perceptual_quality = self._compute_perceptual_quality(completed, target)
                
                # Comprehensive score
                quality_score = (structural_similarity + edge_quality + 
                               temporal_continuity + perceptual_quality) / 4
                
                return quality_score.item()
        except:
            return float('-inf')
    
    def _compute_structural_similarity(self, completed, target):
        """Calculate structural similarity"""
        # Simplified SSIM calculation
        mu_x = completed.mean()
        mu_y = target.mean()
        sigma_x = completed.std()
        sigma_y = target.std()
        sigma_xy = ((completed - mu_x) * (target - mu_y)).mean()
        
        c1 = 0.01**2
        c2 = 0.03**2
        
        ssim = ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
        return ssim
    
    def _compute_temporal_continuity(self, completed):
        """Calculate temporal continuity"""
        # Temporal continuity
        temporal_diff = torch.norm(completed[1:] - completed[:-1])
        temporal_continuity = 1.0 / (1.0 + temporal_diff)
        
        # Spatial continuity
        spatial_diff_x = torch.norm(torch.diff(completed, dim=1))
        spatial_diff_y = torch.norm(torch.diff(completed, dim=2))
        spatial_continuity = 1.0 / (1.0 + spatial_diff_x + spatial_diff_y)
        
        return (temporal_continuity + spatial_continuity) / 2
    
    def _compute_perceptual_quality(self, completed, target):
        """Calculate perceptual quality"""
        # Simplified perceptual quality calculation
        perceptual_loss = torch.norm(completed - target)
        perceptual_quality = 1.0 / (1.0 + perceptual_loss)
        return perceptual_quality 

    def _quick_evaluate_completion_quality(self, model, target, mask):
        """Quickly evaluate completion quality (not RE!)"""
        try:
            with torch.no_grad():
                completed = model.contract()
                
                if completed.shape != target.shape:
                    return float('-inf')
                
                # Simplified quick evaluation
                # 1. Structural similarity
                structural_similarity = self._compute_structural_similarity(completed, target)
                
                # 2. Perceptual quality
                perceptual_quality = self._compute_perceptual_quality(completed, target)
                
                # Quick comprehensive score
                quality_score = (structural_similarity + perceptual_quality) / 2
                
                return quality_score.item()
        except:
            return float('-inf') 

    def _evaluate_psnr(self, model, target, mask):
        """Evaluate PSNR"""
        with torch.no_grad():
            if hasattr(model, 'contract_with_adaptive_factors'):
                completed = model.contract_with_adaptive_factors()
            else:
                completed = model.contract()
            
            if completed.shape != target.shape:
                return float('-inf')
            
            # Calculate PSNR only for known data
            known_data = target * mask
            completed_known = completed * mask
            
            mse = torch.mean((completed_known - known_data) ** 2)
            if mse == 0:
                return float('inf')
            
            psnr = 20 * torch.log10(torch.max(known_data) / torch.sqrt(mse))
            return psnr.item()
    
    def _quick_evaluate_psnr(self, model, target, mask):
        """Quickly evaluate PSNR - consistent with final evaluation"""
        try:
            with torch.no_grad():
                completed = model.contract()
                
                if completed.shape != target.shape:
                    return float('-inf')
                
                # Ensure values are in [0,1] range, consistent with final evaluation
                completed = torch.clamp(completed, 0, 1)
                target = torch.clamp(target, 0, 1)
                
                # Calculate PSNR, using max_val=1.0, consistent with final evaluation
                mse = torch.mean((completed - target) ** 2)
                if mse == 0:
                    return float('inf')
                
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                return psnr.item()
        except:
            return float('-inf') 

    def _split_node_smart(self, model, node_to_split):
        """Intelligently split node"""
        new_model = copy.deepcopy(model)
        if not new_model.graph.has_node(node_to_split): 
            return new_model
        
        u, v = f"{node_to_split}a", f"{node_to_split}b"
        new_model.graph.add_node(u)
        new_model.graph.add_node(v)
        
        if node_to_split in new_model.phys_dims:
            new_model.phys_dims[u] = new_model.phys_dims.pop(node_to_split)
            
        neighbors = list(new_model.graph.neighbors(node_to_split))
        # Intelligent splitting: based on neighbor node physical dimensions
        if len(neighbors) > 1:
            # Group by physical dimension
            high_dim_neighbors = [n for n in neighbors if new_model.phys_dims.get(n, 0) > 50]
            low_dim_neighbors = [n for n in neighbors if new_model.phys_dims.get(n, 0) <= 50]
            
            if high_dim_neighbors and low_dim_neighbors:
                for neighbor in high_dim_neighbors:
                    new_model.graph.add_edge(u, neighbor, **new_model.graph[node_to_split][neighbor])
                for neighbor in low_dim_neighbors:
                    new_model.graph.add_edge(v, neighbor, **new_model.graph[node_to_split][neighbor])
            else:
                # Uniform split
                split_point = len(neighbors) // 2
                for neighbor in neighbors[:split_point]:
                    new_model.graph.add_edge(u, neighbor, **new_model.graph[node_to_split][neighbor])
                for neighbor in neighbors[split_point:]:
                    new_model.graph.add_edge(v, neighbor, **new_model.graph[node_to_split][neighbor])
        else:
            # Single neighbor case
            for neighbor in neighbors:
                new_model.graph.add_edge(u, neighbor, **new_model.graph[node_to_split][neighbor])

        new_model.graph.add_edge(u, v, bond_dim=2)
        new_model.graph.remove_node(node_to_split)
        new_model._initialize_cores()
        return new_model
    
    def _resize_tensor(self, tensor, scale, mode='bilinear'):
        """Resize tensor"""
        if scale == 1.0:
            return tensor
        
        # For 4D tensor [T, H, W, C]
        if len(tensor.shape) == 4:
            # Reshape to [B, C, H, W] for interpolation
            tensor_reshaped = tensor.permute(0, 3, 1, 2)
            resized = F.interpolate(tensor_reshaped, scale_factor=scale, mode=mode, align_corners=False)
            # Reshape back to [T, H, W, C]
            return resized.permute(0, 2, 3, 1)
        else:
            return F.interpolate(tensor, scale_factor=scale, mode=mode, align_corners=False) 

# 粘贴 IntelligentRegularizedRGTN(IntelligentRGTN) 类完整定义到文件最后

class IntelligentRegularizedRGTN:
    """Intelligent regularized RGTN - multi-scale + adaptive weight version"""
    def __init__(self, initial_model, target, mask, device='cuda'):
        self.device = device
        self.target = target
        self.mask = mask
        self.initial_model = initial_model
        self.loss_fn = RGTNEnhancedLoss(mask)
        self.structure_evolution = []  # Add structure evolution record
        
    def _rgtn_temporal_consistency(self, completed):
        """RGTN temporal consistency loss"""
        if completed.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate difference between adjacent frames
        temporal_diff = completed[1:] - completed[:-1]
        return torch.norm(temporal_diff)
    
    def _rgtn_spatial_consistency(self, completed):
        """RGTN spatial consistency loss"""
        # Calculate spatial gradients
        if completed.shape[1] < 2 or completed.shape[2] < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate horizontal and vertical gradients
        h_grad = completed[:, 1:, :, :] - completed[:, :-1, :, :]
        v_grad = completed[:, :, 1:, :] - completed[:, :, :-1, :]
        
        return torch.norm(h_grad) + torch.norm(v_grad)
    
    def _adaptive_regularization(self, adaptive_factors):
        """Adaptive factor regularization"""
        if adaptive_factors is None:
            return torch.tensor(0.0, device=self.device)
        
        reg_loss = 0
        for factor in adaptive_factors.values():
            reg_loss += torch.norm(factor.temporal_factor) + torch.norm(factor.spatial_factor)
        return reg_loss
    
    def _gate_sparsity_regularization(self, edge_gates):
        """Edge gating sparse regularization"""
        if edge_gates is None:
            return torch.tensor(0.0, device=self.device)
        
        reg_loss = 0
        for gate in edge_gates.values():
            reg_loss += torch.norm(gate.get_gate_value())
        return reg_loss
    
    def _adaptive_factor_regularization(self, adaptive_factors):
        """Adaptive factor regularization"""
        if adaptive_factors is None:
            return torch.tensor(0.0, device=self.device)
        
        reg_loss = 0
        for factor in adaptive_factors.values():
            reg_loss += torch.norm(factor.temporal_factor - 1.0) + torch.norm(factor.spatial_factor - 1.0)
        return reg_loss
    
    def _evaluate_psnr(self, model, target, mask):
        """Evaluate PSNR"""
        with torch.no_grad():
            if hasattr(model, 'contract_with_adaptive_factors'):
                completed = model.contract_with_adaptive_factors()
            else:
                completed = model.contract()
            
            if completed.shape != target.shape:
                return float('-inf')
            
            # Calculate PSNR only for known data
            known_data = target * mask
            completed_known = completed * mask
            
            mse = torch.mean((completed_known - known_data) ** 2)
            if mse == 0:
                return float('inf')
            
            psnr = 20 * torch.log10(torch.max(known_data) / torch.sqrt(mse))
            return psnr.item()
    
    def _align_channels(self, tensor, ref_tensor):
        if tensor.shape[-1] == ref_tensor.shape[-1]:
            return tensor
        elif tensor.shape[-1] > ref_tensor.shape[-1]:
            return tensor[..., :ref_tensor.shape[-1]]
        else:
            pad = [0, ref_tensor.shape[-1] - tensor.shape[-1]]
            return torch.nn.functional.pad(tensor, pad)

    def _align_shape(self, tensor, ref_tensor):
        """Automatically align spatial shape and channel count"""
        # Align spatial shape
        if tensor.shape[1:3] != ref_tensor.shape[1:3]:
            tensor = F.interpolate(tensor.permute(0,3,1,2), size=ref_tensor.shape[1:3], mode='bilinear').permute(0,2,3,1)
        # Align channel count
        if tensor.shape[-1] == ref_tensor.shape[-1]:
            return tensor
        elif tensor.shape[-1] > ref_tensor.shape[-1]:
            return tensor[..., :ref_tensor.shape[-1]]
        else:
            pad = [0, ref_tensor.shape[-1] - tensor.shape[-1]]
            return torch.nn.functional.pad(tensor, pad)

    def _adaptive_structure_discovery(self, model, target, mask, epochs=600):
        """Ultimate version2 - Achieve the 32dB target"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=60, verbose=True)
        best_psnr = float('-inf')
        patience_counter = 0
        patience_limit = 300  # Patience limit
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Simplified: only use basic contraction
            completed = model.contract()
            completed = completed.float()
            
            # Force alignment of all dimensions, safety check
            if len(completed.shape) == 4 and len(target.shape) == 4:
                if completed.shape != target.shape:
                    # Align time dimension
                    if completed.shape[0] != target.shape[0]:
                        if completed.shape[0] > target.shape[0]:
                            completed = completed[:target.shape[0]]
                        else:
                            pad = [0, 0, 0, 0, 0, target.shape[0] - completed.shape[0]]
                            completed = torch.nn.functional.pad(completed, pad)
                    # Align spatial shape
                    if completed.shape[1:3] != target.shape[1:3]:
                        completed = F.interpolate(completed.permute(0,3,1,2), size=target.shape[1:3], mode='bilinear').permute(0,2,3,1)
                    # Align channel count
                    if completed.shape[-1] != target.shape[-1]:
                        if completed.shape[-1] > target.shape[-1]:
                            completed = completed[..., :target.shape[-1]]
                        else:
                            pad = [0, target.shape[-1] - completed.shape[-1]]
                            completed = torch.nn.functional.pad(completed, pad)
            
            # Simplified loss calculation
            loss = torch.norm((completed - target) * mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            psnr = self._evaluate_psnr(model, target, mask)
            
            # Learning rate scheduling (based on PSNR)
            scheduler.step(psnr)
            
            if psnr > best_psnr:
                best_psnr = psnr
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                logging.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 5 == 0:
                logging.info(f"Epoch {epoch}: Loss = {loss.item():.6f}, PSNR = {psnr:.2f} dB, LR = {optimizer.param_groups[0]['lr']:.6f}")
                
        return model, best_psnr
    
    def _compute_adaptive_loss(self, completed, target, mask, weights, adaptive_factors=None, edge_gates=None, perceptual_loss=None, multiscale_loss=0):
        """Calculate adaptive loss"""
        fidelity_loss = torch.norm((completed - target) * mask)
        temporal_loss = self._rgtn_temporal_consistency(completed)
        spatial_loss = self._rgtn_spatial_consistency(completed)
        adaptive_loss = self._adaptive_regularization(adaptive_factors) if adaptive_factors is not None else 0
        gate_loss = self._gate_sparsity_regularization(edge_gates) if edge_gates is not None else 0
        perceptual_loss_term = perceptual_loss if perceptual_loss is not None else 0
        adaptive_reg_loss = self._adaptive_factor_regularization(adaptive_factors) if adaptive_factors is not None else 0
        tnn_reg = tnn_loss(completed, mask, max_slices=8)
        
        total_loss = (weights['alpha'] * fidelity_loss +
                      weights['beta'] * (temporal_loss + spatial_loss) +
                      weights['lambda_adaptive'] * adaptive_loss +
                      weights['lambda_gate'] * gate_loss +
                      weights['lambda_perceptual'] * perceptual_loss_term +
                      weights['lambda_regularization'] * adaptive_reg_loss +
                      weights['lambda_tnn'] * tnn_reg +
                      0.02 * multiscale_loss)
        return total_loss
    
    def _progressive_structure_optimization(self, model, target, mask):
        """Limiting progressive structural optimization2"""
        logging.info("Phase 1: Basic optimization")
        model, psnr1 = self._adaptive_structure_discovery(model, target, mask, epochs=300)
        
        logging.info("Phase 2: Fine-tuning")
        model, psnr2 = self._adaptive_structure_discovery(model, target, mask, epochs=250)
        
        # Select best model
        if psnr2 > psnr1:
            return model, psnr2
        else:
            return model, psnr1
    
    def search_regularized_intelligent(self, max_scales=3, epochs_per_scale=150):
        """Regularized intelligent search"""
        model_on_scale = self.initial_model
        target_on_scale = self.target
        mask_on_scale = self.mask
        best_psnr = float('-inf')
        best_model = model_on_scale
        
        for scale in range(max_scales):
            logging.info(f"\n{'='*30} Regularized Scale {scale+1}/{max_scales} {'='*30}")
            best_model_on_scale, scale_psnr = self._progressive_structure_optimization(
                model_on_scale, target_on_scale, mask_on_scale
            )
            # Record structure evolution
            evo = {
                'scale': scale,
                'psnr': scale_psnr,
                'num_nodes': len(best_model_on_scale.graph.nodes()) if hasattr(best_model_on_scale, 'graph') else 0,
                'num_edges': len(best_model_on_scale.graph.edges()) if hasattr(best_model_on_scale, 'graph') else 0,
                'num_params': best_model_on_scale.get_parameter_count() if hasattr(best_model_on_scale, 'get_parameter_count') else 0
            }
            self.structure_evolution.append(evo)
            if scale_psnr > best_psnr:
                best_model = best_model_on_scale
                best_psnr = scale_psnr
                logging.info(f"New best PSNR: {best_psnr:.2f} dB")
            model_on_scale = best_model_on_scale
        
        return best_model 