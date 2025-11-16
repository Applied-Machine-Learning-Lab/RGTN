import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import opt_einsum as oe
import random
import copy
import signal
import time
import logging

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout for this proposal optimization")

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reconstructed, target):
        return torch.norm(reconstructed - target)

class GraphTN(nn.Module):
    def __init__(self, edges, phys_dims, device='cpu'):
        super().__init__()
        self.graph = nx.Graph()
        self.phys_dims = phys_dims
        self.device = device

        for node, dim in self.phys_dims.items():
            self.graph.add_node(node)
            # Add leaf_nodes attribute for robust dimension tracking
            self.graph.nodes[node]['leaf_nodes'] = [node]

        for u, v, data in edges:
            bond_dim = data.get('bond_dim', 2) if isinstance(data, dict) else 2
            self.graph.add_edge(u, v, bond_dim=bond_dim)

        self.cores = nn.ParameterDict()
        self._initialize_cores()

    def _initialize_cores(self):
        new_cores = nn.ParameterDict()
        for node in sorted(self.graph.nodes(), key=str):
            dims = []
            for neighbor in sorted(self.graph.neighbors(node), key=str):
                dims.append(self.graph[node][neighbor]['bond_dim'])
            
            if node in self.phys_dims:
                dims.append(self.phys_dims[node])

            str_node = str(node)
            # If a core with the same key and shape exists, reuse it.
            if str_node in self.cores and self.cores[str_node].shape == tuple(dims):
                new_cores[str_node] = self.cores[str_node]
            else:
                # Otherwise, create a new random one.
                if not dims: # Handle scalar tensors for disconnected nodes
                    new_cores[str_node] = nn.Parameter(torch.randn((), device=self.device))
                else:
                    new_cores[str_node] = nn.Parameter(torch.randn(*dims, device=self.device) * 0.1)
        
        # Atomically replace the old parameter dictionary
        self.cores = new_cores

    def get_einsum_string_and_operands(self):
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        char_idx = 0

        edge_symbols = {}
        for u, v in self.graph.edges():
            if char_idx >= len(alphabet):
                raise ValueError("Ran out of alphabet characters for einsum symbols!")
            edge_symbols[tuple(sorted((str(u), str(v))))] = alphabet[char_idx]
            char_idx += 1

        phys_symbols = {}
        sorted_phys_nodes = sorted(self.phys_dims.keys(), key=str)
        for node in sorted_phys_nodes:
            if char_idx >= len(alphabet):
                raise ValueError("Ran out of alphabet characters for einsum symbols!")
            phys_symbols[str(node)] = alphabet[char_idx]
            char_idx += 1
        
        input_subs = []
        operands = []
        for node in sorted(self.graph.nodes(), key=str):
            str_node = str(node)
            sub_chars = []
            for neighbor in sorted(self.graph.neighbors(node), key=str):
                edge_key = tuple(sorted((str_node, str(neighbor))))
                sub_chars.append(edge_symbols[edge_key])
            
            if node in self.phys_dims:
                sub_chars.append(phys_symbols[str_node])
            
            input_subs.append("".join(sub_chars))
            operands.append(self.cores[str_node])

        output_subs = [phys_symbols[str(key)] for key in sorted_phys_nodes]
        
        einsum_string = f"{','.join(input_subs)}->{''.join(output_subs)}"
        return einsum_string, operands

    def contract(self, use_optimizer=True):
        try:
            einsum_string, operands = self.get_einsum_string_and_operands()
            if not operands: return torch.tensor(1.0, device=self.device)
            # Handle single-node graph case properly
            if len(operands) == 1: 
                return operands[0]

            if use_optimizer:
                # Use 'auto-hq' for a good balance of pathfinding time and contraction speed
                return oe.contract(einsum_string, *operands, optimize='auto-hq', backend='torch')
            else:
                return torch.einsum(einsum_string, *operands)
        except Exception as e:
            print(f"CRITICAL ERROR in contract: {e}")
            einsum_string, _ = self.get_einsum_string_and_operands()
            print(f"Einsum string: '{einsum_string}'")
            print(f"Einsum string generation failed for graph with nodes: {self.graph.nodes()} and edges: {self.graph.edges(data=True)}")
            return torch.tensor(float('inf'), device=self.device)

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def get_config(self):
        config = {
            'phys_dims': self.phys_dims,
            'bond_dims': {frozenset(edge): self.graph[edge[0]][edge[1]]['bond_dim'] for edge in self.graph.edges}
        }
        return config

    def plot_topology(self, save_path=None, title="Tensor Network Topology"):
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42, k=0.9)
        nx.draw_networkx_nodes(self.graph, pos, node_color='skyblue', node_size=800)
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(self.graph, pos, width=1.5, alpha=0.8, edge_color='gray')
        edge_labels = nx.get_edge_attributes(self.graph, 'bond_dim')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red', font_size=10)
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

def initialize_model_from_data(target_data, num_nodes, svd_rank_threshold=1e-3, initial_bond_dim=2):
    """
    Initializes a GraphTN model based on the correlations within the target data,
    inspired by the SVDinsTN paper's greedy approach.

    Args:
        target_data (torch.Tensor): The target tensor data to model.
        num_nodes (int): The number of tensor cores (nodes) to create in the graph.
        svd_rank_threshold (float): The threshold for singular values to be considered significant.
        initial_bond_dim (int): The initial bond dimension for edges created.

    Returns:
        GraphTN: An initialized GraphTN model with a data-driven topology.
    """
    data_shape = target_data.shape
    num_dims = len(data_shape)

    # 1. Assign physical dimensions to nodes
    # We distribute the dimensions of the target tensor among the nodes.
    # This is a heuristic and can be adapted based on domain knowledge.
    phys_dims = {}
    dims_per_node = num_dims // num_nodes
    extra_dims = num_dims % num_nodes
    
    current_dim_idx = 0
    for i in range(num_nodes):
        node_dims_count = dims_per_node + (1 if i < extra_dims else 0)
        phys_dim_indices = list(range(current_dim_idx, current_dim_idx + node_dims_count))
        
        # We will store the original dimension indices in the node attribute for later use.
        # The physical dimension of the core will be the product of these dimensions.
        if phys_dim_indices:
            phys_dims[i] = {
                'size': np.prod([data_shape[d] for d in phys_dim_indices]),
                'original_dims': phys_dim_indices
            }
        current_dim_idx += node_dims_count
    
    # Filter out nodes that didn't get any dimensions
    node_list = sorted([node for node, p_dim in phys_dims.items() if p_dim['original_dims']])
    num_active_nodes = len(node_list)

    # 2. Greedily add edges based on SVD analysis of correlations
    edges = []
    if num_active_nodes >= 2:
        # Calculate a correlation score between all pairs of nodes
        correlation_scores = {}
        for i in range(num_active_nodes):
            for j in range(i + 1, num_active_nodes):
                u_node, v_node = node_list[i], node_list[j]

                u_dims = phys_dims[u_node]['original_dims']
                v_dims = phys_dims[v_node]['original_dims']
                other_dims = [d for d in range(num_dims) if d not in u_dims and d not in v_dims]
                
                # Reshape tensor to partition A (u_dims), B (v_dims), and the rest (other_dims)
                # Permute to bring u's and v's dims together
                permute_order = u_dims + v_dims + other_dims
                permuted_data = target_data.permute(*permute_order)
                
                # Reshape into a matrix for SVD
                dim_a = np.prod([data_shape[d] for d in u_dims]) if u_dims else 1
                dim_b = np.prod([data_shape[d] for d in v_dims]) if v_dims else 1
                dim_c = np.prod([data_shape[d] for d in other_dims]) if other_dims else 1

                # We want to measure the correlation between group U and group V
                matrix_to_svd = permuted_data.reshape(dim_a, -1)
                
                # Perform SVD and check the rank
                try:
                    s = torch.linalg.svdvals(matrix_to_svd)
                    # The "rank" or number of significant singular values is a proxy for correlation
                    rank = torch.sum(s / s[0] > svd_rank_threshold).item()
                    correlation_scores[(u_node, v_node)] = rank
                except torch.linalg.LinAlgError:
                    # If SVD fails, assign a low score
                    correlation_scores[(u_node, v_node)] = 0


        # Add edges for the most correlated pairs
        # Here, we add one less than the number of nodes to ensure a connected graph (like a tree)
        # This is a simple heuristic to start with.
        num_edges_to_add = min(len(correlation_scores), num_active_nodes + 2) # Add a few more edges than a simple tree
        
        sorted_by_correlation = sorted(correlation_scores.items(), key=lambda item: item[1], reverse=True)
        
        for (u, v), score in sorted_by_correlation[:num_edges_to_add]:
            if score > 1: # Only add an edge if the correlation is non-trivial
                edges.append((u, v, {'bond_dim': initial_bond_dim}))

    # 3. Create the GraphTN model
    final_phys_dims = {node: p_dim['size'] for node, p_dim in phys_dims.items() if p_dim['original_dims']}
    initial_model = GraphTN(edges=edges, phys_dims=final_phys_dims, device=target_data.device)
    
    print("--- Initial Model Created from Data ---")
    print(f"Nodes: {initial_model.graph.nodes()}")
    print(f"Edges: {initial_model.graph.edges(data=True)}")
    print(f"Physical Dims: {initial_model.phys_dims}")
    print("------------------------------------")

    return initial_model

class RGTN:
    """
    RGTN (Renormalization Group Tensor Network) class for multi-scale topological search.
    This version is adapted for TENSOR COMPLETION.
    """
    def __init__(self, initial_model, target_data, mask):
        self.best_model = copy.deepcopy(initial_model)
        self.target_data = target_data
        self.mask = mask # Mask for tensor completion
        
        # In completion, the "best solution" is simply the one with the lowest reconstruction error.
        initial_re = self._evaluate_re(initial_model, target_data, mask)
        self.best_solution = {
            "re": initial_re,
            "params": initial_model.get_parameter_count(),
            "config": initial_model.get_config()
        }
        self.search_history = [self.best_solution]

    def _optimize_model(self, model, target, mask, epochs=150, lr=0.01, timeout=60):
        """
        Optimizes the given model's cores to minimize reconstruction error on KNOWN data.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            for epoch in range(epochs):
                optimizer.zero_grad()
                reconstructed = model.contract()
                # Check tensor shape matching
                if reconstructed.shape != target.shape:
                    logging.warning(f"Shape mismatch: reconstructed {reconstructed.shape} vs target {target.shape}")
                    return model, float('inf')
                masked_diff = (reconstructed - target) * mask
                loss = torch.norm(masked_diff)
                loss.backward()
                optimizer.step()
        except TimeoutException as e:
            logging.warning(f"{e} during optimization.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during optimization: {e}", exc_info=True)
            return model, float('inf')
        finally:
            signal.alarm(0)
        final_re = self._evaluate_re(model, target, mask)
        return model, final_re
        
    def _evaluate_re(self, model, target, mask):
        """Evaluates relative error on the known elements."""
        with torch.no_grad():
            reconstructed = model.contract()
            norm_of_difference = torch.norm((reconstructed - target) * mask)
            norm_of_target = torch.norm(target * mask)
            return (norm_of_difference / (norm_of_target + 1e-9)).item()

    def _update_global_best(self, model, re):
        """Updates the best model if the new one has lower reconstruction error."""
        if re < self.best_solution['re']:
            self.best_solution['re'] = re
            self.best_solution['params'] = model.get_parameter_count()
            self.best_solution['config'] = model.get_config()
            self.best_model = copy.deepcopy(model)
            logging.info(f"New best solution found! RE: {re:.6f}, Params: {model.get_parameter_count():,}")
            return True
        return False

    def _propose_one_modification(self, model, phase):
        # This function's logic is complex and assumed correct.
        # It proposes 'split' or 'merge' operations.
        if phase == 'expand':
            # Logic to find a node to split
            nodes_with_phys_dims = list(model.phys_dims.keys())
            if not nodes_with_phys_dims: return None
            node_to_split = random.choice(nodes_with_phys_dims)
            return 'split', self._split_node(model, node_to_split), 1.2
        elif phase == 'compress':
            # Logic to find an edge to merge
            if not model.graph.edges(): return None
            edge_to_merge = random.choice(list(model.graph.edges()))
            u, v = edge_to_merge
            return 'merge', self._merge_nodes(model, u, v), 0.8
        return None

    def _merge_nodes(self, model, u, v):
        # ... (existing implementation, assumed correct)
        new_model = copy.deepcopy(model)
        if not new_model.graph.has_edge(u, v): return new_model # Should not happen if called correctly
        # Merge logic here...
        new_node = f"({u}+{v})"
        new_model.graph.add_node(new_node)
        
        # Combine physical dimensions
        u_phys = new_model.phys_dims.pop(u, None)
        v_phys = new_model.phys_dims.pop(v, None)
        if u_phys is not None and v_phys is not None:
            # This case is complex; for now, let's assign one and drop the other
            new_model.phys_dims[new_node] = u_phys 
        elif u_phys is not None:
            new_model.phys_dims[new_node] = u_phys
        elif v_phys is not None:
            new_model.phys_dims[new_node] = v_phys

        # Re-wire edges
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

    def _split_node(self, model, node_to_split):
        # ... (existing implementation, assumed correct)
        new_model = copy.deepcopy(model)
        if not new_model.graph.has_node(node_to_split): return new_model
        
        u, v = f"{node_to_split}a", f"{node_to_split}b"
        new_model.graph.add_node(u)
        new_model.graph.add_node(v)
        
        if node_to_split in new_model.phys_dims:
            new_model.phys_dims[u] = new_model.phys_dims.pop(node_to_split)
            
        neighbors = list(new_model.graph.neighbors(node_to_split))
        split_point = len(neighbors) // 2
        for neighbor in neighbors[:split_point]:
            new_model.graph.add_edge(u, neighbor, **new_model.graph[node_to_split][neighbor])
        for neighbor in neighbors[split_point:]:
            new_model.graph.add_edge(v, neighbor, **new_model.graph[node_to_split][neighbor])

        new_model.graph.add_edge(u, v, bond_dim=2)
        new_model.graph.remove_node(node_to_split)
        new_model._initialize_cores()
        return new_model

    def _create_node_to_dim_map(self, model):
        """
        Creates a mapping from the current nodes of the model to the original tensor dimensions.
        This version relies on the pre-computed 'leaf_nodes' attribute on each node.
        """
        node_to_dim_map = {}
        for node in model.graph.nodes():
            # The leaf nodes (original dimensions) are pre-calculated and stored.
            if 'leaf_nodes' in model.graph.nodes[node]:
                node_to_dim_map[node] = model.graph.nodes[node]['leaf_nodes']
            else:
                # Fallback for nodes that might not have the attribute (e.g. from older versions)
                node_to_dim_map[node] = [node]
        return node_to_dim_map

    def _get_target_view_info(self, model, node_to_dim_map):
        """
        Based on the current model's topology (and its mapping to original dims),
        this function returns the permutation order and target shape
        to view the original data tensor for comparison.
        """
        sorted_nodes = sorted(node_to_dim_map.keys(), key=str)
        
        permute_order = []
        target_shape = []
        
        original_data_shape = self.target_data.shape
        
        for node in sorted_nodes:
            original_dims_for_node = node_to_dim_map[node]
            permute_order.extend([int(d) for d in original_dims_for_node])
            
            dim_size = 1
            for d in original_dims_for_node:
                dim_size *= original_data_shape[int(d)]
            target_shape.append(dim_size)
            
        return permute_order, tuple(target_shape)

    def _get_coarse_grained_data_and_map(self, data, prev_map=None):
        pass

    def _run_two_phase_search_on_scale(self, model, data, mask, expansion_steps, compression_steps, epochs_per_proposal, timeout_per_proposal):
        current_model = model

        logging.info("\n" + "-"*25 + " Expansion Phase " + "-"*25)
        for i in range(expansion_steps):
            proposal = self._propose_one_modification(current_model, phase='expand')
            if proposal is None:
                logging.warning("Expansion proposal generation failed, skipping this step.")
                continue
            
            proposal_name, proposal_model, epoch_multiplier = proposal
            epochs = int(epochs_per_proposal * epoch_multiplier)
            optimized_model, re = self._optimize_model(proposal_model, data, mask, epochs, timeout=timeout_per_proposal)
            
            current_model = optimized_model
            self._update_global_best(current_model, re)
            logging.info(f"  Expansion [{(i+1):>2}/{expansion_steps}] RE: {re:.6f} | Current Best RE: {self.best_solution['re']:.6f}")
            self.search_history.append({'re': re, 'params': current_model.get_parameter_count(), 'config': current_model.get_config()})

        logging.info("\n" + "-"*25 + " Compression Phase " + "-"*24)
        for i in range(compression_steps):
            proposal = self._propose_one_modification(current_model, phase='compress')
            if proposal is None:
                logging.warning("Compression proposal generation failed, skipping this step.")
                continue

            proposal_name, proposal_model, epoch_multiplier = proposal
            epochs = int(epochs_per_proposal * epoch_multiplier)
            optimized_model, re = self._optimize_model(proposal_model, data, mask, epochs, timeout=timeout_per_proposal)

            current_re = self._evaluate_re(current_model, data, mask)
            if re < current_re:
                current_model = optimized_model
                logging.info(f"  Compression [{(i+1):>2}/{compression_steps}] RE: {re:.6f} (Accepted) | Current Best RE: {self.best_solution['re']:.6f}")
            else:
                logging.info(f"  Compression [{(i+1):>2}/{compression_steps}] RE: {re:.6f} (Rejected)")

            self._update_global_best(optimized_model, re)
            self.search_history.append({'re': re, 'params': proposal_model.get_parameter_count(), 'config': proposal_model.get_config()})
        
        return current_model

    def search(self, max_scales, expansion_steps, compression_steps, epochs_per_proposal=150, timeout_per_proposal=60):
        model_on_scale = self.best_model
        data_on_scale = self.target_data
        mask_on_scale = self.mask

        for scale in range(max_scales):
            logging.info("\n" + "="*30 + f" Scale {scale+1}/{max_scales} " + "="*30)
            
            model_after_scale = self._run_two_phase_search_on_scale(
                model=model_on_scale,
                data=data_on_scale,
                mask=mask_on_scale,
                expansion_steps=expansion_steps,
                compression_steps=compression_steps,
                epochs_per_proposal=epochs_per_proposal,
                timeout_per_proposal=timeout_per_proposal
            )
            
            model_on_scale = model_after_scale

        return self.best_model

    def _fine_grain_topology(self, coarse_model, coarse_to_fine_map, fine_scale_shape):
        pass