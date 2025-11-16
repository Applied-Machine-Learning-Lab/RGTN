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
            if len(operands) == 1: 
                if self.phys_dims:
                    phys_leg_idx = -1
                    num_dims = len(operands[0].shape)
                    permute_order = [phys_leg_idx % num_dims] + [i for i in range(num_dims-1)]
                    return operands[0].permute(permute_order)
                return operands[0]

            if use_optimizer:
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
    def __init__(self, initial_model, target_data):
        self.initial_model = initial_model
        self.target_data = target_data
        self.original_params = torch.prod(torch.tensor(target_data.shape)).item()
        self.device = target_data.device
        self.best_solution = {"re": float('inf'), "cr": float('inf'), "model": None, "loss": float('inf')}
        self.search_history = []
        self.loss_fn = CustomLoss()

    def _optimize_model(self, model, target, epochs=150, lr=0.01, timeout=60, node_to_dim_map=None):
        if timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        try:
            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            patience, no_improve_count = 25, 0
            best_loss = float('inf')

            if node_to_dim_map:
                permute_order, target_shape = self._get_target_view_info(model, node_to_dim_map)
                if permute_order != list(range(len(permute_order))):
                    permuted_target = target.permute(*permute_order)
                else:
                    permuted_target = target
                coarse_target = permuted_target.reshape(target_shape).contiguous()
            else:
                coarse_target = target

            target_norm = torch.norm(coarse_target)
        
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    reconstructed = model.contract()
            
                    loss = self.loss_fn(reconstructed, coarse_target)
            
                    if not torch.isfinite(loss): 
                        return None, float('inf')
            
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
                    current_loss_val = loss.item()
                    if current_loss_val < best_loss:
                        best_loss = current_loss_val
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count >= patience:
                        break
                except Exception as e:
                    return None, float('inf')

            final_re = best_loss / target_norm.item() if target_norm > 0 else 0.0
            if timeout > 0:
                signal.alarm(0)
            return model, final_re
        except TimeoutException as e:
            print(f"      - WARNING: {e}. Skipping this proposal.")
            if timeout > 0:
                signal.alarm(0)
            return None, float('inf')
        except Exception:
            if timeout > 0:
                signal.alarm(0)
            return None, float('inf')

    def _calculate_step_loss(self, re, cr, phase, re_target, lambda_re):
        if phase == 'expand':
            return re + 0.01 * cr
        elif phase == 'compress':
            re_penalty = lambda_re * max(0, re - re_target)
            return cr + re_penalty

    def _update_global_best(self, model, re, cr, re_target, cr_target, lambda_re):
        new_loss = cr + lambda_re * max(0, re - re_target)
        current_best_loss = self.best_solution.get('loss', float('inf'))

        if new_loss < current_best_loss:
            self.best_solution['re'] = re
            self.best_solution['cr'] = cr
            self.best_solution['model'] = copy.deepcopy(model)
            self.best_solution['loss'] = new_loss
            
            is_valid_final = (re <= re_target and cr <= cr_target)
            
            print(f"  New Global Best (Loss: {new_loss:.4f}, Final Criteria Met: {is_valid_final}): RE={re:.4f}, CR={cr:.4%}")

    def _propose_one_modification(self, model, phase):
        model_copy = copy.deepcopy(model)
        
        actions = ['change_bond_dim', 'add_edge', 'remove_edge']
        if phase == 'expand':
            weights = [0.4, 0.6, 0.0] 
        else:
            weights = [0.4, 0.0, 0.6]

        chosen_action = random.choices(actions, weights=weights, k=1)[0]
        
        if chosen_action == 'change_bond_dim':
            if not model_copy.graph.edges:
                return None
            edge_to_modify = random.choice(list(model_copy.graph.edges()))
            u, v = edge_to_modify
            
            current_dim = model_copy.graph[u][v]['bond_dim']
            
            if phase == 'expand':
                multiplier = random.choice([1.25, 1.5, 2.0])
                new_dim = max(2, int(current_dim * multiplier))
                name = f"Increase bond_dim of {edge_to_modify} from {current_dim} to {new_dim}"
            else:
                multiplier = random.choice([0.5, 0.75, 0.8])
                new_dim = max(1, int(current_dim * multiplier))
                if new_dim == current_dim: new_dim = max(1, new_dim-1)
                name = f"Decrease bond_dim of {edge_to_modify} from {current_dim} to {new_dim}"

            if new_dim == 0:
                model_copy.graph.remove_edge(u, v)
            else:
                model_copy.graph[u][v]['bond_dim'] = new_dim
            
            model_copy._initialize_cores()
            return name, model_copy, 1.0

        elif chosen_action == 'add_edge':
            nodes = list(model_copy.graph.nodes())
            if len(nodes) < 2: return None
            
            for _ in range(20):
                u, v = random.sample(nodes, 2)
                if not model_copy.graph.has_edge(u, v):
                    new_dim = random.choice([2,4,6,8])
                    model_copy.graph.add_edge(u, v, bond_dim=new_dim)
                    model_copy._initialize_cores()
                    return f"Add edge ({u}-{v}) with bond_dim={new_dim}", model_copy, 1.5
            return None

        elif chosen_action == 'remove_edge':
            if not model_copy.graph.edges: return None
            
            edge_to_remove = random.choice(list(model_copy.graph.edges()))
            u, v = edge_to_remove
            
            model_copy.graph.remove_edge(u, v)
            model_copy._initialize_cores()
            return f"Remove edge ({u}-{v})", model_copy, 0.8
        
        return None

    def _merge_nodes(self, model, u, v):
        """
        Merges two nodes in the graph, returning a new GraphTN model.
        This is a core part of the coarse-graining step.
        """
        new_model = copy.deepcopy(model)
        
        u_str, v_str = str(u), str(v)
        if u_str > v_str: u_str, v_str = v_str, u_str # Canonical name
        new_node_name = f"({u_str},{v_str})"

        if not (u in new_model.phys_dims and v in new_model.phys_dims):
             print(f"Warning: Cannot merge {u} and {v}, one or both not in phys_dims.")
             return None

        # Pre-fetch leaf nodes before modifying the graph
        u_leaves = new_model.graph.nodes[u].get('leaf_nodes', [u])
        v_leaves = new_model.graph.nodes[v].get('leaf_nodes', [v])

        phys_u = new_model.phys_dims.pop(u)
        phys_v = new_model.phys_dims.pop(v)
        new_model.phys_dims[new_node_name] = phys_u * phys_v
        
        # Store merge metadata for potential splitting later
        new_model.graph.add_node(new_node_name)
        new_model.graph.nodes[new_node_name]['merged_from'] = {
            'nodes': [u, v],
            'phys_dims': [phys_u, phys_v]
        }
        # Correctly compute and store the leaf nodes for the new merged node
        new_model.graph.nodes[new_node_name]['leaf_nodes'] = sorted(list(set(u_leaves + v_leaves)))


        # Rewire edges from u and v to the new node
        new_edges = {}
        original_edges = list(new_model.graph.edges([u, v], data=True))

        for node1, node2, data in original_edges:
            if (node1, node2) == (u, v) or (node1, node2) == (v, u):
                continue # Skip the internal edge being merged

            other_node = node2 if node1 in (u,v) else node1
            
            if other_node in new_edges:
                # If a neighbor was connected to both u and v, merge the bonds.
                # Heuristic: sum the bond dimensions.
                new_edges[other_node]['bond_dim'] += data['bond_dim']
            else:
                new_edges[other_node] = data
        
        new_model.graph.remove_nodes_from([u, v])

        for neighbor, data in new_edges.items():
            new_model.graph.add_edge(new_node_name, neighbor, **data)
            
        new_model._initialize_cores()
        return new_model

    def _split_node(self, model, node_to_split):
        """
        Splits a merged node back into its original constituent nodes using SVD.
        This is a core part of the fine-graining step.
        """
        if 'merged_from' not in model.graph.nodes[node_to_split]:
            print(f"Node {node_to_split} is not a merged node and cannot be split.")
            return None

        new_model = copy.deepcopy(model)
        
        # --- 1. Retrieve metadata and original nodes ---
        merge_info = new_model.graph.nodes[node_to_split]['merged_from']
        nodes_original = merge_info['nodes']
        phys_dims_original = merge_info['phys_dims']
        
        u, v = nodes_original[0], nodes_original[1]
        phys_dim_u, phys_dim_v = phys_dims_original[0], phys_dims_original[1]

        # --- 2. Prepare the core for SVD ---
        core_to_split = new_model.cores[node_to_split].detach()
        
        # Identify which axes of the core tensor belong to which final node.
        neighbor_dims_map = {} # neighbor -> axis_index
        current_axis = 1 # Axis 0 is the physical dimension
        for neighbor in sorted(new_model.graph.neighbors(node_to_split), key=str):
            neighbor_dims_map[neighbor] = current_axis
            current_axis += 1
            
        # We need to know which original neighbor was connected to u and which to v.
        # This requires more metadata from the merge step. For now, we assume a simple case
        # where we can distribute them, or we make a heuristic choice.
        # Let's assume a simple split for now: all neighbors are re-attached to node u,
        # and a new bond is created between u and v.

        # Reshape the core into a matrix for SVD, splitting the physical dimension
        matrix_to_svd = core_to_split.reshape(phys_dim_u, phys_dim_v, -1)
        matrix_to_svd = matrix_to_svd.permute(0, 2, 1).reshape(phys_dim_u * core_to_split.shape[1:].numel(), phys_dim_v)

        # --- 3. Perform SVD ---
        try:
            U, S, Vh = torch.linalg.svd(matrix_to_svd, full_matrices=False)
        except torch.linalg.LinAlgError:
            print(f"SVD failed for splitting node {node_to_split}. Cannot split.")
            return None
        
        # Determine the new bond dimension
        new_bond_dim = S.size(0)
        
        # --- 4. Create new cores and graph structure ---
        # Remove the merged node
        external_neighbors = list(new_model.graph.neighbors(node_to_split))
        new_model.graph.remove_node(node_to_split)
        
        # Add back original nodes
        new_model.graph.add_node(u)
        new_model.graph.nodes[u]['leaf_nodes'] = [u]
        new_model.graph.add_node(v)
        new_model.graph.nodes[v]['leaf_nodes'] = [v]
        
        # Add new bond between them
        new_model.graph.add_edge(u, v, bond_dim=new_bond_dim)
        
        # Reconnect external neighbors (simple heuristic: all to u)
        for neighbor in external_neighbors:
             new_model.graph.add_edge(u, neighbor, bond_dim=2) # Default bond dim

        # Update physical dimensions dict
        new_model.phys_dims.pop(node_to_split)
        new_model.phys_dims[u] = phys_dim_u
        new_model.phys_dims[v] = phys_dim_v

        # Create new parameter tensors
        new_model._initialize_cores() # Re-init all for simplicity
        
        # Manually set the data for the new cores from SVD results
        # This is complex due to dimension ordering. A full implementation
        # needs to carefully reshape U and S*Vh to match the new core shapes.
        # As a starting point, re-initializing is safer.
        
        print(f"Split {node_to_split} into {u} and {v} with new bond dim {new_bond_dim}.")
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

    def _run_two_phase_search_on_scale(self, model, data, expansion_steps, compression_steps, re_target, cr_target, lambda_re, epochs_per_proposal, timeout_per_proposal, node_to_dim_map=None):
        print(f"--- Running 2-Phase Search (Expand: {expansion_steps}, Compress: {compression_steps}) ---")
        
        current_model = copy.deepcopy(model)
        
        # Create the correctly-shaped view of the target tensor for this scale
        if node_to_dim_map:
            permute_order, target_shape = self._get_target_view_info(current_model, node_to_dim_map)
            # Ensure permutation is not redundant
            if permute_order != list(range(len(permute_order))):
                 permuted_target = data.permute(*permute_order)
            else:
                 permuted_target = data
            coarse_data_view = permuted_target.reshape(target_shape).contiguous()
        else:
            coarse_data_view = data

        print(f"\n===== PHASE 1: EXPANSION ({expansion_steps} steps) =====")
        
        best_step_loss = float('inf') 

        for step in range(expansion_steps):
            proposal = self._propose_one_modification(current_model, phase='expand')
            if not proposal:
                print("  No further valid expansion proposals can be generated.")
                break
            
            name, proposal_model, epochs_multiplier = proposal
            epochs = int(epochs_per_proposal * epochs_multiplier)
            print(f"\n--- Expansion Step {step + 1}/{expansion_steps}: Proposing '{name}' ---")
            
            optimized_model, re = self._optimize_model(proposal_model, data, epochs=epochs, timeout=timeout_per_proposal, node_to_dim_map=self._create_node_to_dim_map(proposal_model))

            if optimized_model:
                cr = optimized_model.get_parameter_count() / self.original_params
                self.search_history.append({
                    "re": re, "cr": cr, "config": optimized_model.get_config(), "params": optimized_model.get_parameter_count()
                })
                self._update_global_best(optimized_model, re, cr, re_target, cr_target, lambda_re)
                
                step_loss = self._calculate_step_loss(re, cr, 'expand', re_target, lambda_re)
                print(f"      - Proposal accepted: RE={re:.4f}, CR={cr:.4%} (Step Loss: {step_loss:.4f})")
                
                if step_loss < best_step_loss:
                    best_step_loss = step_loss
                    current_model = optimized_model
                    print(f"      - Continuing search from this new state.")
            else:
                print(f"      - Proposal rejected (optimization failed or timed out).")

        print(f"\n===== PHASE 2: COMPRESSION ({compression_steps} steps) =====")
        
        with torch.no_grad():
            initial_re_for_compression = self.loss_fn(current_model.contract(), coarse_data_view) / torch.norm(coarse_data_view)
            initial_cr_for_compression = current_model.get_parameter_count() / self.original_params
        best_step_loss = self._calculate_step_loss(initial_re_for_compression, initial_cr_for_compression, 'compress', re_target, lambda_re)
        print(f"--- Starting compression phase with initial Step Loss: {best_step_loss:.4f} ---")

        for step in range(compression_steps):
            proposal = self._propose_one_modification(current_model, phase='compress')
            if not proposal:
                print("  No further valid compression proposals can be generated.")
                break
                    
            name, proposal_model, epochs_multiplier = proposal
            epochs = int(epochs_per_proposal * epochs_multiplier)
            print(f"\n--- Compression Step {step + 1}/{compression_steps}: Proposing '{name}' ---")

            optimized_model, re = self._optimize_model(proposal_model, data, epochs=epochs, timeout=timeout_per_proposal, node_to_dim_map=self._create_node_to_dim_map(proposal_model))
            
            if optimized_model:
                cr = optimized_model.get_parameter_count() / self.original_params
                self.search_history.append({
                    "re": re, "cr": cr, "config": optimized_model.get_config(), "params": optimized_model.get_parameter_count()
                })
                self._update_global_best(optimized_model, re, cr, re_target, cr_target, lambda_re)

                step_loss = self._calculate_step_loss(re, cr, 'compress', re_target, lambda_re)
                print(f"      - Proposal accepted: RE={re:.4f}, CR={cr:.4%} (Step Loss: {step_loss:.4f})")

                if step_loss < best_step_loss:
                    best_step_loss = step_loss
                    current_model = optimized_model
                    print(f"      - Continuing search from this new state.")
            else:
                print(f"      - Proposal rejected (optimization failed or timed out).")
        
        return current_model

    def search(self, max_scales, expansion_steps, compression_steps, re_target, cr_target, lambda_re=2.0, epochs_per_proposal=150, timeout_per_proposal=60):
        
        current_model = copy.deepcopy(self.initial_model)
        current_data = self.target_data
        
        with torch.no_grad():
            initial_map = self._create_node_to_dim_map(current_model)
            permute_order, target_shape = self._get_target_view_info(current_model, initial_map)
            initial_target_view = current_data.permute(*permute_order).reshape(target_shape)
            re = self.loss_fn(current_model.contract(), initial_target_view) / torch.norm(initial_target_view)
            cr = current_model.get_parameter_count() / self.original_params
            self._update_global_best(current_model, re.item(), cr, re_target, cr_target, lambda_re)

        print("\n" + "="*60)
        print("Starting RGTN Multi-Scale Search")
        print("="*60)

        for scale in range(max_scales):
            print(f"\n===== SCALE {scale + 1}/{max_scales} (Nodes: {len(current_model.graph.nodes())}) =====")
            
            node_to_dim_map = self._create_node_to_dim_map(current_model)
            
            current_model = self._run_two_phase_search_on_scale(
                model=current_model,
                data=current_data,
                expansion_steps=expansion_steps,
                compression_steps=compression_steps,
                re_target=re_target,
                cr_target=cr_target,
                lambda_re=lambda_re,
                epochs_per_proposal=epochs_per_proposal,
                timeout_per_proposal=timeout_per_proposal,
                node_to_dim_map=node_to_dim_map
            )
            
            if scale < max_scales - 1:
                print("\n--- Coarse-graining for next scale ---")
                
                if len(current_model.graph.edges()) == 0:
                    print("No edges left to merge. Stopping multi-scale search.")
                    break

                # Heuristic: merge nodes connected by the largest bond dimension
                edge_to_merge = max(current_model.graph.edges(data=True), key=lambda x: x[2].get('bond_dim', 0))
                u, v, data = edge_to_merge
                
                print(f"Merging nodes {u} and {v} (connected by bond dim {data.get('bond_dim')}) to create a coarser model.")
                
                merged_model = self._merge_nodes(current_model, u, v)
                if merged_model:
                    current_model = merged_model
                else:
                    print("Failed to merge nodes. Stopping multi-scale search.")
                    break

        print("\n" + "="*60)
        print("RGTN Search Finished")
        print("="*60)
        
        return self.best_solution['model']

    def _fine_grain_topology(self, coarse_model, coarse_to_fine_map, fine_scale_shape):
        pass