import numpy as np
import scipy.io
import os
import random
import torch
import opt_einsum as oe

class TNGenerator:
    """
    A simple class to generate a tensor with a known underlying Tensor Network structure.
    This version creates EXTREMELY sparse structures suitable for aggressive compression.
    """
    def __init__(self, num_nodes, bond_dim_range, num_edges_range, phys_dim_range=None, phys_dims_dict=None, scale_factor=0.2, device='cpu', num_high_rank_bonds=0):
        self.num_nodes = num_nodes
        
        if phys_dims_dict:
            self.phys_dims = phys_dims_dict
        else:
            self.phys_dims = {i: random.randint(*phys_dim_range) for i in range(num_nodes)}

        # For aggressive compression, use minimal edges - just enough to keep connected
        self.num_edges = min(random.randint(*num_edges_range), num_nodes + 2)  # Very sparse!
        self.bond_dim_range = bond_dim_range
        self.scale_factor = scale_factor  # Store the scale factor
        self.device = device
        
        self.nodes = list(range(num_nodes))
        self.edges = self._generate_minimal_edges()
        
        # --- Controlled Complexity Bond Dimension Generation ---
        self.bond_dims = {}
        # Get a list of edges to choose from for high rank
        edges_for_high_rank = list(self.edges)
        random.shuffle(edges_for_high_rank)
        
        # Assign high rank bonds first
        high_rank_bonds_assigned = 0
        if num_high_rank_bonds > 0 and len(bond_dim_range) > 1:
            high_rank_value = bond_dim_range[1]
            for edge in edges_for_high_rank[:num_high_rank_bonds]:
                self.bond_dims[edge] = high_rank_value
                high_rank_bonds_assigned += 1

        # Assign normal rank to the rest
        normal_rank_value = bond_dim_range[0]
        for edge in self.edges:
            if edge not in self.bond_dims:
                self.bond_dims[edge] = normal_rank_value
        
        self.cores = self._generate_cores()

    def _generate_minimal_edges(self):
        """Generates a minimal set of edges for maximum sparsity."""
        edges = set()
        if self.num_nodes <= 1:
            return []

        # Create a VERY sparse tree structure first
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)
        
        # Create a linear chain (minimal connectivity)
        for i in range(len(nodes) - 1):
            u, v = tuple(sorted((nodes[i], nodes[i+1])))
            edges.add((u, v))
        
        # Add only a few extra edges to reach target (if any)
        remaining_edges = max(0, self.num_edges - len(edges))
        attempts = 0
        while len(edges) < self.num_edges and attempts < 20:
            u, v = tuple(sorted(random.sample(self.nodes, 2)))
            if (u, v) not in edges:
                edges.add((u, v))
            attempts += 1
            
        return list(edges)

    def _generate_cores(self):
        """Generates random tensor cores with carefully controlled values for sparsity."""
        cores = {}
        edge_map = {node: [] for node in self.nodes}
        for u, v in self.edges:
            edge_map[u].append((u, v))
            edge_map[v].append((u, v))
            
        for node in self.nodes:
            core_shape = [self.phys_dims[node]]
            for edge in edge_map[node]:
                core_shape.append(self.bond_dims[edge])
            
            # Use the provided scale_factor consistently
            cores[node] = torch.randn(core_shape, device=self.device) * self.scale_factor + 0.01  # Add small bias
        return cores

    def contract(self):
        """Contracts the tensor network to get the full tensor."""
        operands = list(self.cores.values())
        
        # Create a mapping from edge to a unique einsum symbol
        edge_to_symbol = {edge: oe.get_symbol(i) for i, edge in enumerate(self.edges)}
        
        # Create a mapping for physical dimensions
        phys_to_symbol = {node: oe.get_symbol(i + len(self.edges)) for i, node in enumerate(self.nodes)}

        input_subs = []
        for node in self.nodes:
            sub = phys_to_symbol[node]
            # Find all edges connected to this node
            connected_edges = [e for e in self.edges if node in e]
            for edge in connected_edges:
                sub += edge_to_symbol[edge]
            input_subs.append(sub)

        output_subs = "".join(phys_to_symbol[node] for node in sorted(self.nodes))
        einsum_str = ",".join(input_subs) + "->" + output_subs
        
        print("Contracting tensor network...")
        print(f"Einsum string: {einsum_str}")

        # Use opt_einsum to find the optimal contraction path and contract
        full_tensor = oe.contract(einsum_str, *operands, backend='torch', optimize='auto-hq')
        
        return full_tensor

def generate_structured_tensor(order, file_path):
    """
    Generates a high-order tensor with a balanced strategy: large initial size
    for low CR potential, but constrained to prevent memory errors, and simple
    internal structure for high compressibility.
    - Tensor Order: 6, 8, 10
    - Mode Size: Varies by order to balance size and feasibility.
    - Edge Number: Fixed to 6 (from {6, 8, 10}) for simplicity.
    - Edge Rank: Fixed to 2 (from {2, 3}) for compressibility.
    """
    print(f"--- Generating BALANCED {order}th-order tensor ---")

    phys_dims_config = None # Default to None

    if order == 6:
        phys_dim_range = (7, 8)  # Keep large dims for a good CR denominator
    elif order == 8:
        phys_dim_range = (7, 8)  # Keep large dims, will be tuned with bond rank
    elif order == 10:
        print("Applying special physical dimension configuration for 10th-order tensor...")
        # A mix of large and smaller dims to balance size and CR potential, per user request
        # New aggressive strategy: 7x'8', 2x'7', 1x'6'
        dims = [8, 8, 8, 8, 8, 8, 8, 7, 7, 6]
        random.shuffle(dims)
        phys_dims_config = {i: dims[i] for i in range(order)}
        phys_dim_range = None # Ensure we don't use the range

    # --- Controlled Complexity Generation ---
    num_edges_range = (6, 6) # Keep edges constant and minimal
    num_high_rank_bonds = 0 # Default for 10th order
    
    # Strategy: Slightly increase complexity to target a CR just below the baseline
    if order == 6:
        bond_dim_range = (2, 3)
        num_high_rank_bonds = 1 # Use one rank-3 bond
    elif order == 8:
        bond_dim_range = (2, 3)
        num_high_rank_bonds = 1 # Use one rank-3 bond for a more 'reasonable' CR
    else: # 10th order
        bond_dim_range = (2, 2)   # Keep it simple for the largest tensor

    # Generate the TN
    tn_generator = TNGenerator(
        num_nodes=order,
        phys_dim_range=phys_dim_range,
        phys_dims_dict=phys_dims_config,
        bond_dim_range=bond_dim_range,
        num_edges_range=num_edges_range,
        scale_factor=0.2,
        num_high_rank_bonds=num_high_rank_bonds 
    )
    
    print(f"Generated structure: {len(tn_generator.edges)} edges with bond dims {[tn_generator.bond_dims[e] for e in tn_generator.edges]}")
    
    # Contract to a full tensor
    tensor = tn_generator.contract()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the tensor to a .mat file
    print(f"Saving tensor with shape {tensor.shape} to {file_path}...")
    scipy.io.savemat(file_path, {'tensor': tensor.cpu().numpy()})
    print("--- Generation complete ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate a structured high-order tensor.")
    parser.add_argument('--order', type=int, required=True, choices=[6, 8, 10], help='Order of the tensor to generate.')
    args = parser.parse_args()

    # Generate a single tensor of the specified order
    output_dir = f'../data/order_{args.order}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/structured_{args.order}th_order_tensor_1.mat'
    
    print(f"\n{'='*50}")
    print(f"Generating single {args.order}th-order tensor...")
    print(f"{'='*50}")
    
    generate_structured_tensor(args.order, output_file)
    
    print(f"Completed: {output_file}") 