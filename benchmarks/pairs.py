def pair_view(z, pos, batch, num_graphs=None):
    """Per-pair views of a batch of two-atom graphs.

    PyG stores the atoms of each graph contiguously, so with exactly two atoms per graph
    `z` reshapes to [B, 2]. Returns (z [B, 2], pos [B, 2, 3], interatomic distance [B]).
    """
    num_graphs = int(batch.max()) + 1 if num_graphs is None else num_graphs
    if z.numel() != 2 * num_graphs:
        raise ValueError(f"expected two atoms per graph, got {z.numel()} atoms for {num_graphs} graphs")
    pos = pos.view(num_graphs, 2, 3)
    return z.view(num_graphs, 2), pos, (pos[:, 0] - pos[:, 1]).norm(dim=-1)
