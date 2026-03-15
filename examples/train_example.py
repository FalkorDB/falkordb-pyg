"""Example: Training a GNN with FalkorDB as a remote backend.

This example demonstrates how to use falkordb-pyg to train a Graph Neural
Network on data stored in FalkorDB, using PyG's NeighborLoader for
mini-batch sampling.

Prerequisites:
    1. Start FalkorDB:
       docker run -p 6379:6379 -d falkordb/falkordb:edge

    2. Install dependencies:
       uv sync --extra test

Usage:
    uv run python examples/train_example.py

"""

from falkordb import FalkorDB

# ---------------------------------------------------------------------------
# 1. Populate FalkorDB with a small citation graph (skip if already loaded)
# ---------------------------------------------------------------------------


def populate_graph(graph):
    """Create a small citation graph in FalkorDB for demonstration."""
    graph.query("MATCH (n) DETACH DELETE n")  # Clear existing data

    # Create Paper nodes with 4-dimensional feature vectors and labels
    for i in range(6):
        x = [float(j == i % 4) for j in range(4)]
        y = i % 3
        graph.query(f"CREATE (:Paper {{x: {x}, y: {y}, name: 'paper_{i}'}})")

    # Create CITES edges
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (1, 5)]
    for src, dst in edges:
        graph.query(
            f"MATCH (s:Paper {{name: 'paper_{src}'}}), "
            f"(d:Paper {{name: 'paper_{dst}'}}) "
            f"CREATE (s)-[:CITES]->(d)"
        )

    print(f"Created graph with 6 Paper nodes and {len(edges)} CITES edges")


# ---------------------------------------------------------------------------
# 2. Connect to FalkorDB and set up the remote backend
# ---------------------------------------------------------------------------


def main():
    # Connect and populate
    db = FalkorDB(host="localhost", port=6379)
    graph = db.select_graph("citation_graph")
    populate_graph(graph)

    # Create the remote backend
    from falkordb_pyg import FalkorDBFeatureStore, FalkorDBGraphStore

    feature_store = FalkorDBFeatureStore(graph)
    graph_store = FalkorDBGraphStore(graph)

    # -----------------------------------------------------------------------
    # 3. Use the backend with PyG's NeighborLoader
    # -----------------------------------------------------------------------
    from torch_geometric.data.feature_store import TensorAttr
    from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

    # Pre-warm the caches by explicitly fetching features and topology
    x_attr = TensorAttr(group_name="Paper", attr_name="x")
    y_attr = TensorAttr(group_name="Paper", attr_name="y")
    edge_attr = EdgeAttr(
        edge_type=("Paper", "CITES", "Paper"),
        layout=EdgeLayout.COO,
        is_sorted=False,
    )

    x = feature_store._get_tensor(x_attr)
    y = feature_store._get_tensor(y_attr)
    edge_index = graph_store._get_edge_index(edge_attr)

    print(f"Node features x: {x.shape}")  # (6, 4)
    print(f"Node labels y: {y.shape}")  # (6, 1)
    print(f"Edge index: {edge_index.shape}")  # (2, 7)

    # -----------------------------------------------------------------------
    # 4. Simple GNN training loop (using raw tensors from remote backend)
    # -----------------------------------------------------------------------
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=-1)

    model = SimpleGCN(in_channels=4, hidden_channels=16, out_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    labels = y.squeeze(-1).long()

    print("\nTraining for 10 epochs...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1:02d}: Loss = {loss.item():.4f}")

    print("\nDone! The GNN was trained using data stored in FalkorDB.")
    print(
        "In a real use-case, use PyG's NeighborLoader with the "
        "(feature_store, graph_store) tuple for scalable mini-batch training."
    )


if __name__ == "__main__":
    main()
