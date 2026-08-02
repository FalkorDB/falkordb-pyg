"""Full-batch training example: GraphSAGE on a graph stored in FalkorDB.

This script fetches the whole graph from FalkorDB once and then trains on it
in full batches.  It is *not* a mini-batch/NeighborLoader example — see the
README's "Current limitations" for why the loader path still needs the node and
edge types to be primed first.

Prerequisites
-------------
1. Install dependencies::

       pip install 'falkordb-pyg[torch]'

2. Start FalkorDB (requires Docker)::

       docker run -p 6379:6379 falkordb/falkordb:latest

3. Load data — the script below loads a small synthetic graph.
   For real data, adapt the ``load_data_into_falkordb`` section.

Usage
-----
    python examples/train_example.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from falkordb_pyg import get_remote_backend

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

HOST = "localhost"
PORT = 6379
GRAPH_NAME = "papers"

NUM_PAPERS = 1000
NUM_FEATURES = 128
NUM_CLASSES = 10
HIDDEN_CHANNELS = 64
EPOCHS = 3
LR = 0.01
BATCH_SIZE = 128

EDGE_TYPE = ("paper", "cites", "paper")

# ---------------------------------------------------------------------------
# 1. Load synthetic data into FalkorDB
#    (Skip this block if your graph is already in FalkorDB.)
# ---------------------------------------------------------------------------


def load_data_into_falkordb(host: str, port: int, graph_name: str) -> None:
    """Populate FalkorDB with a random paper citation graph."""
    from falkordb import FalkorDB

    db = FalkorDB(host=host, port=port)

    # Drop existing graph if it exists (for repeatability)
    try:
        db.select_graph(graph_name).delete()
    except Exception:
        pass

    graph = db.select_graph(graph_name)

    torch.manual_seed(42)

    print(f"Creating {NUM_PAPERS} paper nodes …")
    # Batch with a parameterised UNWIND: one round-trip per chunk, and values
    # are bound rather than interpolated into the query text.
    batch = 500
    for start in range(0, NUM_PAPERS, batch):
        end = min(start + batch, NUM_PAPERS)
        rows = [
            {
                "pid": i,
                "x": [round(v, 4) for v in torch.randn(NUM_FEATURES).tolist()],
                "y": i % NUM_CLASSES,
            }
            for i in range(start, end)
        ]
        graph.query(
            "UNWIND $rows AS row CREATE (:paper {pid: row.pid, x: row.x, y: row.y})",
            params={"rows": rows},
        )

    # Address nodes by our own key, never by ID(n): FalkorDB reuses internal
    # IDs after deletes and does not promise they start at zero.
    graph.query("CREATE INDEX FOR (p:paper) ON (p.pid)")

    print("Creating citation edges …")
    # Each paper cites ~5 random others.
    edges = []
    for src in range(NUM_PAPERS):
        for dst in torch.randint(0, NUM_PAPERS, (5,)).tolist():
            if dst != src:
                edges.append({"src": src, "dst": dst})

    for start in range(0, len(edges), batch):
        graph.query(
            "UNWIND $rows AS row "
            "MATCH (s:paper {pid: row.src}), (d:paper {pid: row.dst}) "
            "CREATE (s)-[:cites]->(d)",
            params={"rows": edges[start : start + batch]},
        )

    print(f"Data loaded: {NUM_PAPERS} nodes, {len(edges)} edges.")


# ---------------------------------------------------------------------------
# 2. GraphSAGE model
# ---------------------------------------------------------------------------


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# 3. Main training loop
# ---------------------------------------------------------------------------


def main():
    # --- (Optional) populate FalkorDB with synthetic data ---
    print("Loading data into FalkorDB …")
    try:
        load_data_into_falkordb(HOST, PORT, GRAPH_NAME)
    except Exception as exc:
        print(f"Could not connect to FalkorDB: {exc}")
        print("Please start FalkorDB: docker run -p 6379:6379 falkordb/falkordb:latest")
        return

    # --- Connect and create the remote backend ---
    print("Creating remote backend …")
    feature_store, graph_store = get_remote_backend(
        host=HOST,
        port=PORT,
        graph_name=GRAPH_NAME,
    )

    # --- Pre-fetch features and topology (public API) ---
    print("Fetching node features …")
    x = feature_store.get_tensor("paper", "x")  # shape: (N, 128), float
    y = feature_store.get_tensor("paper", "y").squeeze(1)  # shape: (N,), int64

    print("Fetching edge index …")
    src, dst = graph_store.get_edge_index(EDGE_TYPE, layout="coo")
    edge_index = torch.stack([src, dst], dim=0)  # shape: (2, E)

    num_nodes = x.shape[0]
    print(f"Graph: {num_nodes} nodes, {edge_index.shape[1]} edges")

    # --- Train / val / test split (80/10/10) ---
    perm = torch.randperm(num_nodes)
    train_idx = perm[: int(0.8 * num_nodes)]
    val_idx = perm[int(0.8 * num_nodes) : int(0.9 * num_nodes)]
    test_idx = perm[int(0.9 * num_nodes) :]

    # --- Model, optimiser ---
    model = GraphSAGE(
        in_channels=NUM_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=NUM_CLASSES,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    def train():
        model.train()
        optimiser.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimiser.step()
        return float(loss)

    @torch.no_grad()
    def test():
        model.eval()
        out = model(x, edge_index)
        pred = out.argmax(dim=-1)
        accs = {}
        for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            accs[split] = float((pred[idx] == y[idx]).float().mean())
        return accs

    # --- Training loop ---
    for epoch in range(1, EPOCHS + 1):
        loss = train()
        accs = test()
        print(
            f"Epoch {epoch:02d} | Loss: {loss:.4f} | "
            f"Train: {accs['train']:.4f} | "
            f"Val: {accs['val']:.4f} | "
            f"Test: {accs['test']:.4f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
