"""Attention visualization data preparation helpers."""

from __future__ import annotations


def normalize_attention(weights: dict[str, float]) -> dict[str, float]:
    """Generic absolute-global normalization (not canonical directed-GAT normalization)."""
    total = sum(abs(weight) for weight in weights.values())
    if total == 0:
        return {key: 0.0 for key in weights}
    return {key: abs(weight) / total for key, weight in weights.items()}

def plot_attention_edges(explanation,*,top_k=10):
    import matplotlib.pyplot as plt
    records=explanation.edges[:top_k]; fig,ax=plt.subplots(figsize=(8,max(3,len(records)*.35)))
    labels=[f"{r.source_node_id} → {r.target_node_id}" for r in records]
    ax.barh(labels[::-1],[r.aggregated_attention_coefficient for r in records][::-1]); ax.set_xlabel("aggregated attention coefficient")
    semantics="weighted; snapshot economic edge weight used by model" if explanation.uses_economic_edge_weight else "unweighted; snapshot economic edge weight is context only"
    ax.set_title(f"{explanation.model_name} directed attention ({semantics})\nperiod {explanation.period}, layer {explanation.layer_index}, heads {explanation.head_aggregation}")
    fig.text(.5,.01,"Model association/local sensitivity — not a causal estimate",ha="center",fontsize=8); fig.tight_layout(rect=(0,.04,1,1)); return fig

def plot_attention_subgraph(explanation,*,top_k=10,seed=0):
    import matplotlib.pyplot as plt
    import networkx as nx
    graph=nx.MultiDiGraph(); records=explanation.edges[:top_k]
    for r in records: graph.add_edge(r.source_node_id,r.target_node_id,key=r.edge_position,attention=r.aggregated_attention_coefficient,weight=r.economic_edge_weight)
    pos=nx.spring_layout(graph,seed=seed); fig,ax=plt.subplots(figsize=(8,6)); nx.draw_networkx_nodes(graph,pos,ax=ax); nx.draw_networkx_labels(graph,pos,ax=ax)
    nx.draw_networkx_edges(graph,pos,ax=ax,arrows=True,width=[1+4*d["attention"] for *_,d in graph.edges(data=True)])
    semantics="economic edge weight used by model" if explanation.uses_economic_edge_weight else "economic edge weight shown as context only"
    ax.set_title(f"Directed attention subgraph — {semantics}"); ax.axis("off"); fig.text(.5,.01,"Model association/local sensitivity — not a causal estimate",ha="center",fontsize=8); return fig

__all__=["normalize_attention","plot_attention_edges","plot_attention_subgraph"]
