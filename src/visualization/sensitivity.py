from __future__ import annotations
WARNING="Model association/local sensitivity — not a causal estimate"
def _finish(fig): fig.text(.5,.01,WARNING,ha="center",fontsize=8); fig.tight_layout(rect=(0,.04,1,1)); return fig
def plot_shock_prediction_changes(explanation,*,target_name,top_k=10):
    import matplotlib.pyplot as plt
    rows=[r for r in explanation.records if r.target_name==target_name][:top_k]; fig,ax=plt.subplots(); ax.barh([r.node_id for r in rows][::-1],[r.prediction_delta for r in rows][::-1]); ax.set_title(f"Local prediction change — {target_name}"); ax.set_xlabel("signed prediction delta"); return _finish(fig)
def plot_feature_ablation_sensitivity(explanation):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); labels=[f"{r.feature_name} / {r.target_name}" for r in explanation.records]; ax.barh(labels[::-1],[r.mean_absolute_prediction_change for r in explanation.records][::-1]); ax.set_title("One-at-a-time feature ablation sensitivity"); return _finish(fig)
def plot_edge_ablation_sensitivity(explanation):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); labels=[f"{r.mode} {r.edge_positions} / {r.target_name}" for r in explanation.records]; ax.barh(labels,[r.mean_absolute_prediction_change for r in explanation.records]); ax.set_title("Frozen-model edge sensitivity"); return _finish(fig)
def plot_gpr_propagation_coefficients(explanation):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); ax.bar([r.step for r in explanation.records],[r.coefficient for r in explanation.records]); ax.set(xlabel="propagation step",ylabel="softmax coefficient",title="GPR propagation-step weighting (not attention)"); return _finish(fig)
def plot_shock_network(bundle,*,target_name,seed=0):
    import matplotlib.pyplot as plt
    import networkx as nx
    graph=nx.MultiDiGraph(); attention=bundle.attention
    if attention:
        for e in attention.edges: graph.add_edge(e.source_node_id,e.target_node_id,width=e.aggregated_attention_coefficient,economic=e.economic_edge_weight)
    values={r.node_id:r.prediction_delta for r in bundle.perturbation.records if r.target_name==target_name}; graph.add_nodes_from(values)
    pos=nx.spring_layout(graph,seed=seed); fig,ax=plt.subplots(); colors=[values.get(n,0.) for n in graph]; shapes=[n==bundle.source_node_id for n in graph]
    normal=[n for n,s in zip(graph,shapes) if not s]; source=[n for n,s in zip(graph,shapes) if s]
    nx.draw_networkx_nodes(graph,pos,nodelist=normal,node_color=[values.get(n,0) for n in normal],cmap="coolwarm",ax=ax); nx.draw_networkx_nodes(graph,pos,nodelist=source,node_shape="*",node_color=[values.get(n,0) for n in source],cmap="coolwarm",node_size=500,ax=ax); nx.draw_networkx_labels(graph,pos,ax=ax)
    if attention: nx.draw_networkx_edges(graph,pos,width=[1+3*d["width"] for *_,d in graph.edges(data=True)],arrows=True,ax=ax)
    ax.set_title("Node value = signed model prediction delta\nEdge width = aggregated edge attention coefficient\nEconomic edge weight = separate graph context"); ax.axis("off"); return _finish(fig)
__all__=["plot_shock_prediction_changes","plot_shock_network","plot_feature_ablation_sensitivity","plot_edge_ablation_sensitivity","plot_gpr_propagation_coefficients"]
