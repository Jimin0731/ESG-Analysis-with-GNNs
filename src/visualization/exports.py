"""Explicit deterministic text and static-figure exports."""
from __future__ import annotations
import csv,json
from dataclasses import asdict
from pathlib import Path
from src.evaluation.explanation_contracts import ExplanationExportManifest,InterpretabilityValidationError
from .attention import plot_attention_edges,plot_attention_subgraph
from .sensitivity import (plot_shock_prediction_changes,plot_shock_network,plot_feature_ablation_sensitivity,plot_edge_ablation_sensitivity,plot_gpr_propagation_coefficients)

def _csv(path,records):
    rows=[asdict(r) for r in records]
    if not rows:return
    with path.open("x",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
def export_explanation_bundle(bundle,output_directory,*,figure_format="png"):
    if figure_format not in {"png","svg"}: raise InterpretabilityValidationError("figure format must be png or svg")
    root=Path(output_directory); root.mkdir(parents=True,exist_ok=True); generated=[]
    def write_json(name,value):
        path=root/name
        with path.open("x",encoding="utf-8") as f: json.dump(asdict(value),f,ensure_ascii=False,indent=2,allow_nan=False)
        generated.append(name)
    write_json("explanation.json",bundle)
    tables=(("node_perturbation.csv",bundle.perturbation.records),("attention_heads.csv",bundle.attention.heads if bundle.attention else ()),("attention_edges.csv",bundle.attention.edges if bundle.attention else ()),("attention_paths.csv",bundle.attention_paths.records if bundle.attention_paths else ()),("feature_ablation.csv",bundle.feature_ablation.records if bundle.feature_ablation else ()),("edge_ablation.csv",bundle.edge_ablation.records if bundle.edge_ablation else ()),("gpr_coefficients.csv",bundle.propagation.records if bundle.propagation else ()))
    for name,records in tables:
        if records: _csv(root/name,records); generated.append(name)
    figures=[]
    if bundle.attention: figures += [("attention_edges",plot_attention_edges(bundle.attention)),("attention_subgraph",plot_attention_subgraph(bundle.attention)),("shock_network",plot_shock_network(bundle,target_name=bundle.target_names[0]))]
    figures.append(("shock_prediction_changes",plot_shock_prediction_changes(bundle.perturbation,target_name=bundle.target_names[0])))
    if bundle.feature_ablation: figures.append(("feature_ablation",plot_feature_ablation_sensitivity(bundle.feature_ablation)))
    if bundle.edge_ablation: figures.append(("edge_ablation",plot_edge_ablation_sensitivity(bundle.edge_ablation)))
    if bundle.propagation: figures.append(("gpr_coefficients",plot_gpr_propagation_coefficients(bundle.propagation)))
    import matplotlib.pyplot as plt
    for stem,figure in figures:
        name=f"{stem}.{figure_format}"; figure.savefig(root/name); plt.close(figure); generated.append(name)
    manifest=ExplanationExportManifest(tuple(generated)+( "manifest.json",),bundle.warnings)
    write_json("manifest.json",manifest)
    return manifest
__all__=["export_explanation_bundle"]
