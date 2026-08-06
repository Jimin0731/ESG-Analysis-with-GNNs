"""Fast synthetic validation-snapshot interpretability smoke."""
from __future__ import annotations
import json,math,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import torch
from src.models import ModelConfig,build_model
from src.experiments.contracts import SupervisedSnapshot
from src.evaluation import build_shock_explanation
from src.visualization import export_explanation_bundle

def main():
    config=ModelConfig("weighted_gat",2,4,2,0.,("target__synthetic",),attention_heads=2,seed=9)
    model=build_model(config); snapshot=SupervisedSnapshot(2021,"validation",("sector_a","sector_b","sector_c"),torch.tensor([[.1,.2],[.3,.4],[.5,.6]]),torch.zeros(3,1),torch.ones(3,1,dtype=torch.bool),torch.tensor([[0,0,1],[1,2,2]]),torch.tensor([1.,.5,1.]),("safe__activity","safe__intensity"),("target__synthetic",))
    bundle=build_shock_explanation(model,snapshot,source_node_id="sector_a",feature_changes={"safe__activity":.25},feature_ablation=("safe__activity",),edge_ablation={"edge_positions":(0,),"mode":"zero_weight"},top_k=3)
    with tempfile.TemporaryDirectory() as directory:
        manifest=export_explanation_bundle(bundle,directory)
        for name in manifest.generated_files:
            path=Path(directory,name); assert path.is_file() and path.stat().st_size
        text=Path(directory,"explanation.json").read_text(encoding="utf-8"); parsed=json.loads(text); assert "NaN" not in text and "Infinity" not in text
    maximum=max(r.absolute_prediction_delta for r in bundle.perturbation.records); assert math.isfinite(maximum)
    print(f"interpretability_smoke model=weighted_gat validation_period=2021 edges={len(bundle.attention.edges)} paths={len(bundle.attention_paths.records)} max_prediction_delta={maximum:.6f}")
if __name__=="__main__": main()
