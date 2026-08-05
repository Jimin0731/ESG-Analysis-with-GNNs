from __future__ import annotations

import pandas as pd

from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame, validate_metadata_jsonable, validate_node_id, validate_period_series

PILLAR_COLUMNS = ("overall_esg_score", "environmental_score", "social_score", "governance_score", "observation_count")


def build_esg_feature_block(df: pd.DataFrame, *, aggregation: str = "mean", canonical_node_order=None) -> FeatureBlock:
    if aggregation not in {"mean", "median"}:
        raise FeatureValidationError("unsupported ESG aggregation policy")
    validate_period_series(df["period"])
    pillars = [column for column in PILLAR_COLUMNS if column in df]
    if not pillars:
        raise FeatureValidationError("no observed ESG feature columns")
    for column in pillars:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise FeatureValidationError(f"non-numeric ESG column {column}")
    missing_nodes = df["node_id"].isna() | (df["node_id"].astype(str).str.strip() == "")
    unmatched = sorted(df.loc[missing_nodes, "entity_id"].astype(str).tolist()) if "entity_id" in df else []
    valid = df.loc[~missing_nodes].copy()
    valid["node_id"].map(validate_node_id)
    grouped = valid.groupby(["node_id", "period"], sort=True)
    aggregated = grouped[pillars].agg(aggregation).reset_index()
    renamed = {column: f"esg__{column}" for column in pillars}
    aggregated = aggregated.rename(columns=renamed)
    names = tuple(renamed.values())
    counts = grouped.size().reset_index(name="entity_count")
    report = {
        "source_entity_rows": int(len(df)),
        "resulting_node_period_rows": int(len(aggregated)),
        "entity_counts_per_node_period": counts.to_dict("records"),
        "missing_counts_by_pillar": {column: int(valid[column].isna().sum()) for column in pillars},
        "unmatched_or_missing_node_ids": unmatched,
        "aggregation_policy": aggregation,
    }
    validate_metadata_jsonable(report, field_name="ESG aggregation report")
    aggregated = sort_feature_frame(aggregated, canonical_node_order or sorted(aggregated["node_id"].unique()))
    provenance = tuple(FeatureProvenance(name, "esg", "normalized_esg_scores", name.removeprefix("esg__"), aggregation, "annual observation", False, (), "preserve") for name in names)
    return FeatureBlock("esg", aggregated, names, provenance, report=report)
