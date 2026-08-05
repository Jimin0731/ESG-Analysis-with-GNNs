from __future__ import annotations

import pandas as pd

from .contracts import FeatureAssemblyReport, FeatureBlock, FeatureValidationError, sort_feature_frame, validate_annual_period, validate_node_id, validate_unique_periods


def _validate_canonical_nodes(nodes: list[str]) -> list[str]:
    if not nodes:
        raise FeatureValidationError("canonical node order must be non-empty")
    for node in nodes:
        validate_node_id(node)
    if len(nodes) != len(set(nodes)):
        raise FeatureValidationError("canonical node order must be unique")
    return nodes


def assemble_feature_blocks(blocks: list[FeatureBlock], *, canonical_node_order: list[str], periods: list[int], strict: bool = True):
    if not blocks:
        raise FeatureValidationError("at least one feature block required")
    nodes = _validate_canonical_nodes(list(canonical_node_order))
    final_periods = validate_unique_periods(periods)
    all_features: list[str] = []
    for block in blocks:
        if block.frame.duplicated(["node_id", "period"]).any():
            raise FeatureValidationError(f"duplicate keys in block {block.name}")
        all_features.extend(block.feature_names)
    if len(all_features) != len(set(all_features)):
        raise FeatureValidationError("feature name collision")
    canonical = pd.DataFrame([(node, period) for period in final_periods for node in nodes], columns=["node_id", "period"])
    if canonical.duplicated(["node_id", "period"]).any():
        raise FeatureValidationError("canonical keys must be unique")
    final = canonical.copy()
    canonical_keys = set(map(tuple, canonical[["node_id", "period"]].to_numpy().tolist()))
    missing_keys: dict[str, list[dict[str, object]]] = {}
    extra_keys: dict[str, list[dict[str, object]]] = {}
    by_block: dict[str, list[str]] = {}
    node_order = {node: i for i, node in enumerate(nodes)}
    def key_sort(key):
        return (key[1], node_order.get(key[0], len(node_order)), key[0])
    for block in blocks:
        block_keys = set(map(tuple, block.frame[["node_id", "period"]].to_numpy().tolist()))
        missing = sorted(canonical_keys - block_keys, key=key_sort)
        extra = sorted(block_keys - canonical_keys, key=key_sort)
        missing_keys[block.name] = [{"node_id": node, "period": validate_annual_period(period)} for node, period in missing]
        extra_keys[block.name] = [{"node_id": node, "period": validate_annual_period(period)} for node, period in extra]
        if strict and (missing or extra):
            raise FeatureValidationError(f"strict key mismatch for block {block.name}")
        final = final.merge(block.frame[["node_id", "period", *block.feature_names]], on=["node_id", "period"], how="left", validate="one_to_one")
        if len(final) != len(canonical):
            raise FeatureValidationError("feature assembly changed canonical row count")
        by_block[block.name] = list(block.feature_names)
    final = sort_feature_frame(final, nodes)
    report = FeatureAssemblyReport(
        total_final_rows=len(final),
        total_final_features=len(all_features),
        feature_names_by_block=by_block,
        missing_values_by_feature={feature: int(final[feature].isna().sum()) for feature in all_features},
        missing_keys_by_block=missing_keys,
        extra_keys_by_block=extra_keys,
        node_ids=nodes,
        periods=final_periods,
        node_count=len(nodes),
        period_count=len(final_periods),
        final_node_order=nodes,
        final_period_order=final_periods,
    )
    return final, report, tuple(provenance for block in blocks for provenance in block.provenance)
