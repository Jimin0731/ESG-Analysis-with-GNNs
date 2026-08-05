"""Graph construction helpers for ESG company networks."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_sector_graph(frame: pd.DataFrame) -> nx.Graph:
    """Build a company graph with edges connecting companies in nearby sectors."""
    graph = nx.Graph()
    for row in frame.itertuples(index=False):
        graph.add_node(row.company, sector=row.sector, esg_score=row.esg_score)

    sectors = frame.groupby("sector")["company"].apply(list)
    for companies in sectors:
        for source_index, source in enumerate(companies):
            for target in companies[source_index + 1 :]:
                graph.add_edge(source, target, relationship="same_sector")

    return graph
