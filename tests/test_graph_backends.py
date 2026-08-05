import numpy as np, pandas as pd, pytest
from src.graphs import *
from src.graphs.common import *
from src.graphs.contracts import GraphValidationError


def test_contracts_validate_unique_nodes_and_endpoints():
    n=(GraphNode('a','A'), GraphNode('b','B'))
    GraphSnapshot(n, np.array([[0],[1]]), np.array([1.0]), np.array([2.0]), 'x')
    with pytest.raises(GraphValidationError): GraphSnapshot((GraphNode('a','A'),GraphNode('a','B')), np.zeros((2,0),int), np.array([]), np.array([]), 'x')
    with pytest.raises(GraphValidationError): GraphSnapshot(n, np.array([[0],[2]]), np.array([1.0]), np.array([1.0]), 'x')

def test_common_edge_order_duplicate_threshold_density_and_normalization():
    edges=[(1,0,2,2),(0,1,1,1),(1,0,3,3)]
    with pytest.raises(GraphInputError): aggregate_duplicate_edges(edges,'reject')
    assert aggregate_duplicate_edges(edges,'sum')[1] == (1,0,5.0,5.0)
    assert threshold_mask([0,1,3], 'absolute', 1).tolist()==[False,False,True]
    assert density(3,3)==0.5
    assert model_weights([], transform='standardized').size == 0
    assert model_weights([5], transform='standardized').tolist()==[0.0]

def aligned_tables():
    use=pd.DataFrame([[10,2],[1,8]], index=['C1','C2'], columns=['S1','S2'])
    make=pd.DataFrame([[9,1],[2,7]], index=['S1','S2'], columns=['C1','C2'])
    return use, make

def test_use_make_alignment_b_d_a_edges_and_raw():
    use,make=aligned_tables(); snap=build_use_make_graph(use,make, config=UseMakeGraphConfig(include_self_loops=False, weight_transform='raw'))
    assert snap.labels == ['S1','S2']
    assert snap.report.diagnostics['B_shape']==(2,2)
    assert snap.report.diagnostics['D_shape']==(2,2)
    A=snap.source_metadata['unthresholded_A']; B=use.to_numpy(float)/(use.sum(axis=1).to_numpy(float)+make.sum(axis=0).to_numpy(float))[:,None]; D=make.to_numpy(float)/(make.sum(axis=1).to_numpy(float))[:,None]; np.testing.assert_allclose(A, B.T @ D.T)
    assert np.allclose(snap.edge_weight, snap.raw_flow)
    assert snap.report.diagnostics['alignment_coverage']==1.0

def test_use_make_intersection_low_coverage_duplicate_negative_nonfinite():
    use,make=aligned_tables(); use2=use.rename(index={'C2':'DROP'})
    snap=build_use_make_graph(use2, make, config=UseMakeGraphConfig(min_alignment_coverage=.25))
    assert snap.report.diagnostics['alignment_coverage'] < 1
    with pytest.raises(GraphInputError): build_use_make_graph(use2, make, config=UseMakeGraphConfig(min_alignment_coverage=.9))
    with pytest.raises(GraphInputError): build_use_make_graph(pd.DataFrame([[1],[2]], index=['C','C'], columns=['S']), make)
    bad=use.copy(); bad.iloc[0,0]=-1
    with pytest.raises(GraphInputError): build_use_make_graph(bad, make)
    bad=use.astype(float).copy(); bad.iloc[0,0]=np.inf
    with pytest.raises(GraphInputError): build_use_make_graph(bad, make)

def test_use_make_threshold_zero_one_edge_and_dispatcher():
    use,make=aligned_tables()
    zero=build_use_make_graph(use,make, config=UseMakeGraphConfig(threshold_value=999, weight_transform='standardized'))
    assert zero.edge_index.shape==(2,0)
    one=build_use_make_graph(use,make, config=UseMakeGraphConfig(threshold_policy='percentile', threshold_value=70, weight_transform='standardized'))
    assert np.isfinite(one.edge_weight).all()
    with pytest.raises(ValueError): build_economic_graph('bad')

def test_leontief_exact_pinv_reporting_and_invalid_policy():
    L,r=compute_leontief_inverse(np.array([[.1,0],[0,.2]]),'error')
    assert r.method=='inverse' and not r.used_fallback and L.shape==(2,2)
    _,r=compute_leontief_inverse(np.eye(2),'pinv')
    assert r.method=='pinv' and r.used_fallback
    with pytest.raises(GraphInputError): compute_leontief_inverse(np.eye(2),'oops')
    with pytest.raises(Exception): compute_leontief_inverse(np.eye(2),'error')

def test_icio_square_wide_codes_direction_missing_selfloop_threshold():
    sq=pd.DataFrame([[1,2],[0,'..']], index=[' 01','02 '], columns=['01','02'])
    snap=build_icio_graph(sq, config=ICIOGraphConfig(include_self_loops=False))
    assert snap.labels==['01','02']; assert (snap.edge_index[:,0]==np.array([0,1])).all() or (snap.edge_index[:,0]==np.array([0,0])).all()
    assert snap.report.diagnostics['missing_count']==1 and snap.report.diagnostics['zero_count']==1
    wide=pd.DataFrame({'source_industry':['01','02'], '01':[5,0], '02':[7,3]})
    snap2=build_icio_graph(wide, config=ICIOGraphConfig(source_column='source_industry', include_self_loops=True, threshold_value=4))
    assert snap2.labels[0]=='01' and 5 in snap2.raw_flow

def test_icio_invalid_duplicate_ambiguous_no_positive():
    with pytest.raises(GraphInputError): build_icio_graph(pd.DataFrame({'industry':['01','01'], '02':[1,2]}), config=ICIOGraphConfig(source_column='industry'))
    with pytest.raises(GraphInputError): build_icio_graph(pd.DataFrame({'source':['01'], 'source_industry':['01'], '02':[1]}))
    with pytest.raises(GraphInputError): build_icio_graph(pd.DataFrame([[0,0],[0,0]], index=['01','02'], columns=['01','02']))
    with pytest.raises(GraphInputError): build_icio_graph(pd.DataFrame([[1,2],[3,4]], index=['01','01'], columns=['01','02']))


def test_use_make_transposed_orientation_equivalence_and_not_noop():
    use, make = aligned_tables()
    cfg = UseMakeGraphConfig(include_self_loops=True)
    canonical = build_use_make_graph(use, make, config=cfg)
    transposed = build_use_make_graph(
        use.T,
        make.T,
        config=UseMakeGraphConfig(
            use_orientation="sector_by_commodity",
            make_orientation="commodity_by_sector",
            include_self_loops=True,
        ),
    )
    assert transposed.labels == canonical.labels
    np.testing.assert_allclose(transposed.source_metadata["unthresholded_A"], canonical.source_metadata["unthresholded_A"])
    np.testing.assert_array_equal(transposed.edge_index, canonical.edge_index)
    np.testing.assert_allclose(transposed.raw_flow, canonical.raw_flow)
    assert transposed.report.diagnostics["retained_commodities"] == canonical.report.diagnostics["retained_commodities"]
    assert transposed.report.diagnostics["retained_sectors"] == canonical.report.diagnostics["retained_sectors"]


def test_use_make_transposed_alignment_diagnostics_are_canonical():
    use = pd.DataFrame([[10, 2], [1, 8], [5, 6]], index=["C1", "C2", "C_DROP"], columns=["S1", "S2"])
    make = pd.DataFrame([[9, 1, 0], [2, 7, 0], [4, 4, 1]], index=["S1", "S2", "S_DROP"], columns=["C1", "C2", "C_MAKE_ONLY"])
    cfg = UseMakeGraphConfig(min_alignment_coverage=0.5, include_self_loops=True)
    canonical = build_use_make_graph(use, make, config=cfg)
    transposed = build_use_make_graph(
        use.T,
        make.T,
        config=UseMakeGraphConfig(
            use_orientation="sector_by_commodity",
            make_orientation="commodity_by_sector",
            min_alignment_coverage=0.5,
            include_self_loops=True,
        ),
    )
    keys = [
        "removed_use_commodities",
        "removed_use_sectors",
        "removed_make_commodities",
        "removed_make_sectors",
        "retained_commodities",
        "retained_sectors",
    ]
    for key in keys:
        assert transposed.report.diagnostics[key] == canonical.report.diagnostics[key]
    assert canonical.report.diagnostics["removed_use_commodities"] == ["C_DROP"]
    assert canonical.report.diagnostics["removed_make_commodities"] == ["C_MAKE_ONLY"]
    assert canonical.report.diagnostics["removed_make_sectors"] == ["S_DROP"]


def test_use_make_rejects_bad_orientation_labels_and_preserves_input():
    use, make = aligned_tables()
    original = use.copy(deep=True)
    with pytest.raises(GraphInputError):
        UseMakeGraphConfig(use_orientation="bad")
    bad = use.copy()
    bad.index = ["C1", "  "]
    with pytest.raises(GraphInputError, match="empty"):
        build_use_make_graph(bad, make)
    incompatible_make = make.rename(index={"S1": "X1", "S2": "X2"}, columns={"C1": "Y1", "C2": "Y2"})
    with pytest.raises(GraphInputError, match="alignment coverage"):
        build_use_make_graph(use, incompatible_make, config=UseMakeGraphConfig(min_alignment_coverage=1.0))
    pd.testing.assert_frame_equal(use, original)


def test_use_make_hand_calculated_components_and_a_direction():
    use = pd.DataFrame([[2.0, 6.0], [8.0, 4.0]], index=["C1", "C2"], columns=["S1", "S2"])
    make = pd.DataFrame([[10.0, 0.0], [0.0, 20.0]], index=["S1", "S2"], columns=["C1", "C2"])
    snap = build_use_make_graph(use, make, config=UseMakeGraphConfig(include_self_loops=True))
    expected_use_requirements = np.array([[2.0 / 18.0, 6.0 / 18.0], [8.0 / 32.0, 4.0 / 32.0]])
    expected_market_shares = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected_A = expected_use_requirements.T @ expected_market_shares.T
    np.testing.assert_allclose(expected_A, np.array([[1.0 / 9.0, 0.25], [1.0 / 3.0, 0.125]]))
    np.testing.assert_allclose(snap.source_metadata["unthresholded_A"], expected_A)
    assert not np.allclose(expected_A, expected_A.T)


def test_icio_wide_union_preserves_source_and_destination_only_missing_policy():
    wide = pd.DataFrame({"industry": ["01", "02"], "02": [5.0, 0.0], "03": [7.0, 11.0]})
    snap = build_icio_graph(wide, config=ICIOGraphConfig(source_column="industry", include_self_loops=True))
    assert snap.labels == ["01", "02", "03"]
    flows = {(snap.labels[s], snap.labels[t]): flow for (s, t), flow in zip(snap.edge_index.T, snap.raw_flow)}
    assert flows[("01", "03")] == 7.0
    assert flows[("02", "03")] == 11.0
    assert snap.report.diagnostics["missing_cell_count"] == 5
    assert snap.report.diagnostics["zero_flow_count"] == 1
    assert snap.report.diagnostics["source_only_industry_ids"] == ["01"]
    assert snap.report.diagnostics["destination_only_industry_ids"] == ["03"]
    with pytest.raises(GraphInputError, match="missing"):
        build_icio_graph(wide, config=ICIOGraphConfig(source_column="industry", missing_cell_policy="error"))
    with pytest.raises(GraphInputError):
        ICIOGraphConfig(missing_cell_policy="bad")


def test_leontief_integration_and_fallback_reports():
    use = pd.DataFrame([[2.0]], index=["C1"], columns=["S1"])
    make = pd.DataFrame([[0.0]], index=["S1"], columns=["C1"])
    with pytest.raises(GraphInputError, match="positive"):
        build_use_make_graph(use, make)
    L, report = compute_leontief_inverse(np.array([[0.2]]), "error")
    np.testing.assert_allclose(L, np.array([[1.25]]))
    assert report.shape == (1, 1) and report.method == "inverse" and report.fallback_policy == "error"
    with pytest.raises(GraphInputError, match="Leontief"):
        compute_leontief_inverse(np.eye(2), "error")
    _, pinv_report = compute_leontief_inverse(np.eye(2), "pinv")
    assert pinv_report.used_fallback and pinv_report.method == "pinv"
    _, reg_report = compute_leontief_inverse(np.eye(2), "regularized", regularization=0.01)
    assert reg_report.used_fallback and reg_report.method == "regularized" and reg_report.regularization_value == 0.01
    with pytest.raises(GraphInputError):
        compute_leontief_inverse(np.eye(2), "regularized", regularization=0.0)
    with pytest.raises(GraphInputError, match="regularized Leontief"):
        compute_leontief_inverse(np.array([[2.0, 0.0], [0.0, 1.0]]), "regularized", regularization=1.0)


def test_graph_snapshot_focused_failures():
    n = (GraphNode("a", "A"),)
    with pytest.raises(GraphValidationError, match="at least one"):
        GraphSnapshot((), np.zeros((2, 0), dtype=int), np.array([]), np.array([]), "x")
    with pytest.raises(GraphValidationError, match="integer"):
        GraphSnapshot(n, np.array([[0.1], [0.0]]), np.array([1.0]), np.array([1.0]), "x")
    with pytest.raises(GraphValidationError, match="shape"):
        GraphSnapshot(n, np.zeros((3, 0), dtype=int), np.array([]), np.array([]), "x")
    with pytest.raises(GraphValidationError, match="one-dimensional"):
        GraphSnapshot(n, np.zeros((2, 1), dtype=int), np.array(1.0), np.array([1.0]), "x")
    with pytest.raises(GraphValidationError, match="non-empty"):
        GraphSnapshot((GraphNode(" ", "A"),), np.zeros((2, 0), dtype=int), np.array([]), np.array([]), "x")


def test_threshold_percentile_documented_edge_cases_and_reporting():
    assert threshold_mask([0, 1, 2], "percentile", 0).tolist() == [False, True, True]
    assert threshold_mask([1, 2, 3], "percentile", 100).tolist() == [False, False, False]
    assert threshold_mask([5, 5, 5], "percentile", 50).tolist() == [False, False, False]
    assert threshold_mask([7], "percentile", 0).tolist() == [False]
    assert threshold_mask([], "percentile", 50).tolist() == []
    use, make = aligned_tables()
    snap = build_use_make_graph(use, make, config=UseMakeGraphConfig(threshold_policy="percentile", threshold_value=100))
    assert snap.report.diagnostics["flows_dropped_by_threshold"] >= 0


def test_invalid_configuration_contracts_fail_early():
    with pytest.raises(GraphInputError):
        GraphConstructionConfig(backend="bad")
    with pytest.raises(GraphInputError):
        UseMakeGraphConfig(threshold_policy="absolute", threshold_value=-1)
    with pytest.raises(GraphInputError):
        UseMakeGraphConfig(threshold_policy="percentile", threshold_value=101)
    with pytest.raises(GraphInputError):
        UseMakeGraphConfig(duplicate_edge_policy="bad")
    with pytest.raises(GraphInputError):
        UseMakeGraphConfig(weight_transform="bad")
    with pytest.raises(GraphInputError):
        UseMakeGraphConfig(min_alignment_coverage=2)
    with pytest.raises(GraphInputError):
        UseMakeGraphConfig(leontief_fallback="bad")
    UseMakeGraphConfig()
    ICIOGraphConfig()
    with pytest.raises(GraphInputError, match="UseMakeGraphConfig requires"):
        UseMakeGraphConfig(backend="icio")
    with pytest.raises(GraphInputError, match="ICIOGraphConfig requires"):
        ICIOGraphConfig(backend="use_make")
