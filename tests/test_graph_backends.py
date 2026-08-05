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
    A=snap.report.diagnostics['A']; B=use.to_numpy(float)/(use.sum(axis=1).to_numpy(float)+make.sum(axis=0).to_numpy(float))[:,None]; D=make.to_numpy(float)/(make.sum(axis=1).to_numpy(float))[:,None]; np.testing.assert_allclose(A, B.T @ D.T)
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
