"""Economic graph backend dispatcher."""
from .contracts import *
from .use_make import UseMakeGraphConfig, build_use_make_graph, compute_leontief_inverse
from .icio import ICIOGraphConfig, build_icio_graph

def build_economic_graph(backend: str, *args, **kwargs):
    if backend == "use_make": return build_use_make_graph(*args, **kwargs)
    if backend == "icio": return build_icio_graph(*args, **kwargs)
    raise ValueError(f"unsupported graph backend: {backend}")
