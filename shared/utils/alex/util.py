import dolfinx as dlfx
def get_dimension_of_function(f: dlfx.fem.Function) -> int:
    return f.ufl_shape[0]