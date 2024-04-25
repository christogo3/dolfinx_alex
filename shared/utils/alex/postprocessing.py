import dolfinx as dlfx
from mpi4py import MPI
import ufl

def write_mesh_and_get_outputfile_xdmf(domain: dlfx.mesh.Mesh,
                                       outputfile_xdmf_path: str,
                                       comm: MPI.Intercomm,
                                       meshtags: any = None):
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'w')
    xdmfout.write_mesh(domain)
    if( not meshtags is None):
        xdmfout.write_meshtags(meshtags, domain.geometry)
    xdmfout.close()
    return True

def write_phasefield_mixed_solution(domain: dlfx.mesh.Mesh,
                                    outputfile_xdmf_path: str,
                                    w: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    
    
    # split solution to displacement and crack field
    u, s = w.split()
    # u_out = u.collapse()
    
    # vectorfields = [u]
    # vectorfields_names = ["u_test"]
    # write_vector_fields(domain,comm,vectorfields, vectorfields_names, outputfile_xdmf_path)
    
       
    Ue = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    Se = ufl.FiniteElement('CG', domain.ufl_cell(), 1)
    # W_linear = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ue, Se]))
    # w_interp =dlfx.fem.Function(W_linear)

    # w_interp.sub(0).interpolate(w)
    # u_interp, s_interp = w.split()
    
    U = dlfx.fem.FunctionSpace(domain, Ue)
    S = dlfx.fem.FunctionSpace(domain, Se)
    s_interp = dlfx.fem.Function(S)
    u_interp = dlfx.fem.Function(U)
    # # s_interp.interpolate()

    
    s_interp.interpolate(s)
    u_interp.interpolate(u)
    s_interp.name = 's'
    u_interp.name = 'u'
    # s.name='s'
    
    # append xdmf-file
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    xdmfout.write_function(u_interp, t) # collapse reduces to subspace so one can work only in subspace https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2, only one component?
    xdmfout.write_function(s_interp, t)
    xdmfout.close()
    return xdmfout

def getJString(Jx, Jy, Jz):
    out_string = 'Jx: {0:.4e} Jy: {1:.4e} Jz: {2:.4e}'.format(Jx, Jy, Jz)
    return out_string
    
    
def write_tensor_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, tensor_fields_as_functions, tensor_field_names, outputfile_xdmf_path: str):
    TENe = ufl.TensorElement('DG', domain.ufl_cell(), 0)
    TEN = dlfx.fem.FunctionSpace(domain, TENe) 
    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        for n  in range(0,len(tensor_fields_as_functions)):
            tensor_field_function = tensor_fields_as_functions[n]
            tensor_field_name = tensor_field_names[n]
            tensor_field_expression = dlfx.fem.Expression(tensor_field_function, 
                                                        TEN.element.interpolation_points())
            out_tensor_field = dlfx.fem.Function(TEN)
            out_tensor_field.interpolate(tensor_field_expression)
            out_tensor_field.name = tensor_field_name
            
            xdmf_out.write_function(out_tensor_field)

def write_vector_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, vector_fields_as_functions, vector_field_names, outputfile_xdmf_path: str):
    Ve = ufl.VectorElement('DG', domain.ufl_cell(), 0)
    V = dlfx.fem.FunctionSpace(domain, Ve) 
    xdmf_out = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    for n  in range(0,len(vector_fields_as_functions)):
            vector_field_function = vector_fields_as_functions[n]
            vector_field_name = vector_field_names[n]
            vector_field_expression = dlfx.fem.Expression(vector_field_function, 
                                                        V.element.interpolation_points())
            out_vector_field = dlfx.fem.Function(V)
            out_vector_field.interpolate(vector_field_expression)
            out_vector_field.name = vector_field_name
            
            xdmf_out.write_function(out_vector_field)
    xdmf_out.close()
            
def write_scalar_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, scalar_fields_as_functions, scalar_field_names, outputfile_xdmf_path: str):
    Se = ufl.FiniteElement('DG', domain.ufl_cell(), 0)
    S = dlfx.fem.FunctionSpace(domain, Se) 
    xdmf_out = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    for n  in range(0,len(scalar_fields_as_functions)):
            scalar_field_function = scalar_fields_as_functions[n]
            scalar_field_name = scalar_field_names[n]
            scalar_field_expression = dlfx.fem.Expression(scalar_field_function, 
                                                        S.element.interpolation_points())
            out_scalar_field = dlfx.fem.Function(S)
            out_scalar_field.interpolate(scalar_field_expression)
            out_scalar_field.name = scalar_field_name
            
            xdmf_out.write_function(out_scalar_field)
    xdmf_out.close()
        
            
        
        
    