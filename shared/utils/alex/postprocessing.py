import dolfinx as dlfx
from mpi4py import MPI
import ufl

def write_mesh_and_get_outputfile_xdmf(domain: dlfx.mesh.Mesh,
                                       outputfile_xdmf_path: str,
                                       comm: MPI.Intercomm):
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'w')
    xdmfout.write_mesh(domain)
    xdmfout.close()
    return xdmfout

def write_phasefield_mixed_solution(domain: dlfx.mesh.Mesh,
                                    outputfile_xdmf_path: str,
                                    w: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    
    
    
    
    
    
    # split solution to displacement and crack field
    u, s = w.split()
    # u_out = u.collapse()
    
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
        
    