import dolfinx as dlfx
from mpi4py import MPI

def write_mesh_and_get_outputfile_xdmf(domain: dlfx.mesh.Mesh,
                                       outputfile_xdmf_path: str,
                                       comm: MPI.Intercomm):
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'w')
    xdmfout.write_mesh(domain)
    xdmfout.close()
    return xdmfout

def write_phasefield_mixed_solution(outputfile_xdmf_path: str,
                                    w: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    # split solution to displacement and crack field
    u, s = w.split()
    u_out = u.collapse()
    u_out.name = 'u'
    s.name='s'

    # append xdmf-file
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    xdmfout.write_function(u_out, t) # collapse reduces to subspace so one can work only in subspace https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2, only one component?
    xdmfout.write_function(s, t)
    xdmfout.close()
    return xdmfout
        
    