import numpy as np
import meshio
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile


proc = MPI.COMM_WORLD.rank

#mesh, cell_markers, facet_markers = gmshio.read_from_msh("output_mesh.msh", MPI.COMM_WORLD, gdim=2)

# https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html#convert-msh-files-to-xdmf-using-meshio
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    #cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
    return out_mesh

if proc == 0:
    # Read in mesh
    msh = meshio.read("/home/scripts/01-phasefield-fracture-porous/3D-cube-with-hole/mesh.msh")

    # Create and save one file for the mesh, and one file for the facets
    tetra_mesh = create_mesh(msh, "tetra")
    meshio.write("/home/resources/cube_with_hole.xdmf", tetra_mesh)
MPI.COMM_WORLD.barrier()


    
# Now you can use the mesh in FEniCSx

# comm = MPI.COMM_WORLD

# # Load the mesh using meshio
# mesh = meshio.read("output_mesh.msh")

# # Convert to dolfinx mesh
# points = mesh.points.astype(np.float64)
# cells = mesh.cells[0].data.astype(np.int32)  # Assuming your mesh has tetrahedral cells
# cells_flat = cells.flatten()
# adjacency_list = dolfinx.cpp.graph.AdjacencyListLong(cells_flat)

# mesh = dolfinx.mesh.Mesh(dolfinx.cpp.mesh.create_mesh(comm, cells, dolfinx.cpp.mesh.CellType.tetrahedron, points))

# # Print number of elements and nodes
# num_elements = mesh.topology.index_map(mesh.topology.dim).size_local
# num_nodes = mesh.geometry.x.shape[0]
# print("Number of elements:", num_elements)
# print("Number of nodes:", num_nodes)

# Now you can use the mesh in FEniCSx





