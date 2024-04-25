import gmsh
import numpy as np
import os

script_path = os.path.dirname(__file__)

gmsh.initialize()
top_marker = 2
bottom_marker = 1
left_marker = 1

# We create one rectangle for each subdomain
    # gmsh.model.occ.addRectangle(0, 0, 0, 1, 0.5, tag=1)
    # gmsh.model.occ.addRectangle(0, 0.5, 0, 1, 0.5, tag=2)
    # # We fuse the two rectangles and keep the interface between them
    # gmsh.model.occ.fragment([(2, 1)], [(2, 2)])
    # gmsh.model.occ.synchronize()
    
gmsh.model.add("DFG 3D")
cube = gmsh.model.occ.addBox(0,0,0,1.0,1.0,1.0)
hole = gmsh.model.occ.addSphere(0.5,0.5,0.5,0.3)

# gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=1)
# gmsh.model.occ.addDisk(0.5,0.5,0,0.1,0.1, tag=2)
    
gmsh.model.occ.cut([(3,cube)], [(3, hole)])
gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)

volume_marker = 11

gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], volume_marker)
gmsh.model.setPhysicalName(volumes[0][0], volume_marker, "Solid volume")

# gmsh.model.add_physical_group

gmsh.model.occ.synchronize()
gmsh.option.setNumber('Mesh.MeshSizeMin', 0)  # Set the minimum mesh size
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.025) 
    #gmsh.model.occ.cut()
    

    # Mark the top (2) and bottom (1) rectangle
# top, bottom = None, None
# for surface in gmsh.model.getEntities(dim=3):
#     com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
#     if np.allclose(com, [0.5, 0.25, 0]):
#         bottom = surface[1]
#     else:
#         top = surface[1]
#    # gmsh.model.addPhysicalGroup(2, [bottom], bottom_marker)
# gmsh.model.addPhysicalGroup(2, [top], top_marker)
#     # Tag the left boundary
# left = []
# for line in gmsh.model.getEntities(dim=1):
#     com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
#     if np.isclose(com[0], 0):
#         left.append(line[1])
# gmsh.model.addPhysicalGroup(1, left, left_marker)
gmsh.model.mesh.generate(3)
gmsh.write(script_path+"/mesh.msh")
    
gmsh.finalize()