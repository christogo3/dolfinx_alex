import dolfinx as dlfx
import numpy as np
import ufl
import basix.ufl
import yaml

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem.petsc import NonlinearProblem

from Materials import plastic, eigenvalues, integrate_scalar, integrate_tensor
from Utils import log_writer


def mark_bottom_left(x):
    val = np.isclose(x[0], -0.5) & np.isclose(x[1], -0.5)
    return val

def mark_bottom_right(x):
    val = np.isclose(x[0], 0.5) & np.isclose(x[1], -0.5)
    return val


# read parameters from auxetic_para.yaml
with open('inhom_para.yaml', 'r', encoding='utf-8') as file:
   parameters = yaml.safe_load(file)

Emod_mat = parameters['material_matrix']['Emod']
nu_mat = parameters['material_matrix']['nu']
sigY_mat = parameters['material_matrix']['sigY']
Hmod_mat = parameters['material_matrix']['Hmod']

Emod_inc = parameters['material_inclusion']['Emod']
nu_inc = parameters['material_inclusion']['nu']
sigY_inc = parameters['material_inclusion']['sigY']
Hmod_inc = parameters['material_inclusion']['Hmod']

# eps0 =  parameters['fem']['eps0']
sig0 = parameters['fem']['sig0']
element_degree = parameters['fem']['element_degree']
load_steps = parameters['fem']['load_steps']

# Emod = dlfx.fem.Constant(region, 10.0)
# nu = dlfx.fem.Constant(region, 0.3)
# eps0 = 0.1


# Ausgabe-Level festlegen (EROOR, INFO, OFF, WARNING)
dlfx.log.set_log_level(dlfx.log.LogLevel.WARNING)

# Initialisierugn der MPI-Umgebung
comm = MPI.COMM_WORLD

# Netz einlesen
with dlfx.io.XDMFFile(comm, 'input.xdmf', 'r') as mesh_inp:
    region = mesh_inp.read_mesh()
    dim = region.topology.dim 
    fdim = dim-1
    region.topology.create_connectivity(fdim, dim)
    cell_tags = mesh_inp.read_meshtags(region, name='cell_tags')   # 21 for all
    facet_tags = mesh_inp.read_meshtags(region, name='facet_tags')   # 11: right, 12:top, 13: left, 14: bottom

MATe = SCAe = basix.ufl.element('DG', region.topology.cell_name(), 0)
MAT = dlfx.fem.functionspace(region, MATe)

Emod = dlfx.fem.Function(MAT, name='Emod')
Emod.x.array[:] = (22-cell_tags.values[:])*Emod_mat+(cell_tags.values[:]-21)*Emod_inc
nu = dlfx.fem.Function(MAT, name='nu')
nu.x.array[:] = (22-cell_tags.values[:])*nu_mat+(cell_tags.values[:]-21)*nu_inc
sigY = dlfx.fem.Function(MAT, name='sigY')
sigY.x.array[:] = (22-cell_tags.values[:])*sigY_mat+(cell_tags.values[:]-21)*sigY_inc
Hmod = dlfx.fem.Function(MAT, name='Hmod')
Hmod.x.array[:] = (22-cell_tags.values[:])*Hmod_mat+(cell_tags.values[:]-21)*Hmod_inc

print('inclusion', Emod_inc, nu_inc, sigY_inc, Hmod_inc)
print('matrix', Emod_mat, nu_mat, sigY_mat, Hmod_mat)

# Element-Typ (Verschiebungselement) und Funktionenraum erzeugen
Ue = basix.ufl.element('Lagrange', region.topology.cell_name(), element_degree, shape=(dim, ))    
# Elemente auf Integrationsopunkten mit Funktionenraum erzeugen
QTe = basix.ufl.quadrature_element(region.basix_cell(), degree=element_degree, value_shape=(dim, dim, ))
QSe = basix.ufl.quadrature_element(region.basix_cell(), degree=element_degree)

U = dlfx.fem.functionspace(region, Ue) # vector on nodes
QT = dlfx.fem.functionspace(region, QTe) # tensor on quadrature points
QS = dlfx.fem.functionspace(region, QSe) # scalar on quadrature points

# Postprocessing spaces (DG0)
TENe = basix.ufl.element('DG', region.topology.cell_name(), 0, shape=(dim, dim, ))
TEN = dlfx.fem.functionspace(region, TENe)
VECe = basix.ufl.element('DG', region.topology.cell_name(), 0, shape=(dim, ))
VEC = dlfx.fem.functionspace(region, VECe)
SCAe = basix.ufl.element('DG', region.topology.cell_name(), 0)
SCA = dlfx.fem.functionspace(region, SCAe)

# define solution 
u = dlfx.fem.Function(U, name='u')
ep = dlfx.fem.Function(QT)
alpha = dlfx.fem.Function(QS)

# define output fields
disp = dlfx.fem.Function(VEC, name='displacement')
disp_mac = dlfx.fem.Function(VEC, name='displacement macroscopic')
strain = dlfx.fem.Function(TEN, name='strain')
pl_strain = dlfx.fem.Function(TEN, name='plastic strain')
pl_alpha = dlfx.fem.Function(SCA, name='plastic alpha')
stress = dlfx.fem.Function(TEN, name='stress')
# mises= dlfx.fem.Function(SCA, name='von Mises stress')
# Wen = dlfx.fem.Function(SCA, name='Wen')

# setup non-linear problem
para = {'Emod': Emod, 'nu': nu, 'sigY': sigY, 'Hmod': Hmod}

material = plastic(para, u, ep, alpha)
material.report() 
res = material.residual(ufl.dx)

# write material distribution
with dlfx.io.XDMFFile(comm, 'material.xdmf', 'w') as xdmfout:
    xdmfout.write_mesh(region)
    xdmfout.write_function(Emod)
    xdmfout.write_function(nu)
    xdmfout.write_function(sigY)
    xdmfout.write_function(Hmod)

# write output
with dlfx.io.XDMFFile(comm, 'output.xdmf', 'w') as xdmfout:
    xdmfout.write_mesh(region)

# mesh coordinates
x = ufl.SpatialCoordinate(region)
nor = ufl.FacetNormal(region)

# init logger
logger = log_writer('output.log', {'eps_1': 0.0, 'eps_2': 0.0, 'sig_1': 0.0, 'sig_2': 0.0, 'mac_alpha': 0.0})

# init load_counter
load_count = 0

for load_factor1 in np.linspace(-1.0, 1.0, load_steps):

    for load_factor2 in np.linspace(-1.0, 1.0, load_steps):

        mac_stress = ufl.as_tensor([[sig0*load_factor1, 0.0],
                                [0.0, sig0*load_factor2]])

        # trac_mac_expr = dlfx.fem.Expression(mac_stress*nor, U.element.interpolation_points)
        # trac_mac = dlfx.fem.Function(U, name='t_mac') 
        # trac_mac.interpolate(trac_mac_expr)

        # u_mac_expr = dlfx.fem.Expression(mac_strain*x, U.element.interpolation_points, comm=region.comm)
        # u_mac = dlfx.fem.Function(U, name='u_mac')
        # u_mac.interpolate(u_mac_expr)

        # set boundary condition from u_mac
        # right_dofs = dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(11))
        # bc_right = dlfx.fem.dirichletbc(u_mac, right_dofs)
        # top_dofs = dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(12))
        # bc_top = dlfx.fem.dirichletbc(u_mac, top_dofs)
        # left_dofs = dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(13))
        # bc_left = dlfx.fem.dirichletbc(u_mac, left_dofs)
        # bottom_dofs= dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(14))
        # bc_bottom = dlfx.fem.dirichletbc(u_mac, bottom_dofs)

        # collect bc
        # bcs = [bc_right, bc_top, bc_left, bc_bottom]

        # boundary conditions
        bottom_left_point = dlfx.mesh.locate_entities_boundary(region, 0, mark_bottom_left)
        bottom_right_point = dlfx.mesh.locate_entities_boundary(region, 0, mark_bottom_right)

        bottom_left_dofs_x = dlfx.fem.locate_dofs_topological(U.sub(0), 0, bottom_left_point)
        bottom_left_bc_x = dlfx.fem.dirichletbc(0.0, bottom_left_dofs_x, U.sub(0))
        bottom_left_dofs_y = dlfx.fem.locate_dofs_topological(U.sub(1), 0, bottom_left_point)
        bottom_left_bc_y = dlfx.fem.dirichletbc(0.0, bottom_left_dofs_y, U.sub(1))
        bottom_right_dofs_y = dlfx.fem.locate_dofs_topological(U.sub(1), 0, bottom_right_point)
        bottom_right_bc_y = dlfx.fem.dirichletbc(0.0, bottom_right_dofs_y, U.sub(1))

        bcs = [bottom_left_bc_x, bottom_left_bc_y, bottom_right_bc_y]

        # boundary residuals
        du = ufl.TestFunction(U)
        ds_right = ufl.Measure('ds', domain=region, subdomain_data=facet_tags)(11)
        ds_top = ufl.Measure('ds', domain=region, subdomain_data=facet_tags)(12)
        ds_left = ufl.Measure('ds', domain=region, subdomain_data=facet_tags)(13)
        ds_bottom = ufl.Measure('ds', domain=region, subdomain_data=facet_tags)(14)
        
        trac_right = ufl.as_vector([mac_stress[0, 0], mac_stress[1, 0]])
        res_right = ufl.dot(trac_right, du)*ds_right
        trac_top = ufl.as_vector([mac_stress[0, 1], mac_stress[1, 1]])
        res_top = ufl.dot(trac_top, du)*ds_top
        trac_left = ufl.as_vector([-mac_stress[0, 0], -mac_stress[1, 0]])
        res_left = ufl.dot(trac_left, du)*ds_left
        trac_bottom = ufl.as_vector([-mac_stress[0, 1], -mac_stress[1, 1]])
        res_bottom = ufl.dot(trac_bottom, du)*ds_bottom
        
        res_boundary = res_right+res_top+res_left+res_bottom

        # zero all history variables
        material.alpha.x.array[:] = 0.0
        material.ep.x.array[:] = 0.0

        petsc_options= {'snes_type': 'newtonls', 
                            'snes_error_if_not_converged': True,
                            'ksp_error_if_not_converged': True,
                            'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            'pc_factor_mat_solver_type': 'mumps'}

        problem = NonlinearProblem(res+res_boundary, u, bcs=bcs, petsc_options=petsc_options, petsc_options_prefix='bla')

        problem.solve()
        material.update()

        # Ausgabe vorbereiten und Netz schreiben

        disp.interpolate(u)
        # disp_mac.interpolate(u_mac)

        strain_expr = dlfx.fem.Expression(material.eps(u), TEN.element.interpolation_points)
        strain.interpolate(strain_expr)

        stress_expr = dlfx.fem.Expression(material.sig(material.eps(u)-material.ep), TEN.element.interpolation_points)
        stress.interpolate(stress_expr)

        pl_strain_expr = dlfx.fem.Expression(material.ep, TEN.element.interpolation_points)
        pl_strain.interpolate(pl_strain_expr)

        pl_alpha_expr = dlfx.fem.Expression(material.alpha, SCA.element.interpolation_points)
        pl_alpha.interpolate(pl_alpha_expr)

        # Wen_expr = dlfx.fem.Expression(material.Wenergy(material.eps(u)), SCA.element.interpolation_points)
        # Wen.interpolate(Wen_expr)

        # mises_expr = dlfx.fem.Expression(vonMises_sig(material.sig(material.eps(u)), material.nu), SCA.element.interpolation_points)
        # mises.interpolate(mises_expr)

        with dlfx.io.XDMFFile(comm, 'output.xdmf', 'a') as xdmfout:
                xdmfout.write_function(u, load_count)
                xdmfout.write_function(disp, load_count)
                xdmfout.write_function(disp_mac, load_count)
                xdmfout.write_function(strain, load_count)
                xdmfout.write_function(stress, load_count)
                xdmfout.write_function(pl_strain, load_count)
                xdmfout.write_function(pl_alpha, load_count)
        
        # ipdate load_count
        load_count = load_count+1 

        print('')
        print('R E S U L T S ', load_factor1, ' / ', load_factor2)
        #print('cross section')
        # mac_vol = integrate_scalar(dlfx.fem.Constant(region, 1.0), comm, ufl.dx)
        # print(mac_vol)
        # print('macro stress')
        mac_stress = integrate_tensor(stress, comm, ufl.dx)
        # print(mac_stress)
        # print('macro strain')
        mac_strain = integrate_tensor(strain, comm, ufl.dx)
        # print(mac_strain)
        # print('macro ep')
        # mac_ep = integrate_tensor(pl_strain, comm ,ufl.dx)
        # print(mac_ep)
        print('mac_alpha')
        mac_alpha = integrate_scalar(pl_alpha, comm, ufl.dx)
        print(mac_alpha)

        # eig = eigenvalues(ufl.as_tensor(mac_stress))

        logger.data['eps_1'] = mac_strain[0, 0]
        logger.data['eps_2'] = mac_strain[1, 1]
        logger.data['sig_1'] = mac_stress[0, 0]
        logger.data['sig_2'] = mac_stress[1, 1]
        logger.data['mac_alpha'] = mac_alpha
        logger.write()