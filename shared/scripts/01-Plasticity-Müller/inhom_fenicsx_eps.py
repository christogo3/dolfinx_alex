import dolfinx as dlfx
import numpy as np
import ufl
import basix.ufl
import yaml

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem.petsc import NonlinearProblem

#import alex.plasticity

# plastic used to compute residuum, access and update alpha & eps --> alex.plasticity.Large_deformation_3D
# Others self-explanatory
from Materials import plastic, eigenvalues, integrate_scalar, integrate_tensor

# log_writer used to define logger --> Saves history for each timestep
from Utils import log_writer

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

eps0_1 =  parameters['fem']['eps0_1']
eps0_2 =  parameters['fem']['eps0_2']
element_degree = parameters['fem']['element_degree']
load_steps_1 = parameters['fem']['load_steps_1']
load_steps_2 = parameters['fem']['load_steps_2']

# Emod = dlfx.fem.Constant(region, 10.0)
# nu = dlfx.fem.Constant(region, 0.3)
# eps0 = 0.1


# Ausgabe-Level festlegen (EROOR, INFO, OFF, WARNING)
dlfx.log.set_log_level(dlfx.log.LogLevel.WARNING)

# Initialisierugn der MPI-Umgebung
comm = MPI.COMM_WORLD

# Netz einlesen
with dlfx.io.XDMFFile(comm, '/home/resources/Nanomesh.xdmf', 'r') as mesh_inp:
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

# Postprocessing spaces (CG1)
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
#plasticityProblem = alex.plasticity.Plasticity_incremental_3D(sig_y=sigY,hard=Hmod,alpha_n=alpha,e_p_n=ep)
material.report() 
res = material.residual(ufl.dx)
'''um1 =  dlfx.fem.Function(U)
du = ufl.TestFunction(U)
la = (nu/(1-2*nu))*(1/(1+nu))*Emod
mu = 0.5*(1/(1+nu))*Emod
[res, dResdw] = plasticityProblem.prep_newton(u=u,um1=um1,du=du,ddu=0,lam=la,mu=mu)'''

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

# init logger
logger = log_writer('output.log', {'eps_1': 0.0, 'eps_2': 0.0, 'sig_1': 0.0, 'sig_2': 0.0, 'mac_alpha': 0.0, 'mac_eps_11': 0.0, 'mac_eps_22': 0.0, 'mac_eps_12': 0.0, 'mac_sig_11': 0.0, 'mac_eps_22': 0.0, 'mac_eps_12': 0.0})

# init load_counter
load_count = 0

for load_factor_1 in np.linspace(-1.0, 1.0, load_steps_1):

    for load_factor_2 in np.linspace(-1.0, 1.0, load_steps_2):

        if load_steps_1 == 1 and load_steps_2 == 1:
            load_factor_1 = 1.0
            load_factor_2 = 1.0

        mac_strain = ufl.as_tensor([[eps0_1*load_factor_1, 0.0],
                                [0.0, eps0_2*load_factor_2]])


        u_mac_expr = dlfx.fem.Expression(mac_strain*x, U.element.interpolation_points, comm=region.comm)
        u_mac = dlfx.fem.Function(U, name='u_mac')
        u_mac.interpolate(u_mac_expr)

        # set boundary condition from u_mac
        right_dofs = dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(11))
        bc_right = dlfx.fem.dirichletbc(u_mac, right_dofs)
        top_dofs = dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(12))
        bc_top = dlfx.fem.dirichletbc(u_mac, top_dofs)
        left_dofs = dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(13))
        bc_left = dlfx.fem.dirichletbc(u_mac, left_dofs)
        bottom_dofs= dlfx.fem.locate_dofs_topological(U, fdim, facet_tags.find(14))
        bc_bottom = dlfx.fem.dirichletbc(u_mac, bottom_dofs)

        # collect bc
        bcs = [bc_right, bc_top, bc_left, bc_bottom]

        # zero all history variables
        material.alpha.x.array[:] = 0.0
        material.ep.x.array[:] = 0.0

        petsc_options= {'snes_type': 'newtonls', 
                            'snes_error_if_not_converged': True,
                            'ksp_error_if_not_converged': True,
                            'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            'pc_factor_mat_solver_type': 'mumps'}

        problem = NonlinearProblem(res, u, bcs=bcs,petsc_options=petsc_options, petsc_options_prefix='bla')

        problem.solve()
        material.update()

        # Ausgabe vorbereiten und Netz schreiben

        disp.interpolate(u)
        disp_mac.interpolate(u_mac)

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
                xdmfout.write_function(disp, load_count)
                xdmfout.write_function(disp_mac, load_count)
                xdmfout.write_function(strain, load_count)
                xdmfout.write_function(stress, load_count)
                xdmfout.write_function(pl_strain, load_count)
                xdmfout.write_function(pl_alpha, load_count)
        
        # ipdate load_count
        load_count = load_count+1 

        print('')
        print('R E S U L T S ', load_factor_1, ' / ', load_factor_2)
        #print('cross section')
        # mac_vol = integrate_scalar(dlfx.fem.Constant(region, 1.0), comm, ufl.dx)
        # print(mac_vol)
        print('macro strain')
        mac_strain = integrate_tensor(strain, comm, ufl.dx)
        print(mac_strain)
        print('macro stress')
        mac_stress = integrate_tensor(stress, comm, ufl.dx)
        print(mac_stress)
        # print('macro ep')
        # mac_ep = integrate_tensor(pl_strain, comm ,ufl.dx)
        # print(mac_ep)
        print('mac_alpha')
        mac_alpha = integrate_scalar(pl_alpha, comm, ufl.dx)
        print(mac_alpha)

        eig = eigenvalues(ufl.as_tensor(mac_stress))

        sig1, sig2 = eig[0], eig[1]

        if mac_strain[0, 0] < mac_strain[1, 1]:
            sig1, sig2 = eig[1], eig[0]
            
        logger.data['eps_1'] = mac_strain[0, 0]
        logger.data['eps_2'] = mac_strain[1, 1]    
        logger.data['sig_1'] = sig1
        logger.data['sig_2'] = sig2
        logger.data['mac_alpha'] = mac_alpha
        logger.data['mac_eps_11'] = mac_strain[0, 0]
        logger.data['mac_eps_22'] = mac_strain[1, 1]
        logger.data['mac_eps_12'] = mac_strain[0, 1]
        logger.data['mac_sig_11'] = mac_stress[0, 0]
        logger.data['mac_sig_22'] = mac_stress[1, 1]
        logger.data['mac_aig_12'] = mac_stress[0, 1]
        
        logger.write()