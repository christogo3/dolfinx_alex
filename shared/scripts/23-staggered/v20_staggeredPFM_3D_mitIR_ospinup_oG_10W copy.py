from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
import dolfinx as dlf
import dolfinx.nls.petsc
from dolfinx.fem.petsc import LinearProblem
import sys
import ufl
from basix.ufl import element
import numpy as np
from petsc4py import PETSc
import Restart_x_v3
import os
#
# TODO
# RB s lock testen 
# unterschied zu monolithic v20?

###
# todo OneIR example in 3D
# todo staggered approach
# todo mit restart
# todo mit RB 0 für s < lock_val
# todo mit glen über u_old
# todo mit IR und IceTounge
# todo erhöhtes eta_rs
#

#dlf.log.set_log_level(dlf.log.LogLevel.INFO)  # dlf.log.set_log_level(dlf.log.LogLevel.INFO)

# Create mesh with gmsh
mesh_comm = MPI.COMM_WORLD
model_rank = 0

##########################################################################################
# setup for simulation
week = 10
material = 'SVK'  # oder 'SVK' 'NH'
material_description = 'potential'  # 'weak' 'potential'
glen = 'mitGlen'  # 'mitGlen' 'ohneGlen'

case = 'nospinup'  #'nospinup'  #'afterspinup'  #'spinup'
load_restart = False

Gcfaktor = 1000
beta_crit = 1.0

aufloesung = 'mittel'
#meshfile = 'mesh240412/IceTounge_100m_oIR_wasserlinieIR2_3D_5m_klein_halb_neu2_Ausschnitt_mitIR' + aufloesung
meshfile = 'mesh240617/IceTounge_100m_oIR_wasserlinieIR2_3D_klein_halb_Ausschnitt_mitIR_mitRisseninclined' + aufloesung
print('Meshfile', meshfile)

# Gemometry
radius = 250
icerise_mx = 1510  # 3010 #1510     # 3010
icerise_my1 = 1500
x_startpunkt = icerise_mx - radius

resultsfoldername = 'results_v20_staggeredPFM_mitIR_mitRissen_inclined_' + str(week) + 'W_' + glen
out_path = resultsfoldername

filename = 'stag_3D_mitIR_' + case + '_' + str(week) + 'W_' + material + '_' + material_description + '_' + glen + aufloesung + '_' \
               + '_with_ux=1_uzfree_mits1e8_mob0001_5m_gc' + str(Gcfaktor) + '_slock001_etars_0005' + '_beta_' + str(beta_crit)

if case == 'spinup' or case == 'nospinup':
    out_file = case + '_' + resultsfoldername
else:
    out_file = 'spinup_' + resultsfoldername
##########################################################################################
# set parameters
H = 100
v_l = 1.0

rho_ice = 910  # 790-910 kg/m^3 density ice
rho_sw = 1028  # density salt water

g = 9.81  # m/(s^2)

H_sw = rho_ice / rho_sw * H

# read in mesh
with XDMFFile(mesh_comm, meshfile + '.xdmf', "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

with XDMFFile(mesh_comm, meshfile + 'facet.xdmf', "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")
MPI.COMM_WORLD.barrier()

dim = mesh.geometry.dim
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
mesh.topology.create_connectivity(1, 3)
mesh.topology.create_connectivity(1, 2)

#  material parameters
Ee = 1 * 1e9    # Pa
etae = 5e15     # Pa s
nue = 0.325
lam = nue / (1 - 2 * nue) * 1 / (1 + nue) * Ee
mu = 1 / 2 * 1 / (1 + nue) * Ee
K = (lam + 2 / 3 * mu)
eta_g_lock_min = 1e15       # Pa s
eta_g_lock_max = 1e18       # Pa s

E = dlf.fem.Constant(mesh, dlf.default_scalar_type(Ee))         # elasticity spring
eta = dlf.fem.Constant(mesh, dlf.default_scalar_type(etae))     # viscosity damper
nu = dlf.fem.Constant(mesh, dlf.default_scalar_type(nue))       # Possions ration
rho_sw_s = dlf.fem.Constant(mesh, dlf.default_scalar_type(rho_sw))
g_s = dlf.fem.Constant(mesh, dlf.default_scalar_type(g))

tau_b_s = 1             # Pa
tau_b_s_end = 100000    # 365 000
tau_b_days = 5          # 30

# time parameters
day = 86400.0     # s

if case == 'spinup':
    dTmax = day
    time = day * 7 * week
elif case == 'nospinup':
    dTmax = day / 4
    time = day * 7 * week
else:
    dTmax = day / 4
    time = day * 7 * week * 2

dT = 1.0  # 1000.0         # day
dTstart = 1.0  # 1000.0    # day

dt = dlf.fem.Constant(mesh, dlf.default_scalar_type(dT))  # time step
min_dT = 1e-50
cancel_job = False
cancel_job2 = False
counter_restart = 0

t = 0.0
trestart = 0.0
dTchange = False

# phase field parameters
MobParam = 0.0001
crack_length_max = 1.6e+07  # todo realistischer wert
etase = 0.005
etas = dlf.fem.Constant(mesh, dlf.default_scalar_type(etase))  # residual stiffness
s_crack_find = 0.01
lock_tol = 0.01
lock_val = 0.0
locked_points = []
locked_points_index = []
locked_nodes = np.array([], dtype=int)

GcParam = (95 * 1e3) ** 2 / Ee * Gcfaktor  # todo stimmt die größenordnung

Gc = dlf.fem.Constant(mesh, dlf.default_scalar_type(GcParam))
Mob = dlf.fem.Constant(mesh, dlf.default_scalar_type(MobParam))
s0 = dlf.fem.Constant(mesh, dlf.default_scalar_type(1.0))  # initial value for phase field

num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(mesh.topology.dim).num_ghosts  # für MPI wichtig
num_facets = mesh.topology.index_map(dim - 1).size_local + mesh.topology.index_map(fdim).num_ghosts  

def all(x):
    val = np.full(np.shape(x)[1], True)
    return val

entities = dlf.mesh.locate_entities(mesh,dim,all)
#print(mesh.h(mesh.topology.dim, entities))


epsilonParam = mesh_comm.allreduce(np.min(mesh.h(mesh.topology.dim, entities)), op=MPI.MIN)
#epsilonParam = mesh_comm.allreduce(np.min(mesh.h(dim, range(num_cells))), op=MPI.MIN)
epsilon = dlf.fem.Constant(mesh, dlf.default_scalar_type(3 * epsilonParam))  # set transitionzone for PFM
sigmax = np.sqrt(27 * Ee * GcParam / (512 * epsilonParam))  # todo Formel überprüfen

# to find crack tip
node_coor = mesh.geometry.x
node_dist = np.sqrt(node_coor[:, 0] ** 2 + node_coor[:, 1] ** 2 + node_coor[:, 2] ** 2)

#  functions or boundary conditions
urate = dlf.fem.Constant(mesh, dlf.default_scalar_type(0.0))
fz_s = dlf.fem.Constant(mesh, dlf.default_scalar_type(-rho_ice * g))
x = ufl.SpatialCoordinate(mesh)

e_x = ufl.as_vector((1.0, 0.0, 0.0))
e_y = ufl.as_vector((0.0, 1.0, 0.0))
e_z = ufl.as_vector((0.0, 0.0, 1.0))

# numerical parameters
max_iters = 40
min_iters = 6

max_iterschritte = 20  # 50
itterschritte = 0

plot_counter = 0
plot_freq = 1

# print MPI Status
rank = mesh_comm.Get_rank()
size = mesh_comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

if rank == 0:
    print('==================================================')
    print('reduce residual stiffness')
    print('==================================================')
    print('Filename')
    print(filename)
    print('==================================================')
    print('Mesh statistics')
    print('')
    print('Mesh cell hmin: ', str(epsilonParam))
    print('Mesh num cells: ', num_cells)
    print('Mesh num facets: ', num_facets)
    print('==================================================')
    print('Material parameters')
    print('')
    print('Youngs modulus ', Ee)
    print('viscosity ', etae)
    print('eta_g min', str(eta_g_lock_min))
    print('eta_g max', str(eta_g_lock_max))
    print('possions ratio ', nue)
    print('G_c ', str(GcParam))
    print('h_min ', str(epsilonParam))
    print('l_0', str(3 * epsilonParam))
    print('sig_mx ', str(sigmax))
    print('')
    print('tau_b_days ', str(tau_b_days))
    print('tau_b  ', str(tau_b_s_end))
    print('==================================================')
    print('Numerical parameters')
    print('eta_rs ', str(etase))
    print('Newton max iter ', str(max_iters))
    print('staggered max iter ', str(max_iterschritte))
    print('critical beta ', str(beta_crit))
    print('plot frequence ', str(plot_freq))
    sys.stdout.flush()

##########################################################################################
# create Function Spaces
Te = element("DG", mesh.basix_cell(), 0, shape=(dim, dim))
Ve = element("Lagrange", mesh.basix_cell(), 1, shape=(dim,))
Se = element("Lagrange", mesh.basix_cell(), 1)  # , shape=(1,)

V = dlf.fem.functionspace(mesh, Ve)
T = dlf.fem.functionspace(mesh, Te)
S = dlf.fem.functionspace(mesh, Se)

XDMF_Ve = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
XDMF_V = dlf.fem.functionspace(mesh, XDMF_Ve)

XDMF_Te = element("DG", mesh.basix_cell(), 0, shape=(mesh.geometry.dim, mesh.geometry.dim))
XDMF_T = dlf.fem.functionspace(mesh, XDMF_Te)

##########################################################################################
# create Functions
u_ls = dlf.fem.Function(V, name='u')
du_ls = ufl.TrialFunction(V)
ddu_ls = ufl.TestFunction(V)
u_ls_restart = dlf.fem.Function(V)
s_ls_restart = dlf.fem.Function(S)

u_old_restart = dlf.fem.Function(V)
s_old_restart = dlf.fem.Function(S)

C_vold_ls = dlf.fem.Function(T)

C_vold_restart = dlf.fem.Function(T)
s_m1_restart = dlf.fem.Function(S)
u_m1_restart = dlf.fem.Function(V)

u_old = dlf.fem.Function(V)
u_zero = dlf.fem.Function(V)

s_ls = dlf.fem.Function(S, name='s')
ds_ls = ufl.TrialFunction(S)
dds_ls = ufl.TestFunction(S)

s_old = dlf.fem.Function(S)

# to prevent crack healing
s_zero = dlf.fem.Function(S, name='s_zero')
s_zero_restart = dlf.fem.Function(S)

# for saving relevant functions
F_ls = dlf.fem.Function(T, name='F_ls')

P_v_v = dlf.fem.Function(T, name='P_v_v')
S_v_v = dlf.fem.Function(T, name='S_v_v')
Svol = dlf.fem.Function(T, name='S_v_vol')

C_ls = dlf.fem.Function(T, name='C_ls')
C_v_ls = dlf.fem.Function(T, name='C_v_ls')
E_v_v = dlf.fem.Function(T, name='E_v_v')
E_el_v = dlf.fem.Function(T, name='E_el')

eta_g = dlf.fem.Function(S, name='eta_g')

cs_ls_s = dlf.fem.Function(T, name='cs_ls_s')
cs_ls_p = dlf.fem.Function(T, name='cs_ls_p')
e_v_v = dlf.fem.Function(T, name='e_v_v')

XDMF_us = dlf.fem.Function(XDMF_V, name='us')
s_m1 = dlf.fem.Function(S, name='s_old_t')
u_m1 = dlf.fem.Function(V, name='u_old_t')
velocity = dlf.fem.Function(V, name='v')

# Functions for energys
psi_pos = dlf.fem.Function(S, name='psi_pos')
psi_ges = dlf.fem.Function(S, name='psi_ges')
psi_pos_vol = dlf.fem.Function(S, name='psi_pos_vol')
psi_pos_dev = dlf.fem.Function(S, name='psi_pos_dev')

# Function for boundary condition
ftau_b = dlf.fem.Function(S, name='tau_b')

f_IR = dlf.fem.Function(S, name='f_IR')
trac_MK = dlf.fem.Function(V, name='traction_vector_MK')
trac_RK = dlf.fem.Function(V, name='traction_vector_RK')

##########################################################################################
inlet_marker, outlet_marker, sidefront_marker, sideback_marker, top_marker, bottom_marker = 1, 2, 3, 4, 5, 6
back2_marker = 7  # ice front at the back of the domain
ir_marker = 9

inlet_dofs_x = dlf.fem.locate_dofs_topological(V.sub(0), ft.dim, ft.find(inlet_marker))
inlet_dofs_y = dlf.fem.locate_dofs_topological(V.sub(1), ft.dim, ft.find(inlet_marker))
inlet_dofs_z = dlf.fem.locate_dofs_topological(V.sub(2), ft.dim, ft.find(inlet_marker))
sidefront_dofs = dlf.fem.locate_dofs_topological(V.sub(1), ft.dim, ft.find(sidefront_marker))
sideback_dofs = dlf.fem.locate_dofs_topological(V.sub(1), ft.dim, ft.find(sideback_marker))

# set dirichlet boundary condition
bc_inlet_x = dlf.fem.dirichletbc(urate, inlet_dofs_x, V.sub(0))
bc_inlet_y = dlf.fem.dirichletbc(dlf.fem.Constant(mesh, 0.0), inlet_dofs_y, V.sub(1))
bc_inlet_z = dlf.fem.dirichletbc(dlf.fem.Constant(mesh, 0.0), inlet_dofs_x, V.sub(2))

bc_sidefront = dlf.fem.dirichletbc(dlf.fem.Constant(mesh, 0.0), sidefront_dofs, V.sub(1))
bc_sideback = dlf.fem.dirichletbc(dlf.fem.Constant(mesh, 0.0), sideback_dofs, V.sub(1))

bcs = [bc_inlet_x, bc_inlet_y, bc_sideback, bc_sidefront]  # bc_inlet_z,


##########################################################################################
# Finite Deformations
def J(u):
    return ufl.det(F(u))


def F(u):
    return ufl.grad(u) + ufl.Identity(dim)


def C(u):
    return F(u).T * F(u)


def EGL(u):  # Green Lagrange
    return 0.5 * (C(u) - ufl.Identity(dim))


def dEGL(u, v):
    dF = ufl.grad(v)
    return 0.5 * (F(u).T * dF + dF.T * F(u))


# first Piola Kirchhoff
def P_SVK(u, uold, C_vold, dt):
    return F(u) * S_SVK(u, uold, C_vold, dt)


def P_NH(u, uold, C_vold, dt):
    return F(u) * S_NH(u, uold, C_vold, dt)


# right Cauchy Green
def C_v_exp_SVK(u, uold, C_vold, dt):
    if glen == 'mitGlen':
        eta_n = eta_glen_SVK(uold, C_vold, dt)
    else:
        eta_n = eta
    C_vneu = mu / eta_n * dt * (C(u) - 1 / dim * ufl.tr(C(u) * ufl.inv(C_vold)) * C_vold) + C_vold
    Cvneu = C_vneu
    return Cvneu


def C_v_exp_NH(u, uold, C_vold, dt):
    if glen == 'mitGlen':
        eta_n = eta_glen_NH(uold, C_vold, dt)
    else:
        eta_n = eta
    J = ufl.det(F(u))
    C_vneu = mu / eta_n * dt * J ** (-2 / 3) * (
            C_vold - 1 / dim * ufl.tr(C_vold * ufl.inv(C(u))) * ufl.dot(ufl.dot(C_vold, ufl.inv(C(u))),
                                                                        C_vold)) + C_vold
    Cvneu = C_vneu
    return Cvneu


# second Piola Kirchoff Tensor
def S_SVK(u, uold, C_vold, dt):
    C_v_new = C_v_exp_SVK(u, uold, C_vold, dt)  # explizites Verfahren
    C_v_inv = ufl.inv(C_v_new)
    return K / 2 * (ufl.tr(C(u) * C_v_inv) - dim) * C_v_inv + mu * (
            C_v_inv * C(u) * C_v_inv - 1 / dim * ufl.tr(C(u) * C_v_inv) * C_v_inv)


# second Piola Kirchoff Tensor
def S_SVK_vol(u, uold, C_vold, dt):
    C_v_new = C_v_exp_SVK(u, uold, C_vold, dt)  # explizites Verfahren
    C_v_inv = ufl.inv(C_v_new)

    return K / 2 * (ufl.tr(C(u) * C_v_inv) - dim) * C_v_inv


def S_NH(u, uold, C_vold, dt):
    C_v_new = C_v_exp_NH(u, uold, C_vold, dt)  # explizites Verfahren
    C_v_inv = ufl.inv(C_v_new)
    C_inv = ufl.inv(C(u))
    J = ufl.det(F(u))
    return K / 2 * (J ** 2 - 1) * C_inv + mu * J ** (-2 / 3) * (C_v_inv - 1 / dim * ufl.tr(C_v_new * C_inv) * C_inv)


# other tensors
def EulerAlmansi(u):
    b = F(u) * F(u).T
    return 0.5 * (ufl.Identity(dim) - ufl.inv(b))


# bulk energy
def psiSVK2(u, uold, C_vold, dt):
    Kmod = lam + 2 / 3 * mu
    C_v = C_v_exp_SVK(u, uold, C_vold, dt)
    Eel = 1 / 2 * (C(u) - C_v)
    Evol = ufl.tr(Eel)
    Edev = ufl.dev(Eel)
    psielpos = 0.5 * (Kmod * 1 / dim * ufl.inner(pos(Evol) * ufl.Identity(dim),
                                                 pos(Evol) * ufl.Identity(dim)) + 2 * mu * ufl.inner(Edev, Edev))

    psielneg = 0.5 * Kmod * neg(Evol) ** 2
    return psielpos, psielneg  # oderaber psiel= psielpos+ psielneg


def Eelastic(u, uold, C_vold, dt):
    C_v = C_v_exp_SVK(u, uold, C_vold, dt)
    Eel = 1 / 2 * (C(u) - C_v)
    return Eel


def psiSVKpos(u, uold, C_vold, dt):
    Kmod = lam + 2 / 3 * mu
    C_v = C_v_exp_SVK(u, uold, C_vold, dt)
    Eel = 1 / 2 * (C(u) - C_v)
    Evol = ufl.tr(Eel)
    Edev = ufl.dev(Eel)
    psielposvol = 0.5 * (Kmod * pos(Evol) ** 2)
    psielposdev = mu * ufl.inner(Edev, Edev)

    # psielneg = 0.5 * Kmod * neg(Evol) ** 2
    return psielposvol, psielposdev


def psiNH(u, uold, C_vold, dt):
    Kmod = lam + 2 / 3 * mu
    J = ufl.det(F(u))
    U = Kmod / 4 * (J - 1) ** 2 - Kmod / 2 * ufl.ln(J)

    C_v = C_v_exp_NH(u, uold, C_vold, dt)
    C_dach = J ** (-2 / 3) * ufl.dot(C(u), C_v)
    I_C_dach = ufl.tr(C_dach)
    W_dach = mu * (I_C_dach - dim)
    psiel = U + W_dach
    return psiel


# Glen's flow law
def eta_glen_SVK(uold, Cvold, dt):
    C_g = C(uold)

    C_vinv = ufl.inv(Cvold)
    A_ratefactor = 10e-25

    sigma_eff2 = mu ** 2 * (ufl.tr((C_g * C_vinv) * (C_g * C_vinv)) - 1 / 3 * (ufl.tr(C_g * C_vinv)) ** 2)

    etaglen = ufl.conditional(ufl.gt(1 / (A_ratefactor * sigma_eff2), eta_g_lock_min),
                              ufl.conditional(ufl.lt(1 / (A_ratefactor * sigma_eff2), eta_g_lock_max),
                                              1 / (A_ratefactor * sigma_eff2), eta_g_lock_max), eta_g_lock_min)

    return etaglen


def eta_glen_NH(uold, Cvold, dt):
    if t - dT > 0:
        J = ufl.det(F(uold))
        C_g = C(uold)

        C_ginv = ufl.inv(C_g)

        A_ratefactor = 10e-25

        s2 = 2 / 3 * ufl.tr(C_g * ufl.inv(Cvold)) * ufl.tr(C_ginv * Cvold)
        s3 = 1 / 9 * ufl.tr(C_g * ufl.inv(Cvold)) * ufl.tr(C_g * ufl.inv(Cvold)) * ufl.tr(
            C_ginv * Cvold * C_ginv * Cvold)

        sigma_eff2 = mu ** 2 * J ** (-4 / 3) * (3 - s2 - s3)
        etaglen = 1 / (2 * A_ratefactor * sigma_eff2)  # faktor 2
    else:
        etaglen = eta

    return etaglen


def cs_s(u, uold, Cvold, dt):
    if material == 'NH':
        S = S_NH(u, uold, Cvold, dt)
    elif material == 'SVK':
        S = S_SVK(u, uold, Cvold, dt)

    return 1 / ufl.det(F(u)) * F(u) * S * F(u).T


# other usefull functions#
def pos(x):
    return 0.5 * (x + abs(x))


def neg(x):
    return 0.5 * (x - abs(x))


# Set initial Values
def t_init(x):
    if dim == 2:
        values = np.zeros((4, x.shape[1]))
        values[0] = 1.0
        values[3] = 1.0
    else:
        values = np.zeros((9, x.shape[1]))
        values[0] = 1.0
        values[4] = 1.0
        values[8] = 1.0
    return values


def u_init(x):
    values = np.zeros((dim, x.shape[1]))
    return values


# PHASENFELD
#Probleme beim parallelrechnen 
def newcrack(x):
    val = np.less_equal(s_zero.x.array[0:], lock_tol)
    print(val)
    return val

def intact(x):
    # num_points = np.shape(x)[1]
    # val = np.isclose(s_zero.x.array[0:], lock_tol, atol=5e-03)
    val = np.full(np.shape(x)[1], True)
    return val

def all(x):
    return np.full_like(x[0], True)


def degrad(s):
    degrad = s ** 2 + etas
    return degrad


def crack(s):
    crack = ((1 - s) ** 2) / (4 * epsilon) + epsilon * (ufl.dot(ufl.grad(s), ufl.grad(s)))
    return crack


def psisurf(s):
    psisurf = Gc * crack(s)
    return psisurf


# Eigenvalues are roots of characteristic polynomial
# e**2 - tr(A)*e + det(A) = 0
# in 2D: Lösung mithilfe von pq-Formel
def eig_plus(A):
    return (ufl.tr(A) / 2 + ufl.sqrt(ufl.tr(A) ** 2 / 4 - ufl.det(A)))


def eig_minus(A):
    return (ufl.tr(A) - dlf.sqrt(ufl.tr(A) ** 2 - 4 * ufl.det(A))) / 2


# write log-file
def write_log(time, dt, itterschritte, max_itterschritte, E_u, E_s, beta, initialize=True):
    if initialize:
        log_file = open(out_path + '/' + out_file + '_log.txt', 'w')
        write_log(t, dt, itterschritte, max_iterschritte, E_u, E_s, beta, initialize=False)
        log_file.write(
            '# time          |    dt        | iterschritte  | max_iterschritte |  E_u       |  E_s     | beta \n')
        log_file.close()
    else:
        log_file = open(out_path + '/' + out_file + '_log.txt', 'a')
        log_file.write(
            '{0:.6e}     {1:.6e}     {2:4d}            {3:4d}      {4:.6e}     {5:.6e}     {6:.2e} \n'.format(time, dt,
                                                                                                              itterschritte,
                                                                                                              max_itterschritte,
                                                                                                              E_u, E_s,
                                                                                                              beta))
        log_file.close()
    return


def write_log_converged(time, dT, tau_b, itterschritte, crack_length, psi_crack, psi_el_pos, psi_el_neg, timer, force,
                        displacement, initialize=True):
    if initialize:
        log_file = open(out_path + '/' + out_file + '_converged_log.txt', 'w')
        write_log_converged(t, dT, tau_b, itterschritte, crack_length, psi_crack, psi_el_pos, psi_el_neg, timer, force,
                        displacement, initialize=False)
        log_file.write(' \n')
        log_file.write('Filename' + filename + '\n')
        log_file.write('Mesh statistics  \n')
        log_file.write('Mesh num cells: ' + str(num_cells) + ' \n')
        log_file.write('Mesh num facets: ' + str(num_facets) + ' \n')
        log_file.write('Material Parameters \n')
        log_file.write('Youngs modulus ' + str(Ee) + ' \n')
        log_file.write('viscosity ' + str(etae) + ' \n')
        log_file.write('eta_g min ' + str(eta_g_lock_min) + ' \n')
        log_file.write('eta_g max ' + str(eta_g_lock_max) + ' \n')
        log_file.write('possions ratio ' + str(nue) + ' \n')
        log_file.write('GC ' + str(GcParam) + ' \n')
        log_file.write('length scale ' + str(epsilonParam) + ' \n')
        log_file.write('sig_mx ' + str(sigmax) + ' \n')
        log_file.write('tau_b_days ' + str(tau_b_days) + ' \n')
        log_file.write('tau_b  ' + str(tau_b_s_end) + ' \n')
        log_file.write('eta_rs ' + str(etase) + ' \n')
        log_file.write('Newton max iter ' + str(max_iters) + ' \n')
        log_file.write('staggered max iter ' + str(max_iterschritte) + ' \n')
        log_file.write('critical beta ' + str(beta_crit) + ' \n')
        log_file.write('plot frequence' + str(plot_freq) + ' \n')
        log_file.write(
            '# time        |    dt        |  tau_b       | iterschritte    |  crack length   | psi_crack     | psi_el_pos    | psi_el_neg    | timer    | force   | displacement \n')
        log_file.close()
    else:
        log_file = open(out_path + '/' + out_file + '_converged_log.txt', 'a')
        log_file.write(
            '{0:.8e}     {1:.8e}      {2:.3e}     {3:4d}      {4:.8e}      {5:.8e}     {6:.8e}     {7:.8e}   {8:.2e}     {9:.4e}     {10:.4e} \n'.format(
                time, dT, tau_b, itterschritte, crack_length, psi_crack, psi_el_pos, psi_el_neg, timer, force,
                displacement))
        log_file.close()
    return


def calculate_traction(Trac):
    dtr = ufl.TrialFunction(V)
    ddtr = ufl.TestFunction(V)

    L = (ufl.dot(Trac, ddtr) * ds(bottom_marker))
    A = (ufl.dot(dtr, ddtr) * ds(bottom_marker))

    problem = dlf.fem.petsc.LinearProblem(A, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    problem._A.zeroEntries()

    dlf.fem.petsc.assemble_matrix(problem._A, problem._a, bcs=problem.bcs)
    problem._A.assemble()
    # problem._A.setOption(problem._A.Option.NEW_NONZERO_ALLOCATION_ERR, 1)
    # Get diagonal of assembled A matrix
    diagonal = problem._A.getDiagonal()
    diagonal_values = diagonal.array

    # Get zero rows of assembled A matrix.
    zero_rows = problem._A.findZeroRows()
    zero_rows_values_global = zero_rows.array
    local_start = V.dofmap.index_map.local_range[0] * V.dofmap.index_map_bs

    # Maps global numbering to local numbering
    zero_rows_values_local = zero_rows_values_global - \
                             local_start
    diagonal.array[zero_rows_values_local] = np.ones(len(zero_rows_values_local), dtype=PETSc.ScalarType)

    problem._A.setOption(problem._A.Option.NEW_NONZERO_ALLOCATION_ERR, 0)
    problem._A.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)
    problem._A.assemble()

    trac = problem.solve()

    return trac


##########################################################################################
C_vold_ls.interpolate(t_init)
u_old.interpolate(u_init)
u_zero.interpolate(u_init)
u_m1.interpolate(u_init)

c = dlf.fem.Constant(mesh, dlf.default_scalar_type(1.0))
sub_expr = dlf.fem.Expression(c, S.element.interpolation_points())

s_ls.interpolate(sub_expr)
s_zero.interpolate(sub_expr)
s_zero_restart.interpolate(sub_expr)

s_old.x.array[:] = s_ls.x.array[:]
s_m1.x.array[:] = s_ls.x.array[:]
s_ls_restart.x.array[:] = s_ls.x.array[:]

C_vold_restart.x.array[:] = C_vold_ls.x.array[:]
u_m1_restart.x.array[:] = u_m1.x.array[:]
s_m1_restart.x.array[:] = s_m1.x.array[:]

s_ls_restart.x.scatter_forward()
s_old_restart.x.scatter_forward()
s_m1.x.scatter_forward()
s_old.x.scatter_forward()

ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
n = ufl.FacetNormal(mesh)

##########################################################################################
# set up xdmf file
if (case =='spinup' or case == 'nospinup') and not load_restart:
    xdmfout = XDMFFile(mesh_comm, resultsfoldername + '/' + filename + '_solution.xdmf', "w")
    xdmfout.write_mesh(mesh)
    xdmfout.write_meshtags(ft, mesh.geometry)  # in Paraview auf Grid(partial) gehen, Skala von 1-10
    # save initial value
    #XDMF_us.interpolate(dlf.fem.Expression(u_ls, XDMF_V.element.interpolation_points()))
    #xdmfout.write_function(XDMF_us, 0)
    #xdmfout.write_function(s_ls, 0)
    #xdmfout.write_function(s_zero, 0)
    xdmfout.close()

    # init log-file
    if rank == 0:
        write_log(0.0, 0.0, itterschritte, max_iterschritte, 0.0, 0.0, 45.0, initialize=True)
        write_log_converged(0.0, 0.0, 0.0, itterschritte, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initialize=True)

        if not (os.path.exists(resultsfoldername + '/test_restart')):
            os.mkdir(resultsfoldername + '/test_restart')

elif case == 'afterspinup' and not load_restart:
    if rank == 0:
        print('load function to restart after spinup')
        sys.stdout.flush()

    xdmfout = XDMFFile(mesh_comm, resultsfoldername + '/' + filename + '_solution.xdmf', "w")
    xdmfout.write_mesh(mesh)
    xdmfout.write_meshtags(ft, mesh.geometry)  # in Paraview auf Grid(partial) gehen, Skala von 1-10
    xdmfout.close()
    restart = Restart_x_v3.Restart_x(resultsfoldername + '/test_restart/spinup_restart', mesh_comm, w_ls1=u_ls, w_ls=u_ls,
                                  Cvold=C_vold_ls, u_m1=u_m1, s_ls=s_ls, s_m1=s_m1, s_zero=s_zero)
    restart.load_from_restart()
    t = restart.t
    dt.value = dT
    tau_b_s = restart.tau_b # tau_b_s = tau_b_s_end

    # update restart Variable
    trestart = t
    u_ls_restart.x.array[:] = u_ls.x.array[:]
    s_ls_restart.x.array[:] = s_ls.x.array[:]
    C_vold_restart.x.array[:] = C_vold_ls.x.array[:]
    s_m1_restart.x.array[:] = s_m1.x.array[:]
    u_m1_restart.x.array[:] = u_m1.x.array[:]
    s_zero_restart.x.array[:] = s_zero.x.array[:]

    u_ls_restart.x.scatter_forward()
    s_ls_restart.x.scatter_forward()
    C_vold_restart.x.scatter_forward()
    u_m1_restart.x.scatter_forward()
    s_m1_restart.x.scatter_forward()
    s_zero_restart.x.scatter_forward()

    if rank == 0:
        write_log(0.0, 0.0, itterschritte, max_iterschritte, 0.0, 0.0, 45.0, initialize=False)
        write_log_converged(0.0, 0.0, 0.0, itterschritte, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initialize=False)
elif load_restart:
    if rank == 0:
        print('load function to restart')
        sys.stdout.flush()
    restart = Restart_x_v3.Restart_x(resultsfoldername + '/test_restart/'+ case +'_restart', mesh_comm, w_ls1=u_ls, w_ls=u_ls,
                                  Cvold=C_vold_ls, u_m1=u_m1, s_ls=s_ls, s_m1=s_m1, s_zero=s_zero)
    restart.load_from_restart()
    t = restart.t
    dT = restart.dT
    dt.value = dT
    tau_b_s = restart.tau_b #tau_b_s = tau_b_s_end       # todo nur bei restart nach spin up

    # update restart Variable
    trestart = t
    u_ls_restart.x.array[:] = u_ls.x.array[:]
    s_ls_restart.x.array[:] = s_ls.x.array[:]
    C_vold_restart.x.array[:] = C_vold_ls.x.array[:]
    s_m1_restart.x.array[:] = s_m1.x.array[:]
    u_m1_restart.x.array[:] = u_m1.x.array[:]
    s_zero_restart.x.array[:] = s_zero.x.array[:]

    u_ls_restart.x.scatter_forward()
    s_ls_restart.x.scatter_forward()
    C_vold_restart.x.scatter_forward()
    u_m1_restart.x.scatter_forward()
    s_m1_restart.x.scatter_forward()
    s_zero_restart.x.scatter_forward()

    if rank == 0:
        write_log(0.0, 0.0, itterschritte, max_iterschritte, 0.0, 0.0, 45.0, initialize=False)
        write_log_converged(0.0, 0.0, 0.0, itterschritte, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initialize=False)

#######

crack_length = mesh_comm.allreduce(dlf.fem.assemble_scalar(dlf.fem.form(psisurf(s_ls) / Gc * ufl.dx)), op=MPI.SUM)
print('Crack length ', str(crack_length))

dTstart = dT


bcs_s = []

timer = dlf.common.Timer()
timer.start()
##########################################################################################
# start time loop
while t <= time and not (cancel_job):  # and crack_length < crack_length_max:
    MPI.COMM_WORLD.barrier()

    t = t + dT

    if tau_b_s < tau_b_s_end:
        tau_b_s = (t - dTstart) / day * tau_b_s_end / tau_b_days

    # tau_b_s = t/day * tau_b_s_end/tau_b_days

    xdmfout_it = XDMFFile(mesh_comm,resultsfoldername + '/' + case + 'iterrationsteps/' + filename + '_solution_it' + str(t) + '.xdmf',"w")
    xdmfout_it.write_mesh(mesh)
    xdmfout_it.write_meshtags(ft, mesh.geometry)  # in Paraview auf Grid(partial) gehen, Skala von 1-10
    xdmfout_it.close()
    itterschritte = 0
    urate.value = v_l / day * t

    error = 1
    beta = 45

    #crackfacets_update = dlf.mesh.locate_entities(mesh, 1, newcrack) # 0 statt ft.dim
    #crackdofs_update = dlf.fem.locate_dofs_topological(S, 1, crackfacets_update)
    #bccrack_update = dlf.fem.dirichletbc(dlf.default_scalar_type(0.0), crackdofs_update, S)

    all_entities = dlf.mesh.locate_entities(mesh, mesh.topology.dim-1, intact)
    all_dofs_s_local = dlf.fem.locate_dofs_topological(S, mesh.topology.dim-1, all_entities)
    ## all_dofs_s_global = np.array(dofmap.local_to_global(all_dofs_s_local),dtype=np.int32)

    array_s = s_old.x.array[all_dofs_s_local]
    indices_where_zero_in_array_s = np.where(np.isclose(array_s,0.0,atol=lock_tol))
    dofs_s_zero = all_dofs_s_local[indices_where_zero_in_array_s]
    ## array_s_zero=wm1.x.array[dofs_s_zero]

    crackdofs_update = dofs_s_zero 
    bccrack_update = dlf.fem.dirichletbc(dlf.default_scalar_type(0.0), crackdofs_update, S)

    bcs_s.append(bccrack_update)
    cancel_job2 = False

    while itterschritte <= max_iterschritte and error > 1e-8 and beta > beta_crit and not (cancel_job) and not (cancel_job2): #2
        MPI.COMM_WORLD.barrier()
        if rank == 0:
            print('==================================================')
            print('tau_b ', str(tau_b_s))
            print('compute solution for u for t=', str(t))
            print('time step ', str(dT))
            print('iteration step ', str(itterschritte), '/', str(max_iterschritte))
            sys.stdout.flush()

        p_front = ufl.conditional(ufl.le((x[2] + u_ls[2]), 0), - rho_sw * g * (-(x[2] + u_ls[2])), 0)
        p_bottom = - rho_sw * g * (-(x[2] + u_ls[2]))
        f_icerises = ufl.conditional(
            ufl.le(ufl.sqrt((x[0] + u_m1[0] - icerise_mx) ** 2 + (x[1] + u_m1[1] - icerise_my1) ** 2), radius), 1.0,
            0.0)
        f_bottom = ufl.conditional(ufl.le(f_icerises, 0.5), 1.0, 0.0)

        #######################################
        if material == 'SVK':
            psielpos, psielneg = psiSVK2(u_ls, u_m1, C_vold_ls, dt)
        elif material == 'NH':
            psielpos, psielneg = psiNH(u_ls, u_m1, C_vold_ls, dt)
        else:
            if rank == 0:
                print('Materialmodell nicht implementiert. ABBRUCH')
                sys.stdout.flush()
            break

        potin_u = (degrad(s_old) * psielpos + psielneg) * ufl.dx

        MPI.COMM_WORLD.barrier()
        mombal = ufl.derivative(potin_u, u_ls, ddu_ls)

        if t - dT == 0:
            tstern = p_bottom * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * n

            mombal1 = mombal \
                      - ufl.dot(fz_s * ufl.det(F(u_ls)) * e_z, ddu_ls) * ufl.dx \
                      - ufl.dot(p_bottom * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * n, ddu_ls) * ds(bottom_marker) \
                      - ufl.dot((p_front * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * n), ddu_ls) * ds(outlet_marker) \
                      - ufl.dot(tstern * f_icerises, ddu_ls) * ds(ir_marker) \
                      + 10e9 * ufl.dot(ufl.dot(u_ls, e_z) * f_icerises, ddu_ls[2]) * ds(ir_marker) \
                      + 10e9 * ufl.dot(ufl.dot(u_ls, e_z) * f_icerises, ddu_ls[2]) * ds(bottom_marker) \
                      - ufl.dot((p_front * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * n), ddu_ls) * ds(back2_marker)

        else:
            tau_b_s_h = ((ufl.tanh((x[0] - x_startpunkt - 20) / 12) + 1) * tau_b_s / 2 - (
                        ufl.tanh((x[0] - x_startpunkt - 470) / 12) - 1) * tau_b_s / 2) - tau_b_s
            tstern = - tau_b_s_h * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * e_x

            mombal1 = mombal \
                      - ufl.dot(fz_s * ufl.det(F(u_ls)) * e_z, ddu_ls) * ufl.dx \
                      - ufl.dot((p_front * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * n), ddu_ls) * ds(outlet_marker) \
                      - ufl.dot(p_bottom * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * n * f_bottom, ddu_ls) * ds(bottom_marker) \
                      - ufl.dot(tstern * f_icerises, ddu_ls) * ds(bottom_marker) \
                      - ufl.dot(tstern * f_icerises, ddu_ls) * ds(ir_marker) \
                      + 10e9 * ufl.dot(ufl.dot(u_ls, e_z) * f_icerises, ddu_ls[2]) * ds(ir_marker) \
                      + 10e9 * ufl.dot(ufl.dot(u_ls, e_z) * f_icerises, ddu_ls[2]) * ds(bottom_marker) \
                      - ufl.dot((p_front * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * n), ddu_ls) * ds(back2_marker) \

        form_ls = mombal1
        MPI.COMM_WORLD.barrier()

        # Compute Jacobian of F
        J_ls = ufl.derivative(form_ls, u_ls, du_ls)

        problem_u = dlf.fem.petsc.NonlinearProblem(form_ls, u_ls, bcs, J_ls)
        solver_u = dlf.nls.petsc.NewtonSolver(mesh_comm, problem_u)

        solver_u.max_it = max_iters
        solver_u.convergence_criterion = "incremental"
        solver_u.atol = 1e-8  # 1e-4
        solver_u.rtol = 1e-8  # 1e-4
        solver_u.report = True

        # We can customize the linear solver used inside the NewtonSolver by modifying the PETSc options
        # ksp = solver_u.krylov_solver
        # opts = PETSc.Options()  # type: ignore
        # option_prefix = ksp.getOptionsPrefix()
        # opts[f"{option_prefix}ksp_type"] = "preonly"
        # opts[f"{option_prefix}pc_type"] = "lu"
        # psys = PETSc.Sys()  # type: ignore
        # For factorisation prefer MUMPS, then superlu_dist, then default.
        # if psys.hasExternalPackage("mumps"):
        #    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        #    print('USES MUMPS AS SOLVER IN NEWTONSOLVER FOR U')
        # elif sys.hasExternalPackage("superlu_dist"):
        #    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
        # ksp.setFromOptions()

        restart_solution_u = False
        converged_u = False

        if rank == 0:
            print('LÖSEN')
            print('==================================================')
            sys.stdout.flush()

        try:
            MPI.COMM_WORLD.barrier()
            (iters, converged_u) = solver_u.solve(u_ls)
        except RuntimeError:
            restart_solution_u = True

        if not (restart_solution_u):
            u_ls.x.scatter_forward()

            if rank == 0:
                print('compute solution for s for t=', str(t))
                sys.stdout.flush()

            if material == 'SVK':
                psielpos, psielneg = psiSVK2(u_ls, u_m1, C_vold_ls, dt)
            elif material == 'NH':
                psielpos, psielneg = psiNH(u_ls, u_m1, C_vold_ls, dt)
            else:
                if rank == 0:
                    print('Materialmodell nicht implementiert. ABBRUCH')
                    sys.stdout.flush()
                break

            # todo solve s
            potin = (degrad(s_ls) * psielpos + psielneg + psisurf(s_ls)) * ufl.dx
            sdrive = ufl.derivative(potin, s_ls, dds_ls)

            rate = (s_ls - s_m1) / dT * dds_ls * ufl.dx

            Res_s = 1 / Mob * rate + sdrive
            # Res_s = sdrive
            dResdw_s = ufl.derivative(Res_s, s_ls, ds_ls)

            problem_s = dlf.fem.petsc.NonlinearProblem(Res_s, s_ls, bcs_s, dResdw_s)
            solver_s = dlf.nls.petsc.NewtonSolver(mesh_comm, problem_s)

            solver_s.max_it = max_iters
            solver_s.convergence_criterion = "incremental"
            solver_s.atol = 1e-8
            solver_s.rtol = 1e-8
            solver_s.report = True

            restart_solution_s = False
            converged_s = False

            if case=='nospinup' or case=='afterspinup':
                try:
                    (iters, converged_u) = solver_s.solve(s_ls)
                except RuntimeError:
                    restart_solution_s = True

            if not (restart_solution_s):
                s_ls.x.scatter_forward()

                xdmfout_it = XDMFFile(mesh_comm, resultsfoldername + '/' + case + 'iterrationsteps/' + filename + '_solution_it' + str(t) + '.xdmf', "a")

                try:
                    xdmfout_it.write_function(s_ls, itterschritte + counter_restart)
                    xdmfout_it.write_function(u_ls, itterschritte + counter_restart)
                    xdmfout_it.close()
                except RuntimeError:
                    xdmfout_it.close()

                # forward solution to mpi processes (might be necessary)
                s_ls.x.scatter_forward()

                if material == 'SVK':
                    psielpos, psielneg = psiSVK2(u_ls, u_m1, C_vold_ls, dt)
                elif material == 'NH':
                    psielpos, psielneg = psiNH(u_ls, u_m1, C_vold_ls, dt)

                energy = (degrad(s_ls) * psielpos + psielneg + psisurf(s_ls)) * ufl.dx

                # GerasimovDeLorenzis
                if itterschritte == 0:
                    e_0 = dlf.fem.form(energy)
                    E_0 = mesh_comm.allreduce(dlf.fem.assemble_scalar(e_0), op=MPI.SUM)
                    E_N = E_0
                else:
                    e_N = dlf.fem.form(energy)
                    E_N = mesh_comm.allreduce(dlf.fem.assemble_scalar(e_N), op=MPI.SUM)
                    if np.isclose((E_0 - E_N), 0.0) and case == 'spinup':
                        beta = beta_crit
                    else:
                        beta = mesh_comm.allreduce(np.arctan((E_old - E_N) / (E_0 - E_N) * itterschritte) * 180 / np.pi,
                                               op=MPI.MAX)
                    if rank == 0:
                        print('beta ', str(beta))
                        sys.stdout.flush()

                E_old = E_N

                error_u = dlf.fem.form((u_ls - u_old) ** 2 * ufl.dx)
                E_u = np.sqrt(mesh_comm.allreduce(dlf.fem.assemble_scalar(error_u), MPI.SUM))
                error_s = dlf.fem.form((s_ls - s_old) ** 2 * ufl.dx)
                E_s = np.sqrt(mesh_comm.allreduce(dlf.fem.assemble_scalar(error_s), MPI.SUM))
                error = max(E_u, E_s)
                if mesh_comm.rank == 0:
                    print('=======================')
                    print(f"L2-error für u: {E_u:.2e}")
                    print(f"L2-error für s: {E_s:.2e}")
                    print(f'max error:  {error:.2e}')
                    print('=======================')
                    sys.stdout.flush()

                # write log and/or tabular
                if rank == 0:
                    write_log(t, dT, itterschritte, max_iterschritte, E_u, E_s, beta, initialize=False)

                itterschritte = itterschritte + 1
                u_old.x.array[:] = u_ls.x.array[:]
                s_old.x.array[:] = s_ls.x.array[:]
                u_old_restart.x.array[:] = u_ls.x.array[:]
                s_old_restart.x.array[:] = s_ls.x.array[:]
                u_old.x.scatter_forward()
                s_old.x.scatter_forward()
                u_old_restart.x.scatter_forward()
                s_old_restart.x.scatter_forward()


                if rank == 0:
                    print('end of one iteration for u and s')
                    sys.stdout.flush()

                if itterschritte - 1 == max_iterschritte:
                    # restart solution
                    if dT > min_dT:
                        dT = 0.5 * dT
                        dt.value = dT
                    else:
                        cancel_job = True
                        break

                    counter_restart = counter_restart + 100
                    t = trestart  # + dT
                    u_ls.x.array[:] = u_ls_restart.x.array[:]
                    s_ls.x.array[:] = s_ls_restart.x.array[:]
                    u_old.x.array[:] = u_old_restart.x.array[:]
                    s_old.x.array[:] = s_old_restart.x.array[:]
                    u_ls.x.scatter_forward()
                    s_ls.x.scatter_forward()
                    u_old.x.scatter_forward()
                    s_old.x.scatter_forward()
                    itterschritte = 0

                    u_m1.x.array[:] = u_m1_restart.x.array[:]
                    s_m1.x.array[:] = s_m1_restart.x.array[:]
                    s_zero.x.array[:] = s_zero_restart.x.array[:]
                    C_vold_ls.x.array[:] = C_vold_restart.x.array[:]
                    u_m1.x.scatter_forward()
                    s_m1.x.scatter_forward()
                    s_zero.x.scatter_forward()
                    C_vold_ls.x.scatter_forward()
                    cancel_job2 = True

                    if rank == 0:
                        print('restart due to maxiterschritte, decrease dt to ', str(dT))
                        sys.stdout.flush()

            else:  # restart für s
                if dT > min_dT:
                    dT = 0.5 * dT
                    dt.value = dT
                else:
                    cancel_job = True
                    break

                counter_restart = counter_restart + 100
                t = trestart  # + dT
                u_ls.x.array[:] = u_ls_restart.x.array[:]
                s_ls.x.array[:] = s_ls_restart.x.array[:]
                u_old.x.array[:] = u_old_restart.x.array[:]
                s_old.x.array[:] = s_old_restart.x.array[:]
                u_ls.x.scatter_forward()
                s_ls.x.scatter_forward()
                u_old.x.scatter_forward()
                s_old.x.scatter_forward()
                itterschritte = 0

                u_m1.x.array[:] = u_m1_restart.x.array[:]
                s_m1.x.array[:] = s_m1_restart.x.array[:]
                s_zero.x.array[:] = s_zero_restart.x.array[:]
                C_vold_ls.x.array[:] = C_vold_restart.x.array[:]
                u_m1.x.scatter_forward()
                s_m1.x.scatter_forward()
                s_zero.x.scatter_forward()
                C_vold_ls.x.scatter_forward()

                cancel_job2 = True

                if rank == 0:
                    print('restart s, decrease dt to ', str(dT))
                    sys.stdout.flush()

        else:  # restart für u
            if dT > min_dT:
                dT = 0.5 * dT
                dt.value = dT
            else:
                cancel_job = True
                break

            counter_restart = counter_restart + 100
            t = trestart  # + dT
            u_ls.x.array[:] = u_ls_restart.x.array[:]
            s_ls.x.array[:] = s_ls_restart.x.array[:]
            u_old.x.array[:] = u_old_restart.x.array[:]
            s_old.x.array[:] = s_old_restart.x.array[:]
            u_ls.x.scatter_forward()
            s_ls.x.scatter_forward()
            u_old.x.scatter_forward()
            s_old.x.scatter_forward()
            itterschritte = 0

            u_m1.x.array[:] = u_m1_restart.x.array[:]
            s_m1.x.array[:] = s_m1_restart.x.array[:]
            s_zero.x.array[:] = s_zero_restart.x.array[:]
            C_vold_ls.x.array[:] = C_vold_restart.x.array[:]
            u_m1.x.scatter_forward()
            s_m1.x.scatter_forward()
            s_zero.x.scatter_forward()
            C_vold_ls.x.scatter_forward()

            cancel_job2 = True

            if rank == 0:
                print('restart u, decrease dt to ', str(dT))
                sys.stdout.flush()

            MPI.COMM_WORLD.barrier()


    if not(cancel_job2):
        if rank == 0:
            print('==================================================')
            print('end of iteration for timestep t= ', str(t))
            sys.stdout.flush()

        # write log converged file
        crack_length = mesh_comm.allreduce(dlf.fem.assemble_scalar(dlf.fem.form(psisurf(s_ls) / Gc * ufl.dx)), op=MPI.SUM)
        psi_crack = mesh_comm.allreduce(dlf.fem.assemble_scalar(dlf.fem.form(psisurf(s_ls) * ufl.dx)), op=MPI.SUM)

        if material == 'SVK':
            psielpos, psielneg = psiSVK2(u_ls, u_m1, C_vold_ls, dt)
        elif material == 'NH':
            psielpos, psielneg = psiNH(u_ls, u_m1, C_vold_ls, dt)

        psi_el_pos_g = mesh_comm.allreduce(dlf.fem.assemble_scalar(dlf.fem.form(degrad(s_ls) * psielpos * ufl.dx)),
                                           op=MPI.SUM)
        psi_el_neg_g = mesh_comm.allreduce(dlf.fem.assemble_scalar(dlf.fem.form(psielneg * ufl.dx)), op=MPI.SUM)

        displacement = v_l / day * t

        force = 0.0  # todo

        if rank == 0:
            elapsed_time = timer.elapsed()[0]
            write_log_converged(t, dT, tau_b_s, itterschritte, crack_length, psi_crack, psi_el_pos_g, psi_el_neg_g,
                                elapsed_time, force, displacement, initialize=False)

        #
        if material == 'SVK':
            psielpos, psielneg = psiSVK2(u_ls, u_m1, C_vold_ls, dt)
            P_v_v.interpolate(dlf.fem.Expression(P_SVK(u_ls, u_m1, C_vold_ls, dt), XDMF_T.element.interpolation_points()))
            S_v_v.interpolate(dlf.fem.Expression(S_SVK(u_ls, u_m1, C_vold_ls, dt), XDMF_T.element.interpolation_points()))
            C_v_ls.interpolate(
                dlf.fem.Expression(C_v_exp_SVK(u_ls, u_m1, C_vold_ls, dt), XDMF_T.element.interpolation_points()))
            psielposvol, psielposdev = psiSVKpos(u_ls, u_m1, C_vold_ls, dt)
            psi_pos_vol.interpolate(dlf.fem.Expression(psielposvol, XDMF_V.element.interpolation_points()))
            psi_pos_dev.interpolate(dlf.fem.Expression(psielposdev, XDMF_V.element.interpolation_points()))
            eta_g.interpolate(dlf.fem.Expression(eta_glen_SVK(u_m1, C_v_ls, dt), S.element.interpolation_points()))
        elif material == 'NH':
            psielpos, psielneg = psiNH(u_ls, u_m1, C_vold_ls, dt)
            P_v_v.interpolate(dlf.fem.Expression(P_NH(u_ls, u_m1, C_vold_ls, dt), XDMF_T.element.interpolation_points()))
            S_v_v.interpolate(dlf.fem.Expression(S_NH(u_ls, u_m1, C_vold_ls, dt), XDMF_T.element.interpolation_points()))
            C_v_ls.interpolate(
                dlf.fem.Expression(C_v_exp_NH(u_ls, u_m1, C_vold_ls, dt), XDMF_T.element.interpolation_points()))
            # eta_g.interpolate(dlf.fem.Expression(eta_glen_NH(u_ls, C_v_ls, dT), S.element.interpolation_points()))

        # try to get traction
        # Trac_RK = ufl.dot(S_SVK(u_ls, C_vold_ls, dT), n)
        # trac_RK = calculate_traction(Trac_RK)

        # Trac_MK = ufl.dot(P_SVK(u_ls, C_vold_ls, dT), n)
        # trac_MK = calculate_traction(Trac_MK)

        '''
        
        trac = dlf.fem.Function(V, name='traction-vector')
        dtr = ufl.TrialFunction(V)
        ddtr = ufl.TestFunction(V)
    
        L = ((ufl.dot(Trac, ddtr) * ds))
        A = ((ufl.dot(dtr, ddtr) * ds))
    
        problem = dlf.fem.petsc.LinearProblem(A, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        problem._A.zeroEntries()
    
        problem._A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR, 1)
        dlf.fem.petsc.assemble_matrix(problem._A, problem._a, bcs=problem.bcs)
        problem._A.assemble()
    
        # Get diagonal of assembled A matrix
        diagonal = problem._A.getDiagonal()
        diagonal_values = diagonal.array
    
        # Get zero rows of assembled A matrix.
        zero_rows = problem._A.findZeroRows()
        zero_rows_values_global = zero_rows.array
        local_start = V.dofmap.index_map.local_range[0] * V.dofmap.index_map_bs

        # Maps global numbering to local numbering
        zero_rows_values_local = zero_rows_values_global - \
                                 local_start
        diagonal.array[zero_rows_values_local] = np.ones(len(zero_rows_values_local), dtype=PETSc.ScalarType)
        problem._A.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)
        problem._A.assemble()
    
        trac = problem.solve()
    
    
        #A.ident_zeros()
        #dlf.solve(A, t.vector(), L)
        '''

        # nur Berechnen Falls die Größen rausgeschriben werden
        if plot_counter % plot_freq == 0:  # True: #
            XDMF_us.interpolate(dlf.fem.Expression(u_ls,
                                                   XDMF_V.element.interpolation_points()))  # muss seit version 0.8.0.0 gemacht werden da nicht mehr automatischinterpoliert wird.... siehe https://fenicsproject.discourse.group/t/i-o-from-xdmf-hdf5-files-in-dolfin-x/3122/54

            cs_ls_s.interpolate(dlf.fem.Expression(1 / ufl.det(F(u_ls)) * F(u_ls) * S_v_v * F(u_ls).T,
                                                   XDMF_T.element.interpolation_points()))
            cs_ls_p.interpolate(
                dlf.fem.Expression(1 / ufl.det(F(u_ls)) * P_v_v * F(u_ls).T, XDMF_T.element.interpolation_points()))

            E_el_v.interpolate(
                dlf.fem.Expression(Eelastic(u_ls, u_m1, C_vold_ls, dt), XDMF_T.element.interpolation_points()))

            E_v_v.interpolate(dlf.fem.Expression(EGL(u_ls), XDMF_T.element.interpolation_points()))
            e_v_v.interpolate(dlf.fem.Expression(EulerAlmansi(u_ls), XDMF_T.element.interpolation_points()))

            F_ls.interpolate(dlf.fem.Expression(F(u_ls), XDMF_T.element.interpolation_points()))
            C_ls.interpolate(dlf.fem.Expression(C(u_ls), XDMF_T.element.interpolation_points()))

            velocity.interpolate(dlf.fem.Expression((u_ls - u_m1) / dt, XDMF_V.element.interpolation_points()))

            f_IR.interpolate(dlf.fem.Expression(f_icerises, S.element.interpolation_points()))
            psi_pos.interpolate(dlf.fem.Expression(psielpos, S.element.interpolation_points()))
            psi_ges.interpolate(
                dlf.fem.Expression(degrad(s_ls) * psielpos + psielneg + psisurf(s_ls), S.element.interpolation_points()))

            if t - dTstart > 0:
                tau_b_s_h = ((ufl.tanh((x[0] - x_startpunkt - 20) / 12) + 1) * tau_b_s / 2 - (
                        ufl.tanh((x[0] - x_startpunkt - 470) / 12) - 1) * tau_b_s / 2) - tau_b_s
            else:
                tau_b_s_h = 1

            ftau_b.interpolate(dlf.fem.Expression((- tau_b_s_h * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * e_x)[0],
                                                    S.element.interpolation_points()))

            if rank == 0:
                print('save solution for t=', str(t))
                sys.stdout.flush()

            xdmfout = XDMFFile(mesh_comm, resultsfoldername + '/' + filename + '_solution.xdmf', "a")
            try:
                xdmfout.write_function(XDMF_us, t)
                xdmfout.write_function(P_v_v, t)
                xdmfout.write_function(S_v_v, t)
                xdmfout.write_function(C_v_ls, t)
                xdmfout.write_function(cs_ls_s, t)
                xdmfout.write_function(cs_ls_p, t)
                xdmfout.write_function(E_v_v, t)
                xdmfout.write_function(E_el_v, t)
                xdmfout.write_function(e_v_v, t)
                xdmfout.write_function(F_ls, t)
                xdmfout.write_function(C_ls, t)
                xdmfout.write_function(eta_g, t)
                #xdmfout.write_function(ftau_b, t)
                xdmfout.write_function(s_ls, t)
                #xdmfout.write_function(f_IR, t)
                xdmfout.write_function(psi_pos, t)
                xdmfout.write_function(psi_ges, t)
                xdmfout.write_function(velocity, t)
                xdmfout.write_function(psi_pos_vol, t)
                xdmfout.write_function(psi_pos_dev, t)
                xdmfout.close()
            except RuntimeError:
                xdmfout.close()

        plot_counter = plot_counter + 1

        # Update for next time step to prevent crack healing
        s_zero.x.array[:] = s_ls.x.array[:]
        s_zero_restart.x.array[:] = s_zero.x.array[:]

        # Update for next time step
        C_vold_ls.x.array[:] = C_v_ls.x.array[:]
        s_m1.x.array[:] = s_ls.x.array[:]
        u_m1.x.array[:] = u_ls.x.array[:]

        # Update restart variable für iteration
        u_old.x.array[:] = u_ls.x.array[:]
        s_old.x.array[:] = s_ls.x.array[:]
        u_old_restart.x.array[:] = u_ls.x.array[:]
        s_old_restart.x.array[:] = s_ls.x.array[:]
        u_old.x.scatter_forward()
        s_old.x.scatter_forward()
        u_old_restart.x.scatter_forward()
        s_old_restart.x.scatter_forward()

        # update overall restart variable
        trestart = t
        u_ls_restart.x.array[:] = u_ls.x.array[:]
        s_ls_restart.x.array[:] = s_ls.x.array[:]

        u_ls_restart.x.scatter_forward()
        s_ls_restart.x.scatter_forward()
        u_m1.x.scatter_forward()
        s_m1.x.scatter_forward()
        C_vold_ls.x.scatter_forward()

        C_vold_restart.x.array[:] = C_vold_ls.x.array[:]
        s_m1_restart.x.array[:] = s_m1.x.array[:]
        u_m1_restart.x.array[:] = u_m1.x.array[:]

        C_vold_restart.x.scatter_forward()
        u_m1_restart.x.scatter_forward()
        s_m1_restart.x.scatter_forward()

        if rank == 0:
            print('save function to restart')
            sys.stdout.flush()

        if case == 'afterspinup':
            restart = Restart_x_v3.Restart_x(resultsfoldername + '/test_restart/aspinup_restart', mesh_comm, t=t, dT=dT, w=u_ls,
                                          Cv=C_vold_ls, u_m1=u_m1, s_ls=s_ls, s_m1=s_m1, s_zero=s_zero, tau_b=tau_b_s)
            restart.save_for_restart()
        else:
            restart = Restart_x_v3.Restart_x(resultsfoldername + '/test_restart/'+ case +'_restart', mesh_comm, t=t, dT=dT, w=u_ls,
                                      Cv=C_vold_ls, u_m1=u_m1, s_ls=s_ls, s_m1=s_m1, s_zero=s_zero, tau_b=tau_b_s)
            restart.save_for_restart()

        if iters < 8 and 1.5 * dT < dTmax and t / day > tau_b_days:
            dT = 1.5 * dT
            dt.value = dT
            if rank == 0:
                print('increasing dT to', str(dT))
                sys.stdout.flush()
        elif itterschritte < 8 and 1.5 * dT < dTmax:
            dT = 1.5 * dT
            dt.value = dT
            if rank == 0:
                print('increasing dT due to itterschritte to', str(dT))
                sys.stdout.flush()
        elif itterschritte > 20:
            dT = 0.5 * dT
            dt.value = dT
            plot_freq = 5
            if rank == 0:
                print('decreasing dT due to itterschritte to', str(dT))
                sys.stdout.flush()



##########################################################################################
# calculate values to save last step
XDMF_us.interpolate(dlf.fem.Expression(u_ls, XDMF_V.element.interpolation_points()))  # muss seit version 0.8.0.0 gemacht werden da nicht mehr automatischinterpoliert wird.... siehe https://fenicsproject.discourse.group/t/i-o-from-xdmf-hdf5-files-in-dolfin-x/3122/54

cs_ls_s.interpolate(
    dlf.fem.Expression(1 / ufl.det(F(u_ls)) * F(u_ls) * S_v_v * F(u_ls).T, XDMF_T.element.interpolation_points()))
cs_ls_p.interpolate(dlf.fem.Expression(1 / ufl.det(F(u_ls)) * P_v_v * F(u_ls).T, XDMF_T.element.interpolation_points()))

E_el_v.interpolate(dlf.fem.Expression(Eelastic(u_ls, u_m1, C_vold_ls, dT), XDMF_T.element.interpolation_points()))
E_v_v.interpolate(dlf.fem.Expression(EGL(u_ls), XDMF_T.element.interpolation_points()))
e_v_v.interpolate(dlf.fem.Expression(EulerAlmansi(u_ls), XDMF_T.element.interpolation_points()))

F_ls.interpolate(dlf.fem.Expression(F(u_ls), XDMF_T.element.interpolation_points()))
C_ls.interpolate(dlf.fem.Expression(C(u_ls), XDMF_T.element.interpolation_points()))

f_IR.interpolate(dlf.fem.Expression(f_bottom, S.element.interpolation_points()))
psi_pos.interpolate(dlf.fem.Expression(psielpos, S.element.interpolation_points()))
psi_ges.interpolate(
    dlf.fem.Expression(degrad(s_ls) * psielpos + psielneg + psisurf(s_ls), S.element.interpolation_points()))

tau_b_s_h = ((ufl.tanh((x[0] - x_startpunkt - 20) / 8) + 1) * tau_b_s / 2 - (
            ufl.tanh((x[0] - x_startpunkt - 470) / 8) - 1) * tau_b_s / 2) - tau_b_s
ftau_b.interpolate(dlf.fem.Expression((- tau_b_s_h * ufl.det(F(u_ls)) * ufl.inv(F(u_ls)).T * e_x)[0],
                                        S.element.interpolation_points()))

# set up xdmf file
xdmfout = XDMFFile(mesh_comm, resultsfoldername + '/' + filename + '_solution_lastresult.xdmf', "w")
xdmfout.write_mesh(mesh)
xdmfout.write_meshtags(ft, mesh.geometry)  # in Paraview auf Grid(partial) gehen, Skala von 1-10
xdmfout.write_function(XDMF_us, t)
xdmfout.write_function(P_v_v, t)
xdmfout.write_function(S_v_v, t)
xdmfout.write_function(C_v_ls, t)
xdmfout.write_function(cs_ls_s, t)
# xdmfout.write_function(cs_ls_p, t)
xdmfout.write_function(E_v_v, t)
xdmfout.write_function(E_el_v, t)
xdmfout.write_function(e_v_v, t)
xdmfout.write_function(F_ls, t)
xdmfout.write_function(C_ls, t)
xdmfout.write_function(eta_g, t)
xdmfout.write_function(ftau_b, t)
xdmfout.write_function(s_ls, t)
xdmfout.write_function(f_IR, t)
xdmfout.write_function(psi_pos, t)
xdmfout.write_function(psi_ges, t)
xdmfout.write_function(velocity, t)
xdmfout.write_function(psi_pos_vol, t)
xdmfout.write_function(psi_pos_dev, t)
xdmfout.close()

# stopwatch stop
timer.stop()

