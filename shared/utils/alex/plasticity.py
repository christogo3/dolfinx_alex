from typing import Callable
import ufl
import alex.linearelastic as le
import dolfinx.fem as fem
from petsc4py import PETSc
import basix
import numpy as np
import dolfinx as dlfx
from mpi4py import MPI



pos = lambda x: ufl.max_value(x, 0)

def constitutive_update(Δε, old_sig, alpha_n,sigY,H,lam,mu):
    sig_n = old_sig
    sig_tr = sig_n + le.sigma_as_tensor_from_epsilon(Δε,lam,mu)
    sig_dev_tr = ufl.dev(sig_tr)
    sig_eq = ufl.sqrt(3.0 / 2.0 * ufl.inner(sig_dev_tr, sig_dev_tr))
    f_tr = sig_eq - ( sigY + H * alpha_n )
    dgamma = pos(f_tr) / (3 * mu + H)
    n_elas = sig_dev_tr / sig_eq * pos(f_tr) / f_tr
    beta = 3 * mu * dgamma / sig_eq
    new_sig = sig_tr - beta * sig_dev_tr
    
    f_tr = ufl.sqrt(ufl.inner(sig_dev_tr,sig_dev_tr))- ufl.sqrt(2. / 3.) * ( sigY + H * alpha_n )
    dgamma1 = pos(f_tr) / (2. * mu + 2./3. * H)
    n_elas1 = sig_dev_tr / ufl.sqrt(ufl.inner(sig_dev_tr,sig_dev_tr)) * pos(f_tr) / f_tr
    
    new_sig = sig_tr - 2.0 * mu * dgamma1 * n_elas1  # yields the same results as above!
    # new_sig = sig_tr - beta * sig_dev_tr
    
    return new_sig, n_elas, beta, dgamma


def sigma_tang(eps, lam, mu, N_elas, beta, H):
    # N_elas = as_3D_tensor(n_elas)
    return (
        le.sigma_as_tensor_from_epsilon(eps,lam,mu)
        - 3 * mu * (3 * mu / (3 * mu + H) - beta) * ufl.inner(N_elas, eps) * N_elas
        - 2 * mu * beta * ufl.dev(eps)
    )
    
def get_residual_and_tangent(n : ufl.FacetNormal, loading, sig_np1, u_ : ufl.TestFunction, v : ufl.TrialFunction, eps: Callable, ds : ufl.Measure, dx: ufl.Measure, lam, mu, N_elas, beta, H):
    Residual = ufl.inner(eps(u_), sig_np1) * dx - ufl.inner(
        -loading * n, u_) * ds
    tangent_form = ufl.inner(eps(v), sigma_tang(eps(u_), lam, mu, N_elas, beta, H)) * dx
    return Residual, tangent_form
    
    
#basix.make_quadrature()

def get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad):
    basix_celltype = getattr(basix.CellType, domain.topology.cell_types[0].name) # 7.3
    #basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name) # 8.0
    quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad,rule=basix.quadrature.string_to_type("default"))

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    return quadrature_points,cells

def interpolate_quadrature(domain, cells, quadrature_points, ufl_expr):
    expr_expr = fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(domain, cells)
    return expr_eval.flatten()[:]
    # function.x.array[:] = expr_eval.flatten()[:]

def to_history_field_vector_mapper(dim: int):
    def to_vect_2D(X):
        return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 1]])
    def to_vect_3D(X):
        return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[1, 2], X[0, 2], X[0, 1]])
    if dim == 2:
        return to_vect_2D
    elif dim == 3:
        return to_vect_3D
    
def from_history_field_to_3D_tensor_mapper(dim: int):
    def as_3D_tensor_2D(X):
        return ufl.as_tensor([[X[0], X[3], 0], [X[3], X[1], 0], [0, 0, X[2]]])
    def as_3D_tensor_3D(X):
        return ufl.as_tensor([[X[0], X[5], X[4]], [X[5], X[1], X[3]], [X[4], X[3], X[2]]])
    if dim == 2:
        return as_3D_tensor_2D
    elif dim == 3:
        return as_3D_tensor_3D
    
def get_history_field_dimension_for_symmetric_second_order_tensor(dim: int):
    if dim == 2:
        return 4
    elif dim == 3:
        return 6
    

def eps_as_3D_tensor_function(dim: int):
    def eps_2D(v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_tensor([[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, 0]])
    def eps_3D(v):
        return ufl.sym(ufl.grad(v))
    if dim == 2:
        return eps_2D
    elif dim == 3:
        return eps_3D
    

def define_internal_state_variables_basix(gdim, domain, deg_quad, quad_scheme):  
    W0e = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
)
# We = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(alex.plasticity.get_history_field_dimension_for_symmetric_second_order_tensor(gdim),), scheme="default", degree=deg_quad
# )
    
    W0 = fem.functionspace(domain, W0e)
    
    beta = fem.Function(W0, name="beta")
    
    
    return beta


def define_internal_state_variables_basix_b(gdim, domain, deg_quad, quad_scheme):  
    W0e = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
)
    
    We = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(2,2), scheme="default", degree=deg_quad
)
# We = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(alex.plasticity.get_history_field_dimension_for_symmetric_second_order_tensor(gdim),), scheme="default", degree=deg_quad
# )
    
    W0 = fem.functionspace(domain, W0e)
    alpha = fem.Function(W0, name="alpha")
    alpha_tmp = fem.Function(W0, name="alpha_tmp")
    H = fem.Function(W0, name="H")
    
    #W = fem.functionspace(domain, We)
    e_p_11_n = fem.Function(W0, name="e_p_11")
    e_p_22_n = fem.Function(W0, name="e_p_22")
    e_p_12_n = fem.Function(W0, name="e_p_12")
    e_p_33_n = fem.Function(W0, name="e_p_33")

    e_p_11_n_tmp = fem.Function(W0, name="e_p_11_tmp")
    e_p_22_n_tmp = fem.Function(W0, name="e_p_22_tmp")
    e_p_12_n_tmp = fem.Function(W0, name="e_p_12_tmp")
    e_p_33_n_tmp = fem.Function(W0, name="e_p_33_tmp")
    
    return H,alpha,alpha_tmp, e_p_11_n, e_p_22_n, e_p_12_n, e_p_33_n, e_p_11_n_tmp, e_p_22_n_tmp, e_p_12_n_tmp, e_p_33_n_tmp

def define_internal_state_variables_basix_c(gdim, domain, deg_quad, quad_scheme):  
    W0e = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(2,), scheme="default", degree=deg_quad
)
    
    We = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(2,2), scheme="default", degree=deg_quad
)
# We = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(alex.plasticity.get_history_field_dimension_for_symmetric_second_order_tensor(gdim),), scheme="default", degree=deg_quad
# )
    
    W0 = fem.functionspace(domain, W0e)
    alpha = fem.Function(W0, name="alpha")
    alpha_tmp = fem.Function(W0, name="alpha_tmp")
    H = fem.Function(W0, name="H")
    
    #W = fem.functionspace(domain, We)
    e_p_11_n = fem.Function(W0, name="e_p_11")
    e_p_22_n = fem.Function(W0, name="e_p_11")
    e_p_12_n = fem.Function(W0, name="e_p_11")
    
    e_p_11_n_tmp = fem.Function(W0, name="e_p_11_tmp")
    e_p_22_n_tmp = fem.Function(W0, name="e_p_11_tmp")
    e_p_12_n_tmp = fem.Function(W0, name="e_p_11_tmp")
    
    return H,alpha,alpha_tmp, e_p_11_n, e_p_22_n, e_p_12_n, e_p_11_n_tmp, e_p_22_n_tmp, e_p_12_n_tmp

def define_internal_state_variables_basix_d(gdim, domain, deg_quad, quad_scheme):  
    W0e = basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad)

    # TensorFunctionSpace
    Ve_3d = ufl.TensorElement("Quadrature", domain.ufl_cell(), degree=deg_quad, shape=(3,3), quad_scheme=quad_scheme)
    V_3d = dlfx.fem.FunctionSpace(domain, Ve_3d)
    
    # setup history variables for large plasticity
    W0 = fem.functionspace(domain, W0e)
    alpha = fem.Function(W0, name="alpha")
    alpha_tmp = fem.Function(W0, name="alpha_tmp")
    H = fem.Function(W0, name="H")
    
    b_e_n = fem.Function(V_3d,name='b_e_n')
    b_e_n_tmp = fem.Function(V_3d,name='b_e_n_tmp')
    F_n = fem.Function(V_3d,name='F_n')


    num_points = len(F_n.x.array) // 9
    I_33 = np.eye(3).flatten()
    # Initialize arrays
    H.x.array[:] = np.zeros_like(H.x.array[:])
    alpha.x.array[:] = np.zeros_like(alpha.x.array[:])
    alpha_tmp.x.array[:] = np.zeros_like(alpha_tmp.x.array[:])
    b_e_n.x.array[:] = np.tile(I_33, num_points)
    b_e_n_tmp.x.array[:] = np.tile(I_33, num_points)
    F_n.x.array[:] = np.tile(I_33, num_points)
    
    return H,alpha,alpha_tmp, b_e_n, b_e_n_tmp, F_n

    
def define_internal_state_variables(gdim, domain, deg_quad, quad_scheme):  
    # W0e = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
# )
# We = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(alex.plasticity.get_history_field_dimension_for_symmetric_second_order_tensor(gdim),), scheme="default", degree=deg_quad
# )
    W0e = ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=deg_quad, quad_scheme=quad_scheme)
    We = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=deg_quad,dim=get_history_field_dimension_for_symmetric_second_order_tensor(gdim), quad_scheme="default")
    W0 = fem.functionspace(domain, W0e)
    W = fem.functionspace(domain, We)


    sig_np1 = fem.Function(W, name="stress_at_current_timestep")
    
    N_np1 = fem.Function(W, name="normal_to_yield_surface")
    beta = fem.Function(W0, name="beta")
    dGamma = fem.Function(W0, name="plastic_increment")
    
    # history variables 
    sig_n = fem.Function(W, name="stress_at_last_timestep")
    alpha_n = fem.Function(W0, name="Cumulative_plastic_strain")
    
    return sig_np1,sig_n,N_np1,beta,alpha_n,dGamma

def define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, quad_scheme):
    dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": deg_quad, "quadrature_scheme": quad_scheme},
    )
    
    return dx
    
    
    
def constitutive_update_alt(Δε, e_p_n, alpha_n,sig0,H,lam,mu):
    e_np1 = ufl.dev(Δε +  e_p_n)
    s_tr_np1 = 2.0 * mu * (e_np1 - e_p_n)
    norm_s_tr = ufl.sqrt(ufl.inner(s_tr_np1,s_tr_np1))
    f_tr = norm_s_tr- ufl.sqrt(2. / 3.) * ( sig0 + H * alpha_n )
    
    N_tr = s_tr_np1 / norm_s_tr * pos(f_tr) / f_tr
    dGamma = pos(f_tr) / (3 * mu + H)
    
    s_np1 = s_tr_np1 - 2. * mu * dGamma * N_tr
    
    
    alpha_np1 = alpha_n + ufl.sqrt(2.0 / 3.0) * dGamma
    e_p_np1 = e_p_n + dGamma * N_tr
    
    sigma_np1 = le.get_K(lam,mu) * ufl.Identity(3) * ufl.tr(Δε +  e_p_n) + 2. * mu * s_np1
    
    
    return sigma_np1, N_tr, e_p_np1, alpha_np1, dGamma, s_tr_np1


    
def sigma_tang_alt(eps,N,mu,lam,H, dGamma,s_tr_np1):
    norm_s_tr = ufl.sqrt(ufl.inner(s_tr_np1,s_tr_np1))
    return (
        le.sigma_as_tensor_from_epsilon(eps,lam,mu)
        - (2. * mu) ** 2 / (2. * mu +  H * 2. / 3.) * ufl.inner(N, eps) * N
        - (2. * mu) ** 2 / (norm_s_tr) * dGamma *  (ufl.dev(eps)-ufl.inner(N, eps) * N)
        # 
    )
    
    




class Ramberg_Osgood:
    # Constructor method
    def __init__(self, 
                       dx: any = ufl.dx,
                       Id: any = None,
                       tol: any = None,
                        C: any = None,
                       n: any = None,
                 ):
        
        self.dx = dx
 
        self.sigma_undegraded : Callable = self.sigma_undegraded_vol_deviatoric #.sigma_as_tensor # plane strain
        
    def prep_newton(self, u: any, du: any, lam: dlfx.fem.Function, mu: dlfx.fem.Function):
        def residuum():
            
            equi =  (ufl.inner(self.sigma_undegraded_vol_deviatoric(u,lam,mu),  0.5*(ufl.grad(du) + ufl.grad(du).T)))*self.dx # ufl.derivative(pot, u, du)
            
            Res = equi
            return [ Res, None]
            

        
        return residuum()
    
    def eps(self,u):
        return ufl.sym(ufl.grad(u)) #0.5*(ufl.grad(u) + ufl.grad(u).T)
    
    def deveps(self,u):
        return ufl.dev(self.eps(u))
    
    def eqeps(self,u):
        return ufl.sqrt(2.0/3.0 * ufl.inner(self.eps(u),self.eps(u))) 
    
    
    def sigma_undegraded_vol_deviatoric(self,u,lam,mu):
        
        eps = ufl.sym(ufl.grad(u))
        
        C = 0.001 # self.C
        n = 20.0 #self.n
        eps_e_val = ufl.sqrt(2.0/3.0 * ufl.inner(ufl.dev(eps),ufl.dev(eps))) #+ 0.000001
        eps_e = ufl.conditional(ufl.lt(eps_e_val, 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, eps_e_val)
        
        E_mod = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        HH = ((3.0 * mu.value) / E_mod) * (C ** (1.0 / n))
        expo = (1.0 - (1.0/n))
        Z = (2.0 * mu.value) / ( 1.0 + HH * (eps_e) ** expo )
        K = le.get_K(lam=lam,mu=mu) #lam + mu
        sig = K * ufl.tr(eps)* ufl.Identity(2) + Z * ufl.dev(eps)
        return sig
    
    
    def sig_ramberg_osgood_wiki(u, lam, mu,yield_stress_1d,b_hardening_parameter,r_transition_smoothness_parameter):
        # b comparable to hardening modul
        # r lower -> smoother transition
        
        eps = assemble_3D_representation_of_plane_strain_eps(u)
        eps_dev = ufl.dev(eps)
        
        eps_dev_e_val = ufl.sqrt(2.0/3.0*ufl.inner(eps_dev,eps_dev))
        # prevent zero 
        eps_dev_e = ufl.conditional(ufl.lt(eps_dev_e_val, 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, eps_dev_e_val)
        #norm_eps_crit_dev = 0.5
        #yield_stress_1d = mu*2.0*yield_strain_1d
        #norm_sig_dev_crit = yield_stress_1d*np.sqrt(2.0/3.0) # 
        
        #b_hardening_parameter = 0.1     # Strain hardening parameter
        #r = 10.0 
        
        yield_strain_1d = (yield_stress_1d * 2.0 / 3.0) / (2.0*mu)
        
        
        mu_r = (b_hardening_parameter + (1-b_hardening_parameter) / ((1.0 + ufl.sqrt((eps_dev_e/yield_strain_1d) * (eps_dev_e/yield_strain_1d)) ** r_transition_smoothness_parameter )  ** (1.0/r_transition_smoothness_parameter))) * ( mu )
       
        K = le.get_K(lam=lam,mu=mu)
        sig_3D = K * ufl.tr(eps) * ufl.Identity(3)  + 2.0 * mu_r * eps_dev
        
        sig_2D = ufl.as_tensor([[sig_3D[0,0], sig_3D[0,1]],
                            [sig_3D[1,0], sig_3D[1,1]]])
        
        return sig_2D
    
    
    def sig_ramberg_osgood_wiki_matrix(u, lam, mu,yield_stress_1d,b_hardening_parameter,r_transition_smoothness_parameter):
        # b comparable to hardening modul
        # r lower -> smoother transition
        # C = ufl.as_matrix(
        #     [
        #     [lam + 2*mu, lam,        lam,        0,   0,   0],
        #     [lam,        lam + 2*mu, lam,        0,   0,   0],
        #     [lam,        lam,        lam + 2*mu, 0,   0,   0],
        #     [0,          0,          0,          mu,  0,   0],
        #     [0,          0,          0,          0,   mu,  0],
        #     [0,          0,          0,          0,   0,   mu]
        # ]
        # )
        # S = ufl.inv(C)
        
        
        # Tr = ufl.as_matrix(
        #     [
        #     [1, 1, 1,        0,   0,   0],
        #     [1, 1 , 1,        0,   0,   0],
        #     [1, 1 , 1 , 0,   0,   0],
        #     [0,          0,          0,          0,  0,   0],
        #     [0,          0,          0,          0,   0,  0],
        #     [0,          0,          0,          0,   0,   0]
        # ]
        # )
        
        # I =  Tr = ufl.as_tensor(
        #     [
        #     [1,0,0,0,0,0],
        #     [0,1,0,0,0,0],
        #     [0,0,1,0,0,0],
        #     [0,0,0,1,0,0],
        #     [0,0,0,0,1,0],
        #     [0,0,0,0,0,1]
        # ]
        # )
        
        
        # Matrix C (6x6)
        C = np.array([
        [lam.value + 2*mu.value, lam.value,        lam.value,        0,   0,   0],
        [lam.value,        lam.value + 2*mu.value, lam.value,        0,   0,   0],
        [lam.value,        lam.value,        lam.value + 2*mu.value, 0,   0,   0],
        [0,          0,          0,          mu.value,  0,   0],
        [0,          0,          0,          0,   mu.value,  0],
        [0,          0,          0,          0,   0,   mu.value]
        ],dtype=float)

        # Inverse of C
        S = np.linalg.inv(C)

        # Matrix Tr (6x6)
        Tr = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ],dtype=float)

        # Identity matrix I (6x6)
        I = np.eye(6)
        
        
        
        eps = assemble_3D_representation_of_plane_strain_eps(u)
        eps_dev = ufl.dev(eps)
        
        eps_dev_voigt = ufl.as_vector(
            [eps_dev[0,0], 
             eps_dev[1,1], 
             eps_dev[2,2],   
             eps_dev[1,2],   
             eps_dev[0,2], 
             eps_dev[0,1]],
        )
        
        K_0 = le.get_K(lam=lam.value,mu=mu.value)
        
        # M = (I-K_0*ufl.dot(Tr,S))
        # Minv = ufl.inv(M)
        
        # Matrix multiplication (Tr dot S)
        TS = np.dot(Tr, S)

        # Final matrix M
        M = I - K_0 * TS
        Minv = np.linalg.inv(M)

        
        
        
        eps_dev_e_val = ufl.sqrt(2.0/3.0*ufl.inner(eps_dev,eps_dev))
        # prevent zero 
        eps_dev_e = ufl.conditional(ufl.lt(eps_dev_e_val, 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, eps_dev_e_val)
        #norm_eps_crit_dev = 0.5
        #yield_stress_1d = mu*2.0*yield_strain_1d
        #norm_sig_dev_crit = yield_stress_1d*np.sqrt(2.0/3.0) # 
        
        #b_hardening_parameter = 0.1     # Strain hardening parameter
        #r = 10.0 
        
        yield_strain_1d = (yield_stress_1d * 2.0 / 3.0) / (2.0*mu)
        
        
        mu_r = (b_hardening_parameter + (1-b_hardening_parameter) / ((1.0 + ufl.sqrt((eps_dev_e/yield_strain_1d) * (eps_dev_e/yield_strain_1d)) ** r_transition_smoothness_parameter )  ** (1.0/r_transition_smoothness_parameter))) * ( mu )
       
       
        sig_3D_voigt_0 = 2.0 * mu_r * (
            Minv[0,0] * eps_dev_voigt[0] +
            Minv[0,1] * eps_dev_voigt[1] +
            Minv[0,2] * eps_dev_voigt[2] +
            Minv[0,3] * eps_dev_voigt[3] +
            Minv[0,4] * eps_dev_voigt[4] +
            Minv[0,5] * eps_dev_voigt[5]
)

        sig_3D_voigt_1 = 2.0 * mu_r * (
            Minv[1,0] * eps_dev_voigt[0] +
            Minv[1,1] * eps_dev_voigt[1] +
            Minv[1,2] * eps_dev_voigt[2] +
            Minv[1,3] * eps_dev_voigt[3] +
            Minv[1,4] * eps_dev_voigt[4] +
            Minv[1,5] * eps_dev_voigt[5]
        )

        sig_3D_voigt_2 = 2.0 * mu_r * (
            Minv[2,0] * eps_dev_voigt[0] +
            Minv[2,1] * eps_dev_voigt[1] +
            Minv[2,2] * eps_dev_voigt[2] +
            Minv[2,3] * eps_dev_voigt[3] +
            Minv[2,4] * eps_dev_voigt[4] +
            Minv[2,5] * eps_dev_voigt[5]
        )

        sig_3D_voigt_3 = 2.0 * mu_r * (
            Minv[3,0] * eps_dev_voigt[0] +
            Minv[3,1] * eps_dev_voigt[1] +
            Minv[3,2] * eps_dev_voigt[2] +
            Minv[3,3] * eps_dev_voigt[3] +
            Minv[3,4] * eps_dev_voigt[4] +
            Minv[3,5] * eps_dev_voigt[5]
        )

        sig_3D_voigt_4 = 2.0 * mu_r * (
            Minv[4,0] * eps_dev_voigt[0] +
            Minv[4,1] * eps_dev_voigt[1] +
            Minv[4,2] * eps_dev_voigt[2] +
            Minv[4,3] * eps_dev_voigt[3] +
            Minv[4,4] * eps_dev_voigt[4] +
            Minv[4,5] * eps_dev_voigt[5]
        )

        sig_3D_voigt_5 = 2.0 * mu_r * (
            Minv[5,0] * eps_dev_voigt[0] +
            Minv[5,1] * eps_dev_voigt[1] +
            Minv[5,2] * eps_dev_voigt[2] +
            Minv[5,3] * eps_dev_voigt[3] +
            Minv[5,4] * eps_dev_voigt[4] +
            Minv[5,5] * eps_dev_voigt[5]
        )

        
        sig_3D = np.array([
            [sig_3D_voigt_0, sig_3D_voigt_5, sig_3D_voigt_4],
            [sig_3D_voigt_5, sig_3D_voigt_1, sig_3D_voigt_3],
            [sig_3D_voigt_4, sig_3D_voigt_3, sig_3D_voigt_2]
        ])

         
       
        # K = le.get_K(lam=lam,mu=mu_r)
        # sig_3D = K * ufl.tr(eps) * ufl.Identity(3)  + 2.0 * mu_r * eps_dev
        
        sig_2D = ufl.as_tensor([[sig_3D[0,0], sig_3D[0,1]],
                            [sig_3D[1,0], sig_3D[1,1]]])
        
        return sig_2D

    def sig_ramberg_osgood_diewald(u, lam, mu):
        eps = ufl.sym(ufl.grad(u))
        
        C = 0.001 # self.C
        n = 3.5 #self.n
        eps_e_val = ufl.sqrt(2.0/3.0 * ufl.inner(ufl.dev(eps),ufl.dev(eps))) #+ 0.000001
        eps_e = ufl.conditional(ufl.lt(eps_e_val, 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, eps_e_val)
        
        E_mod = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        HH = ((3.0 * mu.value) / E_mod) * (C ** (1.0 / n))
        expo = (1.0 - (1.0/n))
        Z = (2.0 * mu.value) / ( 1.0 + HH * (eps_e) ** expo )
        K = le.get_K(lam=lam,mu=mu) #lam + mu
        sig = K * ufl.tr(eps)* ufl.Identity(2) + Z * ufl.dev(eps)
        return sig
    
    
def f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu):
        eps_np1_3D_plane_strain = assemble_3D_representation_of_plane_strain_eps(u)
        #e_np1 = ufl.dev(ufl.sym(ufl.grad(u)))
        e_np1 = ufl.dev(eps_np1_3D_plane_strain)
        s_tr = 2.0*mu*(e_np1-e_p_n)
        norm_s_tr = ufl.sqrt(ufl.inner(s_tr,s_tr))
        f_tr = norm_s_tr -np.sqrt(2.0/3.0) * (sig_y+hard*alpha_n)
        return f_tr

def f_tr_plast(u,b_e_n,F_n,alpha_tmp,sig_y,hard,mu):
    I_ten = ufl.Identity(3)
    F_np1 = I_ten + ufl.grad(u)

    f_ = F_np1 * ufl.inv(F_n) ## 9.3.16 relative deformation gradient 
    
    # 2. elastic predictor
    f_stroke = ufl.det(f_) ** (-1 / 3) * f_
    b_e_tr = f_stroke * b_e_n * f_stroke.T
    s_tr = mu * ufl.dev(b_e_tr)

    # 3. check for plastic loading
    f_tr = ufl_norm(s_tr) - np.sqrt(2 / 3) * (hard * alpha_tmp + sig_y)
    return f_tr

def assemble_3D_representation_of_plane_strain_eps(u):
    if u.ufl_shape == (2,):
        eps_np1_2D = ufl.sym(ufl.grad(u))
        eps_np1_3D_plane_strain = ufl.as_tensor([[eps_np1_2D[0,0], eps_np1_2D[0,1], 0.0],
                                                [ eps_np1_2D[1,0], eps_np1_2D[1,1], 0.0],
                                                [ 0.0,             0.0,             0.0]])                                   
        return eps_np1_3D_plane_strain
    
    else: return ufl.sym(ufl.grad(u))

def ufl_norm(tensor):
    return ufl.sqrt(ufl.inner(tensor,tensor)) 
    
def update_e_p(u,e_p_n,alpha_n,sig_y,hard,mu):
    e_np1 = ufl.dev(assemble_3D_representation_of_plane_strain_eps(u))
    s_tr = 2.0*mu*(e_np1-e_p_n)
        
    norm_s_tr = ufl.sqrt(ufl.inner(s_tr,s_tr))
        
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    N_np1 = s_tr / norm_s_tr
    eps_p_np1 = ufl.conditional(ufl.le(f_tr,0.0),e_p_n,e_p_n+dgamma*N_np1)
    return eps_p_np1

def linear_problem(TEN,dx,variable,deg_quad):
    # Integral(sigma_interpolated * v) = Integral(sigma * v)
    u_ten = ufl.TrialFunction(TEN)
    v_ten = ufl.TestFunction(TEN)
    
    a_proj = ufl.inner(u_ten, v_ten) * dx

    L_proj = ufl.inner(variable, v_ten) * dx(metadata={"quadrature_degree": deg_quad})

    problem = dlfx.fem.petsc.LinearProblem(a_proj, L_proj, 
                                           petsc_options={"ksp_type": "preonly", 
                                                          "pc_type": "jacobi"})
    
    return problem

def update_alpha(u,e_p_n,alpha_n,sig_y,hard,mu):
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    alpha_np1 = ufl.conditional(ufl.le(f_tr,0.0),alpha_n,alpha_n+np.sqrt(2/3)*dgamma)
    return alpha_np1

def update_alpha2(u,b_e_n,F_n,alpha_n,sig_y,hard,mu):
    I_ten = ufl.Identity(3)
    F_np1 = I_ten + ufl.grad(u)

    f_ = F_np1 * ufl.inv(F_n) ## 9.3.16 relative deformation gradient 
    f_stroke = ufl.det(f_) ** (-1 / 3) * f_
    b_e_tr = f_stroke * b_e_n * f_stroke.T

    f_tr = f_tr_plast(u,b_e_n,F_n,alpha_n,sig_y,hard,mu)
    
    I_e = (1 / 3) * ufl.tr(b_e_tr)
    mu_stroke = I_e * mu
    dgamma = (f_tr / (2 * mu_stroke)) / (1 + (hard / (3 * mu_stroke)))
    alpha_np1 = ufl.conditional(ufl.le(f_tr,0.0),alpha_n,alpha_n+np.sqrt(2/3)*dgamma)
    return alpha_np1

def update_b_e(u,b_e_n,F_n,alpha_tmp,sig_y,hard,mu):
    I_ten = ufl.Identity(3)
    F_np1 = I_ten + ufl.grad(u)

    f_tr = f_tr_plast(u,b_e_n,F_n,alpha_tmp,sig_y,hard,mu)
    f_ = F_np1 * ufl.inv(F_n)
    f_stroke = ufl.det(f_) ** (-1 / 3) * f_
    b_e_tr = f_stroke * b_e_n * f_stroke.T
    s_tr = mu * ufl.dev(b_e_tr)

    norm_s_tr = ufl.conditional(ufl.lt(ufl_norm(s_tr), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl_norm(s_tr))
    n_tr = s_tr / norm_s_tr

    I_e = (1 / 3) * ufl.tr(b_e_tr)
    mu_stroke = I_e * mu
    delta_gamma = (f_tr / (2 * mu_stroke)) / (1 + (hard / (3 * mu_stroke)))

    s_np1 = ufl.conditional(ufl.le(f_tr,0.0),s_tr,s_tr - 2 * mu_stroke * delta_gamma * n_tr)

    b_e_np1 = ufl.conditional(ufl.le(f_tr,0.0),b_e_n,s_np1 / mu + I_e * I_ten)
    return b_e_np1


def sig_plasticity(u,e_p_n,alpha_n,sig_y,hard,lam,mu,mode='2d'):  
    
    eps_np1 = assemble_3D_representation_of_plane_strain_eps(u)
    e_np1 = ufl.dev(eps_np1)
        
    s_tr = 2.0*mu*(e_np1-e_p_n)
        
    norm_s_tr_val = ufl.sqrt(ufl.inner(s_tr,s_tr))
    norm_s_tr = ufl.conditional(ufl.lt(norm_s_tr_val, 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, norm_s_tr_val)
    
    #norm_s_tr = ufl.sqrt(ufl.inner(s_tr,s_tr))
    
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    
    N_np1 = s_tr / norm_s_tr
    s_np1 = ufl.conditional(ufl.le(f_tr,0.0),s_tr,s_tr - 2.0*mu*dgamma*N_np1)
    K = le.get_K(lam=lam,mu=mu)
    sig_3D = K * ufl.tr(eps_np1)*ufl.Identity(3) + s_np1

    sig_2D = ufl.as_tensor([[sig_3D[0,0], sig_3D[0,1]],
                            [sig_3D[1,0], sig_3D[1,1]]])
    
    if mode == '2d': return sig_2D
    else: return sig_3D

def deviator_tensor(x):
    eye = np.eye(x.size)
    y = x-(1/3)*np.trace(x)*eye
    return y

def piola_kirchhoff_2_plasticity(u,b_e_n,F_n,alpha_tmp,sig_y,hard,lam,mu):
    # Determine deformation gradient
    I_ten = ufl.Identity(3)
    F_np1 = I_ten + ufl.grad(u)

    '''
    ka_ = la_ + (2 / 3) * mu_
    f_ = F_np1 @ np.linalg.inv(F_n)
    '''
    ka_ = lam + (2 / 3) * mu
    f_ = F_np1 * ufl.inv(F_n) ## 9.3.16 relative deformation gradient 
    

    # 2. elastic predictor
    '''
    f_stroke = np.linalg.det(f_) ** (-1 / 3) * f_
    b_e_tr = f_stroke @ b_e_n @ f_stroke.T
    s_tr = mu_ * deviator_tensor(b_e_tr)
    F_inv = np.linalg.inv(F_np1)
    '''
    f_stroke = ufl.det(f_) ** (-1 / 3) * f_
    b_e_tr = f_stroke * b_e_n * f_stroke.T
    s_tr = mu * ufl.dev(b_e_tr)
    F_inv = ufl.inv(F_np1)

    # 3. check for plastic loading
    '''
    f_tr = np.linalg.norm(s_tr) - np.sqrt(2 / 3) * (K * alpha_old + sigma_y)
    '''
    f_tr = f_tr_plast(u,b_e_n,F_n,alpha_tmp,sig_y,hard,mu)
    
    # 4. return mapping algorithm
    '''
    if f_tr <= 0: #elastischer Schritt
        s_np1 = s_tr
        b_e_np1 = b_e_tr
        n_tr = s_np1 / (np.linalg.norm(s_np1) + 1*(10**(-8)))

    else: #plastischer Schritt
        I_e = (1 / 3) * np.linalg.trace(b_e_tr)
        mu_stroke = I_e * mu_
        delta_gamma = (f_tr / (2 * mu_stroke)) / (1 + (K / (3 * mu_stroke)))
        n_tr = s_tr / np.linalg.norm(s_tr) ## 9.2.16 associative-flow rule 
        s_np1 = s_tr - 2 * mu_stroke * delta_gamma * n_tr ## 9.3.28 reordered requirement with delta_gamma>0 

        # update intermediate configuration
        ## total/elastic left Cauchy–Green Tensor b/b_e 
        b_e_np1 = s_np1 / mu_ + I_e * I_ten ## 9.3.33 elastic constitutive equation and 9.2.8
    '''
    I_e = (1 / 3) * ufl.tr(b_e_tr)
    mu_stroke = I_e * mu
    delta_gamma = (f_tr / (2 * mu_stroke)) / (1 + (hard / (3 * mu_stroke)))

    norm_s_tr = ufl.conditional(ufl.lt(ufl_norm(s_tr), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl_norm(s_tr))

    n_tr = s_tr / norm_s_tr
    s_np1 = ufl.conditional(ufl.le(f_tr,0.0),s_tr,s_tr - 2 * mu_stroke * delta_gamma * n_tr)
    # s_n converged stresses
    #b_e_np1 = ufl.conditional(ufl.le(f_tr,0.0),b_e_n,s_np1 / mu + I_e * I_ten) ## 9.3.33 elastic constitutive equation and 9.2.8
    # update in history updates machen!
    ## elastic left Cauchy–Green Tensor b_e 

    # 5. elastic mean stress
    J_ = ufl.det(F_np1)
    p_ = (ka_ / 2) * (J_ ** 2 - 1) / J_
    tau_ten = J_ * p_ * I_ten + s_np1 ## uncoupled deviatoric stress-strain relationship 9.2.6
    S_ten = F_inv*tau_ten*F_inv.T

    return S_ten

def update_history_variables(u,b_e_n,b_e_n_tmp,F_n,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu):
    
    alpha_tmp.x.array[:] = alpha_n.x.array[:]
    alpha_expr = update_alpha2(u,b_e_n,F_n,alpha_n,sig_y,hard,mu)
    alpha_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,alpha_expr)

    b_e_expr = update_b_e(u,b_e_n,F_n,alpha_tmp,sig_y,hard,mu)
    expr = dlfx.fem.Expression(b_e_expr, quadrature_points)
    b_e_n_tmp.x.array[:] = b_e_n.x.array[:]
    b_e_n.x.array[:] = expr.eval(domain, cells).flatten()

    I_ten = ufl.Identity(3)
    F_expr = dlfx.fem.Expression(I_ten + ufl.grad(u), quadrature_points)
    F_n.x.array[:] = F_expr.eval(domain, cells).flatten()


def update_e_p_n_and_alpha_arrays_tensorial(u,e_p_n,e_p_n_tmp,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu):
    e_p_n_tmp.x.array[:] = e_p_n.x.array[:]
    
    alpha_tmp.x.array[:] = alpha_n.x.array[:]
    alpha_expr = update_alpha(u,e_p_n=e_p_n_tmp,alpha_n=alpha_n,sig_y=sig_y.value,hard=hard.value,mu=mu)
    alpha_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,alpha_expr)
    
    e_p_np1_expr = update_e_p(u,e_p_n=e_p_n_tmp,alpha_n=alpha_tmp,sig_y=sig_y.value,hard=hard.value,mu=mu)

    expr = dlfx.fem.Expression(e_p_np1_expr, quadrature_points)
    e_p_n.x.array[:] = expr.eval(domain, cells).flatten()


def update_e_p_n_and_alpha_arrays(u,e_p_11_n_tmp,e_p_22_n_tmp,e_p_12_n_tmp,e_p_33_n_tmp,
                           e_p_11_n,e_p_22_n,e_p_12_n,e_p_33_n,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu):
    e_p_11_n_tmp.x.array[:] = e_p_11_n.x.array[:]
    e_p_22_n_tmp.x.array[:] = e_p_22_n.x.array[:]
    e_p_12_n_tmp.x.array[:] = e_p_12_n.x.array[:]
    e_p_33_n_tmp.x.array[:] = e_p_33_n.x.array[:]
    e_p_n_tmp = ufl.as_tensor([[e_p_11_n_tmp, e_p_12_n_tmp, 0], 
                               [e_p_12_n_tmp, e_p_22_n_tmp, 0],
                               [0,         0,    e_p_33_n_tmp]])
    
    alpha_tmp.x.array[:] = alpha_n.x.array[:]
    alpha_expr = update_alpha(u,e_p_n=e_p_n_tmp,alpha_n=alpha_n,sig_y=sig_y.value,hard=hard.value,mu=mu)
    alpha_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,alpha_expr)
    
    e_p_np1_expr = update_e_p(u,e_p_n=e_p_n_tmp,alpha_n=alpha_tmp,sig_y=sig_y.value,hard=hard.value,mu=mu)
    
    e_p_11_expr = e_p_np1_expr[0,0]
    e_p_11_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_11_expr)
    
    e_p_22_expr = e_p_np1_expr[1,1]
    e_p_22_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_22_expr)
    
    e_p_12_expr = e_p_np1_expr[0,1]
    e_p_12_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_12_expr)

    e_p_33_expr = e_p_np1_expr[2,2]
    e_p_33_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_33_expr)

def update_e_p_n_and_alpha_arrays_3d_component_wise(u,e_p_11_n_tmp,e_p_22_n_tmp,e_p_12_n_tmp,e_p_33_n_tmp,e_p_13_n_tmp,e_p_23_n_tmp,
                           e_p_11_n,e_p_22_n,e_p_12_n,e_p_33_n,e_p_13_n,e_p_23_n,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu):
    e_p_11_n_tmp.x.array[:] = e_p_11_n.x.array[:]
    e_p_22_n_tmp.x.array[:] = e_p_22_n.x.array[:]
    e_p_12_n_tmp.x.array[:] = e_p_12_n.x.array[:]
    e_p_33_n_tmp.x.array[:] = e_p_33_n.x.array[:]
    e_p_13_n_tmp.x.array[:] = e_p_13_n.x.array[:]
    e_p_23_n_tmp.x.array[:] = e_p_23_n.x.array[:]
    e_p_n_tmp = ufl.as_tensor([[e_p_11_n_tmp, e_p_12_n_tmp, e_p_13_n_tmp], 
                               [e_p_12_n_tmp, e_p_22_n_tmp, e_p_23_n_tmp],
                               [e_p_13_n_tmp, e_p_23_n_tmp, e_p_33_n_tmp]])
    
    alpha_tmp.x.array[:] = alpha_n.x.array[:]
    alpha_expr = update_alpha(u,e_p_n=e_p_n_tmp,alpha_n=alpha_n,sig_y=sig_y.value,hard=hard.value,mu=mu)
    alpha_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,alpha_expr)
    
    e_p_np1_expr = update_e_p(u,e_p_n=e_p_n_tmp,alpha_n=alpha_tmp,sig_y=sig_y.value,hard=hard.value,mu=mu)
    
    e_p_11_expr = e_p_np1_expr[0,0]
    e_p_11_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_11_expr)
    
    e_p_22_expr = e_p_np1_expr[1,1]
    e_p_22_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_22_expr)
    
    e_p_12_expr = e_p_np1_expr[0,1]
    e_p_12_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_12_expr)

    e_p_33_expr = e_p_np1_expr[2,2]
    e_p_33_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_33_expr)

    e_p_13_expr = e_p_np1_expr[0,2]
    e_p_13_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_13_expr)

    e_p_23_expr = e_p_np1_expr[1,2]
    e_p_23_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_23_expr)


class Plasticity_incremental_2D:
    # Constructor method
    def __init__(self, 
                       sig_y: any,
                       hard: any,
                       alpha_n: any,
                       e_p_n: any,
                       H: any,
                       dx: any = ufl.dx,
                 ):


        # Set all parameters here! Material etc
        self.dx = dx
        self.sig_y = sig_y
        self.hard = hard
        self.e_p_n = e_p_n
        self.alpha_n = alpha_n
        self.H = H
        
        
    def prep_newton(self, u: any, um1: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, lam: dlfx.fem.Function, mu: dlfx.fem.Function, mode='2d'):
        def residuum(u: any, du: any,  um1:any, mode=mode):
            
            delta_u = u - um1
            t1 = self.sigma(u,lam,mu,mode)
            t2 = 0.5*(ufl.grad(du) + ufl.grad(du).T)

            equi =  (ufl.inner(t1, t2))*self.dx # ufl.derivative(pot, u, du)
            H_np1 = self.update_H(u,delta_u=delta_u,lam=lam,mu=mu,mode=mode)
            
            Res = equi
            return [ Res, None]        
        return residuum(u,du,um1)
    
    def sigma(self, u,lam,mu,mode='2d'):
        return  sig_plasticity(u,e_p_n=self.e_p_n,alpha_n=self.alpha_n,sig_y=self.sig_y,hard=self.hard,lam=lam,mu=mu,mode=mode)
        # return 1.0 * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
    
    def eps(self,u):
        return ufl.sym(ufl.grad(u)) #0.5*(ufl.grad(u) + ufl.grad(u).T)
    
    def deveps(self,u):
        return ufl.dev(self.eps(u))
    
    def eqeps(self,u):
        return ufl.sqrt(2.0/3.0 * ufl.inner(self.eps(u),self.eps(u))) 
    
    def update_H(self, u, delta_u,lam,mu,mode='2d'):
        u_n = u-delta_u
        delta_eps = 0.5*(ufl.grad(delta_u) + ufl.grad(delta_u).T)
        W_np1 = ufl.inner(self.sigma(u=u,lam=lam,mu=mu,mode=mode), delta_eps )
        W_n = ufl.inner(self.sigma(u=u_n,lam=lam,mu=mu,mode=mode), delta_eps )
        H_np1 = ( self.H + 0.5 * (W_n+W_np1))
        return H_np1
    
    def psiel(self,u,lam,mu):
        return  self.H
    
    def get_E_el_global(self,u,lam,mu, dx: ufl.Measure, comm: MPI.Intercomm) -> float:
        Pi = dlfx.fem.assemble_scalar(dlfx.fem.form(self.psiel(u,lam,mu) * dx))
        return comm.allreduce(Pi,MPI.SUM)

class Plasticity_incremental_3D:
    # Constructor method
    def __init__(self, 
                       sig_y: any,
                       hard: any,
                       alpha_n: any,
                       e_p_n: any,
                       H: any,
                       dx: any = ufl.dx,
                 ):


        # Set all parameters here! Material etc
        self.dx = dx
        self.sig_y = sig_y
        self.hard = hard
        self.e_p_n = e_p_n
        self.alpha_n = alpha_n
        self.H = H
        
        
    def prep_newton(self, u: any, um1: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, lam: dlfx.fem.Function, mu: dlfx.fem.Function, mode='3d'):
        def residuum(u: any, du: any,  um1:any, mode=mode):
            
            delta_u = u - um1
            t1 = self.sigma(u,lam,mu,mode) # S statt sigma ausgeben
            t2 = 0.5*(ufl.grad(du) + ufl.grad(du).T) # Variation von Green-Lagrange Verzerrungstensor!

            equi =  (ufl.inner(t1, t2))*self.dx # ufl.derivative(pot, u, du)
            H_np1 = self.update_H(u,delta_u=delta_u,lam=lam,mu=mu,mode=mode)
            
            Res = equi
            return [ Res, None]        
        return residuum(u,du,um1)
    
    def sigma(self, u,lam,mu,mode):
        return  sig_plasticity(u,e_p_n=self.e_p_n,alpha_n=self.alpha_n,sig_y=self.sig_y,hard=self.hard,lam=lam,mu=mu,mode=mode) # neue implementierung aufrufen
        # return 1.0 * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
    
    def eps(self,u):
        return ufl.sym(ufl.grad(u)) #0.5*(ufl.grad(u) + ufl.grad(u).T)
    
    def deveps(self,u):
        return ufl.dev(self.eps(u))
    
    def eqeps(self,u):
        return ufl.sqrt(2.0/3.0 * ufl.inner(self.eps(u),self.eps(u))) 
    
    def update_H(self, u, delta_u,lam,mu,mode='3d'):
        u_n = u-delta_u
        delta_eps = 0.5*(ufl.grad(delta_u) + ufl.grad(delta_u).T)
        W_np1 = ufl.inner(self.sigma(u=u,lam=lam,mu=mu,mode=mode), delta_eps )
        W_n = ufl.inner(self.sigma(u=u_n,lam=lam,mu=mu,mode=mode), delta_eps )
        H_np1 = ( self.H + 0.5 * (W_n+W_np1))
        return H_np1
    
    def psiel(self,u,lam,mu):
        return  self.H
    
    def get_E_el_global(self,u,lam,mu, dx: ufl.Measure, comm: MPI.Intercomm) -> float:
        Pi = dlfx.fem.assemble_scalar(dlfx.fem.form(self.psiel(u,lam,mu) * dx))
        return comm.allreduce(Pi,MPI.SUM)

class Large_deformation_3D:
    # Constructor method
    def __init__(self, 
                       sig_y: any,
                       hard: any,
                       alpha_n: any,
                       alpha_tmp: any,
                       F_n: any,
                       b_e_n: any,
                       H: any,
                       dx: any = ufl.dx,
                 ):


        # Set all parameters here! Material etc
        self.dx = dx
        self.sig_y = sig_y
        self.hard = hard
        self.b_e_n = b_e_n
        self.alpha_n = alpha_n
        self.alpha_tmp = alpha_tmp
        self.F_n = F_n
        self.H = H
        
        
    def prep_newton(self, u: any, um1: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, lam: dlfx.fem.Function, mu: dlfx.fem.Function):
        def residuum(u: any, du: any, ddu:any, um1:any):
            I_ten = ufl.Identity(3)
            F_np1 = I_ten + ufl.grad(u)

            #delta_u = u - um1
            S = self.S(u,lam,mu) # 2nd Piola Kirchhoff
            E_var = 0.5 * (ufl.grad(du).T * F_np1 + F_np1.T * ufl.grad(du)) # Variation von Green-Lagrange Verzerrungstensor
            Res =  ufl.inner(S, E_var)*self.dx
            #H_np1 = self.update_H(u,delta_u=delta_u,lam=lam,mu=mu)

            J = ufl.derivative(Res, u, ddu) # more accurate Jacobian

            return [Res, J]
        return residuum(u,du,ddu,um1)
    
    def S(self,u,lam,mu):
        S = piola_kirchhoff_2_plasticity(u,b_e_n=self.b_e_n,F_n=self.F_n,alpha_tmp=self.alpha_tmp,sig_y=self.sig_y,hard=self.hard,lam=lam,mu=mu)
        return S
    
    def eps(self,u):
        return ufl.sym(ufl.grad(u)) #0.5*(ufl.grad(u) + ufl.grad(u).T)
    
    def deveps(self,u):
        return ufl.dev(self.eps(u))
    
    def eqeps(self,u):
        return ufl.sqrt(2.0/3.0 * ufl.inner(self.eps(u),self.eps(u))) 
    
    def update_H(self, u, delta_u,lam,mu):
        u_n = u-delta_u
        delta_eps = 0.5*(ufl.grad(delta_u) + ufl.grad(delta_u).T)
        W_np1 = ufl.inner(self.S(u=u,lam=lam,mu=mu), delta_eps )
        W_n = ufl.inner(self.S(u=u_n,lam=lam,mu=mu), delta_eps )
        H_np1 = ( self.H + 0.5 * (W_n+W_np1))
        return H_np1
    
    def psiel(self,u,lam,mu):
        return  self.H
    
    def get_E_el_global(self,u,lam,mu, dx: ufl.Measure, comm: MPI.Intercomm) -> float:
        Pi = dlfx.fem.assemble_scalar(dlfx.fem.form(self.psiel(u,lam,mu) * dx))
        return comm.allreduce(Pi,MPI.SUM)