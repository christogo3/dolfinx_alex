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

    
    return H,alpha,alpha_tmp

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

def assemble_3D_representation_of_plane_strain_eps(u):
    eps_np1_2D = ufl.sym(ufl.grad(u))
    eps_np1_3D_plane_strain = ufl.as_tensor([[eps_np1_2D[0,0], eps_np1_2D[0,1], 0.0],
                                                    [ eps_np1_2D[1,0], eps_np1_2D[1,1], 0.0],
                                                    [ 0.0,             0.0,             0.0]])
                                            
    return eps_np1_3D_plane_strain
        
    
def update_e_p(u,e_p_n,alpha_n,sig_y,hard,mu):
    e_np1 = ufl.dev(assemble_3D_representation_of_plane_strain_eps(u))
    s_tr = 2.0*mu*(e_np1-e_p_n)
        
    norm_s_tr = ufl.sqrt(ufl.inner(s_tr,s_tr))
        
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    N_np1 = s_tr / norm_s_tr
    eps_p_np1 = ufl.conditional(ufl.le(f_tr,0.0),e_p_n,e_p_n+dgamma*N_np1)
    return eps_p_np1


def update_alpha(u,e_p_n,alpha_n,sig_y,hard,mu):
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    alpha_np1 = ufl.conditional(ufl.le(f_tr,0.0),alpha_n,alpha_n+np.sqrt(2/3)*dgamma)
    return alpha_np1


def sig_plasticity(u,e_p_n,alpha_n,sig_y,hard,lam,mu):  
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
    
    sig_2D = ufl.as_tensor([[sig_3D[0,0], sig_3D[0,1], sig_3D[0,2]],
                            [sig_3D[1,0], sig_3D[1,1], sig_3D[0,2]],
                            [sig_3D[2,0], sig_3D[2,1], sig_3D[2,2]]])
    
    return sig_2D


def update_e_p_n_and_alpha_arrays_tensorial(u,e_p_n,e_p_n_tmp,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu):
    e_p_n_tmp.x.array[:] = e_p_n.x.array[:]
    
    alpha_tmp.x.array[:] = alpha_n.x.array[:]
    alpha_expr = update_alpha(u,e_p_n=e_p_n_tmp,alpha_n=alpha_n,sig_y=sig_y.value,hard=hard.value,mu=mu)
    alpha_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,alpha_expr)
    

    e_p_np1_expr = update_e_p(u,e_p_n=e_p_n_tmp,alpha_n=alpha_tmp,sig_y=sig_y.value,hard=hard.value,mu=mu)
    
    # get function space and e_p_n shape
    V = e_p_n.function_space
    shape = V._ufl_element._shape
    num_rows, num_cols = shape[0], shape[1]


    # iteratively update components
    for i in range(num_rows):
        for j in range(num_cols):

            # map the tensor coordinates to a single integer for use in .sub()
            k = i * num_cols + j
            # Get dofmap for the (i, j) component
            V_sub, map_ij = V.sub(k).collapse()

            e_p_n_expr = e_p_np1_expr[i,j]
            interpolation = interpolate_quadrature(domain,cells,quadrature_points,e_p_n_expr)
            l1 = interpolation.size
            l2 = len(map_ij)
            if l1 < l2:
                interpolation_extended = np.append(interpolation, (np.zeros(l2-l1)))
                # print(f'Maximum in interpolated e_p_n array: {np.max(interpolation_extended)}')
            e_p_n.x.array[map_ij] = interpolation_extended

def update_e_p_n_and_alpha_arrays(u,e_p_11_n_tmp,e_p_22_n_tmp,e_p_12_n_tmp,e_p_33_n_tmp,e_p_13_n_tmp,e_p_23_n_tmp,
                           e_p_11_n,e_p_22_n,e_p_12_n,e_p_33_n,e_p_13_n,e_p_23_n,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu):
    e_p_11_n_tmp.x.array[:] = e_p_11_n.x.array[:]
    e_p_22_n_tmp.x.array[:] = e_p_22_n.x.array[:]
    e_p_12_n_tmp.x.array[:] = e_p_12_n.x.array[:]
    e_p_33_n_tmp.x.array[:] = e_p_33_n.x.array[:]
    e_p_13_n_tmp.x.array[:] = e_p_33_n.x.array[:]
    e_p_23_n_tmp.x.array[:] = e_p_33_n.x.array[:]
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
'''# map the tensor coordinates to a single integer for use in .sub()
            k = i * num_cols + j
            # Get dofmap for the (i, j) component
            V_sub, map_ij = V.sub(k).collapse()
            # Temporary function for subspace values
            temp_func = fem.Function(V_sub)
            
            #Create temp expression and wrap it as dolfinx expression
            temp_ufl_expr = ufl.tensors.as_scalar(e_p_np1_expr[i, j])
            temp_dolfinx_expr = fem.Expression(temp_ufl_expr, domain.geometry.x, V_sub.element.interpolation_points()) # quadrature points = gauss Punkte statt interpolation points an den Rändern
            
            # Interpolate function
            temp_func.interpolate(temp_dolfinx_expr)

            # Assign updated values
            temp_array = temp_func.x.array[:]
            e_p_n.x.array[map_ij] = temp_array'''
    
    
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
        
        
    def prep_newton(self, u: any, um1: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, lam: dlfx.fem.Function, mu: dlfx.fem.Function ):
        def residuum(u: any, du: any,  um1:any):
            
            delta_u = u - um1
            t1 = self.sigma(u,lam,mu)
            t2 = 0.5*(ufl.grad(du) + ufl.grad(du).T)

            equi =  (ufl.inner(t1, t2))*self.dx # ufl.derivative(pot, u, du)
            H_np1 = self.update_H(u,delta_u=delta_u,lam=lam,mu=mu)
            
            Res = equi
            return [ Res, None]        
        return residuum(u,du,um1)
    
    def sigma(self, u,lam,mu):
        return  sig_plasticity(u,e_p_n=self.e_p_n,alpha_n=self.alpha_n,sig_y=self.sig_y,hard=self.hard,lam=lam,mu=mu)
        # return 1.0 * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
    
    def eps(self,u):
        return ufl.sym(ufl.grad(u)) #0.5*(ufl.grad(u) + ufl.grad(u).T)
    
    def deveps(self,u):
        return ufl.dev(self.eps(u))
    
    def eqeps(self,u):
        return ufl.sqrt(2.0/3.0 * ufl.inner(self.eps(u),self.eps(u))) 
    
    def update_H(self, u, delta_u,lam,mu):
        u_n = u-delta_u
        delta_eps = 0.5*(ufl.grad(delta_u) + ufl.grad(delta_u).T)
        W_np1 = ufl.inner(self.sigma(u=u,lam=lam,mu=mu), delta_eps )
        W_n = ufl.inner(self.sigma(u=u_n,lam=lam,mu=mu), delta_eps )
        H_np1 = ( self.H + 0.5 * (W_n+W_np1))
        return H_np1
    
    def psiel(self,u,lam,mu):
        return  self.H
    
    def get_E_el_global(self,u,lam,mu, dx: ufl.Measure, comm: MPI.Intercomm) -> float:
        Pi = dlfx.fem.assemble_scalar(dlfx.fem.form(self.psiel(u,lam,mu) * dx))
        return comm.allreduce(Pi,MPI.SUM)

    
