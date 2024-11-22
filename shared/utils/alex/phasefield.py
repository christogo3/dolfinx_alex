import dolfinx as dlfx
import ufl
from typing import Callable
import alex.linearelastic as le
import numpy as np
import alex.tensor as tensor
from mpi4py import MPI
import math
import alex.util as ut

def degrad_quadratic(s: any, eta: dlfx.fem.Constant) -> any:
    degrad = s**2+eta
    return degrad

def degrad_cubic(s: any, eta: dlfx.fem.Constant, beta=0.01) -> any:
    degrad = beta * ((s ** 2) * s - (s ** 2)) + 3.0*(s**2) - 2.0*(s**2)*s + eta
    return degrad

def sig_c_quadr_deg(Gc, mu, epsilon):
    return 9.0/16.0 * math.sqrt(Gc*2.0*mu/(6.0*epsilon))

def sig_c_cubic_deg(Gc, mu, epsilon):
    return 81.0/50.0 * math.sqrt(Gc*2.0*mu/(15.0*epsilon))


def get_Gc_for_given_sig_c_quadr(sig_c, mu, epsilon):
    return (256.0 * epsilon / (27.0 * mu)) * sig_c**2


def psisurf(s: dlfx.fem.Function, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant) -> any:
    psisurf = Gc.value*(((1-s)**2)/(4*epsilon.value)+epsilon.value*(ufl.dot(ufl.grad(s), ufl.grad(s))))
    return psisurf

def surf(s: dlfx.fem.Function, epsilon: dlfx.fem.Constant) -> any:
    surf = (((1-s)**2)/(4*epsilon.value)+epsilon.value*(ufl.dot(ufl.grad(s), ufl.grad(s))))
    return surf

def get_surf_area(s: dlfx.fem.Function, epsilon: dlfx.fem.Constant, dx: ufl.Measure, comm: MPI.Intercomm) -> float:
    A = dlfx.fem.assemble_scalar(dlfx.fem.form(surf(s,epsilon) * dx))
    return comm.allreduce(A,MPI.SUM)

def psisurf_from_function(s: dlfx.fem.Function, Gc: dlfx.fem.Function, epsilon: dlfx.fem.Constant) -> any:
    '''
        Gc from function
    '''
    psisurf = Gc*(((1-s)**2)/(4*epsilon.value)+epsilon.value*(ufl.dot(ufl.grad(s), ufl.grad(s))))
    return psisurf


class StaticPhaseFieldProblem3D:
    # Constructor method
    def __init__(self, degradationFunction: Callable,
                       psisurf: Callable
                 ):
        self.degradation_function = degradationFunction
        # self.psi_el_undegraded = le.psiel
        # self.sigma_undegraded = le.sigma_as_tensor3D
        # self.eps_voigt = le.eps_voigt_3D
        # self.cmat_funct = le.cmat_voigt_3D
        
        # Set all parameters here! Material etc
        self.psisurf = psisurf
        
    def prep_newton(self, w: any, wm1: any, dw: ufl.TestFunction, ddw: ufl.TrialFunction, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant, eta: dlfx.fem.Constant, iMob: dlfx.fem.Constant, delta_t: dlfx.fem.Constant):
        def residuum(u: any, s: any, du: any, ds: any, sm1: any):
            pot = (self.psiel_degraded(s,eta,u,lam.value,mu.value)+self.psisurf(s,Gc,epsilon))*ufl.dx
            equi = ufl.derivative(pot, u, du)
            sdrive = ufl.derivative(pot, s, ds)
            rate = (s-sm1)/delta_t*ds*ufl.dx
            Res = iMob*rate+sdrive+equi
            dResdw = ufl.derivative(Res, w, ddw)
            return [ Res, dResdw]
            
        u, s = ufl.split(w)
        um1, sm1 = ufl.split(wm1)
        du, ds = ufl.split(dw)
        
        return residuum(u,s,du,ds,sm1)
    
    def sigma_degraded(self, u,s,lam: dlfx.fem.Constant,mu: dlfx.fem.Constant, eta):
        return self.degradation_function(s=s,eta=eta) * le.sigma_as_tensor(u=u,lam=lam,mu=mu)
        # return 1.0 * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
        
    def psiel_degraded(self,s,eta,u,lam,mu):
        return self.degradation_function(s,eta) * le.psiel(u,le.sigma_as_tensor(u=u,lam=lam,mu=mu))
    
    def getEshelby(self, w: any, eta: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
        u, s = ufl.split(w)
        eshelby = self.psiel_degraded(s,eta,u,lam.value,mu.value) * ufl.Identity(3) - ufl.dot(ufl.grad(u).T,self.sigma_degraded(u,s,lam.value,mu.value, eta))
        return ufl.as_tensor(eshelby)
    
    
    
def getCohesiveConfStress(s: any, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant):
    return psisurf(s,Gc,epsilon) * ufl.Identity(3) - 2.0 * Gc.value * epsilon.value * ufl.outer(ufl.grad(s), ufl.grad(s))
    
def get_G_ad_3D_volume_integral(cohesiveConfStress: any , dx: ufl.Measure, comm: MPI.Intracomm):
    return tensor.get_volume_integral_of_div_of_tensors_3D(cohesiveConfStress,dx,comm)
    # G_ad_x = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(cohesiveConfStress)[0] ) * dx ) ))
    # G_ad_y = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(cohesiveConfStress)[1] ) * dx ) ))
    # G_ad_z = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(cohesiveConfStress)[2] ) * dx )))
    # return G_ad_x, G_ad_y, G_ad_z

def getDissipativeConfForce(s: any, sm1: any, Mob: dlfx.fem.Constant, dt: float):
    return (s -sm1) / dt * (1/Mob.value) * ufl.grad(s)

def getDissipativeConfForce_volume_integral(dissipativeConfForce: any, dx: ufl.Measure,  comm: MPI.Intracomm):
    G_dis_x = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( dissipativeConfForce[0] ) * dx ) ))
    G_dis_y = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( dissipativeConfForce[1] ) * dx ) ))
    G_dis_z = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( dissipativeConfForce[2] ) * dx )))
    return tensor.assemble_global_sum_dimX1([G_dis_x, G_dis_y, G_dis_z],comm)
     

def get_dynamic_crack_locator_function(wm1: dlfx.fem.Function, s_zero: dlfx.fem.Function):
    def newcrack(x):
        lock_tol = 0.0
        s_zero.x.scatter_forward()
        # wm1.x.scatter_forward()
        # u, s = wm1.split()
        
        # val = np.isclose(s.collapse().x.array[0:], lock_tol, atol=0.005) # works only on one process
        val = np.isclose(s_zero.x.array[0:], lock_tol, atol=0.005)
        return val
    return newcrack
    
         
def irreversibility_bc(domain: dlfx.mesh.Mesh, W: dlfx.fem.FunctionSpace, wm1: dlfx.fem.Function) -> dlfx.fem.DirichletBC:
    def all(x):
        return np.full_like(x[0],True)
    
    wm1.x.scatter_forward()
    # dofmap : dlfx.cpp.common.IndexMap = W.dofmap.index_map
    
    # all_entities = dlfx.mesh.locate_entities(domain,domain.topology.dim-1,all)
    # all_dofs_s_local = dlfx.fem.locate_dofs_topological(W.sub(1),domain.topology.dim-1,all_entities)
    all_entities = dlfx.mesh.locate_entities(domain,0,all)
    all_dofs_s_local = dlfx.fem.locate_dofs_topological(W.sub(1),0,all_entities)
    
    # all_dofs_s_global = np.array(dofmap.local_to_global(all_dofs_s_local),dtype=np.int32)
    
    array_s = wm1.x.array[all_dofs_s_local]
    indices_where_zero_in_array_s = np.where(np.isclose(array_s,0.0,atol=0.001))
    dofs_s_zero = all_dofs_s_local[indices_where_zero_in_array_s]
    
    # array_s_zero=wm1.x.array[dofs_s_zero]
    crackdofs_update = dofs_s_zero #np.array(dofmap.local_to_global(dofs_s_zero),dtype=np.int32) #dofs_s_zero
    bccrack_update = dlfx.fem.dirichletbc(dlfx.default_scalar_type(0.0), crackdofs_update, W.sub(1))
    return bccrack_update


class StaticPhaseFieldProblem2D:
    # Constructor method
    def __init__(self, degradationFunction: Callable,
                       psisurf: Callable, 
                       split = False,
                    #    domain
                 ):
        self.degradation_function = degradationFunction

        # Set all parameters here! Material etc
        self.psisurf : Callable = psisurf
        self.sigma_undegraded : Callable = le.sigma_as_tensor # plane strain
        self.split = split
        # self.z = dlfx.fem.Constant(domain,0.0)
        # self.Id = ufl.as_matrix([[1,self.z],
        #             [self.z,1]])

        
    def prep_newton(self, w: any, wm1: any, dw: ufl.TestFunction, ddw: ufl.TrialFunction, lam: dlfx.fem.Function, mu: dlfx.fem.Function, Gc: dlfx.fem.Function, epsilon: dlfx.fem.Constant, eta: dlfx.fem.Constant, iMob: dlfx.fem.Constant, delta_t: dlfx.fem.Constant, H: dlfx.fem.Function = None):
        def residuum(u: any, s: any, du: any, ds: any, sm1: any):
            pot = (self.psiel_degraded(s,eta,u,lam,mu)+self.psisurf(s,Gc,epsilon))*ufl.dx
            equi = ufl.derivative(pot, u, du)
            if H is not None: # Irreversibility
                potH = (self.psiel_degraded_history_field(s,eta,u,lam,mu,H) + self.psisurf(s,Gc,epsilon))*ufl.dx
                sdrive = ufl.derivative(pot,s,ds)
            else:
                sdrive = ufl.derivative(pot, s, ds)
            rate = (s-sm1)/delta_t*ds*ufl.dx
            Res = iMob*rate+sdrive+equi
            dResdw = ufl.derivative(Res, w, ddw)
            return [ Res, dResdw]
            
        u, s = ufl.split(w)
        um1, sm1 = ufl.split(wm1)
        du, ds = ufl.split(dw)
        
        return residuum(u,s,du,ds,sm1)
    
    def sigma_as_tensor_test(self, u: dlfx.fem.Function, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant ):
        eps = ufl.sym(ufl.grad(u))
        # other usefull functions#
        
        
        # I = self.Id
        val = lam * ufl.Identity(2) + 2*mu*eps
        return val
    
    def sigma_degraded(self, u,s,lam,mu, eta):
        #return self.degradation_function(s=s,eta=eta) * self.sigma_as_tensor_test(u,lam,mu)
        if self.split:
            return self.sigma_degraded_vol_split(u,s,lam,mu,eta)
        else:
           return self.degradation_function(s=s,eta=eta) * self.sigma_as_tensor_test(u,lam,mu) 
        # return self.degradation_function(s=s,eta=eta) * self.sigma_undegraded(u=u,lam=lam,mu=mu)
    
    def sigma_degraded_vol_split(self,u,s,lam,mu,eta):
        def pos(x):
            return 0.5 * (x + abs(x))


        def neg(x):
            return 0.5 * (x - abs(x))
        
        K = le.get_K_2D(lam=lam,mu=mu)
        eps = le.eps_as_tensor(u)
        epsD =  ufl.dev(eps)
        trEps = ufl.tr(eps)
        # dim = ut.get_dimension_of_function(u)
        
        # vol_eps = ufl.tr(eps)
        #sigma_degraded = K * ufl.conditional(ufl.lt(trEps,0.0),trEps,ufl.as_ufl(0.0)) * ufl.Identity(dim) #+ self.degradation_function(s=s,eta=eta) * ( ufl.conditional(ufl.ge(trEps,0.0),trEps,0.0) * ufl.Identity(dim) + 2.0 * mu * epsD)
        #sigma_degraded = K * ufl.conditional(ufl.lt(trEps,0.0),trEps,0.0) * self.Id + self.degradation_function(s=s,eta=eta) * ( ufl.conditional(ufl.ge(trEps,0.0),trEps,0.0) * self.Id + 2.0 * mu * epsD)
        sigma_degraded = K * neg(trEps) * ufl.Identity(2)+ self.degradation_function(s=s,eta=eta) * (K * pos(trEps) * ufl.Identity(2) + 2.0 * mu * epsD)
        return sigma_degraded
        
    def psiel_degraded(self,s,eta,u,lam,mu):
        return 0.5 * ufl.inner(self.sigma_degraded(u,s,lam,mu,eta),le.eps_as_tensor(u))
        #return self.degradation_function(s,eta) * le.psiel(u,self.sigma_undegraded(u=u,lam=lam,mu=mu))
    
    def psiel_degraded_history_field(self,s,eta,u,lam,mu,H):
        psiel = le.psiel(u,self.sigma_undegraded(u=u,lam=lam,mu=mu))
        return self.degradation_function(s,eta) * ufl.conditional(ufl.ge(psiel,H),true_value=psiel,false_value=H)
    
    def getEshelby(self, w: any, eta: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
        u, s = ufl.split(w)
        # Wen = self.degradation_function(s,eta) * self.psiel_undegraded(u,self.eps_voigt,self.cmat_funct(lam=lam,mu=mu)) 
        eshelby = self.psiel_degraded(s,eta,u,lam,mu) * ufl.Identity(2) - ufl.dot(ufl.grad(u).T,self.sigma_degraded(u,s,lam,mu, eta))
        return ufl.as_tensor(eshelby)
    
    
    def getGlobalFractureSurface(s: dlfx.fem.Function, Gc: dlfx.fem.Function, epsilon: dlfx.fem.Constant, dx: ufl.Measure):
        S = dlfx.fem.assemble_scalar(dlfx.fem.form(psisurf_from_function(s,Gc,epsilon)))
        return S
    
class StaticPhaseFieldProblem3D_CZM:
    # Constructor method
    def __init__(self):
        self.degradation_function = StaticPhaseFieldProblem3D_CZM.degradation_function_czm
        # self.psi_el_undegraded = le.psiel
        # self.sigma_undegraded = le.sigma_as_tensor3D
        # self.eps_voigt = le.eps_voigt_3D
        # self.cmat_funct = le.cmat_voigt_3D
        
        # Set all parameters here! Material etc
        self.psisurf = StaticPhaseFieldProblem3D_CZM.psisurf
        
    def degradation_function_czm(s: any, eta: dlfx.fem.Constant, epsilon: dlfx.fem.Constant, l_ch: dlfx.fem.Constant):
        alpha = 1.0 - s
        g = (1.0-alpha) ** 2 / ( (1.0-alpha) ** 2 + (4.0 * l_ch)/(math.pi * epsilon ) * alpha * (1.0 - alpha / 2.0) )
        return g
    
    def psisurf(s: dlfx.fem.Function, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant) -> any:
        alpha = 1.0 - s
        c_w = math.pi
        w = 2.0 * alpha - alpha ** 2
        psisurf = Gc/c_w*(w/epsilon+epsilon*(ufl.dot(ufl.grad(alpha), ufl.grad(alpha))))
        return psisurf
        
        
    def prep_newton(self, w: any, wm1: any, dw: ufl.TestFunction, ddw: ufl.TrialFunction, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant, eta: dlfx.fem.Constant, iMob: dlfx.fem.Constant, delta_t: dlfx.fem.Constant, l_ch: dlfx.fem.Constant):
        def residuum(u: any, s: any, lam_1, lam_2, du: any, ds: any, dlam_1, dlam_2, sm1: any):
            alpha = 1.0 - s
            alpha_m1 = 1.0 - sm1
            pot = (self.psiel_degraded(s,eta,u,lam.value,mu.value, epsilon.value, l_ch.value)+self.psisurf(s,Gc,epsilon) + lam_1 * (alpha - alpha_m1) + lam_2 * (alpha - 1.0))*ufl.dx
            equi = ufl.derivative(pot, u, du)
            sdrive = ufl.derivative(pot, s, ds)
            constraint = ufl.derivative(pot, lam_2, dlam_2) + ufl.derivative(pot, lam_1, dlam_1)
            rate = (s-sm1)/delta_t*ds*ufl.dx
            Res = iMob*rate+sdrive+equi+constraint
            dResdw = ufl.derivative(Res, w, ddw)
            return [ Res, dResdw]
            
        u, s, lam_1, lam_2 = ufl.split(w)
        um1, sm1, _, _ = ufl.split(wm1)
        du, ds, d_lam1, d_lam2 = ufl.split(dw)
        
        return residuum(u,s,lam_1, lam_2, du,ds,d_lam1, d_lam2, sm1)
    
    def sigma_degraded(self, u,s,lam: dlfx.fem.Constant,mu: dlfx.fem.Constant, eta):
        return self.degradation_function(s=s,eta=eta) * le.sigma_as_tensor(u=u,lam=lam,mu=mu)
        # return 1.0 * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
        
    def psiel_degraded(self,s,eta,u,lam,mu,epsilon, l_ch):
        return self.degradation_function(s,eta, epsilon, l_ch) * le.psiel(u,le.sigma_as_tensor(u=u,lam=lam,mu=mu))
    
    # def getEshelby(self, w: any, eta: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
    #     u, s = ufl.split(w)
    #     # Wen = self.degradation_function(s,eta) * self.psiel_undegraded(u,self.eps_voigt,self.cmat_funct(lam=lam,mu=mu)) 
    #     eshelby = self.psiel_degraded(s,eta,u,lam.value,mu.value) * ufl.Identity(3) - ufl.dot(ufl.grad(u).T,self.sigma_degraded(u,s,lam.value,mu.value, eta))
    #     return ufl.as_tensor(eshelby)
    
    
    
    
    

        
        
        
    
    