import dolfinx as dlfx
import ufl
from typing import Callable
import alex.linearelastic as le
import numpy as np

def degrad_quadratic(s: any, eta: dlfx.fem.Constant) -> any:
    degrad = s**2+eta
    return degrad


def psisurf(s: any, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant) -> any:
    psisurf = Gc.value*(((1-s)**2)/(4*epsilon.value)+epsilon.value*(ufl.dot(ufl.grad(s), ufl.grad(s))))
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
    
    def sigma_degraded(self, u,s,lam,mu, eta):
        return self.degradation_function(s=s,eta=eta) * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
        # return 1.0 * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
        
    def psiel_degraded(self,s,eta,u,lam,mu):
        return self.degradation_function(s,eta) * le.psiel(u,le.sigma_as_tensor3D(u=u,lam=lam,mu=mu))
    
    def getEshelby(self, w: any, eta: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
        u, s = ufl.split(w)
        # Wen = self.degradation_function(s,eta) * self.psiel_undegraded(u,self.eps_voigt,self.cmat_funct(lam=lam,mu=mu)) 
        eshelby = self.psiel_degraded(s,eta,u,lam.value,mu.value) * ufl.Identity(3) - ufl.dot(ufl.grad(u).T,self.sigma_degraded(u,s,lam.value,mu.value, eta))
        return ufl.as_tensor(eshelby)
    
    # def getEshelby( u: any, s: any, eta: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
    #     return deg(s) * psiel(u) * ufl.Identity(3) - ufl.dot(ufl.grad(u).T, deg(s) * sigma_wo_deg(u))
        
    # def deg(s,eta):
    #     return degrad_quadratic(s,eta)
    
    # def sigma_wo_deg(u):
    #         return lam.value * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3) + 2.0 * mu.value * ufl.sym(ufl.grad(u)) 
        
    # def psiel(u):
    #     return 0.5*ufl.inner(sigma_wo_deg(u), ufl.sym(ufl.grad(u)))
    #     # u,s = ufl.split(w)
    
    
def getCohesiveConfStress(s: any, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant):
    return psisurf(s,Gc,epsilon) * ufl.Identity(3) - 2.0 * Gc.value * epsilon.value * ufl.outer(ufl.grad(s), ufl.grad(s))
    
def get_G_ad_3D_volume_integral(cohesiveConfStress: any , dx: ufl.Measure):
    G_ad_x = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(cohesiveConfStress)[0] ) * dx ) ))
    G_ad_y = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(cohesiveConfStress)[1] ) * dx ) ))
    G_ad_z = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(cohesiveConfStress)[2] ) * dx )))
    return G_ad_x, G_ad_y, G_ad_z

def getDissipativeConfForce(s: any, sm1: any, Mob: dlfx.fem.Constant, dt: float):
    return (s -sm1) / dt * (1/Mob.value) * ufl.grad(s)

def getDissipativeConfForce_volume_integral(dissipativeConfForce: any, dx: ufl.Measure):
    G_dis_x = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( dissipativeConfForce[0] ) * dx ) ))
    G_dis_y = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( dissipativeConfForce[1] ) * dx ) ))
    G_dis_z = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( dissipativeConfForce[2] ) * dx )))
    return G_dis_x, G_dis_y, G_dis_z
     

def get_dynamic_crack_locator_function(wm1: dlfx.fem.Function):
    def newcrack(x):
        lock_tol = 0.0
        u, s = wm1.split()
        val = np.isclose(s.collapse().x.array[0:], lock_tol, atol=0.005)
        return val
    return newcrack
    
         
def irreversibility_bc(domain: dlfx.mesh.Mesh, W: dlfx.fem.FunctionSpace, wm1: dlfx.fem.Function) -> dlfx.fem.DirichletBC:
    crackfacets_update = dlfx.mesh.locate_entities(domain,domain.topology.dim-1, get_dynamic_crack_locator_function(wm1))
    crackdofs_update = dlfx.fem.locate_dofs_topological(W.sub(1),domain.topology.dim-1,crackfacets_update)
    bccrack_update = dlfx.fem.dirichletbc(dlfx.default_scalar_type(0.0), crackdofs_update, W.sub(1))
    return bccrack_update
    
    
    
    
    

        
        
        
    
    