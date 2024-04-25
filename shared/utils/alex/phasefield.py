import dolfinx as dlfx
import ufl
from typing import Callable
import alex.linearelastic as le

def degrad_quadratic(s: any, eta: dlfx.fem.Constant) -> any:
    degrad = s**2+eta
    return degrad


def psisurf(s: any, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant) -> any:
    psisurf = Gc*(((1-s)**2)/(4*epsilon)+epsilon*(ufl.dot(ufl.grad(s), ufl.grad(s))))
    return psisurf


class StaticPhaseFieldProblem3D:
    # Constructor method
    def __init__(self, degradationFunction: Callable,
                       psisurf: Callable
                 ):
        self.degradation_function = degradationFunction
        self.psiel_voigt = le.psiel_voigt
        self.eps_voigt = le.eps_voigt_3D
        self.cmat_funct = le.cmat_voigt_3D
        self.psisurf = psisurf
        
    def prep_newton(self, w: any, wm1: any, dw: ufl.TestFunction, ddw: ufl.TrialFunction, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant, eta: dlfx.fem.Constant, iMob: dlfx.fem.Constant, delta_t: dlfx.fem.Constant):
        def residuum(u: any, s: any, du: any, ds: any, sm1: any):
            pot = (self.degradation_function(s,eta)*self.psiel_voigt(u,self.eps_voigt,self.cmat_funct(lam,mu))+self.psisurf(s,Gc,epsilon))*ufl.dx
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
    
    def sigma_as_tensor(self, u,s,lam,mu, eta):
        return self.degradation_function(s=s,eta=eta) * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
    
    def getEshelby(self, w: any, eta: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
        u, s = ufl.split(w)
        Wen = self.degradation_function(s,eta) * self.psiel_voigt(u,self.eps_voigt,self.cmat_funct(lam=lam,mu=mu)) 
        eshelby = Wen * ufl.Identity(3) - ufl.grad(u).T*self.sigma_as_tensor(u,s,lam,mu, eta)
        return eshelby

        
        
        
    
    