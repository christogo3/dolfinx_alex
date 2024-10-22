import dolfinx as dlfx
import ufl
import alex.util as ut


def psiel(u: dlfx.fem.Function, sigma: any):
    return 0.5*ufl.inner(sigma, ufl.sym(ufl.grad(u)))

def sigma_as_tensor(u: dlfx.fem.Function, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant ):
        eps = ufl.sym(ufl.grad(u))
        val = lam * ufl.tr(eps) * ufl.Identity(ut.get_dimension_of_function(u)) + 2*mu*eps
        return val

def F(u):
    return ufl.Identity(dim=ut.get_dimension_of_function(u)) + ufl.grad(u)

def J(u):
    return ufl.det(F(u))

def C(u):
    return ufl.dot(ufl.transpose(F(u)),F(u))
    

def E(u):
    return 1/2 * (C(u) - ufl.Identity(dim=ut.get_dimension_of_function(u)))

def psi_Saint_Venant(u: dlfx.fem.Function, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
    psiel = 0.5 * lam * ufl.tr(E(u)) * ufl.tr(E(u)) + mu * ufl.tr(ufl.dot(E(u),E(u)))
    return psiel

def psi_Neo_Hooke(u: dlfx.fem.Function, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
    IC = ufl.tr(C(u))
    W = mu / 2.0 * (IC - 2) + lam /  4 * (J(u)*J(u) - 1.0) - lam / 2.0 * ufl.ln(J(u)) - mu*ufl.ln(J(u))
    return W

def S(u: dlfx.fem.Function, psi_fct):
    S = ufl.derivative(psi_fct,E(u))
    return S

def P(u: dlfx.fem.Function,psi_fct):
    return ufl.dot(F(u), S(u,psi_fct))
    

class ElasticProblem:
    # Constructor method
    def __init__(self):
        self.traction = 0.0
        self.psi = psi_Neo_Hooke
        return 
        
    def prep_newton(self, u: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant, 
                    rho0: dlfx.fem.Constant = 0.0, 
                    accel: dlfx.fem.Function = None, 
                    dx: ufl.Measure = ufl.dx):
        def residuum(u: any, du: ufl.TestFunction, ddu: ufl.TrialFunction):
            pot = self.psi(u,lam=lam,mu=mu)*dx - self.traction
            equi = ufl.derivative(pot, u, du)
            if rho0 != 0.0 and accel is not None:
                inertia = rho0 * ufl.inner(accel, du) * dx
                Res = inertia + equi
            else:
                Res = equi
            dResdw = ufl.derivative(Res, u, ddu)
            return [ Res, dResdw]        
        return residuum(u,du,ddu)
    
    def set_traction_bc(self, P0: any, u: dlfx.fem.Function, N: ufl.FacetNormal, ds: ufl.Measure = ufl.ds):
        # t0 -> stress tensor
        self.traction = ufl.inner(ufl.dot(P0,N),u)*ds # TODO use test function here? no since derivative?
        
