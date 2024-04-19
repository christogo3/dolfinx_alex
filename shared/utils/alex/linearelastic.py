import dolfinx as dlfx
import ufl 
from typing import Callable

def cmat_voigt_3D(lam: dlfx.fem.Constant, mu: dlfx.fem.Constant) -> any:
    return ufl.as_matrix([[lam+2*mu, lam, lam, 0.0, 0.0, 0.0],
                      [lam, lam+2*mu,     lam, 0.0, 0.0, 0.0],
                      [lam, lam, lam+2*mu, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,      mu,  0.0, 0.0],
                      [0.0, 0.0, 0.0,      0.0,  mu, 0.0],
                      [0.0, 0.0, 0.0,      0.0,  0.0, mu],
                      ])

# compute strain in Voigt notation
def eps_voigt_3D(u: any) -> any:
    eps = ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2), u[1].dx(2)+u[2].dx(1), u[2].dx(0)+u[0].dx(2), u[0].dx(1)+u[1].dx(0)])
    return eps


# compute strain energy in Voigt noation
def psiel_voigt(u: any, eps_funct: Callable, cmat: any) -> any:
    psiel = 0.5*ufl.dot(eps_funct(u), cmat*eps_funct(u))
    return psiel