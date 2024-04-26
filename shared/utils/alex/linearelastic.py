import dolfinx as dlfx
import ufl 
from typing import Callable
import numpy as np
import alex.phasefield

from dolfinx.fem.petsc import assemble_vector

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

def get_nu(lam: float, mu: float):
    return lam/(2*(lam+mu))

def get_emod(lam: float, mu: float):
    return  mu * (3.0 * lam + 2.0 * mu) / (lam + mu)

def get_J_3D(eshelby_as_function: Callable, outer_normal: ufl.FacetNormal):
    Jx = (eshelby_as_function[0,0]*outer_normal[0]+eshelby_as_function[0,1]*outer_normal[1]+eshelby_as_function[0,2]*outer_normal[2])*ufl.ds
    Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form(Jx))
    Jy = (eshelby_as_function[1,0]*outer_normal[0]+eshelby_as_function[1,1]*outer_normal[1]+eshelby_as_function[1,2]*outer_normal[2])*ufl.ds
    Jya = dlfx.fem.assemble_scalar(dlfx.fem.form(Jy))
    Jz = (eshelby_as_function[2,0]*outer_normal[0]+eshelby_as_function[2,1]*outer_normal[1]+eshelby_as_function[2,2]*outer_normal[2])*ufl.ds
    Jza = dlfx.fem.assemble_scalar(dlfx.fem.form(Jz))
    return Jxa, Jya, Jza

def get_J_3D_volume_integral(eshelby_as_function: Callable, dx: ufl.Measure):
    Jx = ufl.div(eshelby_as_function)[0] * dx
    Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form(Jx))
    Jy = ufl.div(eshelby_as_function)[1] * dx
    Jya = dlfx.fem.assemble_scalar(dlfx.fem.form(Jy))
    Jz = ufl.div(eshelby_as_function)[2] * dx
    Jza = dlfx.fem.assemble_scalar(dlfx.fem.form(Jz))
    return Jxa, Jya, Jza

def get_J_3D_volume_integral_tf(eshelby_as_function: Callable, v: ufl.TestFunction, dx: ufl.Measure):
    grad_v = ufl.grad(v)
    Jx = (eshelby_as_function[0,0]*grad_v[0]+eshelby_as_function[0,1]*grad_v[1]+eshelby_as_function[0,2]*grad_v[2])*dx
    Jxa = assemble_vector(dlfx.fem.form(Jx))
    
    Jy = (eshelby_as_function[1,0]*grad_v[0]+eshelby_as_function[1,1]*grad_v[1]+eshelby_as_function[1,2]*grad_v[2])*dx
    Jya = assemble_vector(dlfx.fem.form(Jy))
    
    Jz = (eshelby_as_function[2,0]*grad_v[0]+eshelby_as_function[2,1]*grad_v[1]+eshelby_as_function[2,2]*grad_v[2])*dx
    Jza = assemble_vector(dlfx.fem.form(Jz))
    
    return -Jxa.sum(), -Jya.sum(), -Jza.sum()

def sigma_as_tensor3D(u: float, lam:float, mu:float ):
        eps = ufl.sym(ufl.grad(u))
        val = lam * ufl.tr(eps) * ufl.Identity(3) * 2*mu*eps
        return val