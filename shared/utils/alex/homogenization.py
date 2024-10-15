import ufl
import alex.util
import dolfinx as dlfx
import numpy as np
import alex.linearelastic as le

def compute_averaged_sigma(u,lam,mu, vol, dx: ufl.Measure = ufl.dx):
    if alex.util.get_dimension_of_function(u) == 3:
        dim_voigt = 6
    else:
        dim_voigt = 3
    sigma_for_unit_strain = np.zeros((dim_voigt,))
    for k in range(0,len(sigma_for_unit_strain)):
        sigma_for_unit_strain[k] = dlfx.fem.assemble_scalar(dlfx.fem.form(le.sigma_as_voigt(u,lam,mu)[k] * dx)) / vol
    return sigma_for_unit_strain


def unit_macro_strain_tensor_for_voigt_eps(domain: dlfx.mesh.Mesh, voigt_index: int):
    if domain.topology.dim == 3:
        Eps_Voigt = np.zeros((6,))
        Eps_Voigt[voigt_index] = 1.0
        return dlfx.fem.Constant(domain,np.array([[Eps_Voigt[0], Eps_Voigt[5]/2.0, Eps_Voigt[4]/2.0],
                                              [Eps_Voigt[5]/2.0, Eps_Voigt[1], Eps_Voigt[3]/2.0],
                                              [Eps_Voigt[4]/2.0, Eps_Voigt[3]/2.0, Eps_Voigt[2]]]))
    elif domain.topology.dim == 2:
        Eps_Voigt = np.zeros((3,))
        Eps_Voigt[voigt_index] = 1.0
        return dlfx.fem.Constant(domain,np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.0],
                                              [Eps_Voigt[2]/2.0, Eps_Voigt[1]]]))
        
    
def lam_hom(cmat):
    if len(cmat[0])==6:
        data = np.array([cmat[0,1], cmat[0,2] , cmat[1,0] , cmat[2,0] , cmat[1,2] , cmat[2,1]]) 
    elif len(cmat[0])==3:
        data = np.array([cmat[0,1] , cmat[1,0]])
    
    lam_hom = np.mean(data)
    lam_hom_std_dev = np.std(data, ddof=1)
    return lam_hom, lam_hom_std_dev

def mu_hom(cmat):
    if len(cmat[0])==6:
        data = np.array([cmat[3,3] , cmat[4,4] , cmat[5,5]])
    elif len(cmat[0])==3:
        data = np.array([cmat[2,2]])
    mu_hom = np.mean(data)
    mu_hom_std_dev = np.std(data, ddof=1)
    return mu_hom, mu_hom_std_dev

def E_hom(cmat):
    lam_h, _ = lam_hom(cmat)
    mu_h, _  = mu_hom(cmat)
    return le.get_emod(lam_h,mu_h)

def nu_hom(cmat):
    lam_h, _ = lam_hom(cmat)
    mu_h, _ = mu_hom(cmat)
    return le.get_nu(lam_h,mu_h)

def average_of_values_that_should_be_zero_isotropic_hom(cmat):
    if len(cmat[0])==6:
        data = np.array([ [0.0, 0.0, 0.0, cmat[0,3], cmat[0,4], cmat[0,5]],
                      [0.0, 0.0, 0.0, cmat[1,3], cmat[1,4], cmat[1,5]],
                      [0.0, 0.0, 0.0, cmat[2,3], cmat[2,4], cmat[2,5]],
                      [cmat[3,0], cmat[3,1], cmat[3,2], 0.0, cmat[3,4], cmat[3,5]],
                      [cmat[4,0], cmat[4,1], cmat[4,2], cmat[4,3], 0.0, cmat[4,5]],
                      [cmat[5,0], cmat[5,1], cmat[5,2], cmat[5,3], cmat[5,4], 0.0],
                      ])
    elif len(cmat[0])==3:
        data = np.array([[0.0, 0.0,  cmat[0,2] ],
                         [0.0, 0.0,  cmat[1,2] ],
                         [cmat[2,0], cmat[2,1], 0.0],
                      ])
    values_that_should_be_zero_average = np.mean(data)
    values_that_should_be_zero_std = np.std(data, ddof=1)
    return values_that_should_be_zero_average, values_that_should_be_zero_std


def print_results(cmat):
    lam_h, lam_h_std_dev = lam_hom(cmat)
    mu_h, mu_h_std_dev = mu_hom(cmat)
    E_h = E_hom(cmat)
    nu_h = nu_hom(cmat)
    zero_avg, zero_std = average_of_values_that_should_be_zero_isotropic_hom(cmat)
    
    result_string = (
        f"Lam Hom: {lam_h:.4f}, Standard Deviation: {lam_h_std_dev:.4f}\n"
        f"Mu Hom: {mu_h:.4f}, Standard Deviation: {mu_h_std_dev:.4f}\n"
        f"E Hom: {E_h:.4f}\n"
        f"Nu Hom: {nu_h:.4f}\n"
        f"Average of Values That Should Be Zero: {zero_avg:.4f}, "
        f"Standard Deviation: {zero_std:.4f}"
    )
    
    return result_string

