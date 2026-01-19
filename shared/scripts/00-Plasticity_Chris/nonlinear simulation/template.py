import numpy as np
#import sympy as sp
#from FElix.FElix_misc.FElix_mat_utils import *
#from FElix.FElix_misc.FElix_symbolic_mat_utils import *
#from scipy.optimize import fsolve


def deviator_tensor(x):
    eye = np.eye(x.size)
    y = x-(1/3)*np.trace(x)*eye
    return y

## Zusammenfassung der Methode in simo_hughes Seite 319
## s_ten ist die Haupt Iterationsvariable, ähnlich zu unserem sigma_plasticity
def mat_large_strain_plast_3D(F, mat_par, hist):
    """
    description of material routine (Return-mapping Algorithm for J2-Flow Theory. Isotropic Hardening)

    F: total deformation gradient (3x3 second order tensor)
        F = F_e @ F_p, multiplicative splitting in elastic (F_e) and plastic (F_p) parts
        C_p = F_p.T @ F_p right plastic Cauchy-Green Tensor
        b_e = F_e @ F_e.T = F @ C_p_-1 @ F.T elastic left Cauchy-Green Tensor

    f: relative deformation gradient
        f_n+1 = I_ten + nabla(u) = F_n+1 @ F_n_inv

    isotropic stress response
        plastic flow is isochoric:
        det(F_p) = det(C_p) = 1  ==>  J = det(F) = det(F_e)


    (1) Simo, J.C.; Hughes, T.J.R. Computational Inelasticity. Interdisciplinary Applied Mathematics Vol.7, 300-320
        ISBN 0-387-97520-09
    (2) Wriggers, P. Nichtlineare Finite-Element-Methoden. Springer-Verlag Berlin Heidelberg New York 59ff, 226ff
        ISBN 3-540-67747-X

    values received F_n, F_n+1 , b_e_trial

    F_e, F_p one is required

    preliminary step / initial step (t_0)

    F_n - total deformation gradient --> known
    F_p - plastic part of deformation gradient --> identity tensor (no plastic deformation at the beginning)
        F = F_e @ F_p ==> F = F_e
    C = F.T @ F     right Cauchy-Green Tensor
    b = F @ F.T     left Cauchy-Green Tensor
    b_e = F_e @ F_e.T   left elastic Cauchy-Green Tensor
        b = b_e

    s_trial = mu_ * dev(b_e_trial)  // dev(A) = A - 1/3 * tr(A) * I_ten
    f_trial = ||s_trial|| - sqrt(s/2)*(K*alpha_n+sigma_Y)   K - isotropic hardening modulus, alpha_n - hardining parameter, sigma_Y - flow stress

    if f_trial <= 0:
        set (~)_n+1 = (~)_trial
    else:
        return mapping algorithm
        I_e = 1/3 * tr(b_e_trial)
        mu_trial = I_e * mu_

        delta_gamma = (f_trial/2*mu_trial)/(1+K/3*mu_trial)
        n = s_trial / ||s_trial||

        s_new = s_trial - 2*mu_trial*delta_gamma*n
        alpha_new = alpa_n + sqrt(2/3)*delta_gamma

    J_new = det(F_n+1)
    p_new = U´(J_new) // p_new = 1/2*kappa*((J_e^2-1)/J_e) , J_e = det(F_e)   //   U´(~) - volumetric part of sotred-energy function (W = U(J_e) + W_trial(b_e_trial)
    tau_new = J_new*p_new*I_ten + s_new

    b_e_trial_new = s_new/mu_ + I_e*I_ten

    in history list: b_e_trial_new, tau_new, p_new, J_new, alpha_new, s_new
    """

    I_ten = np.eye(3)  # identity tensor
    #II_ten = (np.einsum("ac,bd -> abcd", I_ten, I_ten) + np.einsum("ad,bc -> abcd", I_ten, I_ten))/2

    # material parameters / history variables
    F_old = hist[0]
    b_e_old = hist[2]
    alpha_old = hist[1]
    mu_ = mat_par[1]
    K = mat_par[3]
    sigma_y = mat_par[2]
    la_ = mat_par[0]

    ka_ = la_ + (2 / 3) * mu_
    f_ = F @ np.linalg.inv(F_old) ## 9.3.16 relative deformation gradient 

    # 2. elastic predictor
    f_stroke = np.linalg.det(f_) ** (-1 / 3) * f_
    #f_det = np.linalg.det(f_)  # for test_large_strain_plast
    b_e_tr = f_stroke @ b_e_old @ f_stroke.T
    s_tr = mu_ * deviator_tensor(b_e_tr)
    F_inv = np.linalg.inv(F)


    # 3. check for plastic loading
    f_tr = np.linalg.norm(s_tr) - np.sqrt(2 / 3) * (K * alpha_old + sigma_y)

    if f_tr <= 0: #elastischer Schritt
        s_np1 = s_tr
        b_e_np1 = b_e_tr
        n_tr = s_np1 / (np.linalg.norm(s_np1) + 1*(10**(-8)))
        ## Right Cauchy Green Tensor C?

    else: #plastischer Schritt
        I_e = (1 / 3) * np.linalg.trace(b_e_tr)
        mu_stroke = I_e * mu_
        delta_gamma = (f_tr / (2 * mu_stroke)) / (1 + (K / (3 * mu_stroke)))

        n_tr = s_tr / np.linalg.norm(s_tr) ## 9.2.16 associative-flow rule 
        s_np1 = s_tr - 2 * mu_stroke * delta_gamma * n_tr ## 9.3.28 reordered requirement with delta_gamma>0 
        ## s_ten = s_n+1,  s_n converged stresses
        # update intermediate configuration
        ## total/elastic left Cauchy–Green Tensor b/b_e 
        b_e_np1 = s_np1 / mu_ + I_e * I_ten ## 9.3.33 elastic constitutive equation and 9.2.8


    # 5. elastic mean stress
    J_ = np.linalg.det(F)
    p_ = (ka_ / 2) * (J_ ** 2 - 1) / J_
    tau_ten = J_ * p_ * I_ten + s_np1 ## uncoupled deviatoric stress-strain relationship 9.2.6
    S_ten = F_inv@tau_ten@F_inv.T

    # update history variables
    hist_new = hist.copy()
    hist_new[0] = F
    hist_new[2] = b_e_np1

    return S_ten, hist_new

