import numpy as np
import alex.phasefield as phasefield

mu=1.0
E = 2.0*mu # or 2.5
H = 0.6
sig_y = 1.0
Gc = 1.0
epsilon = 0.1

term0 = 1.0/3.0 * (E+H) / (E*H)
term1 = ((sig_y**2) / H + 5.0/18.0 *Gc/epsilon)
term2 = 4.0 * (sig_y ** 2 / H) ** 2
term3 = 8/9*(sig_y**2*Gc) / (H*epsilon)
term4 = 25.0/324 * (Gc/epsilon) ** 2
term5 = sig_y / H

eps_c = np.sqrt(term0 * (term1 + np.sqrt(term2 + term3 + term4  ))) - term5
print("eps_c ductile:" + str(eps_c))

sig_c = (E*H) / (E+H) * (eps_c) + sig_y*(E/(E+H))
print("sig_c ductile:" + str(sig_c))


sig_c_linear_elastic = phasefield.sig_c_cubic_deg(Gc,mu,epsilon)
print("sig_c brittle:" + str(sig_c_linear_elastic))



Gc_equivalent = phasefield.get_Gc_for_given_sig_c_cub(sig_c,mu,epsilon)
print("G_c brittle with same sig_c:" + str(Gc_equivalent))
print("sig_c with corrected G_c:" + str(phasefield.sig_c_cubic_deg(Gc_equivalent,mu,epsilon)))