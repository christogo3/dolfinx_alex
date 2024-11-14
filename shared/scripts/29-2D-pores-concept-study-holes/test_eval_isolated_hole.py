def sig_ff_for_isolated_hole(dhole,epsilon,sig_loc_bar):
    a = dhole/2.0
    def antiderivative(r):
        # a = dhole / 2.0
        return 2.0 * r - a ** 2 / r - a ** 4 / (r ** 3)
    sig_ff = (sig_loc_bar) * (10.0 * epsilon * 2.0) / (antiderivative(a+10.0*epsilon) - antiderivative(a))
    return sig_ff

dhole = 1.0
epsilon = 0.1

sig_ff = sig_ff_for_isolated_hole(dhole,1.0*epsilon,sig_loc_bar=1.0)

a=1.0