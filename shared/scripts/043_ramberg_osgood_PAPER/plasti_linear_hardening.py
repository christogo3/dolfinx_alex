def strain_tensor(eps0):
    return np.array([
        [eps0, 0, 0],
        [0, -nu0 * eps0, 0],
        [0, 0, -nu0 * eps0]
    ])