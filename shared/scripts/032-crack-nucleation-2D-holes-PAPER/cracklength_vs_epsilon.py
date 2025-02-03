import numpy as np
import matplotlib.pyplot as plt

def compute_da(epsilon, C1, C2, a):
    """
    Compute da for a given epsilon, C1, C2, and a.

    Solves the equation:
    0.5 * C1 / epsilon = 0.5 * C2 / (a + da) + da
    """
    lhs = 0.5 * C1 / epsilon

    def equation(da):
        return lhs - (0.5 * C2 / (a + da) + da)

    # Use numerical root finding to solve for da
    from scipy.optimize import fsolve
    da_initial_guess = 0.1  # Initial guess for da
    da_solution, = fsolve(equation, da_initial_guess)
    return da_solution

def plot_da_vs_epsilon(epsilon_values, C1, C2, a, output_file):
    """
    Plot da vs. epsilon for given parameters and save to a file.

    Parameters:
        epsilon_values: Array of epsilon values (positive only).
        C1, C2, a: Parameters in the equation.
        output_file: File path to save the plot.
    """
    da_values = []
    for epsilon in epsilon_values:
        if epsilon > 0:  # Avoid division by zero
            da = compute_da(epsilon, C1, C2, a)
            da_values.append(da)
        else:
            da_values.append(np.nan)  # Handle invalid values

    # Plot da vs epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_values, da_values, label=f"C1={C1}, C2={C2}, a={a}")
    plt.xlabel("Epsilon (\u03B5)")
    plt.ylabel("da")
    plt.title("Plot of da vs. Epsilon")
    plt.legend()
    plt.grid()

    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    # Parameters
    C1 = 0.06
    C2 = 0.00001
    a = 5.0

    # Epsilon range
    epsilon_min = 0.0
    epsilon_max = 0.5
    num_points = 20

    epsilon_values = np.linspace(epsilon_min, epsilon_max, num_points)

    # Output file
    output_file = "test.png"

    # Plot and save
    plot_da_vs_epsilon(epsilon_values, C1, C2, a, output_file)
    print(f"Plot saved to {output_file}")
