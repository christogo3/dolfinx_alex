import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import json
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

script_path = os.path.dirname(__file__)
search_path = os.path.join(script_path,'JSONS/n192', '*.json')

json_files = glob.glob(search_path)

def read_jsons(directory,mode):
    results = np.zeros((len(directory),3))
    i = 0

    for json_path in directory:
        
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            final_state = data.get('final_yield_state', None)
            if final_state != None:
                if mode == 'strain':
                    final_vector = final_state.get('eps_mac_eigenvalues_current', None)
                elif mode == 'stress':
                    tensor = final_state.get('sigma_avg_reduced_volume', None)
                    eig_array = np.linalg.eigvals(tensor)
                    final_vector = eig_array.tolist()
                else: final_vector = None
            
            if final_vector != None:
                results[i,:] = final_vector
            
            i = i+1

    return results
        

def scatter_plot(results):
    fig = plt.figure("failure_surface")
    ax = fig.add_subplot(111, projection='3d')

    # Plot x, y, z
    ax.scatter(results[:,0], results[:,1], results[:,2])

    ax.set_xlabel("eps_1")
    ax.set_ylabel("eps_2")
    ax.set_zlabel("eps_3")
    plt.title("Failure Surface")
    plt.savefig(os.path.join(script_path, 'scatter.png'))

def hull_plot(results,elev=None,azim=None,fontsize=14,mode='strain'):
    # Convex Hull plot
    hull = ConvexHull(results)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points
    ax.scatter(results[:,0], results[:,1], results[:,2])

    ax.plot_trisurf(results[:,0], results[:,1], results[:,2], 
                    triangles=hull.simplices, 
                    cmap='viridis',    # Applies a color gradient based on height
                    alpha=0.6,         # Transparency helps see internal structures
                    edgecolor='black', # Keeps the wireframe definition
                    linewidth=0.2)

    if mode == 'strain':
        ax.set_xlabel("$\epsilon_1$",fontsize=fontsize)
        ax.set_ylabel("$\epsilon_2$",fontsize=fontsize)
        ax.set_zlabel("$\epsilon_3$",fontsize=fontsize)
        savename = 'strain'
    elif mode == 'stress':
        ax.set_xlabel("$\sigma_1$",fontsize=fontsize)
        ax.set_ylabel("$\sigma_2$",fontsize=fontsize)
        ax.set_zlabel("$\sigma_3$",fontsize=fontsize)
        savename = 'stress'

    if elev and azim != None:
        ax.view_init(elev,azim)

    plt.tight_layout()
    plt.savefig(os.path.join(script_path, '3D_' + savename + '_surface.png'),dpi=300)

def projected_plots(results,align_with_3d_view=False):

    # 2D Projections
    # Create a figure with two subplots for the classical projections
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # -------------------------------------------------------------
    # PROJECTION 1: The pi-Plane (Deviatoric Plane)
    # -------------------------------------------------------------
    # Transform 3D principal strains to 2D pi-plane coordinates (u, v)
    if align_with_3d_view:
        # Aligns directly with ax3d.view_init(elev=35.264, azim=45)
        u = (results[:, 0] - results[:, 1]) / np.sqrt(2)
        v = (results[:, 0] + results[:, 1] - 2 * results[:, 2]) / np.sqrt(6)
    else:
        # Standard Haigh-Westergaard (u along primary axis)
        u = (2 * results[:, 0] - results[:, 1] - results[:, 2]) / np.sqrt(6)
        v = (results[:, 1] - results[:, 2]) / np.sqrt(2)
    pi_points = np.column_stack((u, v))

    # Compute the 2D Convex Hull on the pi-plane
    hull_pi = ConvexHull(pi_points)

    # Plot all data points on the pi-plane
    ax1.scatter(u, v, color='gray', alpha=0.5, label='Data Points')

    # Plot the convex hull boundary
    for simplex in hull_pi.simplices:
        ax1.plot(pi_points[simplex, 0], pi_points[simplex, 1], 'r-', lw=2)

    # Fill the yield surface
    ax1.fill(pi_points[hull_pi.vertices, 0], pi_points[hull_pi.vertices, 1], 'red', alpha=0.15, label='Yield Domain')

    # Add the hydrostatic center point
    ax1.plot(0, 0, 'k+', markersize=10, label='Hydrostatic Axis')

    ax1.set_title("$\pi$-Plane (Deviatoric) Projection")
    ax1.set_xlabel("u (Shear)",fontsize=12)
    ax1.set_ylabel("v (Shear)",fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('equal', 'box')
    ax1.legend()

    # -------------------------------------------------------------
    # PROJECTION 2: Biaxial Projection (eps_1 vs. eps_2)
    # -------------------------------------------------------------
    # We extract the first two principal strains
    biaxial_points = results[:, [0, 1]]

    # Compute the 2D Convex Hull for the biaxial projection
    hull_bi = ConvexHull(biaxial_points)

    # Plot all data points
    ax2.scatter(results[:, 0], results[:, 1], color='gray', alpha=0.5)

    # Plot the convex hull boundary
    for simplex in hull_bi.simplices:
        ax2.plot(biaxial_points[simplex, 0], biaxial_points[simplex, 1], 'b-', lw=2)

    # Fill the yield surface
    ax2.fill(biaxial_points[hull_bi.vertices, 0], biaxial_points[hull_bi.vertices, 1], 'blue', alpha=0.15)

    ax2.set_title("Biaxial Projection ($\epsilon_1$ vs. $\epsilon_2$)")
    ax2.set_xlabel("$\epsilon_1$",fontsize=12)
    ax2.set_ylabel("$\epsilon_2$",fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_aspect('equal', 'box')


    # Save the crisp 2D projections
    plt.tight_layout()
    plt.savefig(os.path.join(script_path, '2D_projections.png'), dpi=300)
    plt.show()

#elevation = np.degrees(np.arcsin(1 / np.sqrt(3)))  # ~35.264 degrees
#azimuth = 45.0

results = read_jsons(json_files,mode='strain')
#hull_plot(results,mode='strain')
projected_plots(results)