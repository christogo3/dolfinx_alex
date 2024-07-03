class Material:
    def __init__(self, name, density, youngs_modulus, shear_modulus, poissons_ratio, thermal_conductivity, 
                 coeff_thermal_expansion, melting_point, tensile_strength, yield_strength, fracture_toughness, 
                 hardness, elongation_at_break):
        self.name = name
        self.density = density  # kg/m^3
        self.youngs_modulus = youngs_modulus * 1e9  # Convert GPa to Pa
        self.shear_modulus = shear_modulus * 1e9  # Convert GPa to Pa
        self.poissons_ratio = poissons_ratio
        self.thermal_conductivity = thermal_conductivity  # W/m·K
        self.coeff_thermal_expansion = coeff_thermal_expansion  # /°C
        self.melting_point = melting_point  # °C
        self.tensile_strength = [ts * 1e6 for ts in tensile_strength]  # Convert MPa to Pa
        self.yield_strength = [ys * 1e6 for ys in yield_strength]  # Convert MPa to Pa
        self.fracture_toughness = fracture_toughness  # MPa√m
        self.hardness = hardness  # HB
        self.elongation_at_break = elongation_at_break  # %

    def get_material_properties(self):
        return {
            "Name": self.name,
            "Density (kg/m^3)": self.density,
            "Young's Modulus (Pa)": self.youngs_modulus,
            "Shear Modulus (Pa)": self.shear_modulus,
            "Poisson's Ratio": self.poissons_ratio,
            "Thermal Conductivity (W/m·K)": self.thermal_conductivity,
            "Coefficient of Thermal Expansion (/°C)": self.coeff_thermal_expansion,
            "Melting Point (°C)": self.melting_point,
            "Tensile Strength (Pa)": self.tensile_strength,
            "Yield Strength (Pa)": self.yield_strength,
            "Fracture Toughness (MPa√m)": self.fracture_toughness,
            "Hardness (HB)": self.hardness,
            "Elongation at Break (%)": self.elongation_at_break
        }

    def calculate_fracture_resistance(self):
        E = self.youngs_modulus
        K_IC_lower, K_IC_upper = self.fracture_toughness
        G_c_lower = (K_IC_lower * 1e6) ** 2 / E  # Convert MPa√m to Pa√m and calculate G_c
        G_c_upper = (K_IC_upper * 1e6) ** 2 / E  # Convert MPa√m to Pa√m and calculate G_c
        return G_c_lower, G_c_upper

# Define an instance for AlSi10
AlSi10 = Material(
    name="AlSi10",
    density=2660,  # kg/m^3
    youngs_modulus=75,  # GPa
    shear_modulus=28,  # GPa
    poissons_ratio=0.33,
    thermal_conductivity=125,  # W/m·K
    coeff_thermal_expansion=22e-6,  # /°C
    melting_point=575,  # °C
    tensile_strength=(180, 240),  # MPa
    yield_strength=(120, 180),  # MPa
    fracture_toughness=(15, 25),  # MPa√m
    hardness=(60, 80),  # HB
    elongation_at_break=(1, 5)  # %
)


def compute_lame_parameters(youngs_modulus, poisson_ratio):
    """
    Compute the Lamé parameters (λ and μ) from Young's modulus (E) and Poisson's ratio (ν).

    Parameters:
    youngs_modulus (float): Young's modulus (E)
    poisson_ratio (float): Poisson's ratio (ν)

    Returns:
    tuple: A tuple containing the first Lamé parameter (λ) and the second Lamé parameter (μ)
    """
    # Ensure inputs are valid
    if youngs_modulus <= 0:
        raise ValueError("Young's modulus must be positive.")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("Poisson's ratio must be between -1 and 0.5 (exclusive).")
    
    # Compute the second Lamé parameter (μ), also known as the shear modulus
    mue = youngs_modulus / (2 * (1 + poisson_ratio))
    
    # Compute the first Lamé parameter (λ)
    lam = (youngs_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    
    return lam, mue

# # Retrieve properties and calculate fracture resistance for AlSi10
# properties = AlSi10.get_material_properties()
# Gc_lower, Gc_upper = AlSi10.calculate_fracture_resistance()

# # Print material properties and fracture resistance
# print("AlSi10 Material Properties:")
# for prop, value in properties.items():
#     print(f"{prop}: {value}")

# print("\nFracture Resistance Gc (J/m^2):")
# print(f"Lower Bound: {Gc_lower:.6f} J/m^2")
# print(f"Upper Bound: {Gc_upper:.6f} J/m^2")
