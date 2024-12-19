import matplotlib.pyplot as plt
import numpy as np
import os 

script_path = os.path.dirname(__file__)

# Set up matplotlib to use the pgf backend
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # or "lualatex" or "xelatex"
    "font.family": "serif",       # Use serif/main font for text elements
    "text.usetex": True,          # Use LaTeX to write all text
    "pgf.rcfonts": False,         # Don't set fonts from rc parameters
})

# Generate data for the plot
x = np.linspace(0, 10, 100)
y1 = x**2
y2 = np.sqrt(x)

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x, y1, label=r"$x^2$")
plt.plot(x, y2, label=r"$\sqrt{x}$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title("Sample Plot")
plt.legend()

# Save the plot as a PGF file
plt.savefig(os.path.join(script_path,"test_pic.pgf"))
