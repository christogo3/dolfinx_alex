# Start from dolfinx base image
FROM dolfinx/dolfinx:v0.7.3

# Set working directory
WORKDIR /home

# Clean APT cache and ensure system is up to date
RUN apt-get update && apt-get clean

# Remove any partial TeX Live installation (if present)
RUN apt-get remove --purge -y texlive* && apt-get autoremove -y && apt-get clean

# Install TeX Live Full and required dependencies
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    libgl1 \
    xvfb \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install numpy pyfiglet uvw tqdm vtk "pyvista<0.42.0" meshio python_papi pandas scipy pygmsh
RUN pip install --no-binary=h5py h5py

# Set PYTHONPATH
ENV PYTHONPATH="/home/utils:${PYTHONPATH}"

# Keep container running
CMD ["tail", "-f", "/dev/null"]





