# Use the dolfinx/dolfinx:stable image as base
# FROM dolfinx/dolfinx:stable
# FROM dolfinx/dolfinx:v0.7.3
FROM dolfinx/dolfinx:v0.7.3
#FROM dolfiny/dolfiny

RUN apt clean

# RUN apt-key update

# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C

# Ensure APT is using HTTPS for the repositories
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates

# Set the working directory in the container
WORKDIR /home

# Copy the current directory contents into the container at /home
#COPY . .


# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    xvfb \
    texlive \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install numpy pyfiglet uvw tqdm vtk pyvista meshio python_papi pandas scipy pygmsh
RUN pip install --no-binary=h5py h5py

# Set PYTHONPATH to include /home/utils
ENV PYTHONPATH="/home/utils:${PYTHONPATH}"

# Command to keep the container running
CMD ["tail", "-f", "/dev/null"]


