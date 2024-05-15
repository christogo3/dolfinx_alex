# Use the dolfinx/dolfinx:stable image as base
FROM dolfinx/dolfinx:stable

# Set the working directory in the container
WORKDIR /home

# Copy the current directory contents into the container at /home
COPY . .

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    xvfb \
    && rm -rf /var/lib/apt/lists/*


# Install X server and other necessary packages
# RUN apt-get update && apt-get install -y xserver-xorg-core xserver-xorg-video-dummy openmpi-bin

# Set DISPLAY environment variable
# ENV DISPLAY=127.0.0.1:0.0

# Install required Python packages
RUN pip install numpy pyfiglet uvw tqdm vtk pyvista meshio h5py

# Set PYTHONPATH to include /home/utils
ENV PYTHONPATH="/home/utils:${PYTHONPATH}"

# Command to keep the container running
CMD ["tail", "-f", "/dev/null"]


