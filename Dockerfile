# Use CUDA 12.1 development image with Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    ninja-build \
    gcc-11 g++-11 \
    libsndfile1 \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    cmake \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda && \
    rm /tmp/miniconda.sh

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

# Create and configure the conda environment
RUN /root/miniconda/bin/conda create -n street-gaussian python=3.8 -y && \
    /root/miniconda/bin/conda run -n street-gaussian conda install \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Set working directory and copy files
WORKDIR /app
COPY . .

# Install project dependencies
RUN /root/miniconda/bin/conda run -n street-gaussian pip install -r requirements.txt && \
    /root/miniconda/bin/conda run -n street-gaussian pip install ./submodules/diff-gaussian-rasterization && \
    /root/miniconda/bin/conda run -n street-gaussian pip install ./submodules/simple-knn && \
    /root/miniconda/bin/conda run -n street-gaussian pip install ./submodules/simple-waymo-open-dataset-reader

# Install COLMAP
RUN git clone https://github.com/colmap/colmap.git /colmap && \
    mkdir /colmap/build && \
    cd /colmap/build && \
    cmake .. -GNinja -DGUI_ENABLED=OFF -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_INSTALL_PREFIX=/colmap-install && \
    ninja install && \
    echo 'export PATH=/colmap-install/bin:$PATH' >> /root/.bashrc

# Install GroundingDino
# RUN /root/miniconda/bin/conda run -n street-gaussian bash -c "cd /app/GroundingDino && pip install -e ."

# Verify PyTorch CUDA compatibility
RUN /root/miniconda/bin/conda run -n street-gaussian python -c "import torch; print(torch.cuda.is_available())"

# Set entrypoint
CMD ["bash"]

# docker build -t street-gaussian-env .
# docker run --gpus all -it --name street-gaussian-container -v ~/street_gaussians:/street_gaussians -v /mnt/wato-drive2/waymo_processed:/mnt/wato-drive2/waymo_processed -w /street_gaussians --rm street-gaussian-env
# source /root/miniconda/bin/activate street-gaussian
