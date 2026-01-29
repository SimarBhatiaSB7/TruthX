FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory inside the container
WORKDIR /workspaces/TruthX

# Install system tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Pre-install the TruthX requirements
# This assumes you have the 'transformers' and 'FastChat' folders in your path
RUN pip install bitsandbytes accelerate scipy scikit-learn pyyaml