# M2F2_Det Dockerfile - 本地开发和远程训练统一环境
# 基于NVIDIA官方镜像，确保CUDA兼容性
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 设置非交互式安装
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# 创建Python环境（使用conda-forge避免ToS问题）
RUN conda create -n M2F2 python=3.10 -y -c conda-forge --override-channels
SHELL ["conda", "run", "-n", "M2F2", "/bin/bash", "-c"]

# 分阶段安装依赖以避免冲突

# 1. 安装NumPy（PyTorch 2.1.0需要<2.0版本）
RUN pip install "numpy<2.0"

# 2. 安装PyTorch（基础依赖，使用CUDA 12.1）
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装Flash Attention编译所需依赖
RUN pip install ninja packaging wheel psutil einops

# 4. 安装Flash Attention（需要编译）
RUN pip install flash-attn==2.5.7 --no-build-isolation

# 5. 安装其他依赖
COPY requirements.txt /tmp/requirements.txt
RUN grep -v "^torch==" /tmp/requirements.txt | \
    grep -v "^torchvision==" | \
    grep -v "^flash-attn" | \
    grep -v "^numpy" | \
    grep -v "^einops==" | \
    grep -v "^ninja==" | \
    grep -v "^packaging==" | \
    grep -v "^wheel" | \
    grep -v "^psutil==" > /tmp/requirements_filtered.txt && \
    pip install -r /tmp/requirements_filtered.txt

# 设置工作目录
WORKDIR /workspace/M2F2_Det

# 默认激活conda环境
RUN echo "conda activate M2F2" >> ~/.bashrc

# 暴露端口（Gradio等）
EXPOSE 7860

# 默认命令
CMD ["/bin/bash"]
