FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    sqlite3 \
    libbz2-dev \
    libsqlite3-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CXX=/opt/rocm/llvm/bin/clang++
ENV CC=/opt/rocm/llvm/bin/clang
