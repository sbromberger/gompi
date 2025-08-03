# FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04
FROM mcr.microsoft.com/devcontainers/go:1-1.24-bookworm

# ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

# Optionally install the cmake for vcpkg

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install clangd"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
RUN apt update && apt -y upgrade && export DEBIAN_FRONTEND=noninteractive && apt -y install openmpi-bin openmpi-doc libopenmpi-dev

WORKDIR /tmp
RUN wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5.1.tar.gz
RUN tar xzf /tmp/osu-micro-benchmarks-7.5.1.tar.gz
WORKDIR /tmp/osu-micro-benchmarks-7.5.1
RUN ./configure CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx
RUN make && make install
RUN go install golang.org/x/tools/cmd/stringer@latest


# RUN update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100
# RUN update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-19 100