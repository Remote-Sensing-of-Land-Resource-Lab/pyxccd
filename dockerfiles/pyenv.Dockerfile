# syntax=docker/dockerfile:1.3.0-labs

# This dockerfile uses new-ish buildkit syntax. 
# Details on how to run are on the bottom of the file.
# (docker devs: todo unconsequential heredocs)

FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.10.5
ARG PYENV_VERSION=v2.3.3

ENV HOME=/root
ENV PYENV_ROOT=/root/.pyenv

## Install Prerequisites 
RUN <<EOF
#!/bin/bash
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        subversion \
        wget curl \
        make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev \
        libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl 
apt-get clean 
rm -rf /var/lib/apt/lists/*
EOF

## Install pyenv
RUN <<EOF
#!/bin/bash
git clone https://github.com/pyenv/pyenv.git -b $PYENV_VERSION $PYENV_ROOT 
(cd $PYENV_ROOT && src/configure && make -C src)
EOF


## Use pyenv to compile an optimized Python
RUN <<EOF
#!/bin/bash
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$($PYENV_ROOT/bin/pyenv init -)"

PROFILE_TASK="-m test.regrtest --pgo test_array test_base64 test_binascii test_binhex test_binop test_c_locale_coercion test_csv test_json test_hashlib test_unicode test_codecs test_traceback test_decimal test_math test_compile test_threading test_time test_fstring test_re test_float test_class test_cmath test_complex test_iter test_struct test_slice test_set test_dict test_long test_bytes test_memoryview test_io test_pickle"

PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-computed-gotos --with-lto"

PYTHON_CFLAGS="-march=native -O2 -pipe" 

PROFILE_TASK=$PROFILE_TASK \
PYTHON_CFLAGS="${PYTHON_CFLAGS}" \
PYTHON_CONFIGURE_OPTS="${PYTHON_CONFIGURE_OPTS}" \
pyenv install $PYTHON_VERSION 

EOF


## Create a default Python virtualenv
RUN <<EOF
#!/bin/bash
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$($PYENV_ROOT/bin/pyenv init -)"
pyenv global $PYTHON_VERSION

PYENV_PREFIX=$(pyenv prefix)
python -m venv $PYENV_PREFIX/envs/pyenv$PYTHON_VERSION

BASHRC_CONTENTS='
# Add the pyenv command to our environment if it exists
export HOME="/root"
export PYENV_ROOT="$HOME/.pyenv"
if [ -d "$PYENV_ROOT" ]; then
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$($PYENV_ROOT/bin/pyenv init -)"
    source $PYENV_ROOT/completions/pyenv.bash
    export PYENV_PREFIX=$(pyenv prefix)
fi

# Optionally auto-activate the chosen pyenv pyenv environment
if [ -d "$PYENV_PREFIX/envs/pyenv$PYTHON_VERSION" ]; then
    source $PYENV_PREFIX/envs/pyenv$PYTHON_VERSION/bin/activate
fi
'
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
# Write a secondary script for non-interactive usage
echo "$BASHRC_CONTENTS" >> $HOME/activate
EOF


################
### __DOCS__ ###
################
RUN <<EOF


# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/
echo "

# docker login
# docker pull docker/dockerfile:1.3.0-labs

cd $HOME/code/pyxccd
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "pyenv:310" \
    --build-arg PYTHON_VERSION=3.10.5 \
    -f ./dockerfiles/pyenv.Dockerfile .

docker run --runtime=nvidia -it pyenv:310 bash  .
# docker buildx build -t "pyenv3.10" -f ./pyenv.Dockerfile --build-arg BUILD_STRICT=1 .
"


EOF

