# syntax=docker/dockerfile:1.3.0-labs

# This dockerfile uses new-ish buildkit syntax. 
# Details on how to run are on the bottom of the file.
# (docker devs: todo unconsequential heredocs)

FROM pyenv:310

ENV HOME=/root

RUN <<EOF
#!/bin/bash
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
        build-essential zlib1g-dev libgsl-dev gfortran \
        ffmpeg tmux jq tree
apt-get clean 
rm -rf /var/lib/apt/lists/*
EOF

SHELL ["/bin/bash", "--login", "-c"]

RUN echo $(pwd)

COPY requirements.txt /pyxccd/
COPY requirements /pyxccd/requirements

# Setup primary dependencies
RUN <<EOF
#!/bin/bash
source $HOME/activate
cd /pyxccd
# Always use the latest Python build tools
python -m pip install pip -U 
python -m pip install -r requirements.txt
python -m pip install -r requirements/build.txt
python -m pip install -r requirements/headless.txt
python -m pip install -r requirements/runtime.txt
python -m pip install -r requirements/optional.txt
python -m pip install -r requirements/tests.txt
python -m pip install setuptools==63.2.0  
EOF


# Copy the pyxccd source (only copy the .git folder so we are careful to not
# copy any host build artifacts)
COPY .git /pyxccd/.git

WORKDIR /pyxccd

# Use the .git repo to setup a clean checkout
RUN git reset --hard HEAD

RUN <<EOF
#!/bin/bash
source $HOME/activate
pip install -r requirements/build.txt
python -m pip install setuptools==63.2.0  
chmod +x run_developer_setup.sh
./run_developer_setup.sh
EOF


## Run simple tests
RUN <<EOF
#!/bin/bash
source $HOME/activate
tree
echo "Start simple tests"
python -c "import pyxccd; print(pyxccd.__file__)"
python -c "import pyxccd; print(pyxccd.__version__)"
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

DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "pyxccd:310" \
    -f ./dockerfiles/pyxccd.Dockerfile .

docker run -w /pyxccd -it pyxccd:310 bash
# docker buildx build -t "pyenv3.10" -f ./pyenv.Dockerfile --build-arg BUILD_STRICT=1 .
"


EOF

