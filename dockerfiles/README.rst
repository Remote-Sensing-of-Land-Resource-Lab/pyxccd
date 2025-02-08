These dockerfiles help to produce a reproducible build environment for pyxccd.

The base pyenv.dockerfile sets up an Ubuntu 20.04 image with an optimized pyenv
virtual environment that is ready for use. This is used as the base image for
the pyxccd.Dockerfile, which roughly corresponds to how a user would install it
on their system given an existing virtualenv.


This process should work end-to-end using the following commands.

Starting from the root of a fresh clone of the pyxccd repo: 

.. code:: bash

    # Build the pyenv base image.
    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t "pyenv:310" \
        --build-arg PYTHON_VERSION=3.10.5 \
        -f ./dockerfiles/pyenv.Dockerfile .

    # Build the pyxccd image on top of pyenv
    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t "pyxccd:310" \
        -f ./dockerfiles/pyxccd.Dockerfile .

    # Launch a shell in the docker container 
    docker run -w /pyxccd -it pyxccd:310 bash


Note that the pyenv image uses ``nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04``
as the base image, which could be changed, as nvidia or cuda support is not
directly needed by this project.
