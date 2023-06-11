# https://jupyter-docker-stacks.readthedocs.io/en/latest/
# Note this takes about 30 minutes to build

FROM jupyter/scipy-notebook:python-3.10
MAINTAINER John Kitchin <jkitchin@andrew.cmu.edu>

# Set the default shell to bash instead of sh so the source commands work
SHELL ["/bin/bash", "-c"]

RUN python -m pip install --upgrade pip \
	&& python -m pip install pycse pint quantities uncertainties \
	nbconvert nbconvert[webpdf] \
	jupyter-book matplotlib numpy ghp-import pandas sympy autograd numpydoc joblib plotly \
	"jupyter_server>=2.0.0" \
	&& jupyter server extension disable nbclassic \


# Now all the jupyter lab settings
COPY jupyter_lab_config.py /home/jovyan/.jupyter/jupyter_lab_config.py


# # change the final working dir to the place you mount your CWD
WORKDIR /home/jovyan/work

