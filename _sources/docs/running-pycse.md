Running pycse
=================

pycse is just a Python library. You can pip install it (see https://github.com/jkitchin/pycse#installation), then import it in your code like any other package.

I like to use it in Jupyter notebooks, especially through Jupyter lab. Of course you can run your own Jupyter lab. 

Docker
------

An alternative approach to using your own Jupyter lab is to run Jupyter lab from a Docker image. pycse provides a Docker file (https://github.com/jkitchin/pycse/tree/master/docker) and image (https://hub.docker.com/repository/docker/jkitchin/pycse/general) that you can use instead. Of course, you need a working Docker Desktop installation (see https://www.docker.com/).

When you install pycse, you should also get a command-line utility `pycse` to launch Jupyter lab from that Docker image with the current working directory mounted in the lab session. 
