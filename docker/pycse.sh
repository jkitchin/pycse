#!/bin/bash

# Accept optional Python version argument (e.g., 3.12, 3.13, 3.14)
# Defaults to 'latest' if no argument provided
VERSION=${1:-latest}

export JUPYTER_TOKEN=`uuidgen`
PORT=`shuf -i 8000-9000 -n 1`

docker run -d --name pycse -it --rm -p ${PORT}:8888 -e JUPYTER_TOKEN -v "${PWD}":/home/jovyan/work pycse:${VERSION}
sleep 2
echo "opening on http://localhost:${PORT}/lab?token=${JUPYTER_TOKEN}"
open http://localhost:${PORT}/lab?token=${JUPYTER_TOKEN}

docker attach pycse
#end
