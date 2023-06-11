#!/usr/bin/env python
"""CLI for pycse.

In theory this should probably use docker-py, but this worked first. It is only
lightly tested.

There is not a build in way to update the image. You have to manage this with
docker commands.

To get the latest image:

> docker pull jkitchin/pycse:latest


Sometimes you have to manually delete an old container if it doesn't close
properly. This should do it:

> docker rm -f pycse

"""

import os
import uuid
import numpy as np
import time
import webbrowser
import subprocess


def pycse():
    """CLI to launch a Docker image with Jupyter lab in the CWD.
    This assumes you have a working Docker installation."""
    PWD = os.getcwd()
    PORT = np.random.randint(8000, 9000)

    JUPYTER_TOKEN = str(uuid.uuid4())
    os.environ["JUPYTER_TOKEN"] = JUPYTER_TOKEN

    # Check setup and get image if needed
    try:
        subprocess.run(
            ["docker", "image", "inspect", "jkitchin/pycse"],
            capture_output=True,
        )
    except:
        subprocess.run(
            ["docker", "pull", "jkitchin/pycse"], capture_output=True
        )

    cmd1 = (
        f"docker run -d --name pycse -it --rm -p {PORT}:8888 "
        f"-e JUPYTER_TOKEN -v {PWD}:/home/jovyan/work"
        "jkitchin/pycse"
    )

    print("Starting Jupyter lab. Type C-c to quit")
    subprocess.Popen(cmd1.split())

    time.sleep(2)

    url = f"http://localhost:{PORT}/lab?token={JUPYTER_TOKEN}"
    webbrowser.open(url)

    subprocess.run(["docker", "attach", "pycse"])


if __name__ == "__main__":
    pycse()
