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

The Docker Desktop app is also helpful for this kind of stuff.

TODO: make this a click app, with cli arguments to update the image? e.g.

> pycse --update
> pycse path/to/working-dir

"""

import os
import uuid
import numpy as np
import time
import webbrowser
import subprocess
import shutil
import sys


def pycse():
    """CLI to launch a Docker image with Jupyter lab in the CWD.
    This assumes you have a working Docker installation."""
    if shutil.which("docker") is None:
        raise Exception(
            "docker was not found."
            " Please install it from https://www.docker.com/"
        )

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

    # Check if the container is already running
    p = subprocess.run(
        ["docker", "ps", "--format", '"{{.Names}}"'], capture_output=True
    )
    if "pycse" in p.stdout.decode("utf-8"):
        ans = input(
            "There is already a pycse container running."
            "Do you want to kill it? (y/n)"
        )
        if ans.lower() == "y":
            subprocess.run("docker rm -f pycse".split())
        else:
            print(
                "There can only be one pycse container running at a time."
                "Connecting to it."
            )

            # this outputs something like 0.0.0.0:8987
            p = subprocess.run(
                "docker port pycse 8888".split(), capture_output=True
            )
            output = p.stdout.decode("utf-8").strip()
            PORT = output.split(":")[-1]

            # We need the token for the running container
            p = subprocess.run(
                ["docker", "exec", "pycse", "printenv", "JUPYTER_TOKEN"],
                capture_output=True,
            )
            JUPYTER_TOKEN = p.stdout.decode("utf-8").strip()

            url = f"http://localhost:{PORT}/lab?token={JUPYTER_TOKEN}"
            webbrowser.open(url)
            sys.exit()

    # Start a new container.
    PWD = os.getcwd()
    PORT = np.random.randint(8000, 9000)

    JUPYTER_TOKEN = str(uuid.uuid4())
    os.environ["JUPYTER_TOKEN"] = JUPYTER_TOKEN

    cmd = (
        f"docker run -d --name pycse -it --rm -p {PORT}:8888 "
        f"-e JUPYTER_TOKEN -v {PWD}:/home/jovyan/work "
        "jkitchin/pycse"
    )

    print("Starting Jupyter lab. Type C-c to quit.")
    subprocess.Popen(cmd.split())

    time.sleep(2)

    url = f"http://localhost:{PORT}/lab?token={JUPYTER_TOKEN}"
    webbrowser.open(url)

    subprocess.run(["docker", "attach", "pycse"])


if __name__ == "__main__":
    pycse()
