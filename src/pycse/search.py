"""A Search library for Jupyter lab.

Eventually I want to add a command you can run that opens a tab.

it would be something like Search, it would have to take some input I assume
"""

import glob
import ipykernel
import nbformat
import os
import re
import requests
from jupyter_server import serverapp as app
from contextlib import contextmanager


@contextmanager
def cwd(path):
    """Context manager to temporarily change working directory to PATH."""
    _cwd = os.getcwd()
    try:
        os.chdir(_cwd)
        yield
    except Exception as e:
        raise e
    finally:
        os.chdir(_cwd)


def get_kernel_id():
    """Get the current kernel id.

    Based on the connection file. These look like:
    kernel-539488f9-b76d-4a55-bea9-e5f5dcaf3dc1.json
    and we extract out the piece like 539488f9-b76d-4a55-bea9-e5f5dcaf3dc1.
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    fname, _ = os.path.splitext(connection_file)
    kernel_id = fname.replace("kernel-", "")
    return kernel_id


def get_server():
    """Get the running server.
    It appears you may have many running servers.
    """
    srvs = list(app.list_running_servers())
    if len(srvs) == 1:
        return srvs[0]
    else:
        raise Exception("More than one running server found: {srvs}")


def get_running_notebooks():
    """Get a list of all the open notebooks in the running server.
    It appears that each notebook runs in its own kernel.
    """
    srv = get_server()
    notebooks = requests.get(srv["url"] + "api/sessions?token=" + srv["token"]).json()
    return notebooks


def get_notebook_path():
    kernel_id = get_kernel_id()
    srv = get_server()
    notebooks = get_running_notebooks()
    NBS = [nb for nb in notebooks if nb["kernel"]["id"] == kernel_id]
    if len(NBS) == 1:
        nb = NBS[0]
        return os.path.join(srv["root_dir"], nb["notebook"]["path"])
    else:
        raise Exception("Multiple notebooks found: {NBS}")


def get_notebook_paths(path=None, recursive=True):
    """Get a list of paths to all notebooks in the path.
    If recursive is True, find them recursively.
    if path is None, use the server root directory.
    """
    if path is None:
        path = get_server()["root_dir"]

    with cwd(path):
        nbs = glob.glob("**/*.ipynb", recursive=recursive)
        return nbs


def search_headings(pattern, ipynb_path):
    """Search for PATTERN in the headings of IPYNB_PATH.

    Note that headings are defined as markdown lines that start with #.

    Returns the first match.
    """

    nb = nbformat.read(ipynb_path, as_version=4)
    md = [cell for cell in nb["cells"] if cell["cell_type"] == "markdown"]
    src_lines = sum(([mc["source"].split("\n") for mc in md]), [])
    headings = [sl for sl in src_lines if sl.startswith("#")]

    for heading in headings:
        m = re.search(pattern, heading)
        if m:
            return m


def search_markdown(pattern, ipynb_path):
    """Search for PATTERN in the markdown cells of IPYNB_PATH.

    Returns the first match.
    """

    nb = nbformat.read(ipynb_path, as_version=4)
    md = "\n".join([cell["source"] for cell in nb["cells"] if cell["cell_type"] == "markdown"])
    return re.search(pattern, md)
