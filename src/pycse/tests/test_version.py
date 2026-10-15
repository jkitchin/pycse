"""pycse.__version__ must match the installed package metadata."""

from importlib.metadata import version

import pycse


def test_version_matches_metadata():
    """A hardcoded __version__ once reported 2.3.5 in a 2.11.2 install."""
    assert pycse.__version__ == version("pycse")
