"""A proof of concept pycse MCP server.

1. provide help with pycse functions.
2. a set of DOE functions.

"""

import platform
import sys
import shutil
import os
import json
from mcp.server.fastmcp import FastMCP, Image
from typing import Tuple, List, Union
from pydantic import BaseModel, Field
import pandas as pd
import pycse
import pkgutil
import importlib
import inspect
import io

from pycse.sklearn.lhc import LatinSquare
from pycse.sklearn.surface_response import SurfaceResponse

import matplotlib

matplotlib.use("Agg")

# Initialize FastMCP server
mcp = FastMCP("pycse")


class Factor(BaseModel):
    """Represents a single experimental factor with its levels."""

    name: str = Field(..., description="Name of the factor (e.g., 'Red', 'Temperature')")
    levels: List[Union[int, float]] = Field(..., description="List of factor levels")


class LatinSquareSpec(BaseModel):
    """Complete specification for a Latin square design.

    It is a list of the factors and their levels.

    """

    factors: List[Factor] = Field(..., description="List of experimental factors")


# this is a clunky way to save state between calls. It is not persistent, and
# probably can be broken without trying too hard. e.g. using it multiple times,
# or mixing lhc and sr. Another day I should look into using a class.
STATE = {}


@mcp.tool()
def design_lhc(inputs: LatinSquareSpec):
    """Design a LatinSquare design from the inputs.

    inputs is a list of tuples of the form: (varname, levels).

    For example, you might specify:

    Create a Latin square where Red is 0, 0.5, 1, Green is 0.0, 0.5, 1, and Blue
    is [0, 0.5, 1] and we measure 515nm at the output.

    This function returns the experiments that you should do.
    """
    # I think this is a little clunky for saving state
    global STATE

    factors = inputs.factors

    d = {factor.name: factor.levels for factor in factors}

    ls = LatinSquare(d)
    STATE["ls"] = ls

    design = ls.design()
    STATE["design"] = design

    return design


class LatinSquareResult(BaseModel):
    """Specification for a result.

    A result is identified by an experiment number, and a corresponding float
    result.

    """

    experiment: int = Field(..., description="Experiment number")
    result: float = Field(..., description="Result as a float")


class LatinSquareResults(BaseModel):
    """List of the results.

    This is a list of (experiment number, result).

    """

    results: List[LatinSquareResult] = Field(..., description="List of results")


@mcp.tool()
def analyze_lhc(lsr: LatinSquareResults):
    """Analyze the LatinSquare results.

    The results have to be provided in a way that looks like a list of
    (experiment #, result) can be parsed by the LLM.

    Returns an analysis of variance (ANOVA).

    """
    global STATE

    df = pd.DataFrame(
        [(result.experiment, result.result) for result in lsr.results],
        columns=["Experiment", "Result"],
    )

    merged = STATE["design"].merge(df, left_index=True, right_on="Experiment")

    ls = STATE["ls"]

    X = merged[ls.labels]
    y = merged["Result"]

    ls.fit(X, y)
    return ls.anova()


class SurfaceResponseInputs(BaseModel):
    """Class to represent the list of inputs for the surface response model."""

    inputs: List[str] = Field(..., description="List of input names")


class SurfaceResponseOutputs(BaseModel):
    """Class to represent the names of the output variables in the surface response model."""

    outputs: List[str] = Field(..., description="List of output names")


class SurfaceResponseBound(BaseModel):
    """Class to represent the bounds of a variable."""

    minmax: Tuple[float, float] = Field(..., description="Bounds (min, max) for one variable")


class SurfaceResponseBounds(BaseModel):
    """Class to represent all the bounds of the all the variables."""

    bounds: List[SurfaceResponseBound] = Field(..., description="List of bounds")


@mcp.tool()
def design_sr(
    inputs: SurfaceResponseInputs, outputs: SurfaceResponseOutputs, bounds: SurfaceResponseBounds
):
    """Design a surface response design of experiments.

    Example:
    design a pycse surface response experiment where red changes from 0.0 to 1.0,
    blue from 0.0 to 0.5 and g changes from 0.0 to 1.0 and we measure the 515nm
    channel.


    """
    global STATE

    b = [list(b.minmax) for b in bounds.bounds]
    sr = SurfaceResponse(inputs=inputs.inputs, outputs=outputs.outputs, bounds=b)

    STATE["sr"] = sr
    STATE["sr_design"] = sr.design(shuffle=False)
    return STATE["sr_design"]


class SurfaceResponseResult(BaseModel):
    """Specification for a result.

    A result is identified by an experiment number, and a corresponding float
    result.

    Note this works for only one result column.

    """

    experiment: int = Field(..., description="Experiment number")
    result: float = Field(..., description="Result as a float")


class SurfaceResponseResults(BaseModel):
    """List of the results.

    This is a list of (experiment number, result).

    """

    results: List[SurfaceResponseResult] = Field(..., description="List of results")


@mcp.tool()
def analyze_sr(data: SurfaceResponseResults):
    """Analyze the surface response results.

    Returns a table of ANOVA results.
    """
    global STATE
    results = [[d.result] for d in data.results]

    STATE["sr"].set_output(results)
    STATE["sr"].fit()
    return STATE["sr"].summary()


@mcp.tool()
def sr_parity():
    """Return a parity plot as an image.
    You must run the analyze_sr tool before this one.
    """
    global STATE
    fig = STATE["sr"].parity()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.getvalue()
    return Image(data=png_bytes, format="png")


@mcp.tool()
def pycse_help():
    """Get help about pycse functions.

    This returns a dictionary of function names and docstrings.
    """
    func_dict = {}
    for finder, modname, ispkg in pkgutil.walk_packages(
        pycse.__path__, prefix=pycse.__name__ + "."
    ):
        # This module seems to hang the function
        if "sandbox" in modname:
            continue
        try:
            print(finder, modname)
            module = importlib.import_module(modname)
        except Exception:
            # skip modules that error on import
            continue

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # only include functions actually defined in pycse
            print(name)
            if obj.__module__.startswith("pycse"):
                qualname = f"{obj.__module__}.{obj.__name__}"
                func_dict[qualname] = inspect.getdoc(obj) or ""

    s = """The following list of functions are available. They are formatted
    as function : docstring"""
    for fq, doc in func_dict.items():
        s += f"{fq} : {doc if doc else '<no doc>'}\n\n"

    return s


# Run / install / uninstall the server


def main():
    """Install, uninstall, or run the server.

    This is the cli. If you call it with install or uninstall as an argument, it
    will do that in the Claude Desktop. With no arguments it just runs the
    server.
    """
    if platform.system() == "Darwin":
        cfgfile = "~/Library/Application Support/Claude/claude_desktop_config.json"
    elif platform.system() == "Windows":
        cfgfile = r"%APPDATA%\Claude\claude_desktop_config.json"
    else:
        raise Exception("Only Mac and Windows are supported for the pycse mcp server")

    cfgfile = os.path.expandvars(cfgfile)
    cfgfile = os.path.expanduser(cfgfile)

    if os.path.exists(cfgfile):
        with open(cfgfile, "r") as f:
            cfg = json.loads(f.read())
    else:
        cfg = {}

    # Called with no arguments just run the server
    if len(sys.argv) == 1:
        mcp.run(transport="stdio")

    elif sys.argv[1] == "install":
        setup = {"command": shutil.which("pycse_mcp")}

        if "mcpServers" not in cfg:
            cfg["mcpServers"] = {}

        cfg["mcpServers"]["pycse"] = setup
        with open(cfgfile, "w") as f:
            f.write(json.dumps(cfg, indent=4))

        print(
            f"\n\nInstalled litdb. Here is your current {cfgfile}."
            " Please restart Claude Desktop."
        )
        print(json.dumps(cfg, indent=4))

    elif sys.argv[1] == "uninstall":
        if "mcpServers" not in cfg:
            cfg["mcpServers"] = {}

        if "pycse" in cfg["mcpServers"]:
            del cfg["mcpServers"]["pycse"]
            with open(cfgfile, "w") as f:
                f.write(json.dumps(cfg, indent=4))

        print(f"Uninstalled litdb. Here is your current {cfgfile}.")
        print(json.dumps(cfg, indent=4))

    else:
        print("I am not sure what you are trying to do. Please use install or uninstall.")
