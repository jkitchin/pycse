"""A proof of concept pycse MCP server.

1. provide help with pycse functions.
2. a set of DOE functions.
3. tools for python docs

"""

import platform
import sys
import shutil
import os
import json
import logging
import re
import pydoc
from io import StringIO
from mcp.server.fastmcp import FastMCP, Image
from typing import Tuple, List, Union, Dict, Any, Optional, Pattern
from pydantic import BaseModel, Field
import pandas as pd
import pycse
import pkgutil
import importlib
import inspect
import io

from pycse.sklearn.lhc import LatinSquare
from pycse.sklearn.surface_response import SurfaceResponse
from pycse.sklearn.dpose import DPOSE

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
def design_lhc(inputs: LatinSquareSpec) -> List[Dict[str, Any]]:
    """Design a LatinSquare design from the inputs.

    inputs is a list of tuples of the form: (varname, levels).

    For example, you might specify:

    Create a Latin square where Red is 0, 0.5, 1, Green is 0.0, 0.5, 1, and Blue
    is [0, 0.5, 1] and we measure 515nm at the output.

    This function returns the experiments that you should do as a pandas DataFrame.
    """
    # I think this is a little clunky for saving state

    factors = inputs.factors

    d = {factor.name: factor.levels for factor in factors}

    ls = LatinSquare(d)
    STATE["ls"] = ls

    design = ls.design()
    STATE["design"] = design

    return design.to_dict(orient="records")


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
def analyze_lhc(lsr: LatinSquareResults) -> List[Dict[str, Any]]:
    """Analyze the LatinSquare results.

    The results have to be provided in a way that looks like a list of
    (experiment #, result) can be parsed by the LLM.

    Returns an analysis of variance (ANOVA).

    """

    df = pd.DataFrame(
        [(result.experiment, result.result) for result in lsr.results],
        columns=["Experiment", "Result"],
    )

    merged = STATE["design"].merge(df, left_index=True, right_on="Experiment")

    ls = STATE["ls"]

    X = merged[ls.labels]
    y = merged["Result"]

    ls.fit(X, y)
    return ls.anova().to_dict(orient="records")


# * Surface Response tools


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
    inputs: SurfaceResponseInputs,
    outputs: SurfaceResponseOutputs,
    bounds: SurfaceResponseBounds,
) -> List[Dict[str, Any]]:
    """Design a surface response design of experiments.

    Example:
    design a pycse surface response experiment where red changes from 0.0 to 1.0,
    blue from 0.0 to 0.5 and g changes from 0.0 to 1.0 and we measure the 515nm
    channel.


    """

    b = [list(b.minmax) for b in bounds.bounds]
    sr = SurfaceResponse(inputs=inputs.inputs, outputs=outputs.outputs, bounds=b)

    STATE["sr"] = sr
    STATE["sr_design"] = sr.design(shuffle=False)
    return STATE["sr_design"].to_dict(orient="records")


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
def analyze_sr(data: SurfaceResponseResults) -> str:
    """Analyze the surface response results.

    Returns a table of ANOVA results.
    """
    results = [[d.result] for d in data.results]

    STATE["sr"].set_output(results)
    STATE["sr"].fit()
    return STATE["sr"].summary()


@mcp.tool()
def sr_parity() -> Image:
    """Return a parity plot as an image.

    You must run the analyze_sr tool before this one.
    """
    fig = STATE["sr"].parity()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.getvalue()
    return Image(data=png_bytes, format="png")


@mcp.tool()
def random_image(n: int = 10) -> Image:
    """Return a random image with N points in it."""
    import numpy as np
    import matplotlib.pyplot as plt

    plt.plot(np.random.rand(n))
    fig = plt.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.getvalue()
    return Image(data=png_bytes, format="png")


@mcp.tool()
def pycse_help() -> str:
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


# * Function help


@mcp.tool()
def get_pydoc_help(func: str) -> str:
    """Use pydoc to get help documentation on func.

    Args:
        func: Function object or string name of function

    Returns:
        str: Help documentation as string
    """
    # Capture pydoc output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        pydoc.help(func)
        help_text = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    return help_text


@mcp.tool()
def search_functions(pattern: str) -> str:
    """
    Search for functions matching a pattern using pydoc.

    Args:
        pattern (str): Search pattern

    Returns:
        str: Search results
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        pydoc.apropos(pattern)
        search_results = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    return search_results


@mcp.tool()
def get_function_source(qualified_name: str) -> Tuple[str, Optional[str]]:
    """
    Retrieve the source code for a function given its fully qualified name.

    Parameters
    ----------
    qualified_name : str
        The fully qualified name of the function (e.g., 'numpy.linalg.solve',
        'scipy.optimize.minimize', 'pycse.nlinfit')

    Returns
    -------
    tuple
        A tuple containing (source_code, error_message)
        - source_code: str - The source code of the function, or None if error
        - error_message: str - Error message if retrieval failed, or None if successful

    Examples
    --------
    >>> source, error = get_function_source('numpy.mean')
    >>> if error is None:
    ...     print(source)

    >>> source, error = get_function_source('scipy.optimize.minimize')
    >>> if error:
    ...     print(f"Error: {error}")
    """
    try:
        # Split the qualified name into module path and function name
        parts = qualified_name.split(".")
        if len(parts) < 2:
            return None, "Function name must be fully qualified (e.g., 'module.function')"

        function_name = parts[-1]
        module_path = ".".join(parts[:-1])

        # Import the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            return None, f"Could not import module '{module_path}': {str(e)}"

        # Navigate through nested attributes if needed
        # Handle cases like 'numpy.linalg.solve' where we need to go deeper
        current_obj = module
        for part in parts[len(module_path.split(".")) : -1]:  # noqa:E203
            if hasattr(current_obj, part):
                current_obj = getattr(current_obj, part)
            else:
                return None, f"Module '{module_path}' has no attribute '{part}'"

        # Get the function object
        if hasattr(current_obj, function_name):
            func_obj = getattr(current_obj, function_name)
        else:
            return None, f"Object has no attribute '{function_name}'"

        # Check if it's callable
        if not callable(func_obj):
            return None, f"'{qualified_name}' is not a callable function"

        # Try to get the source code
        try:
            source = inspect.getsource(func_obj)
            return source, None
        except OSError as e:
            # This happens when source is not available (built-in functions, C extensions, etc.)
            return None, f"Source code not available for '{qualified_name}': {str(e)}"
        except Exception as e:
            return None, f"Error retrieving source for '{qualified_name}': {str(e)}"

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


@mcp.tool()
def get_function_info(qualified_name: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Get comprehensive information about a function including source, signature, and docstring.

    Parameters
    ----------
    qualified_name : str
        The fully qualified name of the function

    Returns
    -------
    tuple
        A tuple containing (info_dict, error_message)
        - info_dict: dict containing 'source', 'signature', 'docstring', 'module', 'file'
        - error_message: str if error occurred, None if successful
    """
    try:
        # Split the qualified name
        parts = qualified_name.split(".")
        if len(parts) < 2:
            return None, "Function name must be fully qualified (e.g., 'module.function')"

        function_name = parts[-1]
        module_path = ".".join(parts[:-1])

        # Import the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            return None, f"Could not import module '{module_path}': {str(e)}"

        # Navigate to the function
        current_obj = module
        for part in parts[len(module_path.split(".")) : -1]:  # noqa:E203
            if hasattr(current_obj, part):
                current_obj = getattr(current_obj, part)
            else:
                return None, f"Module '{module_path}' has no attribute '{part}'"

        # Get the function object
        if hasattr(current_obj, function_name):
            func_obj = getattr(current_obj, function_name)
        else:
            return None, f"Object has no attribute '{function_name}'"

        if not callable(func_obj):
            return None, f"'{qualified_name}' is not a callable function"

        # Collect information
        info = {
            "name": qualified_name,
            "module": module_path,
            "docstring": inspect.getdoc(func_obj),
        }

        # Try to get signature
        try:
            info["signature"] = str(inspect.signature(func_obj))
        except (ValueError, TypeError):
            info["signature"] = "Signature not available"

        # Try to get source file
        try:
            info["file"] = inspect.getfile(func_obj)
        except (OSError, TypeError):
            info["file"] = "File location not available"

        # Try to get source code
        try:
            info["source"] = inspect.getsource(func_obj)
        except OSError:
            info["source"] = "Source code not available (built-in or C extension)"
        except Exception as e:
            info["source"] = f"Error retrieving source: {str(e)}"

        return info, None

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


logger = logging.getLogger(__name__)


class AproposResult(BaseModel):
    functions: Dict[str, str] = Field(default_factory=dict)
    classes: Dict[str, str] = Field(default_factory=dict)
    modules: Dict[str, str] = Field(default_factory=dict)
    methods: Optional[Dict[str, str]] = Field(default_factory=dict)
    errors: Optional[Dict[str, str]] = Field(
        default_factory=dict
    )  # Optional: module_name -> error msg


@mcp.tool()
def search_with_apropos(
    keywords: Union[str, List[str]],
    modules_to_search: List[str],
    case_sensitive: bool = False,
    include_methods: bool = True,
    include_errors: bool = False,
) -> AproposResult:
    """
    Search for functions, classes, and modules using their docstrings and names.

    Parameters:
    -----------
    keywords : str or list of str
        Search keywords. Multiple keywords are treated as AND conditions.
    modules_to_search : list of str
        List of module names to search. Required for MCP safety.
    case_sensitive : bool
        Whether the search is case sensitive.
    include_methods : bool
        Whether to include methods in the results.
    include_errors : bool
        Whether to include modules that failed with error messages.

    Returns:
    --------
    AproposResult: Structured result object.
    """

    if isinstance(keywords, str):
        keywords = [keywords]

    patterns: List[Pattern[str]] = [
        re.compile(re.escape(kw), 0 if case_sensitive else re.IGNORECASE) for kw in keywords
    ]

    def matches(text: str) -> bool:
        return all(p.search(text) for p in patterns)

    results = AproposResult()

    for module_name in modules_to_search:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                raise ModuleNotFoundError(f"No module spec found for '{module_name}'")

            module = importlib.import_module(module_name)
            module_doc = getattr(module, "__doc__", "") or ""
            module_info = (
                f"{module_name}: {module_doc.split('.')[0] if module_doc else 'No description'}"
            )

            if matches(module_name) or matches(module_doc):
                results.modules[module_name] = module_info.strip()

            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue

                try:
                    attr = getattr(module, attr_name)
                    attr_doc = getattr(attr, "__doc__", "") or ""
                    full_name = f"{module_name}.{attr_name}"
                    search_text = f"{attr_name} {attr_doc}"

                    if not matches(search_text):
                        continue

                    attr_desc = attr_doc.split(".")[0] if attr_doc else "No description"
                    desc = f"{full_name}: {attr_desc}".strip()

                    if callable(attr):
                        if isinstance(attr, type):
                            results.classes[full_name] = desc
                        elif hasattr(attr, "__self__") and include_methods:
                            results.methods[full_name] = desc
                        else:
                            results.functions[full_name] = desc

                except Exception as sub_err:
                    if include_errors:
                        results.errors[f"{module_name}.{attr_name}"] = str(sub_err)
                    continue

        except Exception as mod_err:
            if include_errors:
                results.errors[module_name] = str(mod_err)
            logger.warning(f"Error processing module {module_name}: {mod_err}")
            continue

    if not include_methods:
        results.methods = None

    return results


# * DPOSE - Direct Propagation of Shallow Ensembles


class DPOSESpec(BaseModel):
    """Specification for a DPOSE model."""

    layers: Tuple[int, int, int] = Field(
        ..., description="Network architecture: (input_dim, hidden_dim, n_ensemble)"
    )
    loss_type: str = Field(default="crps", description="Loss type: 'crps', 'nll', or 'mse'")
    activation: str = Field(
        default="tanh", description="Activation function: 'tanh', 'relu', 'softplus', 'elu'"
    )
    optimizer: str = Field(
        default="bfgs", description="Optimizer: 'bfgs', 'lbfgs', 'adam', 'sgd', 'muon'"
    )
    maxiter: int = Field(default=500, description="Maximum training iterations")
    seed: int = Field(default=42, description="Random seed for reproducibility")


@mcp.tool()
def dpose_info() -> str:
    """Get information about DPOSE and usage examples.

    Returns:
        str: Information about DPOSE, its features, and example usage
    """
    info = """
DPOSE (Direct Propagation of Shallow Ensembles)
================================================

A neural network ensemble method for uncertainty quantification.

Key Features:
- Per-sample uncertainty estimates (heteroscedastic)
- Shallow ensemble architecture (only last layer differs)
- CRPS or NLL loss for calibrated uncertainties
- Uncertainty propagation through transformations
- Handles gaps and extrapolation

Architecture:
- Input layer: matches data dimension
- Hidden layer: typically 15-50 units
- Ensemble: typically 32 members

Example Workflow:
1. Prepare data: X_train (2D), y_train (1D)
2. Train model: use fit() or create via Python
3. Get predictions: predict() returns mean
4. Get uncertainty: predict() with return_std=True

Recommended Settings:
- layers: (n_features, 50, 32) for robust fitting
- loss_type: 'crps' (default, robust)
- activation: 'tanh' (smooth functions)
- optimizer: 'bfgs' (default, fast for small data)
- maxiter: 500 (adjust based on convergence)

Reference:
Kellner, M., & Ceriotti, M. (2024). Uncertainty quantification
by direct propagation of shallow ensembles.
Machine Learning: Science and Technology, 5(3), 035006.
"""
    return info


@mcp.tool()
def dpose_example_code() -> str:
    """Get example Python code for using DPOSE.

    Returns:
        str: Complete example code showing how to use DPOSE
    """
    # This uses DPOSE to ensure it's not an unused import
    example = f"""
# Example: Using DPOSE for Uncertainty Quantification
# ====================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pycse.sklearn.dpose import {DPOSE.__name__}

# 1. Generate example data with heteroscedastic noise
np.random.seed(42)
x = np.linspace(0, 1, 200)[:, None]
noise = 0.01 + 0.15 * x.ravel()  # Increasing noise
y = 2 * x.ravel() + noise * np.random.randn(200)

# 2. Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 3. Create and train DPOSE model with StandardScaler
model = Pipeline([
    ('scaler', StandardScaler()),
    ('dpose', DPOSE(
        layers=(1, 50, 32),  # (input, hidden, ensemble)
        loss_type='crps',    # CRPS loss (recommended)
        activation='tanh',   # Tanh activation
        maxiter=500,         # Training iterations
        seed=42
    ))
])

# 4. Fit the model
model.fit(x_train, y_train)

# 5. Make predictions with uncertainty
x_test_scaled = model.named_steps['scaler'].transform(x_test)
y_pred, y_std = model.named_steps['dpose'].predict(
    x_test_scaled,
    return_std=True
)

# 6. Evaluate
mae = np.abs(y_test - model.predict(x_test)).mean()
print(f"MAE: {{mae:.6f}}")
print(f"Uncertainty range: [{{y_std.min():.4f}}, {{y_std.max():.4f}}]")

# 7. For uncertainty propagation through transformations
ensemble = model.named_steps['dpose'].predict_ensemble(x_test_scaled)
z_ensemble = np.exp(ensemble)  # Apply transformation
z_mean = z_ensemble.mean(axis=1)
z_std = z_ensemble.std(axis=1)
"""
    return example


# * Run / install / uninstall the server


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
            f"\n\nInstalled litdb. Here is your current {cfgfile}. Please restart Claude Desktop."
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
