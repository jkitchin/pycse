"""Tests for the pycse MCP server module (pycse.mcp).

These exercise the tool functions directly and the ``pycse_mcp`` entry point
with mocking; no server is actually started.
"""

import asyncio
import importlib
import subprocess
import sys
import textwrap
from unittest.mock import patch

import pytest

m = pytest.importorskip("pycse.mcp")


def test_pycse_help_writes_nothing_to_stdout(capsys):
    """stdout is the JSON-RPC stream on stdio transport; tools must not print."""
    real_import = importlib.import_module

    def noisy_import(name, *args, **kwargs):
        print(f"noise while importing {name}")  # e.g. a module printing on import
        return real_import(name, *args, **kwargs)

    with patch.object(m.pkgutil, "walk_packages", return_value=[(None, "pycse.utils", False)]):
        with patch.object(m.importlib, "import_module", side_effect=noisy_import):
            text = m.pycse_help()

    out, err = capsys.readouterr()
    assert out == ""
    assert "noise while importing" in err
    assert isinstance(text, str)
    assert "pycse.utils.feq" in text


def test_module_has_no_bare_prints_in_tools():
    """Only main()'s install/uninstall paths may print to stdout."""
    import inspect

    for name in ["pycse_help", "get_pydoc_help", "search_functions", "design_latin_square"]:
        assert "print(" not in inspect.getsource(getattr(m, name)), name


def test_main_runs_server_on_linux():
    """Running the server must not depend on the platform."""
    with patch("platform.system", return_value="Linux"):
        with patch.object(sys, "argv", ["pycse_mcp"]):
            with patch.object(m.mcp, "run") as run:
                m.main()
    run.assert_called_once_with(transport="stdio")


def test_main_install_on_linux_is_a_clean_error(capsys):
    with patch("platform.system", return_value="Linux"):
        with patch.object(sys, "argv", ["pycse_mcp", "install"]):
            with pytest.raises(SystemExit) as exc:
                m.main()
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "macOS and Windows" in err


@pytest.mark.parametrize("cmd", ["install", "uninstall"])
def test_main_install_uninstall_messages(cmd, tmp_path, capsys):
    cfg = tmp_path / "claude_desktop_config.json"
    with patch.object(m, "claude_desktop_config_path", return_value=str(cfg)):
        with patch.object(sys, "argv", ["pycse_mcp", cmd]):
            m.main()
    out = capsys.readouterr().out
    assert "litdb" not in out
    assert "pycse MCP server" in out


def test_main_reports_missing_mcp_extra(capsys):
    with patch.object(m, "_MCP_IMPORT_ERROR", ImportError("No module named 'mcp'")):
        with patch.object(sys, "argv", ["pycse_mcp"]):
            with pytest.raises(SystemExit) as exc:
                m.main()
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert 'pip install "pycse[mcp]"' in err


@pytest.mark.slow  # spawns a fresh interpreter (~1 s)
def test_entry_point_without_mcp_package():
    """With mcp not installed, pycse_mcp prints a hint instead of a traceback."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["mcp"] = None  # simulate the [mcp] extra not being installed
        sys.argv = ["pycse_mcp"]
        from pycse.mcp import main
        main()
        """
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.returncode == 1, p.stderr
    assert 'pip install "pycse[mcp]"' in p.stderr
    assert "Traceback" not in p.stderr


def test_factor_accepts_categorical_levels():
    f = m.Factor(name="Catalyst", levels=["A", "B", "C"])
    assert f.levels == ["A", "B", "C"]
    f = m.Factor(name="T", levels=[1, 2.5, 3])
    assert f.levels == [1, 2.5, 3]


def test_design_latin_square_and_alias():
    spec = m.LatinSquareSpec(
        factors=[
            m.Factor(name="T", levels=[20, 40, 60]),
            m.Factor(name="P", levels=[1, 2, 3]),
            m.Factor(name="Catalyst", levels=["A", "B", "C"]),
        ]
    )
    records = m.design_latin_square(spec)
    assert len(records) == 9
    assert {r["Catalyst"] for r in records} == {"A", "B", "C"}
    assert m.design_lhc(spec) == records


def test_latin_square_tools_registered():
    pytest.importorskip("mcp")
    tools = {t.name: t for t in asyncio.run(m.mcp.list_tools())}
    for name in ["design_latin_square", "analyze_latin_square"]:
        assert name in tools
        assert "deprecated" not in tools[name].description.lower()
    for name in ["design_lhc", "analyze_lhc"]:
        assert name in tools
        assert "Latin square; deprecated alias" in tools[name].description


def test_pycse_help_skips_modules_with_broken_lazy_attributes():
    """A module whose attribute access fails (missing optional dep) is skipped."""
    import types

    class Broken(types.ModuleType):
        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named 'torch'")

        def __dir__(self):
            return ["SomeLazyModel"]

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "pycse.broken":
            return Broken(name)
        return real_import(name, *args, **kwargs)

    walk = [(None, "pycse.broken", False), (None, "pycse.utils", False)]
    with patch.object(m.pkgutil, "walk_packages", return_value=walk):
        with patch.object(m.importlib, "import_module", side_effect=fake_import):
            text = m.pycse_help()
    assert "pycse.utils.feq" in text
