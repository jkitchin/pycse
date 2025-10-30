#!/usr/bin/env python
"""CLI for pycse.

A Click-based CLI for managing pycse Docker containers and MCP server.
"""

import os
import platform
import sys
import uuid
import shutil
import json
import subprocess
import time
import webbrowser

import numpy as np
import requests
import click


@click.group()
@click.version_option()
def pycse():
    """pycse CLI - Manage Docker-based Jupyter Lab and MCP server."""
    pass


@pycse.command()
@click.option(
    "--working-dir",
    "-w",
    default=None,
    type=click.Path(exists=True),
    help="Working directory to mount (defaults to current directory)",
)
def launch(working_dir):
    """Launch Jupyter Lab in a Docker container (default command)."""
    if shutil.which("docker") is None:
        click.secho(
            "Error: docker was not found. Please install it from https://www.docker.com/",
            fg="red",
            err=True,
        )
        sys.exit(1)

    # Check setup and get image if needed
    try:
        subprocess.run(
            ["docker", "image", "inspect", "jkitchin/pycse"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        click.echo("Docker image not found. Pulling jkitchin/pycse...")
        subprocess.run(["docker", "pull", "jkitchin/pycse"], check=True)

    # Check if the container is already running
    p = subprocess.run(["docker", "ps", "--format", '"{{.Names}}"'], capture_output=True)
    if "pycse" in p.stdout.decode("utf-8"):
        if click.confirm("A pycse container is already running. Kill it?"):
            subprocess.run(["docker", "rm", "-f", "pycse"])
        else:
            click.echo("Connecting to existing container...")
            # Get the port
            p = subprocess.run(["docker", "port", "pycse", "8888"], capture_output=True)
            output = p.stdout.decode("utf-8").strip()
            PORT = output.split(":")[-1]

            # Get the token
            p = subprocess.run(
                ["docker", "exec", "pycse", "printenv", "JUPYTER_TOKEN"],
                capture_output=True,
            )
            JUPYTER_TOKEN = p.stdout.decode("utf-8").strip()

            url = f"http://localhost:{PORT}/lab?token={JUPYTER_TOKEN}"
            webbrowser.open(url)
            return

    # Start a new container
    PWD = working_dir or os.getcwd()
    PORT = np.random.randint(8000, 9000)

    JUPYTER_TOKEN = str(uuid.uuid4())
    os.environ["JUPYTER_TOKEN"] = JUPYTER_TOKEN

    cmd = (
        f"docker run -d --name pycse -it --rm -p {PORT}:8888 "
        f"-e JUPYTER_TOKEN -v {PWD}:/home/jovyan/work "
        "jkitchin/pycse"
    )

    click.echo("Starting Jupyter Lab. Press Ctrl+C to quit.")
    subprocess.Popen(cmd.split())

    time.sleep(2)

    url = f"http://localhost:{PORT}/lab?token={JUPYTER_TOKEN}"

    # Wait for the server to be ready
    for i in range(10):
        try:
            if requests.get(url).status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    webbrowser.open(url)
    subprocess.run(["docker", "attach", "pycse"])


@pycse.command()
def pull():
    """Pull the latest pycse Docker image."""
    if shutil.which("docker") is None:
        click.secho(
            "Error: docker was not found. Please install it from https://www.docker.com/",
            fg="red",
            err=True,
        )
        sys.exit(1)

    click.echo("Pulling latest jkitchin/pycse Docker image...")
    try:
        subprocess.run(["docker", "pull", "jkitchin/pycse:latest"], check=True)
        click.secho("Successfully pulled the latest image!", fg="green")
    except subprocess.CalledProcessError:
        click.secho("Failed to pull the Docker image.", fg="red", err=True)
        sys.exit(1)


@pycse.command()
@click.option("--force", "-f", is_flag=True, help="Force removal without confirmation")
def rm(force):
    """Remove a stuck pycse container."""
    if shutil.which("docker") is None:
        click.secho(
            "Error: docker was not found. Please install it from https://www.docker.com/",
            fg="red",
            err=True,
        )
        sys.exit(1)

    # Check if container exists
    p = subprocess.run(["docker", "ps", "-a", "--format", '"{{.Names}}"'], capture_output=True)
    if "pycse" not in p.stdout.decode("utf-8"):
        click.echo("No pycse container found.")
        return

    if not force and not click.confirm("Remove the pycse container?"):
        click.echo("Cancelled.")
        return

    try:
        subprocess.run(["docker", "rm", "-f", "pycse"], check=True)
        click.secho("Successfully removed pycse container!", fg="green")
    except subprocess.CalledProcessError:
        click.secho("Failed to remove the container.", fg="red", err=True)
        sys.exit(1)


@pycse.group()
def mcp():
    """Manage pycse MCP server for Claude Desktop."""
    pass


def get_mcp_config_path():
    """Get the path to Claude Desktop config file."""
    if platform.system() == "Darwin":
        cfgfile = "~/Library/Application Support/Claude/claude_desktop_config.json"
    elif platform.system() == "Windows":
        cfgfile = r"%APPDATA%\Claude\claude_desktop_config.json"
    else:
        click.secho(
            "Error: Only macOS and Windows are supported for the pycse MCP server.",
            fg="red",
            err=True,
        )
        sys.exit(1)

    cfgfile = os.path.expandvars(cfgfile)
    cfgfile = os.path.expanduser(cfgfile)
    return cfgfile


@mcp.command()
def install_mcp():
    """Install pycse MCP server in Claude Desktop."""
    cfgfile = get_mcp_config_path()

    if os.path.exists(cfgfile):
        with open(cfgfile, "r") as f:
            cfg = json.loads(f.read())
    else:
        cfg = {}

    setup = {"command": shutil.which("pycse_mcp")}

    if "mcpServers" not in cfg:
        cfg["mcpServers"] = {}

    if "pycse" in cfg["mcpServers"]:
        click.secho("Warning: pycse MCP server is already installed.", fg="yellow")
        if not click.confirm("Overwrite the existing configuration?"):
            click.echo("Cancelled.")
            return

    cfg["mcpServers"]["pycse"] = setup

    with open(cfgfile, "w") as f:
        f.write(json.dumps(cfg, indent=4))

    click.secho("\nSuccessfully installed pycse MCP server in Claude Desktop!", fg="green")
    click.echo(f"\nConfig file: {cfgfile}")
    click.echo(json.dumps(cfg, indent=4))
    click.echo("\n" + click.style("Please restart Claude Desktop.", fg="yellow", bold=True))


@mcp.command()
def uninstall_mcp():
    """Uninstall pycse MCP server from Claude Desktop."""
    cfgfile = get_mcp_config_path()

    if os.path.exists(cfgfile):
        with open(cfgfile, "r") as f:
            cfg = json.loads(f.read())
    else:
        click.echo("No Claude Desktop config file found.")
        return

    if "mcpServers" not in cfg or "pycse" not in cfg["mcpServers"]:
        click.echo("pycse MCP server is not installed.")
        return

    del cfg["mcpServers"]["pycse"]

    with open(cfgfile, "w") as f:
        f.write(json.dumps(cfg, indent=4))

    click.secho("\nSuccessfully uninstalled pycse MCP server!", fg="green")
    click.echo(f"\nConfig file: {cfgfile}")
    click.echo(json.dumps(cfg, indent=4))


@pycse.group()
def skill():
    """Manage pycse skill for Claude Code."""
    pass


@skill.command()
def install_skill():
    """Install pycse skill in Claude Code (~/.claude/skills/pycse/)."""
    # Get the path to the SKILL.md file in the package
    import pycse
    import pathlib

    skill_source = pathlib.Path(pycse.__file__).parent / "SKILL.md"

    if not skill_source.exists():
        click.secho(f"Error: SKILL.md not found at {skill_source}", fg="red", err=True)
        sys.exit(1)

    # Create the skills directory
    skills_dir = pathlib.Path.home() / ".claude" / "skills" / "pycse"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Copy the SKILL.md file
    skill_dest = skills_dir / "SKILL.md"

    if skill_dest.exists():
        click.secho("Warning: pycse skill is already installed.", fg="yellow")
        if not click.confirm("Overwrite the existing skill?"):
            click.echo("Cancelled.")
            return

    shutil.copy2(skill_source, skill_dest)

    click.secho("\nSuccessfully installed pycse skill in Claude Code!", fg="green")
    click.echo(f"\nSkill location: {skill_dest}")
    click.echo("\n" + click.style("Restart Claude Code to use the skill.", fg="yellow", bold=True))


@skill.command()
def uninstall_skill():
    """Uninstall pycse skill from Claude Code."""
    import pathlib

    skills_dir = pathlib.Path.home() / ".claude" / "skills" / "pycse"

    if not skills_dir.exists():
        click.echo("pycse skill is not installed.")
        return

    if not click.confirm(f"Remove {skills_dir}?"):
        click.echo("Cancelled.")
        return

    shutil.rmtree(skills_dir)
    click.secho("\nSuccessfully uninstalled pycse skill!", fg="green")


if __name__ == "__main__":
    pycse()
