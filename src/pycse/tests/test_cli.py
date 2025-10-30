"""Tests for CLI module.

The CLI module provides a Docker-based Jupyter lab launcher. These tests
use mocking to avoid requiring Docker to be installed and running.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import subprocess
from click.testing import CliRunner


class TestCLI:
    """Tests for pycse CLI functionality."""

    @pytest.fixture
    def cli_runner(self):
        """Provide a Click CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def mock_docker_available(self):
        """Mock docker being available in PATH."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/docker"
            yield mock_which

    @pytest.fixture
    def mock_docker_unavailable(self):
        """Mock docker not being available in PATH."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            yield mock_which

    @pytest.fixture
    def mock_subprocess_run(self):
        """Mock subprocess.run to avoid actual Docker calls."""
        with patch("subprocess.run") as mock_run:
            # Default: image exists, no containers running
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = b""
            mock_run.return_value = mock_result
            yield mock_run

    @pytest.fixture
    def mock_subprocess_popen(self):
        """Mock subprocess.Popen to avoid starting containers."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            yield mock_popen

    @pytest.fixture
    def mock_requests_get(self):
        """Mock requests.get to simulate Jupyter server response."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            yield mock_get

    @pytest.fixture
    def mock_webbrowser(self):
        """Mock webbrowser.open to avoid opening browser."""
        with patch("webbrowser.open") as mock_open:
            yield mock_open

    @pytest.fixture
    def mock_time_sleep(self):
        """Mock time.sleep to speed up tests."""
        with patch("time.sleep") as mock_sleep:
            yield mock_sleep

    def test_module_imports(self):
        """Test that the CLI module imports without error."""
        from pycse import cli

        assert hasattr(cli, "pycse")

    def test_pycse_function_exists(self):
        """Test that pycse() function is accessible."""
        from pycse.cli import pycse

        assert callable(pycse)

    def test_docker_not_found_raises_exception(self, cli_runner, mock_docker_unavailable):
        """Test that missing Docker raises an exception."""
        from pycse.cli import pycse

        result = cli_runner.invoke(pycse, ["launch"])
        assert result.exit_code == 1
        assert "docker was not found" in result.output

    def test_image_check_when_available(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that image check runs when Docker is available."""
        from pycse.cli import pycse

        # Mock: image exists
        mock_subprocess_run.return_value.returncode = 0

        cli_runner.invoke(pycse, ["launch"])

        # Should have called docker image inspect
        calls = mock_subprocess_run.call_args_list
        inspect_call = any("image" in str(call) and "inspect" in str(call) for call in calls)
        assert inspect_call, "Should check if image exists"

    def test_image_pull_when_missing(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that image is pulled when not available."""
        from pycse.cli import pycse

        # Mock: image inspect fails (image doesn't exist)
        def run_side_effect(*args, **kwargs):
            cmd = args[0]
            if "inspect" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = b""
            return result

        mock_subprocess_run.side_effect = run_side_effect

        cli_runner.invoke(pycse, ["launch"])

        # Should have called docker pull
        calls = mock_subprocess_run.call_args_list
        pull_call = any("pull" in str(call) for call in calls)
        assert pull_call, "Should pull image when it doesn't exist"

    def test_existing_container_no_kill(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test behavior when container exists and user doesn't want to kill it."""
        from pycse.cli import pycse

        # Mock: container is running
        def run_side_effect(*args, **kwargs):
            cmd = args[0]
            result = MagicMock()
            result.returncode = 0

            if "ps" in cmd:
                result.stdout = b'"pycse"'
            elif "port" in cmd:
                result.stdout = b"0.0.0.0:8888"
            elif "printenv" in cmd:
                result.stdout = b"test-token-123"
            else:
                result.stdout = b""

            return result

        mock_subprocess_run.side_effect = run_side_effect

        # Mock user declining to kill container
        with patch("click.confirm", return_value=False):
            cli_runner.invoke(pycse, ["launch"])

        # Should have opened browser with existing token
        mock_webbrowser.assert_called_once()
        url = mock_webbrowser.call_args[0][0]
        assert "token=test-token-123" in url
        assert "8888" in url

    def test_existing_container_with_kill(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that existing container is killed when user confirms."""
        from pycse.cli import pycse

        # Mock: container is running first time, then not
        call_count = [0]

        def run_side_effect(*args, **kwargs):
            cmd = args[0]
            result = MagicMock()
            result.returncode = 0

            if "ps" in cmd:
                call_count[0] += 1
                # First call: container exists, second call: doesn't exist
                result.stdout = b'"pycse"' if call_count[0] == 1 else b""
            else:
                result.stdout = b""

            return result

        mock_subprocess_run.side_effect = run_side_effect

        # Mock user confirming to kill container
        with patch("click.confirm", return_value=True):
            cli_runner.invoke(pycse, ["launch"])

        # Should have called docker rm -f pycse
        calls = [str(call) for call in mock_subprocess_run.call_args_list]
        rm_call = any("rm" in call and "-f" in call and "pycse" in call for call in calls)
        assert rm_call, "Should kill existing container when user confirms"

    def test_new_container_startup(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test starting a new container."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        cli_runner.invoke(pycse, ["launch"])

        # Should have called Popen to start container
        mock_subprocess_popen.assert_called_once()

        # Check that command includes required elements
        popen_cmd = mock_subprocess_popen.call_args[0][0]
        assert "docker" in popen_cmd[0]
        assert "run" in popen_cmd
        assert "-d" in popen_cmd  # Detached mode
        assert "--name" in popen_cmd
        assert "pycse" in popen_cmd
        assert "-p" in popen_cmd  # Port mapping

    def test_port_assignment(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that a random port is assigned in the correct range."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        with patch("numpy.random.randint") as mock_randint:
            mock_randint.return_value = 8765
            cli_runner.invoke(pycse, ["launch"])

            # Should call randint with correct range
            mock_randint.assert_called_once_with(8000, 9000)

            # Port should appear in the docker command
            popen_cmd = mock_subprocess_popen.call_args[0][0]
            assert "8765:8888" in " ".join(popen_cmd)

    def test_jupyter_token_generation(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that a Jupyter token is generated and used."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = "test-uuid-12345"
            cli_runner.invoke(pycse, ["launch"])

            # Token should be in browser URL
            url = mock_webbrowser.call_args[0][0]
            assert "token=test-uuid-12345" in url

    def test_volume_mount_current_directory(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that current directory is mounted as volume."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        cli_runner.invoke(pycse, ["launch"])

        # Check volume mount in command
        popen_cmd = mock_subprocess_popen.call_args[0][0]
        cmd_str = " ".join(popen_cmd)

        assert "-v" in popen_cmd
        # Should mount current directory to /home/jovyan/work
        cwd = os.getcwd()
        assert f"{cwd}:/home/jovyan/work" in cmd_str

    def test_webbrowser_opens_correct_url(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that browser opens with correct URL."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        with patch("numpy.random.randint", return_value=8765):
            with patch("uuid.uuid4", return_value="test-token"):
                cli_runner.invoke(pycse, ["launch"])

                # Should open browser with correct URL
                mock_webbrowser.assert_called_once()
                url = mock_webbrowser.call_args[0][0]

                assert url.startswith("http://localhost:8765/lab")
                assert "token=test-token" in url

    def test_waits_for_jupyter_server(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that CLI waits for Jupyter server to be ready."""
        from pycse.cli import pycse

        # Mock: no existing container, server takes 3 tries to be ready
        mock_subprocess_run.return_value.stdout = b""

        call_count = [0]

        def get_side_effect(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            # First 2 calls fail, 3rd succeeds
            response.status_code = 200 if call_count[0] >= 3 else 500
            return response

        mock_requests_get.side_effect = get_side_effect

        cli_runner.invoke(pycse, ["launch"])

        # Should have made at least 3 requests
        assert mock_requests_get.call_count >= 3

        # Should have slept between attempts
        assert mock_time_sleep.call_count >= 2

    def test_attaches_to_container(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that CLI attaches to the container at the end."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        cli_runner.invoke(pycse, ["launch"])

        # Should have called docker attach pycse as the last subprocess call
        last_call = mock_subprocess_run.call_args_list[-1]
        cmd = last_call[0][0]
        assert "docker" in cmd
        assert "attach" in cmd
        assert "pycse" in cmd

    def test_image_name(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that correct Docker image name is used."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        cli_runner.invoke(pycse, ["launch"])

        # Should use jkitchin/pycse image
        popen_cmd = mock_subprocess_popen.call_args[0][0]
        cmd_str = " ".join(popen_cmd)
        assert "jkitchin/pycse" in cmd_str

    def test_container_flags(
        self,
        cli_runner,
        mock_docker_available,
        mock_subprocess_run,
        mock_subprocess_popen,
        mock_requests_get,
        mock_webbrowser,
        mock_time_sleep,
    ):
        """Test that correct Docker flags are used."""
        from pycse.cli import pycse

        # Mock: no existing container
        mock_subprocess_run.return_value.stdout = b""

        cli_runner.invoke(pycse, ["launch"])

        popen_cmd = mock_subprocess_popen.call_args[0][0]
        # Should use -it (interactive terminal), --rm (remove on exit)
        assert "-it" in popen_cmd
        assert "--rm" in popen_cmd

    def test_main_entrypoint_exists(self):
        """Test that cli.py has __main__ entrypoint."""
        from pycse import cli
        import inspect

        # Read the source to check for __main__ block
        source = inspect.getsource(cli)
        assert 'if __name__ == "__main__"' in source
        assert "pycse()" in source


class TestCLIEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_docker_available(self):
        """Mock docker being available."""
        with patch("shutil.which", return_value="/usr/local/bin/docker"):
            yield

    def test_image_inspect_exception_handling(self):
        """Test that exception during image inspect triggers pull."""
        from pycse.cli import pycse

        runner = CliRunner()
        with patch("shutil.which", return_value="/usr/local/bin/docker"):
            with patch("subprocess.run") as mock_run:
                with patch("subprocess.Popen"):
                    with patch("requests.get") as mock_get:
                        with patch("webbrowser.open"):
                            with patch("time.sleep"):
                                mock_get.return_value.status_code = 200

                                # Mock: inspect raises CalledProcessError, ps returns empty
                                def run_side_effect(*args, **kwargs):
                                    cmd = args[0]
                                    if "inspect" in cmd:
                                        raise subprocess.CalledProcessError(1, cmd)
                                    result = MagicMock()
                                    result.returncode = 0
                                    result.stdout = b""
                                    return result

                                mock_run.side_effect = run_side_effect

                                runner.invoke(pycse, ["launch"])

                                # Should have attempted to pull
                                calls = [str(c) for c in mock_run.call_args_list]
                                pull_attempted = any("pull" in c for c in calls)
                                assert pull_attempted

    def test_max_wait_time_for_server(self):
        """Test that waiting for server has a maximum retry count."""
        from pycse.cli import pycse

        runner = CliRunner()
        with patch("shutil.which", return_value="/usr/local/bin/docker"):
            with patch("subprocess.run") as mock_run:
                with patch("subprocess.Popen"):
                    with patch("requests.get") as mock_get:
                        with patch("webbrowser.open"):
                            with patch("time.sleep"):
                                # Mock: no container, server never becomes ready
                                mock_run.return_value.stdout = b""
                                mock_get.return_value.status_code = 500

                                runner.invoke(pycse, ["launch"])

                                # Should have made exactly 10 attempts (0-9 range)
                                assert mock_get.call_count == 10
