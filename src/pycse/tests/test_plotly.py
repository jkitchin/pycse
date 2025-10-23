"""Tests for plotly module."""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock


class TestPlotlyModule:
    """Tests for pycse.plotly module."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        # Cleanup
        os.chdir(original_cwd)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_figure(self):
        """Create a mock plotly Figure."""
        # Import inside fixture to avoid import errors if plotly not available
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9], mode="lines"),
            layout=go.Layout(title="Test Plot"),
        )
        return fig

    def test_module_imports(self):
        """Test that the module imports without error."""
        import pycse.plotly

        assert hasattr(pycse.plotly, "myshow")

    def test_monkey_patch_applied(self):
        """Test that go.Figure.show is monkey-patched."""
        import plotly.graph_objects as go
        import pycse.plotly

        # The show method should be the myshow function
        assert go.Figure.show == pycse.plotly.myshow

    @patch("pycse.plotly.display.Image")
    @patch("pycse.plotly.display.FileLink")
    @patch("pycse.plotly.pio.to_image")
    @patch("pycse.plotly.pio.to_html")
    def test_myshow_creates_directory(
        self, mock_to_html, mock_to_image, mock_filelink, mock_display_image, temp_dir, mock_figure
    ):
        """Test that myshow creates .ob-jupyter directory."""
        import pycse.plotly

        # Mock the HTML output
        mock_to_html.return_value = "<html>test plot</html>"
        mock_to_image.return_value = b"fake_png_data"

        # Call myshow
        pycse.plotly.myshow(mock_figure)

        # Check that .ob-jupyter directory was created
        assert os.path.exists(".ob-jupyter")
        assert os.path.isdir(".ob-jupyter")

    @patch("pycse.plotly.display.Image")
    @patch("pycse.plotly.display.FileLink")
    @patch("pycse.plotly.pio.to_image")
    @patch("pycse.plotly.pio.to_html")
    def test_myshow_saves_html_file(
        self, mock_to_html, mock_to_image, mock_filelink, mock_display_image, temp_dir, mock_figure
    ):
        """Test that myshow saves HTML file with correct hash."""
        import pycse.plotly
        from hashlib import md5

        # Mock the HTML output
        test_html = "<html>test plot content</html>"
        mock_to_html.return_value = test_html
        mock_to_image.return_value = b"fake_png_data"

        # Call myshow
        pycse.plotly.myshow(mock_figure)

        # Calculate expected hash
        expected_hash = md5(test_html.encode("utf-8")).hexdigest()
        expected_file = os.path.join(".ob-jupyter", expected_hash + ".html")

        # Check that file was created
        assert os.path.exists(expected_file)

        # Check file contents
        with open(expected_file, "r", encoding="utf-8") as f:
            contents = f.read()
        assert contents == test_html

    @patch("pycse.plotly.display.Image")
    @patch("pycse.plotly.display.FileLink")
    @patch("pycse.plotly.pio.to_image")
    @patch("pycse.plotly.pio.to_html")
    def test_myshow_creates_filelink(
        self, mock_to_html, mock_to_image, mock_filelink, mock_display_image, temp_dir, mock_figure
    ):
        """Test that myshow creates FileLink display."""
        import pycse.plotly
        from hashlib import md5

        test_html = "<html>test</html>"
        mock_to_html.return_value = test_html
        mock_to_image.return_value = b"png_data"

        # Call myshow
        pycse.plotly.myshow(mock_figure)

        # Check that FileLink was called
        expected_hash = md5(test_html.encode("utf-8")).hexdigest()
        expected_file = os.path.join(".ob-jupyter", expected_hash + ".html")
        mock_filelink.assert_called_once_with(expected_file, result_html_suffix="")

    @patch("pycse.plotly.display.Image")
    @patch("pycse.plotly.display.FileLink")
    @patch("pycse.plotly.pio.to_image")
    @patch("pycse.plotly.pio.to_html")
    def test_myshow_returns_png_image(
        self, mock_to_html, mock_to_image, mock_filelink, mock_display_image, temp_dir, mock_figure
    ):
        """Test that myshow returns PNG Image display object."""
        import pycse.plotly

        mock_to_html.return_value = "<html>test</html>"
        fake_png = b"fake_png_bytes"
        mock_to_image.return_value = fake_png
        mock_image_obj = MagicMock()
        mock_display_image.return_value = mock_image_obj

        # Call myshow
        result = pycse.plotly.myshow(mock_figure)

        # Check that to_image was called with correct params
        mock_to_image.assert_called_once_with(mock_figure, "png", engine="kaleido")

        # Check that Image was called with PNG data
        mock_display_image.assert_called_once_with(fake_png)

        # Check that Image object is returned
        assert result == mock_image_obj

    @patch("pycse.plotly.display.Image")
    @patch("pycse.plotly.display.FileLink")
    @patch("pycse.plotly.pio.to_image")
    @patch("pycse.plotly.pio.to_html")
    def test_myshow_reuses_existing_directory(
        self, mock_to_html, mock_to_image, mock_filelink, mock_display_image, temp_dir, mock_figure
    ):
        """Test that myshow doesn't fail if .ob-jupyter already exists."""
        import pycse.plotly

        # Pre-create the directory
        os.mkdir(".ob-jupyter")

        mock_to_html.return_value = "<html>test</html>"
        mock_to_image.return_value = b"png"

        # Should not raise an error
        pycse.plotly.myshow(mock_figure)

        # Directory should still exist
        assert os.path.exists(".ob-jupyter")

    @patch("pycse.plotly.display.Image")
    @patch("pycse.plotly.display.FileLink")
    @patch("pycse.plotly.pio.to_image")
    @patch("pycse.plotly.pio.to_html")
    def test_myshow_with_figure_show_method(
        self, mock_to_html, mock_to_image, mock_filelink, mock_display_image, temp_dir
    ):
        """Test calling show() on a Figure object uses myshow."""
        import plotly.graph_objects as go

        # Create a real figure
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[1, 2]))

        mock_to_html.return_value = "<html>fig</html>"
        mock_to_image.return_value = b"png"
        mock_display_image.return_value = MagicMock()

        # Call show() - should use the monkey-patched method
        result = fig.show()

        # Verify mocks were called
        assert mock_to_html.called
        assert mock_to_image.called
        assert result is not None

    @patch("pycse.plotly.display.Image")
    @patch("pycse.plotly.display.FileLink")
    @patch("pycse.plotly.pio.to_image")
    @patch("pycse.plotly.pio.to_html")
    def test_myshow_hash_consistency(
        self, mock_to_html, mock_to_image, mock_filelink, mock_display_image, temp_dir, mock_figure
    ):
        """Test that same HTML produces same hash/filename."""
        import pycse.plotly
        from hashlib import md5

        test_html = "<html>consistent test</html>"
        mock_to_html.return_value = test_html
        mock_to_image.return_value = b"png"

        # Call twice
        pycse.plotly.myshow(mock_figure)
        pycse.plotly.myshow(mock_figure)

        # Should create same file (same hash)
        expected_hash = md5(test_html.encode("utf-8")).hexdigest()
        expected_file = os.path.join(".ob-jupyter", expected_hash + ".html")

        # File should exist and only one file with this hash
        assert os.path.exists(expected_file)

        # Check that only expected files exist
        files = os.listdir(".ob-jupyter")
        html_files = [f for f in files if f.endswith(".html")]
        assert len(html_files) == 1
        assert html_files[0] == expected_hash + ".html"
