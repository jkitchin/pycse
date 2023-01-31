"""Module for using plotly with orgmode.

This monkey-patches go.Figure.show to provide a png image for org-mode, and an
html file that is saved that you can click on in org-mode to see the interactive
version.

"""
import os

from hashlib import md5

from IPython import display

import plotly.graph_objects as go
import plotly.io as pio


def myshow(self, *args, **kwargs):
    """Make a PNG image to display for plotly."""
    html = pio.to_html(self)
    mhash = md5(html.encode("utf-8")).hexdigest()
    if not os.path.isdir(".ob-jupyter"):
        os.mkdir(".ob-jupyter")
    fhtml = os.path.join(".ob-jupyter", mhash + ".html")

    with open(fhtml, "w", encoding="utf-8") as f:
        f.write(html)

    display.FileLink(fhtml, result_html_suffix="")

    return display.Image(pio.to_image(self, "png", engine="kaleido"))


go.Figure.show = myshow
