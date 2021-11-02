import os

from hashlib import md5

from IPython.display import Image, FileLink

import plotly.graph_objects as go
import plotly.io as pio


def myshow(self, *args, **kwargs):
    html = pio.to_html(self)
    mhash = md5(html.encode('utf-8')).hexdigest()
    if not os.path.isdir('.ob-jupyter'):
        os.mkdir('.ob-jupyter')
    fhtml = os.path.join('.ob-jupyter', mhash + '.html')

    with open(fhtml, 'w') as f:
        f.write(html)

    display(FileLink(fhtml, result_html_suffix=''))

    return Image(pio.to_image(self, 'png', engine='kaleido'))


go.Figure.show = myshow

