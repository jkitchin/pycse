import io
from IPython.core.magic import register_line_magic

from nbconvert import HTMLExporter, PDFExporter
import nbformat
import os
import requests
import shlex
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse

from google.colab import drive
from google.colab import files
from googleapiclient.http import MediaIoBaseDownload

from google.colab import auth
from googleapiclient.discovery import build


auth.authenticate_user()
drive_service = build('drive', 'v3')


##################################################################
# Utilities
##################################################################


def aptinstall(apt_pkg):
    '''Utility to install a package and check for success.'''
    print(f'Installing {apt_pkg}. Please be patient.')
    s = subprocess.run(['apt-get', 'install', apt_pkg],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if s.returncode != 0:
        raise Exception(f'{apt_pkg} installation failed.\n'
                        f'{s.stdout.decode()}\n'
                        f'{s.stderr.decode()}')


@register_line_magic
def gimport(fid_or_url):
    '''Load the python code at fid or url.
    Also a line magic.'''
    with gopen(fid_or_url) as f:
        py = f.read()
        g = globals()
        exec(py, g)

##################################################################
# Exporting functions
##################################################################


def current_notebook():
    '''Returns current notebook name and file id.
    from kora.drive.
    '''
    d = requests.get('http://172.28.0.2:9000/api/sessions').json()[0]
    fid = d['path'].split('=')[1]
    fname = d['name']
    return fname, fid


def notebook_string(fid):
    '''Return noteook json data in string form for notebook at FID.'''
    request = drive_service.files().get_media(fileId=fid)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while done is False:
        # _ is a placeholder for a progress object that we ignore.
        # (Our file is small, so we skip reporting progress.)
        _, done = downloader.next_chunk()

    # Rewind
    downloaded.seek(0)
    ipynb = downloaded.read()  # nb in string form
    return ipynb


def pdf_from_html(pdf=None, debug=False):
    '''Export the current notebook as a PDF.
    pdf is the name of the PDF to export.
    The pdf is not saved in GDrive. Conversion is done from an HTML export.
    '''
    fname, fid = current_notebook()
    ipynb = notebook_string(fid)

    exporter = HTMLExporter()

    nb = nbformat.reads(ipynb, as_version=4)
    body, resources = exporter.from_notebook_node(nb)

    html = fname.replace(".ipynb", ".html")
    if pdf is None:
        pdf = html.replace(".html", ".pdf")

    tmpdirname = tempfile.TemporaryDirectory().name

    if not os.path.isdir(tmpdirname):
        os.mkdir(tmpdirname)

    ahtml = os.path.join(tmpdirname, html)
    apdf = os.path.join(tmpdirname, pdf)

    with open(ahtml, 'w') as f:
        f.write(body)

    if not shutil.which('xvfb-run'):
        aptinstall('xvfb')

    if not shutil.which('wkhtmltopdf'):
        aptinstall('wkhtmltopdf')

    s = subprocess.run(['xvfb-run', 'wkhtmltopdf', ahtml, apdf],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if s.returncode != 0:
        raise Exception('PDF conversion failed.\n'
                        f'{s.stdout}\n'
                        f'{s.stderr}')

    if os.path.exists(apdf):
        files.download(apdf)
    else:
        print('no pdf found.')
        print(ahtml)
        print(apdf)


def pdf_from_latex(pdf=None):
    '''Export the notebook to PDF via LaTeX.
    This is not fast because you have to install texlive.'''

    if not shutil.which('xelatex'):
        aptinstall('texlive-xetex')

    fname, fid = current_notebook()
    ipynb = notebook_string(fid)

    exporter = PDFExporter()

    nb = nbformat.reads(ipynb, as_version=4)
    body, resources = exporter.from_notebook_node(nb)

    if pdf is None:
        pdf = fname.replace(".ipynb", ".pdf")

    tmpdirname = tempfile.TemporaryDirectory().name

    if not os.path.isdir(tmpdirname):
        os.mkdir(tmpdirname)

    apdf = os.path.join(tmpdirname, pdf)

    if os.path.exists(apdf):
        os.unlink(apdf)

    with open(apdf, 'wb') as f:
        f.write(body)

    if os.path.exists(apdf):
        files.download(apdf)
    else:
        print(f'{apdf} not found')


@register_line_magic
def pdf(line):
    '''Line magic to export a colab to PDF.
    You can have an optional arg -l to use LaTeX, defaults to html->PDF.
    You can have an optional last argument for the filename of the pdf
    '''
    args = shlex.split(line)

    if args and args[-1].endswith('.pdf'):
        pdf = args[-1]
    else:
        pdf = None

    if '-l' in args:
        pdf_from_latex(pdf)
    else:
        pdf_from_html(pdf)


##################################################################
# File utilities
##################################################################

def fid_from_url(url):
    '''Return a file ID for a file on GDrive from its url.'''
    u = urlparse(url)

    # This is a typical sharing link
    # https://drive.google.com/file/d/1q_qE9RGdfV_8Vv3zuApf-LqXBwqo8HO2/view?usp=sharing
    if (u.netloc == 'drive.google.com') and (u.path.startswith('/file/d/')):
        return u.path.split('/')[3]

    # This is a download link
    # https://drive.google.com/uc?id=1LLOGvaXsaEhUQXd7AmN_offy2IzNEu0K
    elif (u.netloc == 'drive.google.com') and (u.path == '/uc'):
        q = u.query
        # I think this could have other things separated by &
        qs = q.split('&')
        for item in qs:
            if item.startswith('id='):
                return item[3:]

    # A colab url
    elif (u.netloc == 'colab.research.google.com'):
        return u.path.split('/')[2]

    elif 'folders' in u.path:
        raise Exception('You seem to be opening a folder.')

    else:
        raise Exception(f'Cannot parse {url} yet.')


def gopen(fid_or_url):
    '''Open a file on Gdrive by its ID or sharing link.
    Returns a file-like object you can read from.
    Note this reads the whole file into memory, so it may not
    be good for large files. Returns an io.StringIO
    '''
    if fid_or_url.startswith('http'):
        fid = fid_from_url(fid_or_url)
    else:
        fid = fid_or_url

    request = drive_service.files().get_media(fileId=fid)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while done is False:
        # _ is a placeholder for a progress object that we ignore.
        # (Our file is small, so we skip reporting progress.)
        _, done = downloader.next_chunk()

    # I prefer strings to bytes.
    downloaded.seek(0)
    return io.TextIOWrapper(downloaded)

# Path utilities
# This is tricky, paths are not deterministic in GDrive the way we are used to.
# There is also some differences in My Drive and Shared drives, and files
# shared with you.


def _get_path(fid_or_url):
    """Return the path to an fid or url.
    The path is relative to the mount point."""
    if fid_or_url.startswith('http'):
        fid = fid_from_url(fid_or_url)
    else:
        fid = fid_or_url

    x = drive_service.files().get(fileId=fid,
                                  supportsAllDrives=True,
                                  fields='parents,name').execute()

    dirs = [x['name']]  # start with the document name

    while x.get('parents', None):

        if len(x['parents']) > 1:
            print(f'Warning, multiple parents found {x["parents"]}')

        x = drive_service.files().get(fileId=x['parents'][0],
                                      supportsAllDrives=True,
                                      fields='id,parents,name').execute()

        if ('parents' not in x) and x['name'] == 'Drive':
            # this means your file is in a shared drive I think.
            drives = drive_service.drives().list().execute()['drives']
            for drv in drives:
                if drv['id'] == x['id']:
                    dirs += [drv['name'], 'Shared drives']
        else:
            dirs += [x['name']]

    if not os.path.isdir('/gdrive'):
        drive.mount('/gdrive')

    dirs += ['/gdrive']

    dirs.reverse()
    return os.path.sep.join(dirs)
