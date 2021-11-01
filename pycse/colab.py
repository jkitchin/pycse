from datetime import datetime
import glob
import io
from IPython import get_ipython
from IPython.core.magic import register_line_magic
from IPython.display import HTML, IFrame
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


DRIVE = None


def gdrive():
    '''Get the drive service, authenticate if needed.'''
    global DRIVE
    if DRIVE is None:
        auth.authenticate_user()
        DRIVE = build('drive', 'v3')
    return DRIVE


##################################################################
# Utilities
##################################################################

def aptupdate():
    s = subprocess.run(['apt-get', 'update'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if s.returncode != 0:
        raise Exception(f'apt-get update failed.\n'
                        f'{s.stdout.decode()}\n'
                        f'{s.stderr.decode()}')

    
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


# @register_line_magic
# def gimport(fid_or_url):
#     '''Load the python code at fid or url.
#     Also a line magic.'''
#     with gopen(fid_or_url) as f:
#         py = f.read()
#         g = globals()
#         exec(py, g)

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
    drive_service = gdrive()
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


def pdf_from_html(pdf=None, verbose=False, plotly=False, javascript_delay=10000):
    '''Export the current notebook as a PDF.
    pdf is the name of the PDF to export.
    plotly uses the plotly exporter
    The pdf is not saved in GDrive. Conversion is done from an HTML export.
    javascript_delay is in ms, and is how long to wait in wkhtmltopdf to let
    javascript, especially mathjax finish.
    '''
    if verbose:
        print('PDF via wkhtmltopdf')

    fname, fid = current_notebook()
    ipynb = notebook_string(fid)

    if plotly:
        subprocess.run(['pip', 'install', 'plotlyhtmlexporter'])
        from plotlyhtmlexporter import PlotlyHTMLExporter
        exporter = PlotlyHTMLExporter()
    else:
        exporter = HTMLExporter()

    nb = nbformat.reads(ipynb, as_version=4)
    body, resources = exporter.from_notebook_node(nb)

    if verbose:
        print(f'args: pdf={pdf}, verbose={verbose}')

    if pdf is None:
        html = fname.replace(".ipynb", ".html")
        pdf = html.replace(".html", ".pdf")
    else:
        html = pdf.replace(".pdf", ".html")

    if verbose:
        print(f'using html = {html}')

    tmpdirname = tempfile.TemporaryDirectory().name

    if not os.path.isdir(tmpdirname):
        os.mkdir(tmpdirname)

    ahtml = os.path.join(tmpdirname, html)
    apdf = os.path.join(tmpdirname, pdf)
    css = os.path.join(tmpdirname, 'custom.css')

    with open(ahtml, 'w') as f:
        f.write(body)

    with open(css, 'w') as f:
        f.write('\n'.join(resources['inlining']['css']))

    aptupdate()
    
    if not shutil.which('xvfb-run'):        
        aptinstall('xvfb')

    if not shutil.which('wkhtmltopdf'):
        aptinstall('wkhtmltopdf')

    if verbose:
        print(f'Running with delay: {javascript_delay}')

    s = subprocess.run(['xvfb-run', 'wkhtmltopdf',
                        '--enable-javascript',
                        '--no-stop-slow-scripts',
                        '--javascript-delay', str(javascript_delay),
                        ahtml, apdf],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

    if verbose and s.returncode != 0:
        print(f'Conversion exited with non-zero status: {s.returncode}.\n'
              f'{s.stdout.decode()}\n'
              f'{s.stderr.decode()}')

    if os.path.exists(apdf):
        files.download(apdf)
    else:
        print('no pdf found.')
        print(ahtml)
        print(apdf)


def pdf_from_latex(pdf=None, verbose=False):
    '''Export the notebook to PDF via LaTeX.
    This is not fast because you have to install texlive.
    verbose is not used right now.
    '''
    print('PDF via LaTeX')
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


def pdf(line=''):
    '''Line magic to export a colab to PDF.
    You can have an optional arg -l to use LaTeX, defaults to html->PDF.
    You can have an optional arg -p to use plotlyhtmlexporter
    You can have an optional arg -d integer for a delay in seconds for the html to pdf.
    
    You can have an optional last argument for the filename of the pdf
    '''
    args = shlex.split(line)

    if args and args[-1].endswith('.pdf'):
        pdf = args[-1]
    else:
        pdf = None

    verbose = '-v' in args

    if verbose:
        print(f'%pdf args = {args}')

    if '-l' in args:
        pdf_from_latex(pdf, verbose)
        
    
    else:
        if '-d' in args:
            i = args.index('-d')
            # The delay should be in microseconds.
            delay = int(args[i + 1]) * 1000
        else:
            delay = 10000
        plotly = '-p' in args
        pdf_from_html(pdf, verbose, plotly, delay)
                   


# this is hackery so that CI works.
# it is an error to do this when there is not IPython
try:
    pdf = register_line_magic(pdf)
except:
    pass


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
    # https://colab.research.google.com/drive/1YcD5OXL-CNBO2h_OXZFb-mY6-LqgcLkB#scrollTo=0qkiF99z01pc
    elif (u.netloc == 'colab.research.google.com'):
        return u.path.split('/')[2]

    # 'https://docs.google.com/document/d/1lvDK2GisDM5aBnImtHNwOmLsU9jxg1NaPC46rB4bVqw/edit?usp=sharing'
    elif (u.netloc == 'docs.google.com') and u.path.startswith('/document/d/'):
        p = u.path
        p = p.replace('/document/d/', '')
        p = p.replace('/edit', '')
        return p

    # https://docs.google.com/spreadsheets/d/1qSaBe73Pd8L3jJyOL68klp6yRArW7Nce/edit#gid=1923176268
    elif (u.netloc == 'docs.google.com') and u.path.startswith('/spreadsheets/d/'):
        p = u.path
        p = p.replace('/spreadsheets/d/', '')
        p = p.replace('/edit', '')
        return p

    #https://docs.google.com/presentation/d/1poP1gvWlfeZCR_5FsIzlRPMAYlBUR827wKPjbWGzW9M/edit#slide=id.p
    elif (u.netloc == 'docs.google.com') and u.path.startswith('/presentation/d/'):
        p = u.path
        p = p.replace('/presentation/d/', '')
        p = p.replace('/edit', '')
        return p

    # https://drive.google.com/drive/u/0/folders/1aTs-_bhjT1GXy2P2hStzn31qAihRq2sl
    elif (u.netloc == 'drive.google.com') and 'folders' in u.path:
        return u.path.split('folders')[1][1:]

    else:
        raise Exception(f'Cannot parse {url} yet.')


def gopen(fid_or_url_or_path, mode='r'):
    '''Open a file on Gdrive by its ID, sharing link or path.
    Returns a file-like object you can read from.
    Note this reads the whole file into memory, so it may not
    be good for large files. Returns an io.StringIO if mode is "r"
    or io.BytesIO if mode is "rb".
    '''
    if mode not in ['r', 'rb']:
        raise Exception(f'mode must be "r" or "rb"')

    if fid_or_url_or_path.startswith('http'):
        fid = fid_from_url(fid_or_url_or_path)
    else:
        # it could be a path
        if os.path.isfile(fid_or_url_or_path):
            fid = get_id(fid_or_url_or_path)
        else:
            # assume it is an fid
            fid = fid_or_url_or_path
            print('fid: ', fid)

    drive_service = gdrive()
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
    if mode == 'r':
        return io.TextIOWrapper(downloaded)
    else:
        return downloaded

# Path utilities
# This is tricky, paths are not deterministic in GDrive the way we are used to.
# There is also some differences in My Drive and Shared drives, and files
# shared with you.


def get_path(fid_or_url):
    """Return the path to an fid or url.
    The path i's relative to the mount point."""
    if fid_or_url.startswith('http'):
        fid = fid_from_url(fid_or_url)
    else:
        fid = fid_or_url

    drive_service = gdrive()
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
    p = os.path.sep.join(dirs)

    # Sometimes, it appears we are missing an extension, because the name does
    # not always include the extension. We glob through matches to get the match
    # in this case.
    if not os.path.exists(p):
        for f in glob.glob(f'{p}*'):
            if get_id(f) == fid:
                return f
    else:
        return p


def get_id(path):
    '''Given a path, return an id to it.'''
    drive_service = gdrive()

    if not shutil.which('xattr'):
        aptinstall('xattr')

    path = os.path.abspath(path)

    if os.path.isfile(path):
        return subprocess.getoutput(f"xattr -p 'user.drive.id' '{path}'")

    elif os.path.isdir(path):
        # Strip the / gdrive off
        path = path.split('/')[2:]

        if path[0] == 'My Drive' and len(path) == 1:
            return 0

        if path[0] == 'My Drive':
            drive_id = 'root'
            id = 'root'

        elif path[0] == 'Shared drives':
            drives = drive_service.drives().list().execute()['drives']
            for drv in drives:
                if drv['name'] == path[1]:
                    drive_id = drv['id']
                    id = drv['id']
                    break

        path = path[1:]

        found = False
        for d in path:
            dsf = drive_service.files()
            args = dict(q=f"'{id}' in parents")
            if drive_id != 'root':
                args['corpora'] = 'drive'
                args['supportsAllDrives'] = True
                args['includeItemsFromAllDrives'] = True
                args['driveId'] = drive_id

            file_list = dsf.list(**args).execute()

            found = False
            for file1 in file_list.get('files', []):
                if file1['name'] == d:
                    found = True
                    id = file1['id']
                    break

        if found:
            return id

        else:
            raise Exception(f'Something went wrong with {path}')

    else:
        raise Exception(f'{path} does not seem to be a file or directory')


def get_link(path):
    '''Returns a clickable link for path.'''
    fid = get_id(os.path.abspath(path))
    drive_service = gdrive()
    x = drive_service.files().get(fileId=fid,
                                  supportsAllDrives=True,
                                  fields='webViewLink').execute()
    url = x.get('webViewLink', 'No web link found')
    return HTML(f"<a href={url} target=_blank>{path}</a>")


def gchdir(path=None):
    '''Change working dir to path.
    if path is None, default to working directory of current notebook.
    '''
    if path is None:
        path = os.path.dirname(get_path(current_notebook()[1]))

    if os.path.isabs(path):
        os.chdir(path)
    else:
        os.chdir(os.path.abspath(path))


def gdownload(*FILES, **kwargs):
    '''Download files. Each arg can be a path, or pattern.
    If you have more than one file, a zip is downloaded.
    You can specify a zip file name as a kwarg:

    gdownload("*", zip="test.zip")

    The zip file will be deleted unless you use keep=True as a kwarg.

    '''
    fd = []
    for f in FILES:
        for g in glob.glob(f):
            fd += [g]

    if (len(fd) == 1) and (os.path.isfile(fd[0])):
        files.download(fd[0])
    else:
        if 'zip' in kwargs:
            zipfile = kwargs['zip']
        else:
            now = datetime.now()
            zipfile = now.strftime("%m-%d-%YT%H-%M-%S.zip")

        if os.path.exists(zipfile):
            os.unlink(zipfile)

        s = subprocess.run(['zip', '-r', zipfile, *fd],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        if s.returncode != 0:
            print(f'zip did not fully succeed:\n'
                  f'{s.stdout.decode()}\n'
                  f'{s.stderr.decode()}\n')
        files.download(zipfile)
#        if not kwargs.get('keep', False):
#            os.unlink(zip)

##################################################################
# Get to a shell
##################################################################
def gconsole():
    '''Open a shell in colab.
    Adapted from https://github.com/airesearch-in-th/kora/blob/master/kora/console.py'''

    url = "https://github.com/gravitational/teleconsole/releases/download/0.4.0/teleconsole-v0.4.0-linux-amd64.tar.gz"
    os.system(f"curl -L {url} | tar xz")  # download & extract
    os.system("mv teleconsole /usr/local/bin/")  # in PATH

    # Set PS1, directory
    with open("/root/.bashrc", "a") as f:
        f.write('PS1="\e[1;36m\w\e[m# "\n')
        f.write("cd /content \n")
        f.write("PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin:/opt/bin \n")


    process = subprocess.Popen("teleconsole", shell=True,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(6):
        line = process.stdout.readline()

    url = line.decode().strip().split()[-1]
    print("Console URL:", url)
    return IFrame(url, width=800, height=600)


##################################################################
# Fancy outputs
##################################################################

def gsuite(fid_or_url, width=1200, height=1000):
    '''Return an iframe that renders the item in a colab.'''
    drive_service = gdrive()
    if fid_or_url.startswith('http'):
        url = fid_or_url
    else:
        # Assume we have an fid
        x = drive_service.files().get(fileId=fid_or_url,
                                      supportsAllDrives=True,
                                      fields='webViewLink').execute()
        url = x.get('webViewLink', 'No web link found.')

    display(HTML(f'''<a href="{url}" target="_blank">Link</a><br>'''))

    g = requests.get(url)
    xframeoptions = g.headers.get('X-Frame-Options', '').lower()
    if  xframeoptions in ['deny', 'sameorigin']:
        print(f'X-Frame-Option = {xframeoptions}\nEmbedding in IFrame is not allowed for {url}.')
    else:
        return IFrame(url, width, height)
