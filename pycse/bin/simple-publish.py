#!python

'''
publish [--tex] [--pdf] [--org] *.py

Convert a python script to a published format that has pretty-printed code, captured output, including figures.



The ideas here (and some of the code) is based on pyreport:
http://gael-varoquaux.info/computers/pyreport/

I could not figure out how to make pyreport work with both pylab and matplotlib.pyplot, and write this script instead.
'''
import cStringIO, os, sys, traceback
import random, string
import pylab

original_savefig = pylab.savefig
def show():
    'monkey-patch for saving figs so they get captured'
    global FIGURELIST, BASENAME, BASEPATH, SALT
    figure_name = os.path.join(BASEPATH,
                               '%s-%d.%s' % (BASENAME,
                                             len(FIGURELIST),
                                             'png'))
    figure_name = figure_name.replace('\\','/')                
    FIGURELIST += (figure_name, )
    original_savefig(figure_name)

def savefig(*args,**kwargs):
    'monkeypatch to make sure we get a figure too'
    global FIGURELIST, BASE, SALT
    figure_name = os.path.join(BASEPATH,
                               '%s-%d.%s' % (BASENAME,
                                             len(FIGURELIST),
                                             'png'))
    figure_name = figure_name.replace('\\','/')
                    
    original_savefig(*args, **kwargs)
    original_savefig(figure_name)
    FIGURELIST += (figure_name, )
    
import pylab
pylab.show = show
pylab.savefig = savefig

import matplotlib.pyplot 
matplotlib.pyplot.show = show
matplotlib.pyplot.savefig = savefig

##############################################################
# output functions

def tex(code, output, error, exception):
    'Create a tex document of the code'
    global FIGURELIST, BASENAME, data

    texfile = BASENAME + '.tex'

    escaped_path = os.path.abspath(INPUT).replace('\\','/')

    d = locals()
    d.update(globals())
    
    s = r'''\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{fixltx2e}}
\usepackage{{graphicx}}
\usepackage{{minted}}
\usepackage{{float}}
\usepackage{{fancyhdr}}
\usepackage{{underscore}}
\pagestyle{{fancyplain}}
\rhead{{\today}}
\begin{{document}}
\begin{{verbatim}}
name: {data[NAME]}
userid: {userid}
path: {escaped_path}
ip-addr: {ipaddr}
hostname: {hostname}
mac: {mac}
date-submitted: {date_submitted}
\end{{verbatim}}


\section{{Code}}

\begin{{minted}}[frame=lines,fontsize=\scriptsize,linenos=true]{{python}}
{code}
\end{{minted}}

\section{{Output}}

\begin{{verbatim}}
{output}
\end{{verbatim}}
'''.format(**d)

    if err:
        s += r'''
\section{{Errors}}    
\begin{{Verbatim}}[fontcom=\color{{red}}
{0}
\end{{Verbatim}}
'''.format(err)

    if exception:
        s += r'''
\section{{Exceptions}}    
\begin{{Verbatim}}[formatcom=\color{{red}}]
{0}
\end{{Verbatim}}
'''.format(exception)

    for figure in FIGURELIST:
        s += r'''
\begin{{figure}}[H]
    \includegraphics[width=0.9\textwidth]{{{0}}}
\end{{figure}}
'''.format(figure)
    
    s += r'\end{document}'

    with open(texfile, 'w') as f:
        f.write(s)

    return s

def pdf(code, output, error, exception):
    'create pdf from latex document'
    print 'Creating pdf output'
    global BASENAME
    tex(code, output, error, exception)
    import subprocess

    texfile = BASENAME + '.tex'
    #status = subprocess.call(["pdflatex.exe", texfile])
    os.system('pdflatex -shell-escape {0}'.format(texfile))
    
    for f in [texfile,
              texfile.replace('.tex','.aux'),
              texfile.replace('.tex','.log'),]:
        if os.path.exists(f):
            os.unlink(f)

    print 'opening ',texfile.replace('.tex','.pdf')
    try:
        os.startfile(texfile.replace('.tex','.pdf'))
    except:
        print texfile.replace('.tex','.pdf'), ' not found'
        
def org(code, output, error, exception):
    '''outputs an org-file zip archive'''
    global INPUT, BASENAME
    
    escaped_path = os.path.abspath(INPUT).replace('\\','/')
    d = locals()
    d.update(globals())
    
    s = '''#+STARTUP: showall inlineimages
#+FILE: {escaped_path}
#+AUTHOR: {data[NAME]}
#+COURSE: {data[COURSE]}
#+ASSIGNMENT: {data[ASSIGNMENT]}
#+ANDREWID: {data[ANDREWID]}
    
#+BEGIN_EXAMPLE
userid: {userid}
path: {escaped_path}
ip-addr: {ipaddr}
hostname: {hostname}
mac: {mac}
date-submitted: {date_submitted}
#+END_EXAMPLE
    
* Code
#+BEGIN_SRC python
{code}
#+END_SRC

* Results
#+BEGIN_EXAMPLE
{output}
#+END_EXAMPLE
'''.format(**d)

    if err:
        s += r'''
* Errors
#+BEGIN_EXAMPLE
{0}
#+END_EXAMPLE
'''.format(err)

    if exception:
        s += r'''
* Exceptions
#+BEGIN_EXAMPLE
{0}
#+END_EXAMPLE
'''.format(exception)

    if FIGURELIST:
        s += '* Figures\n'     
        for figure in FIGURELIST:
            s += r'''
[[file:{0}]]
'''.format(figure)


    zipfile = BASENAME + '.zip'
    orgfile = BASENAME + '.org'

    # create the orgfile
    with open(orgfile, 'w') as f:
        f.write(s)

    # create the zip file
    from zipfile import ZipFile
    with ZipFile(zipfile, 'w') as zip:
        zip.write(orgfile)
        for figure in FIGURELIST:            
            zip.write(figure)

    os.unlink(orgfile)
    print 'Org-archive created'

def run(INPUT):    
    print 'running ',INPUT
    with open(INPUT) as f:
        code = f.read()

    # run code and get output:    
    code_out = cStringIO.StringIO()
    code_err = cStringIO.StringIO()
    sys.stdout = code_out
    sys.stderr = code_err

    try:
        formatted_exception = None
        exec code 
    except:
        formatted_exception = traceback.format_exc()

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
            
    output = code_out.getvalue()
    err = code_err.getvalue()
            
    code_out.close()
    code_err.close()

    # close all figures to avoid cross-contamination when running
    # many scripts. There is some "leakage" of variables between scripts.
    # pyreport has some kind of sandbox
    from pylab import close
    close('all')
    print 'Done running'
    return code, output, err, formatted_exception




##################################################################
##################################################################
import argparse
from uuid import getnode as get_mac
import re, socket
import uuid
import datetime

userid = os.environ.get('USER','no user found')
date_submitted = datetime.datetime.now()


# mac address of submitting computer
mac = get_mac()
try:
    hostname, aliases, ipaddr =  socket.gethostbyaddr(socket.gethostbyname(socket.gethostname()))
except:
    hostname, aliases, ipaddr = None, None, None

if ipaddr:
    ipaddr = ipaddr[0] # it is usually a list. I think it is ok to take the first element.
data = {'mac':mac,
        'hostname':hostname,
        'ipaddr':ipaddr,
        'userid':userid,
        'date_submitted':date_submitted}

PROPERTIES = ['COURSE',
              'ASSIGNMENT',
              'ANDREWID',              
              'NAME']

##################################################################

parser = argparse.ArgumentParser(description='submit your python script and output in tex, pdf or org-mode archive file')

parser.add_argument('--tex', action='store_true',
                    help = 'output texfile')

parser.add_argument('--pdf', action='store_true',
                    help = 'output pdf.')

parser.add_argument('--org', action='store_true',
                    help = 'output org-mode archive')

parser.add_argument('--submit', action='store_true',
                    help='submit assignment')

parser.add_argument('files', nargs='*',                    
                    help='scripts to submit')

args = parser.parse_args()
    
for INPUT in args.files:
    # check for compliance of data
    with open(INPUT) as f:
        text = f.read()
    
        for prop in PROPERTIES:
            regexp = '#\+{0}:(.*)'.format(prop)
            m = re.search(regexp, text)
            if m:
                data[prop] = m.group(1).strip()
            else:
                raise Exception('''You are missing #+{0}: in your file. please add it and try again.'''.format(prop))

    # list of paths to figures saved here
    FIGURELIST = []
    # global variables that are needed above.
    BASENAME = '{ANDREWID}-{COURSE}-{ASSIGNMENT}'.format(**data)
    BASEPATH, fname = os.path.split(INPUT)
    SALT = ''.join([random.choice(string.letters) for x in range(4)])
    os.mkdir(SALT)
    
    code, output, err, formatted_exception = run(INPUT)
    
    if args.tex:
        pdf(code, output, err, formatted_exception)

    if args.pdf:
        pdf(code, output, err, formatted_exception)

    if args.org:
        org(code, output, err, formatted_exception)

    # make org the default
    if not (args.tex or args.pdf or args.org):
        org(code, output, err, formatted_exception)

    # clean up figures
    for figure in FIGURELIST:
        os.unlink(figure)
    os.rmdir(SALT)


    if args.submit:
        import requests
        
        url = 'http://localhost:8080/upload'
        files = {'file': open(BASENAME + '.zip', 'rb')}
        r = requests.post(url, files=files, data=data)

        print r.status_code
        print r.text


    
    




