# http://ipython.org/ipython-doc/dev/interactive/qtconsole.html#display
from IPython.display import display
from subprocess import Popen, PIPE

# set images to inline by default
c = c = get_ipython().config
c.IPKernelApp.pylab = 'inline'

ip = get_ipython()

# active true float division
exec ip.compile('from __future__ import division', '<input>', 'single') \
    in ip.user_ns

############################################################

def magic_publish(self, args):
    '''magic function to publish a python file in ipython

    This is some wicked hackery. You cannot directly call the publish module
    more than once because of namespace pollution. We cannot directly
    call publish.py with subprocess because it does not recognize it as an executable,
    even though you can call it from a shell.

    so we basically pipe a script to a python interpreter and execute it
    in its own process, with a new namespace each time.

    this is not pretty, but it works for now.
    '''
    
    # this is some new code inspired by some magic methods in Ipython. 
    # It seems to be a cleaner approach.
    code = '''from pycse.publish import publish
publish(u'{0}')'''.format(args)
    
    p = Popen('python', stdout=PIPE, stderr=PIPE, stdin=PIPE)
    out, err = p.communicate(code)
    print out, err
ip.define_magic('publish', magic_publish)

###########################################################################
from setuptools.command import easy_install

def magic_easy_install(self, package):
    easy_install.main( [args, package] )
    
ip.define_magic('easy_install', magic_easy_install)
    
##################################################################
def magic_pycse_update(self, args):
    # for installing magic IPython stuff

    from setuptools.command import easy_install
    cmd = [args, 'pycse'] if args else ['pycse']
    easy_install.main(cmd)

    # my customized pyreport
    package = 'https://github.com/jkitchin/pyreport/archive/master.zip'
    cmd = [args, package] if args else [package]
    easy_install.main(cmd)


    import IPython, os
    IPydir = os.path.join(IPython.utils.path.get_ipython_dir(),
                      'profile_default',
                      'startup')
                      
    print 'Installing ipython magic to : ',IPydir

    if not os.path.exists(IPydir):
        raise Exception('No ipython directory found')

    url = 'https://raw.github.com/jkitchin/pycse/master/pycse/00-pycse-magic.py'

    import urllib
    urllib.urlretrieve (url, os.path.join(IPydir,'00-pycse-magic.py')) 
    print 'Ipython magic installed now!'

    # extra packages
    for pkg in ['quantities', 
                'uncertainties']:
        cmd = [args, pkg] if args else [pkg]
        easy_install.main(cmd)

    print 'Extra packages now installed.'
    
ip.define_magic('pycse_update', magic_pycse_update)

##################################################################
## pycse_test magic

def magic_pycse_test(self, args):
    PASSED = True
    import pycse
    s = []
    s += ['pycse version: {0}'.format(pycse.__version__)]
    try:
        p = Popen('pdflatex --version', stdout=PIPE, stderr=PIPE, stdin=PIPE)
        s += ['Found pdflatex']
    except:
        PASSED = False
        s += ['No pdflatex found']

    import numpy
    s += ['numpy version: {0}'.format(numpy.__version__)]

    import scipy
    s += ['scipy version: {0}'.format(scipy.__version__)]

    import matplotlib
    s += ['matplotlib version: {0}'.format(matplotlib.__version__)]

    import IPython
    s += ['IPython version: {0}'.format(IPython.__version__)]

    import quantities
    s += ['quantities version: {0}'.format(quantities.__version__)]

    import uncertainties
    s += ['uncertainties version: {0}'.format(uncertainties.__version__)]
    
    return '\n'.join(s)

ip.define_magic('pycse_test', magic_pycse_test)

###########################################################################    
## load some common libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import quantities as u
import uncertainties as unc

print 'pycse-magic loaded.'
