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
def pycse_update(self, *args):
    # for installing magic IPython stuff

    from setuptools.command import easy_install
    easy_install.main( [args, "pycse"] )

    # my customized pyreport
    package = 'https://github.com/jkitchin/pyreport/archive/master.zip'
    easy_install.main( [args,package] )


    import IPython, os
    IPydir = os.path.join(IPython.utils.path.get_ipython_dir(),
                      'profile_default',
                      'startup')
                      
    print 'Installing ipython magic to : ',IPydir

    if not os.path.exists(IPydir):
        raise Exception('No ipython directory found')
    
    import pycse
    p = pycse.__file__
    a = os.path.join(os.path.split(p)[0],'00-pycse-magic.py')
    import shutil
    shutil.copy(a, os.path.join(IPydir,'00-pycse-magic.py'))

    print 'Ipython magic installed now!'

    # extra packages
    easy_install.main( ["-U","quantities"] )
    easy_install.main( ["-U","uncertainties"] )
    print 'Extra packages now installed.'
    
ip.define_magic('pycse_update', pycse_update)

##################################################################
## pycse_test magic

def magic_pycse_test(self, args):
    PASSED = True
    try:
        p = Popen('pdflatex --version', stdout=PIPE, stderr=PIPE, stdin=PIPE)
    except:
        PASSED = False
        print 'No pdflatex found'
    print 'Your installation checked out: ', PASSED

ip.define_magic('pycse_test', magic_pycse_test)

###########################################################################    
## load some common libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import quantities as u
import uncertainties as unc

print 'pycse-magic loaded.'
