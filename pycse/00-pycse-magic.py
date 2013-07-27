ip = get_ipython()

def magic_publish(self, args):
    '''magic function to publish a python file in ipython

    This is some wicked hackery. You cannot directly call the publish module
    more than once because of namespace pollution. We cannot directly
    call publish.py with subprocess because it does not recognize it as an executable,
    even though you can call it from a shell.

    so we find the location of of the publish.py script and execute it
    in its own process, with a new namespace each time.

    this is not pretty, but it works for now.
    '''
    import os, subprocess
    import pycse
    path, init = os.path.split(pycse.__file__)
    cmd = ['python',
           os.path.join(path,'publish.py')]
    cmd += [str(x) for x in args.split()]
    
    status = subprocess.check_call(cmd)
    if status != 0:
        print 'Something went wrong in publishing.'
        
ip.define_magic('publish', magic_publish)

###########################################################################
from setuptools.command import easy_install

def magic_easy_install(self, package):
    easy_install.main( ["-U", package] )
    
ip.define_magic('easy_install', magic_easy_install)
    
import pip

def magic_pip(self, package):
    pip.main(['install','--upgrade', package]) 

ip.define_magic('pip', magic_pip)
    
###########################################################################    

print 'pycse-magic loaded.'