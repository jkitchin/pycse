from PYCSE import *
__version__ = '1.25.4'
# myextension.py

def pycse_update(*args):
    # for installing magic IPython stuff
    # In ipython run this
    # %load https://raw.github.com/jkitchin/pycse/master/install-pycse.py

    print args
    from setuptools.command import easy_install
    easy_install.main( ["-U","pycse"] )

    # my customized pyreport
    package = 'https://github.com/jkitchin/pyreport/archive/master.zip'
    easy_install.main( ["-U",package] )


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
    easy_install.main( ["-U","quantities"] )
    easy_install.main( ["-U","uncertainties"] )
    print 'Extra packages now installed.'


def load_ipython_extension(ip):
    # The `ipython` argument is the currently active `InteractiveShell`
    # instance, which can be used in any way. This allows you to register
    # new magics or aliases, for example.
    ip.define_magic('pycse_update', pycse_update)
    
    # active true float division
    exec ip.compile('from __future__ import division', '<input>', 'single') \
        in ip.user_ns

def unload_ipython_extension(ip):
    # If you want your extension to be unloadable, put that logic here.
    pass
