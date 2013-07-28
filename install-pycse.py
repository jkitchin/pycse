## from setuptools.command import easy_install

## for package in ['quantities',
##                 'uncertainties',
##                 'https://github.com/jkitchin/pycse/archive/master.zip',
##                 'https://github.com/jkitchin/pyreport/archive/master.zip']:
##     easy_install.main( ["-U", package] )

import pip

for package in ['quantities',
                'uncertainties',
                'https://github.com/jkitchin/pycse/archive/master.zip',
                'https://github.com/jkitchin/pyreport/archive/master.zip']:

    pip.main(['install','--upgrade', package]) 
