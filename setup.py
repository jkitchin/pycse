from distutils.core import setup
import os

# for installing magic IPython stuff
import IPython
IPydir = os.path.join(IPython.utils.path.get_ipython_dir(),
                      'profile_default',
                      'startup')
print IPydir
                      

   
setup(name = 'pycse',
      version='1.0',
      description='python computations in science and engineering',
      url='http://github.com/jkitchin/pycse',
      maintainer='John Kitchin',
      maintainer_email='jkitchin@andrew.cmu.edu',
      license='GPL',
      platforms=['linux'],
      packages=['pycse'],
      scripts=['pycse/publish.py',
               #'pycse/bin/submit.py',
               #'pycse/bin/pycse-server.py'
               ],
      data_files=[(IPydir, ['pycse/00-pycse-magic.py'])],
      long_description='''python computations in science and engineering''')
