from distutils.core import setup

setup(name = 'pycse',
      version='0.1',
      description='python computations in science and engineering',
      url='http://github.com/jkitchin/pycse',
      maintainer='John Kitchin',
      maintainer_email='jkitchin@andrew.cmu.edu',
      license='GPL',
      platforms=['linux'],
      packages=['pycse'],
      scripts=['pycse/bin/publish.py',
               'pycse/bin/submit.py',
               'pycse/bin/pycse-server.py'],
      long_description='''python computations in science and engineering''')
