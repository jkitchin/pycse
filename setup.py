# Copyright 2015, John Kitchin
# (see accompanying license files for details).
from setuptools import setup

setup(name='pycse',
      version='2.0',
      description='python computations in science and engineering',
      url='http://github.com/jkitchin/pycse',
      maintainer='John Kitchin',
      maintainer_email='jkitchin@andrew.cmu.edu',
      license='GPL',
      platforms=['linux'],
      packages=['pycse'],
      setup_requires=['nose>=1.0'],
      data_files=['requirements.txt', 'LICENSE'],
      long_description='''\
python computations in science and engineering
===============================================

This package provides functions that are useful in science and engineering
computations.

See http://kitchingroup.cheme.cmu.edu/pycse for documentation.

      ''')

# python setup.py register to setup user
# to push to pypi - (shell-command "python setup.py sdist upload")
