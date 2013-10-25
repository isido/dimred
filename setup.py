#! /usr/bin/env python

from distutils.core import setup
setup(name='dimred',
      description='Dimensionality reduction utilities',
      long_description="""A collection of utilities related to dimensionality reduction. Currently provides only trustworthiness and continuity calculations""",
      version='0.0.2',
      author='Ilja Sidoroff',
      author_email='ilja.sidoroff@iki.fi',
      url='none',
      license='GNU Public Licence',
      platforms=['any'],
      packages=['dimred'],
      scripts=['dr-error', 'dr-reduce', 'dr-plot']
      )
