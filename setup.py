import os
import re
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    init_py = open(os.path.join(here, 'concarne', '__init__.py')).read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''
try:
    # obtain long description from README and CHANGES
    README = open(os.path.join(here, 'README.rst')).read()
    #CHANGES = open(os.path.join(here, 'CHANGES.rst')).read()
except IOError:
    README = ''
    #CHANGES = ''

install_requires = [
    'numpy',
    'Theano',  # we require a development version, see requirements.txt
    'Lasagne' 
    ]

tests_require = [
#    'mock',
#    'pytest',
#    'pytest-cov',
#    'pytest-pep8',
    ]

setup(name='concarne',
    version=version,
    description='Lightweight contextual learning framework, based on Theano and Lasagne',
    long_description=README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author='Sebastian Hoefer, Rico Jonschkowski',
    author_email='sebastian.hoefer@tu-berlin.de, rico.jonschkowski@tu-berlin.de',
    url='https://gitlab.tubit.tu-berlin.de/rbo-lab/concarne',
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
#    extras_require={
#        'testing': tests_require,
#        },
    )