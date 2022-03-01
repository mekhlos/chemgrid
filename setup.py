from setuptools import find_packages
from setuptools import setup

setup(
    name='chemgrid',
    version='0.1.0',
    description='ChemGrid multi-agent game',
    url='https://github.com/mekhlos/chemgrid',
    packages=find_packages(),
    install_requires=['pygame', 'gym'],
)
