import setuptools

setuptools.setup(
    name='chemgrid',
    version='0.1.0',
    description='ChemGrid multi-agent game',
    url='https://github.com/mekhlos/chemgrid',
    packages=setuptools.find_packages(),
    install_requires=['pygame', 'gym', 'matplotlib', 'seaborn'],
)
