from setuptools import setup, find_packages

install_requires = [
    # "numpy==1.22.1", "pandas==1.4.0", "biopandas==0.2.9", # required by hmtploc
    #    'mdtraj', 'lxml', 'numpy', 'numba', 'openmm', 'openmmforcefields', 'rdkit'
]
setup(
    name='steermd',
    version='0.0.1',
    author='dptech.net',
    author_email='hermite@dptech.net',
    description=('Implicit solvent steered MD.'),
    license=None,
    keywords='Protein structure, Protein-protein interaction',
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    #packages=packages,
    entry_points={'console_scripts': ['steermd = steermd.main:main']},
    include_package_data=True)
