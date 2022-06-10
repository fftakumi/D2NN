from setuptools import setup, find_packages

setup(
    name='d2nn',
    version='0.1.0',
    license='proprietary',

    packages=find_packages(where='one_dim'),
    package_dir={'': 'one_dim'}
)