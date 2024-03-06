from setuptools import setup

with open(file = './README.md', mode = 'r') as f:
    description = f.read()

setup(
    name = 'dmap',
    version = 'v0.0.1',
    author = 'Nathaniel Tomczak',
    description = 'A dimensional mapping tool.',
    long_description = description,
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    packages = ['dmap'],
    python_requires = '>=3.7',
    install_requires = [], 
    extras_require = {'tests': ['pytest']}
)