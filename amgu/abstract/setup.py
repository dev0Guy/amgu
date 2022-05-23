from setuptools import setup

setup(
    name='abstract',
    version='1.0',
    description='Abstract model for future project to use',
    author='Guy Arieli',
    author_email='guyarieli17@yahoo.com',
    packages=['abstract'],  #same as name
    install_requires=['ray', 'gym', 'numpy','torch'], #external packages as dependencies
)