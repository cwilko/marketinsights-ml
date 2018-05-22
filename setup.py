""" Hello World app for deploying Python functions as APIs on Bluemix """ 
# Always prefer setuptools over distutils 
from setuptools import setup 
from codecs import open 
from os import path 
here = path.abspath(path.dirname(__file__)) 
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read() 

setup( 
	name='marketinsights-model-api', 
	version='1.0.0', 
	description='Train, Deploy, and Score models', long_description=long_description, 
	url='https://github.com/cwilko/marketinsights-ml', license='MIT' 
)