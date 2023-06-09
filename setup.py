from glob import glob

from setuptools import setup, find_packages

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setup(name='convo_wizard',
      version="1.0",
      description='a library for toxicity forecasting (and conversational aiding)',
      author='Tushaar Gangavarapu',
      scripts=glob('scripts/**/*.py', recursive=True),
      py_modules=[],
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=requirements,
      url='https://github.com/CornellNLP/craft-experiments')
