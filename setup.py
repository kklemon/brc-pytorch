from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / 'README.md').read_text()

setup(
    name='brc',
    packages=find_packages(),
    version='0.1.2',
    license='MIT',
    description='Implementation of the bistable recurrent cell (BRC) in PyTorch',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Kristian Klemon',
    author_email='kristian.klemon@gmail.com',
    url='https://github.com/kklemon/brc-pytorch',
    keywords=['artificial intelligence', 'deep learning'],
    install_requires=['torch']
)
