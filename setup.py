from setuptools import setup, find_packages

setup(
    name='sci-ml',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sciml=src.main:main',
        ],
    },
)