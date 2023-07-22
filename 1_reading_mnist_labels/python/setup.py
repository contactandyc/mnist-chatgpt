from setuptools import setup, find_packages

setup(
    name='file_reader',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'file_reader = file_reader.main:main',
        ],
    },
)
