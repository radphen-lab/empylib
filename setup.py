#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

package_name = 'empylib'
PACKAGES = [package_name]
ROOT = Path(__file__).parent

def get_init_val(val, packages=PACKAGES):
    pkg_init = ROOT / PACKAGES[0] / '__init__.py'
    value = '__%s__' % val
    with pkg_init.open(encoding='utf-8') as fn:
        for line in fn:
            if line.startswith(value):
                return line.split('=', 1)[1].strip().strip("'\"")
    raise RuntimeError(f'Unable to find {value} in {pkg_init}')

setup(
    name=get_init_val('title'),
    version=get_init_val('version'),
    description=get_init_val('description'),
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author=get_init_val('author'),
    url=get_init_val('url'),
    project_urls={
        'Source': get_init_val('url'),
        'Changelog': f"{get_init_val('url')}/blob/master/CHANGELOG.txt",
    },
    package_data={
        package_name: [
            'py.typed',
            'nk_files/*.nk',
            'spectra_data/*.txt',
            'spectra_data/*.xls',
            'IS_reflectance_reference/*.csv',
        ],
    },
    license=get_init_val('license'),
    license_files=('LICENCE.txt',),
    python_requires='>=3.10,<3.15',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='computational electromagnetics mie scattering radiative transfer optics',
    install_requires=[
        'numpy>=1.24',
        'pandas>=1.5',
        'scipy>=1.10',
        'pyyaml>=6',
        'requests>=2.28',
        'typing-extensions>=4.0',
        'colour-science>=0.4.6',  # for color science utilities
        'radphen-iadpython>=0.7.0,<0.8',
    ],
    packages=find_packages(include=[package_name, package_name + '.*']),
)
