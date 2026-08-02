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
            '*.pyi',
            'py.typed',
            'nk_files/*.nk',
            'spectra_data/*.txt',
            'spectra_data/*.xls',
            'IS_reflectance_reference/*.csv',
        ],
    },
    license=get_init_val('license'),
    license_files=('LICENCE.txt',),
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='computational electromagnetics mie scattering radiative transfer optics',
    install_requires=[
        'numpy>=1.15.0',     # or >=1.20.0 if you want a more modern floor
        'pandas>=0.24.0',    # or >=1.0.0 for better support
        'scipy>=1.1.0',
        'pyyaml>=5.1',
        'requests>=2.18.0',
        'typing-extensions>=4.0',
        'colour-science>=0.4.6',  # for color science utilities
        'iadpython @ git+https://github.com/PanxoPanza/iadpython.git' # forked iadpython version
    ],
    packages=find_packages(include=[package_name, package_name + '.*']),
)
