from pathlib import Path
from setuptools import setup, find_packages

metadata_path = Path(__file__).parent / 'src' / 'opensignals' / '__about__.py'
metadata = {}
with metadata_path.open() as file:
    raw_code = file.read()
exec(raw_code, metadata)
metadata = {key.strip('_'): value for key, value in metadata.items()}
metadata['name'] = metadata.pop('package_name')

setup(
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/councilofelders/opensignals',
    project_urls={
        'Bug Tracker': 'https://github.com/councilofelders/opensignals/issues',
    },
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[path.stem for path in Path('src').glob('*.py')],
    python_requires='>=3.7',
    zip_safe=True,
    include_package_data=True,
    install_requires=[
        'docopt',
        'pandas==1.2.2',
        'numpy==1.20.1',
        'pyarrow==3.0.0',
        'requests',
        'tqdm',
    ],
    extras_require=dict(
        test=[
            'pytest==5.1.2',
            'pytest-cov==2.7.1',
            'pytest-flake8==1.0.6',
            'pytest-mypy==0.4.0',
            'pydocstyle==4.0.1',
            'pep8-naming==0.8.1',
            'pytest-docstyle==2.0.0',
            'flake8 == 3.8.1',
        ],
    ),
    entry_points={
        'console_scripts': [
            'opensignals = opensignals.__main__:main',
        ]
    },
    **metadata,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
    ],
)
