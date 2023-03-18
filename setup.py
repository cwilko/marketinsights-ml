from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='marketinsights-ml',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Train, Deploy, and Score models',  # Required

    install_requires=requirements,

    extras_require={
        'server': ['Flask>=2.2.3', 'Flask-Cors>=3.0.10', 'flask-restx>=1.1.0', 'Flask-SSLify>=0.1.5', 'marketinsights-remote @ git+https://github.com/cwilko/marketinsights-remote.git'],
        'tf': ['tradeframework @ git+https://github.com/cwilko/tradeframework.git', 'marketinsights-remote @ git+https://github.com/cwilko/marketinsights-remote.git']

        # Successfully installed Flask-2.2.3 Werkzeug-2.2.3 aniso8601-9.0.1
        # click-8.1.3 Flask-Cors-3.0.10 Flask-SSLify-0.1.5 flask-restx-1.1.0
        # importlib-metadata-6.0.0 itsdangerous-2.1.2 jsonschema-4.17.3

    },

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_namespace_packages(exclude=['contrib', 'docs', 'test', 'build', 'docker', 'notebooks', 'models'])  # Required

)
