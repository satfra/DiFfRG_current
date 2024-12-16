from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "DiFfRG python utilities package"
LONG_DESCRIPTION = "The DiFfRG python utilities package is a collection of python utilities for the DiFfRG project. It includes utilities for data analysis, plotting, and more."

# Setting up
setup(
    name="DiFfRG",
    version=VERSION,
    author="Franz R. Sattler",
    author_email="<sattler@thphys.uni-heidelberg.de>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "adaptive[notebook]", "pandas", "seaborn", "jupyter_bokeh", "vtk"],
    keywords=["python", "DiFfRG"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
