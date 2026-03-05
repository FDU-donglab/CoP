"""Setup script for Noise Genome Estimator package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="noise-genome-estimator",
    version="1.0.0",
    author="Yuanjie Gu",
    author_email="yuanjie.gu@fudan.edu.cn",
    description="Deep learning model for noise parameter estimation in images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FDU-donglab/CoP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "noise-estimator-train=train:main",
        ],
    },
)
