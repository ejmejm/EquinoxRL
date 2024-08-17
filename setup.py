from setuptools import setup, find_packages

setup(
    name="discovery",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author="Edan Meyer",
    author_email="ejmejm98@gmail.com",
    description="Research on the disovery problem implemented with JAX and Equinox",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ejmejm/discovery",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
