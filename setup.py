from setuptools import find_packages, setup

setup(
    name="bcad",
    version="1",
    description="Bootsrap Comparison of Attractor Dimensions",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author="Yair Daon",
    author_email="firstname.lastname@gmail.com",
    license="MIT"
)
