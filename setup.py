from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="PheTK",
    version="0.1.1",
    author="Tran, Tam",
    description="PheTK - Phenotype Toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nhgritctran/PheTK",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "PheTK": ["phecode/*.csv"]
    },
    install_requires=requirements,
    python_requires=">=3.7"
)
