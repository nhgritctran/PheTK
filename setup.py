from setuptools import setup, find_packages

setup(
    name="PheTK",
    version="0.1.32",
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
    install_requires=["adjusttext",
                      "connectorx",
                      "google-cloud-bigquery",
                      "matplotlib",
                      "numpy",
                      "pandas",
                      "polars",
                      "pyarrow",
                      "tqdm",
                      "statsmodels"],
    python_requires=">=3.7"
)
