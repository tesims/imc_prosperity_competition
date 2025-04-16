from setuptools import setup, find_packages

setup(
    name="imc_prosperity",
    version="0.1.0",
    description="Compute features on L2 order‑book CSVs and split by product",
    author="Your Name",
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pyspark>=3.0.0"
    ],
    entry_points={
        "console_scripts": [
            "l2feat=orderbook_features.cli:main"
        ]
    },
    include_package_data=True,
)
