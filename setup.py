from setuptools import setup, find_packages

setup(
    name="imc_prosperity",
    version="0.1.0",
    description="Compute features on L2 order‑book CSVs and split by product",
    author="Your Name",
    python_requires=">=3.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "l2feat=orderbook_features.cli:main",
            "synthgen=synthetic_data.cli:main"
        ]
    },
    install_requires=[
        "pyspark>=3.0.0",
        "tensorflow",
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    include_package_data=True,
)
