from setuptools import setup, find_packages

setup(
    name="antibody_al",
    version="1.0.0",
    description="Explainable Active Learning for Antibody Binding Affinity Prediction",
    author="Harshit Singh",
    author_email="explore.research.ai@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "gpytorch>=1.12",
        "scikit-learn>=1.3",
        "numpy>=1.26",
        "scipy>=1.11",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "shap>=0.44",
        "matplotlib>=3.8",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
