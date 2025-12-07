from setuptools import setup, find_packages

setup(
    name="graphvelo",
    version="0.2.0",
    description="Geometric Deep Learning for RNA Velocity Denoising & Simulation",
    author="Beyza Kaya",
    packages=find_packages(),
    install_requires=[
        "scanpy",
        "scvelo",
        "torch",
        "torch-geometric",
        "matplotlib",
        "seaborn",
        "scipy",
        "numpy",
        "pandas"
    ],
    python_requires='>=3.8',
)