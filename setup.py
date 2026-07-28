from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cuda-ml-kernels",
    version="0.1.0",
    author="Om Uniyal",
    author_email="om.uniyal19@gmail.com",
    description="GPU-accelerated implementations of classic ML algorithms using CUDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OmUniyal/cuda-ml-kernels",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
)