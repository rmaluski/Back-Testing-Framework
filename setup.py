from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys

# Get version from __init__.py
def get_version():
    with open("trading_core/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# C++ extensions
ext_modules = [
    Pybind11Extension(
        "trading_core._core",
        [
            "src/core/event_loop.cpp",
            "src/core/market_data_feed.cpp",
            "src/core/order_book.cpp",
            "src/core/fill_engine.cpp",
            "src/core/portfolio.cpp",
            "src/core/strategy_api.cpp",
            "src/bindings/python_bindings.cpp",
        ],
        include_dirs=[
            "src/include",
            "third_party/pybind11/include",
        ],
        language="c++",
        cxx_std=17,
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            "-DNDEBUG",
        ] if not sys.platform.startswith("win") else [
            "/O2",
            "/DNDEBUG",
        ],
    ),
]

setup(
    name="trading-core",
    version=get_version(),
    description="High-performance event-driven backtesting framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ryan Maluski",
    author_email="ryan.maluski@example.com",
    url="https://github.com/rmaluski/Back-Testing-Framework",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "plotly>=5.15.0",
        "jinja2>=3.1.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "ray>=2.6.0",
    ],
    extras_require={
        "cli": ["click>=8.1.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bt=trading_core.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    include_package_data=True,
    package_data={
        "trading_core": ["templates/*.html", "configs/*.yaml"],
    },
) 