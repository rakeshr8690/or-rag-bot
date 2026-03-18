"""
Setup script for OR RAG Bot package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="or-rag-bot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RAG-based Operations Research chatbot with LLM and optimization solver integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/or-rag-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "or-rag-bot-setup=src.main:main",
            "or-rag-bot-serve=src.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.html", "*.css", "*.js"],
    },
)