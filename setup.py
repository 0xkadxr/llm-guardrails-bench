"""Package setup for llm-guardrails-bench."""

from pathlib import Path

from setuptools import find_packages, setup

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="llm-guardrails-bench",
    version="0.1.0",
    author="Abdelkader Benmeriem",
    author_email="abdelkader.benmeriem@example.com",
    description=(
        "A standardized benchmark suite for evaluating LLM safety "
        "guardrails and defense mechanisms."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kadirou12333/llm-guardrails-bench",
    project_urls={
        "Bug Tracker": "https://github.com/kadirou12333/llm-guardrails-bench/issues",
        "Documentation": "https://github.com/kadirou12333/llm-guardrails-bench#readme",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    include_package_data=True,
    package_data={
        "guardrails_bench": ["datasets/data/*.json"],
    },
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    keywords=[
        "llm",
        "ai-safety",
        "benchmark",
        "guardrails",
        "red-teaming",
        "jailbreak",
        "content-filter",
    ],
)
