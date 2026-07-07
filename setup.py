# ZX-2026-0303: 基于多模态感知的机械臂洗碗机智能化摆盘
#
# Usage:  pip install -e .
# Requires: Isaac Lab installed at ~/IsaacLab/

from setuptools import setup, find_packages

setup(
    name="dishwasher",
    version="0.1.0",
    description="Multi-modal perception robotic dishwasher placement (ZX-2026-0303)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "opencv-python>=4.8",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
    ],
)
