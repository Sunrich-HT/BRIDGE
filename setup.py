from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bridge-persona",
    version="0.1.0",
    description="BRIDGE: Triangular Fixed-Point Refinement for Long-Horizon Persona Consistency (ICML 2026)",
    author="Yinghui Jiang, Bocheng Xu, Jianye Xie, Haotong Sun",
    url="https://github.com/Sunrich-HT/BRIDGE",
    python_requires=">=3.10",
    packages=find_packages(include=["bridge", "bridge.*"]),
    install_requires=reqs,
    license="Apache-2.0",
)
