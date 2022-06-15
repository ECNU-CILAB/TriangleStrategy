from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="TriangleStrategy",
    version="1.0.1",
    author="Artiprocher",
    author_email="zjduan@stu.ecnu.edu.cn",
    description="TriangleStrategy is a high-efficiency reinforcement learning based algorithmic trading library.",
    url="https://github.com/ECNU-CILAB/TriangleStrategy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages()
)
