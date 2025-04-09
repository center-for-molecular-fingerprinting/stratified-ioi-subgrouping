from setuptools import setup, find_packages

setup(
    name="stratified_ioi_subgrouping",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["variability_tree_splitter"],
    install_requires=["pandas", "numpy"],
    author="Zita I. Zarandy",
    description="A decision-tree-like splitter to stratify subjects based on IOI",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/center-for-molecular-fingerprinting/stratified-ioi-subgrouping",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
