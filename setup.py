import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CRBMG",
    version="0.1",
    author="ZHANG Jun-jie",
    author_email="zjacob@mail.ustc.edu.cn",
    description="Coupled Relativistic Boltzmann Maxweel's on GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Juenjie/CRBMG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0",
        "Operating System :: Linux",
    ],
)
