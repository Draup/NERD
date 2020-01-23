import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NERD", # Replace with your own username
    version="0.1",
    author="Deepak Mishra",
    author_email="deepak_mishra_@outlook.com",
    description="An package to interactively train NER/TextClassification models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Draup-Zinnov/NERD",
    packages=setuptools.find_packages(),
    package_data={
            "NERD": ["html_templates/*"],
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)