from setuptools import setup, find_packages

setup(
    name="my_vietnamese_nlp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyvi",
        "scikit-learn",
        "tensorflow"
    ],
    author="Your Name",
    description="A Vietnamese NLP library for text processing and tokenization",
)
