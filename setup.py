# setup.py
from setuptools import setup, find_packages

setup(
    name="vietnamese_nlp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyvi",
        "scikit-learn",
        "tensorflow"
    ],
    author="Dght1104_SuperCow",
    description="Thư viện xử lý ngôn ngữ tự nhiên cho tiếng Việt",
)
