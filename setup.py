from setuptools import setup, find_packages

setup(
    name="framework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "controlflow",
        "docker",
        "fastapi",
        "sqlalchemy",
        "uvicorn",
        "python-dotenv",
        "httpx",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "python-multipart",
    ],
) 