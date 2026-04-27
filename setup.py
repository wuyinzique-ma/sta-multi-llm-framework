from setuptools import find_packages, setup

setup(
    name="semanticher",
    version="0.1.0",
    author="Yin Wu",
    author_email="",
    description="Semantic Annotation for Energy Data with Diverse Large Language Models",
    license="MIT",
    url="https://github.com/your-repo/sta-multi-llm-framework",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)