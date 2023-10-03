from setuptools import setup, find_packages

setup(
    name='hotshot_xl',
    version='1.0',
    packages=find_packages(include=['hotshot_xl*',]),
    author="Natural Synthetics Inc",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "diffusers>=0.21.4",
        "transformers>=4.33.3",
        "einops"
    ],
)