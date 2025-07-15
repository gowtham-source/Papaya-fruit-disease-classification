from setuptools import setup, find_packages

setup(
    name="papaya_models",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "albumentations",
        "opencv-python",
        "numpy",
        "Pillow",
        "tqdm",
        "tensorboard",
    ],
)
