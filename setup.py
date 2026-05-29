from setuptools import setup, find_packages

# Optional: read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="AnytimeTrajectoryPredictor",
    version="0.1.0",
    url="https://github.com/Pmk2021/VisualThinkingProject",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "torch",
        "torchvision",
        "tqdm",
        "wandb",
        "Pillow",
        "ultralytics",
        "torch_geometric"
    ],
    extras_require={
        "waymo": [
            "pandas",
            "pyarrow",
            "Pillow",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
