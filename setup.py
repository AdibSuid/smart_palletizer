from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies for computer vision and point cloud processing
install_deps = [
    "numpy>=1.21.0",
    "open3d>=0.18.0",
    "opencv-python>=4.5.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "pillow>=8.0.0",
    "scikit-learn>=1.0.0",
    "scikit-image>=0.19.0",
]

setup(
    name="smart_palletizer",
    version="0.1.0",
    author="Smart Palletizer Team",
    author_email="developer@neura-robotics.com",
    maintainer="Smart Palletizer Team",
    maintainer_email="team@neura-robotics.com",
    description=(
        "A computer vision toolkit for box detection and 6D pose estimation "
        "in robotic palletizing applications using classical CV methods."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdibSuid/smart_palletizer",
    project_urls={
        "Bug Tracker": "https://github.com/AdibSuid/smart_palletizer/issues",
        "Documentation": "https://github.com/AdibSuid/smart_palletizer/blob/main/METHODOLOGY.md",
        "Source Code": "https://github.com/AdibSuid/smart_palletizer",
    },
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_deps,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="computer-vision robotics point-cloud pose-estimation icp detection",
    entry_points={
        'console_scripts': [
            'smart-palletizer=smart_palletizier.pipeline:main',
        ],
    },
)
