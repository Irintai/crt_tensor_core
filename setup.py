# setup.py
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    """Custom build command to handle the optional Cython extension."""
    
    def build_extensions(self):
        try:
            # Try to build with Cython
            from Cython.Build import cythonize
            
            # Define the extension module
            extensions = [
                Extension(
                    "crt_tensor_core.kernels",
                    ["crt_tensor_core/kernels.pyx"],
                    include_dirs=[],
                    libraries=[],
                    language="c++"
                )
            ]
            
            # Cythonize the extension module
            self.extensions = cythonize(extensions)
            
            # Build extensions
            super().build_extensions()
        except ImportError:
            print("Cython not found. Using Python fallback for kernels.")

setup(
    name="crt_tensor_core",
    version="0.1.0",
    author="Andrew 'Irintai' Orth",
    author_email="drewski871@gmail.com",
    description="A tensor library for Cosmological Recursion Theory",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/irintai/crt_tensor_core",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[],
    ext_modules=[],
    cmdclass={"build_ext": BuildExt},
)