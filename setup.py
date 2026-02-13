import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ions',  
     version='0.5.1',
     py_modules = ["ions"],
     install_requires = [
                         "ase",
                         "numpy",
			             "spglib",
                         "networkx",
                         "scipy",
                         "pymatgen"
                        ],
     author="Artem Dembitskiy",
     author_email="art.dembitskiy@gmail.com",
     description="A python library for studying ionic conductors",
     key_words = ['ionic', 'radius', 'bond valence', 'percolation'],
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dembart/ions",
     package_data={"ions": ["*.txt", "*.rst", '*.md', "*"], 
     '':['ions/data/*.pkl'], 
     '':['ions/data/*.yaml'], 
     '':['ions/data/*.json'], 
     '':['ions/data/*.text'], 
     '':['ions/data/*.csv'], 
      '':['ions/data/*.cif'], 

     },
     classifiers=[
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "Programming Language :: Python :: 3.11",
         "Programming Language :: Python :: 3.12",
         "Programming Language :: Python :: 3.13",
         "Programming Language :: Python :: 3.14",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    include_package_data=True,
    packages=setuptools.find_packages(),
 )





