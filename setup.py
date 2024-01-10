import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ions',  
     version='0.2.4',
     py_modules = ["ions"],
     install_requires = [
                         "ase>=3.22.1",
                         "numpy",
			             "spglib",
                         "networkx",
                         "scipy"
                        ],
     author="Artem Dembitskiy",
     author_email="art.dembitskiy@gmail.com",
     description="A python library for studying percolation in solids",
     key_words = ['ionic', 'radius', 'bond valence'],
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
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "Programming Language :: Python :: 3.11",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    include_package_data=True,
    packages=setuptools.find_packages(),
 )





