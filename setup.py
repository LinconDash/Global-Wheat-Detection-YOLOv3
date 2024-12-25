from setuptools import find_packages, setup
from typing import List 

def get_requirements(file_path:str) -> List[str]:
    """
        Returns the list of required packages listed in requirements.txt
    """
    HYPHEN_E_DOT = "-e."
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="objectdetectionYOLOv3",
    version="0.0.1",
    author="Lincon",
    author_email="lincondash02@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)