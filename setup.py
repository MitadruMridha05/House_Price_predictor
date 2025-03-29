from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str) ->List[str]:
    '''
    This add required packages from requirement.txt
    '''
    requirement=list()
    with open(file_path) as file_obj:
        requirement=file_obj.readlines()
        requirement=[r.replace('\n','') for r in requirement]
    return requirement


setup(
    name="house_price_predictor",
    version="0.0.1",
    author="Mitadru_Mridha",
    author_email="mitadrumridha@outlook.com",
    packages=find_packages(),
    install_requires=get_requirements("requirement.txt")
)

