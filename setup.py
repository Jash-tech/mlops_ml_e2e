from setuptools import find_packages,setup
from typing import List


HYPHEN_E_DOT='-e .'

def get_requirements(filepath:str)->List[str]:
    requirements=[]
    with open (filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(
    name='GemStonePrediction',
    version='0.0.2',
    author='jash suke',
    author_email='jashsuke@gmail.com',
    install_requires=get_requirements('requirements_dev.txt'),
    packages=find_packages()
)