from setuptools import setup, find_packages

setup(
    name="urgentpk",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18',
        'requests',
    ],
    
    author="Jiahe Wang, Chenda Li, Wei Wang, Wangyou Zhang",
    description="URGENT-PK: Perceptually-Aligned Ranking Model Designed for Speech Enhancement Competition",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/urgent-challenge/URGENT-PK",
    
    entry_points={
        'console_scripts': [
            'urgentpk_rank = urgentpk.rank:main'
            'urgentpk_train = urgentpk.train_urgentpk:main'
        ]
    },
    
    # include_package_data=True,
    # package_data={'my_package': ['data/*.json']}
)