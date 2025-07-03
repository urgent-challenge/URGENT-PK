from setuptools import setup, find_packages

setup(
    name="urgentpk",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'fire>=0.7.0',
        'librosa>=0.9.2',
        'lightning>=2.3.3',
        'numpy>=1.23.5',
        'pandas>=2.0.3',
        'scipy>=1.10.1',
        'setuptools>=75.1.0',
        'soundfile>=0.12.1',
        'torch>=2.1.0',
        'torchaudio>=2.1.0',
        'tqdm>=4.66.5',
        # 'utmos>=1.1.10',
        'requests',
    ],
    python_requires='>=3.8',
    
    author="Jiahe Wang, Chenda Li, Wei Wang, Wangyou Zhang, Samuele Cornell, Marvin Sach, Robin Scheibler, Kohei Saijo, Yihui Fu, Zhaoheng Ni, Anurag Kumar, Tim Fingscheidt, Shinji Watanabe, Yanmin Qian",
    description="URGENT-PK: Perceptually-Aligned Ranking Model Designed for Speech Enhancement Competition",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/urgent-challenge/URGENT-PK",
    
    entry_points={
        'console_scripts': [
            'urgentpk_rank=urgentpk.rank:main',
            'urgentpk_train=urgentpk.train_urgentpk:main',
        ]
    },
    
    # include_package_data=True,
    # package_data={'my_package': ['data/*.json']}
)