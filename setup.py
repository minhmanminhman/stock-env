import setuptools
REQUIRED_PACKAGES = [
    'numpy==1.23.3',
    'pandas==1.4.4',
    'scipy==1.9.1',
    'empyrical',
    'pyfolio',
    'stable-baselines3[extra]',
    'mt4_hst',
    'pandas-ta==0.3.14b0',
    'finplot==1.8.2',
]

PACKAGE_NAME = 'stock_env'
PACKAGE_VERSION = '1.0.0'

setuptools.setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description='Stock Env',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
)
