from glob import glob

from setuptools import find_packages, setup

package_name = 'go1_odometry'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='team@example.com',
    description='Go1 odometry (T4)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'odometry_node = go1_odometry.node:main',
        ],
    },
)
