from glob import glob

from setuptools import find_packages, setup

package_name = 'go1_navigation'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/behavior_trees', glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='team@example.com',
    description='Go1 navigation (T5)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'aruco_node = go1_navigation.aruco_node:main',
            'map_odom_node = go1_navigation.map_odom_node:main',
        ],
    },
)
