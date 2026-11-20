from glob import glob

from setuptools import setup

package_name = 'go1_safety'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models/unitree_go1', glob('models/unitree_go1/*.*')),
        ('share/' + package_name + '/models/unitree_go1/assets', glob('models/unitree_go1/assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'safety_node = go1_safety.safety_node:main',
        ],
    },
)
