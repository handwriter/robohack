from setuptools import find_packages, setup

package_name = 'go1_locomotion'
share = 'share/' + package_name

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (share, ['package.xml']),
        (share + '/launch', ['launch/t3.launch.py']),
        (share + '/policy', ['policy/t3_weights.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='team@example.com',
    description='Unitree Go1 locomotion policy for ROBOHACK T3',
    license='MIT',
    entry_points={'console_scripts': ['walking_node = go1_locomotion.walking_node:main']},
)
