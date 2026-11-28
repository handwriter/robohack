import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'go1_odometry'))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'go1_navigation'))
