"""T5: ArUco, map->odom, конфигурация Nav2 и запуск — без ROS, Nav2 и Gazebo.

Nav2 и Gazebo здесь не запускаются: проверяются распознавание на отрендеренных метках,
математика TF, согласованность конфигов с ТЗ и проводка узлов на заглушках ROS.
"""
import importlib.util
import json
import math
import os
import re
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

import marker_scene
from go1_navigation import frames
from go1_navigation.aruco import (ArucoDetector, DetectionHub, TemporalConfirmer, format_message,
                                  image_to_gray, quad_ok)

ROOT = os.path.join(os.path.dirname(__file__), '..')
NAV = os.path.join(ROOT, 'src', 'go1_navigation')
COURSE_IDS = list(range(19))          # метки трассы 1 м: aruco_0 .. aruco_18


# ----------------------------------------------------------------------- ArUco
@pytest.fixture(scope='module')
def detector():
    return ArucoDetector()


@pytest.mark.parametrize('camera,dist,yaw', [
    ('front', 1.0, 0), ('front', 2.0, 30), ('front', 3.0, 0), ('front', 1.5, 60),
    ('side', 0.5, 0), ('side', 0.5, 60), ('side', 1.0, 40), ('side', 1.5, 0)])
def test_course_markers_are_detected(detector, camera, dist, yaw):
    for mid in COURSE_IDS:
        img = marker_scene.render(mid, dist, yaw, camera, seed=mid)
        assert [d.id for d in detector.detect(img)] == [mid], (mid, camera, dist, yaw)


def test_two_markers_in_one_frame(detector):
    a = marker_scene.render(3, 1.2, 0, 'front', offset=(-0.35, 0.0))
    b = marker_scene.render(11, 1.2, 0, 'front', offset=(0.35, 0.0))
    img = np.minimum(a, b)
    assert sorted(d.id for d in detector.detect(img)) == [3, 11]


def test_no_false_ids_on_clutter(detector):
    rng = np.random.default_rng(0)
    false = 0
    for k in range(200):
        kind = k % 4
        if kind == 0:
            img = rng.integers(0, 256, (480, 640)).astype(np.uint8)            # белый шум
        elif kind == 1:
            img = (rng.random((480, 640)) > 0.5).astype(np.uint8) * 255        # чёрно-белый шум
        elif kind == 2:                                                        # мелкая шахматка
            c = 4 + k % 9
            img = (((np.indices((480, 640)) // c).sum(axis=0) % 2) * 255).astype(np.uint8)
        else:                                                                  # плавный градиент + шум
            img = np.clip(np.linspace(0, 255, 640)[None, :].repeat(480, 0)
                          + rng.normal(0, 8, (480, 640)), 0, 255).astype(np.uint8)
        false += len(detector.detect(img))
    assert false == 0


def test_empty_scene_has_no_detections(detector):
    assert detector.detect(np.full((400, 464), 150, np.uint8)) == []


def test_quad_geometry_filter():
    good = np.array([[0, 0], [30, 0], [30, 30], [0, 30]], float)
    assert quad_ok(good)
    assert not quad_ok(np.array([[0, 0], [30, 0], [0, 30], [30, 30]], float))        # самопересечение
    assert not quad_ok(np.array([[0, 0], [5, 0], [5, 5], [0, 5]], float))            # слишком мелкая
    assert not quad_ok(np.array([[0, 0], [100, 0], [100, 10], [0, 10]], float))      # слишком вытянутая
    assert not quad_ok(np.array([[0, 0], [30, 0], [30, np.nan], [0, 30]], float))


def test_temporal_confirmer_requires_repeated_sightings():
    c = TemporalConfirmer(confirm=2, window=0.7)
    assert c.update(0.0, {5}) == set()                  # единичное обнаружение не публикуется
    assert c.update(0.1, {5}) == {5}
    assert c.update(2.0, {5}) == set()                  # старые попадания устарели
    assert c.update(2.1, {5, 9}) == {5}                 # новый ID ещё не подтверждён


def test_hub_publishes_only_current_detections():
    h = DetectionHub(hold=0.3)
    h.update('/camera_left', 10.0, {1, 2})
    h.update('/camera', 10.1, {2, 3})
    assert h.current(10.2) == [1, 2, 3]
    assert h.current(10.35) == [2, 3]                   # ID 1 больше не виден — исчез из списка
    assert h.current(11.0) == []                        # накопления нет


def test_message_format_matches_tz():
    assert json.loads(format_message([12, 3, 7, 3])) == {'ids': [3, 7, 12]}
    assert json.loads(format_message([])) == {'ids': []}


def test_image_decoding():
    class Img:
        pass

    gray = marker_scene.render(0, 1.0)
    msg = marker_scene.to_rgb8_message(gray, Img)
    assert np.array_equal(image_to_gray(msg), gray)
    msg.encoding = 'bgr8'
    assert np.array_equal(image_to_gray(msg), gray)
    # строка с выравниванием (step больше ширины)
    wide = Img()
    wide.height, wide.width, wide.encoding = 4, 5, 'mono8'
    wide.step = 8
    wide.data = bytes(range(32))
    out = image_to_gray(wide)
    assert out.shape == (4, 5) and out[1, 0] == 8 and out[3, 4] == 28
    msg.encoding = '32FC1'
    assert image_to_gray(msg) is None
    msg.encoding, msg.data = 'rgb8', b'\x00' * 10
    assert image_to_gray(msg) is None                   # данных меньше, чем заявлено


# ------------------------------------------------------------------------ TF
def test_map_to_odom_math():
    assert frames.map_to_odom((0, 0, 0), (0, 0, 0)) == (0.0, 0.0, 0.0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        pm = (*rng.uniform(-5, 5, 2), rng.uniform(-3, 3))
        po = (*rng.uniform(-5, 5, 2), rng.uniform(-3, 3))
        t = frames.map_to_odom(pm, po)
        back = frames.compose(t, po)                    # T_map_odom * T_odom_trunk = T_map_trunk
        assert abs(back[0] - pm[0]) < 1e-9 and abs(back[1] - pm[1]) < 1e-9
        assert abs(frames.wrap(back[2] - pm[2])) < 1e-9
    assert abs(frames.yaw_from_quat(0, 0, math.sin(0.4), math.cos(0.4)) - 0.8) < 1e-12


# ------------------------------------------------------------- конфиги Nav2
@pytest.fixture(scope='module')
def params():
    with open(os.path.join(NAV, 'config', 'nav2_params.yaml')) as f:
        return yaml.safe_load(f)


def test_nav2_frames_and_topics(params):
    for node in ('bt_navigator', 'behavior_server'):
        assert params[node]['ros__parameters']['robot_base_frame'] == 'trunk'
    assert params['bt_navigator']['ros__parameters']['global_frame'] == 'map'
    assert params['bt_navigator']['ros__parameters']['odom_topic'] == '/odom'
    for name, frame in (('local_costmap', 'odom'), ('global_costmap', 'map')):
        p = params[name][name]['ros__parameters']
        assert p['global_frame'] == frame and p['robot_base_frame'] == 'trunk'
        assert p['obstacle_layer']['scan']['topic'] == '/scan'
        assert p['obstacle_layer']['scan']['data_type'] == 'LaserScan'


def test_nav2_every_node_uses_sim_time(params):
    def walk(d):
        if isinstance(d, dict):
            if 'ros__parameters' in d:
                yield d['ros__parameters']
            for v in d.values():
                yield from walk(v)
    nodes = list(walk(params))
    assert len(nodes) >= 7
    assert all(n.get('use_sim_time') is True for n in nodes)


def test_nav2_limits_fit_champ(params):
    # CHAMP (gait.yaml): vx 0.5, vy 0.25, wz 1.0
    c = params['controller_server']['ros__parameters']['FollowPath']
    assert c['desired_linear_vel'] <= 0.5 and c['allow_reversing'] is False
    s = params['velocity_smoother']['ros__parameters']
    assert s['max_velocity'][0] <= 0.5 and s['max_velocity'][1] == 0.0 and s['max_velocity'][2] <= 1.0
    assert abs(s['min_velocity'][0]) <= 0.5 and s['min_velocity'][1] == 0.0


def test_goal_tolerance_inside_judge_threshold(params):
    g = params['controller_server']['ros__parameters']['general_goal_checker']
    assert g['xy_goal_tolerance'] <= 0.5                          # ТЗ: <= 0.5 м
    assert g['yaw_goal_tolerance'] <= math.radians(30)            # ТЗ: <= 30 град


def test_global_costmap_covers_course_and_goal(params):
    p = params['global_costmap']['global_costmap']['ros__parameters']
    assert p['rolling_window'] is False
    x0, y0 = p['origin_x'], p['origin_y']
    x1, y1 = x0 + p['width'], y0 + p['height']
    # трасса из мира course_w10: x -1.6..23.9, y -4.4..2.3; цель (22.7, 0)
    assert x0 < -1.6 and x1 > 23.9 and y0 < -4.4 and y1 > 2.3
    assert x0 < 22.7 < x1 and y0 < 0 < y1
    assert 'static_layer' not in p['plugins']
    assert params['planner_server']['ros__parameters']['GridBased']['allow_unknown'] is True
    assert isinstance(p['width'], int) and isinstance(p['origin_x'], float)   # типы для rclcpp


def test_footprint_fits_one_metre_corridor(params):
    for name in ('local_costmap', 'global_costmap'):
        fp = yaml.safe_load(params[name][name]['ros__parameters']['footprint'])
        ys = [abs(p[1]) for p in fp]
        assert max(ys) * 2 < 0.65         # зазор у перегородок трассы ~0.65 м


def test_behavior_tree_is_valid_and_has_no_spin_or_backup():
    root = ET.parse(os.path.join(NAV, 'behavior_trees', 'navigate_no_backup.xml')).getroot()
    tags = {e.tag for e in root.iter()}
    assert {'ComputePathToPose', 'FollowPath', 'RateController', 'PipelineSequence'} <= tags
    assert not ({'Spin', 'BackUp', 'DriveOnHeading'} & tags)
    # все узлы дерева входят в набор стандартных BT-узлов Nav2 Humble
    known = {'root', 'BehaviorTree', 'RecoveryNode', 'PipelineSequence', 'RateController',
             'ComputePathToPose', 'ReactiveFallback', 'GoalUpdated', 'Wait', 'FollowPath',
             'ClearEntireCostmap', 'RoundRobin', 'Sequence'}
    assert tags <= known, tags - known
    assert root.get('main_tree_to_execute') == 'MainTree'


# --------------------------------------------------------------- launch-файл
def load_launch_module():
    saved = {k: sys.modules.get(k) for k in ('launch', 'launch.actions', 'launch.substitutions',
                                              'launch_ros', 'launch_ros.actions',
                                              'ament_index_python', 'ament_index_python.packages')}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m

    cls = lambda n: type(n, (), {'__init__': lambda self, *a, **k: self.__dict__.update(  # noqa: E731
        args=a, kw=k)})
    mod('launch', LaunchDescription=cls('LaunchDescription'))
    mod('launch.actions', DeclareLaunchArgument=cls('DeclareLaunchArgument'),
        OpaqueFunction=cls('OpaqueFunction'))
    mod('launch.substitutions', LaunchConfiguration=cls('LaunchConfiguration'))
    mod('launch_ros')
    mod('launch_ros.actions', Node=cls('Node'))
    mod('ament_index_python')
    mod('ament_index_python.packages', get_package_share_directory=lambda n: NAV)
    spec = importlib.util.spec_from_file_location('t5_launch', os.path.join(NAV, 'launch', 't5.launch.py'))
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return m


def test_launch_params_file_gets_bt_path():
    m = load_launch_module()
    path = m.build_params_file(NAV)
    try:
        with open(path) as f:
            p = yaml.safe_load(f)
        bt = p['bt_navigator']['ros__parameters']['default_nav_to_pose_bt_xml']
        assert os.path.isabs(bt) and bt.endswith('navigate_no_backup.xml')
        assert p['controller_server']['ros__parameters']['FollowPath']['desired_linear_vel'] == 0.3
    finally:
        os.remove(path)
    # узлы жизненного цикла совпадают с запускаемыми серверами Nav2
    assert m.LIFECYCLE_NODES == ['controller_server', 'planner_server', 'behavior_server',
                                 'bt_navigator', 'velocity_smoother']


def test_launch_starts_all_required_nodes():
    m = load_launch_module()
    ctx = types.SimpleNamespace()
    sys.modules['launch.substitutions'] = types.ModuleType('x')
    # LaunchConfiguration('nav').perform(ctx) -> 'true'
    m.LaunchConfiguration = lambda name: types.SimpleNamespace(perform=lambda c: 'true')
    actions = m._setup(ctx)
    pairs = {(a.kw['package'], a.kw['executable']) for a in actions}
    assert ('go1_odometry', 'odometry_node') in pairs
    assert ('go1_navigation', 'map_odom_node') in pairs and ('go1_navigation', 'aruco_node') in pairs
    for exe in ('controller_server', 'planner_server', 'behavior_server', 'bt_navigator',
                'velocity_smoother', 'lifecycle_manager'):
        assert any(e == exe for _, e in pairs), exe
    lm = [a for a in actions if a.kw['executable'] == 'lifecycle_manager'][0]
    assert lm.kw['parameters'][0]['bond_timeout'] == 0.0
    ctrl = [a for a in actions if a.kw['executable'] == 'controller_server'][0]
    assert ('cmd_vel', 'cmd_vel_nav') in ctrl.kw['remappings']
    sm = [a for a in actions if a.kw['executable'] == 'velocity_smoother'][0]
    assert ('cmd_vel_smoothed', 'cmd_vel') in sm.kw['remappings']


def test_setup_and_entry_points_are_consistent():
    setup = open(os.path.join(NAV, 'setup.py')).read()
    for mod in ('aruco_node', 'map_odom_node'):
        assert os.path.exists(os.path.join(NAV, 'go1_navigation', mod + '.py'))
        assert re.search(r"'%s = go1_navigation\.%s:main'" % (mod, mod), setup)
    pkg = open(os.path.join(NAV, 'package.xml')).read()
    assert '<name>go1_navigation</name>' in pkg and 'navigation2' in pkg
