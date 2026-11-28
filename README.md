# Решение T2 + T3 + T4 + T5

Сборка:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t robohack:t2-t3-t5 .
```

Проверка:

```bash
docker run --rm \
  --platform linux/amd64 \
  --network none \
  --entrypoint /entrypoint.sh \
  robohack:t2-t3-t5 ping
```

Ожидаемый ответ `ping`: `supported_tasks` содержит `T2`, `T3`, `T4`, `T5`;
`self_trained_rl` равно `true`, `preferred_speed_mps` равно `0.4`.

## T2. Safety Block

ROS 2 узел `go1_safety` (папка `go1_safety/`) принимает команды из `/lowcmd_bare` и публикует безопасные команды в `/lowcmd`:

- сообщения с NaN / Inf отбрасываются, публикуется последняя принятая команда;
- целевые углы ограничиваются диапазонами суставов, `kp` в [0, 1000], `kd` в [0, 100], `tau` в [-100, 100], `dq` в [-20, 20];
- изменение цели между соседними командами не больше 5°, большие скачки отрабатываются постепенно;
- переходы проверяются на самоколлизии в MuJoCo по 655 парам из `t2_collision_pairs.csv` (шаг 0.05°);
- `/lowcmd` публикуется с частотой 500 Гц;
- если входных команд нет дольше 0.25 с, удерживается последняя цель с `dq = 0`, `tau = 0`.

Запуск с симулятором:

```bash
python3 run_local.py --task T2 --participant-image robohack:t2-t3-t5 --gui
```

Тесты:

```bash
pip install "mujoco==3.3.7" numpy pytest
cd go1_safety
PYTHONPATH=. python3 -m pytest test
```

## T3. Ходьба с RL-политикой

Узел `go1_walking` принимает `/cmd_vel` (`geometry_msgs/msg/Twist`) и `/lowstate`
(`unitree_go/msg/LowState`). Политика формирует цели для 12 суставов в порядке
`FR, FL, RR, RL` и публикует только `/lowcmd_bare`. Неизменённый Safety Block T2
проверяет команды и публикует `/lowcmd`. Оба узла используют `/clock`. Камеры и
истинное положение робота не используются во время исполнения политики.

Параметры походки и обратной связи обучены по награде в MuJoCo методом
эволюционной стратегии. Веса включены в образ; сеть и установка дополнительных
ML-библиотек на старте не нужны. Выбранная скорость — **0,4 м/с**.

Запуск в симуляторе из репозитория организаторов:

```bash
python3 run_local.py --task T3 --participant-image robohack:t2-t3-t5 --gui --logs
```

Контейнер запускает T3 командой `/entrypoint.sh T3`. Воспроизводимая локальная
проверка выполняется через команду запуска симулятора выше.

## T4. Оценка движения робота

Пакет `go1_odometry` оценивает положение и ориентацию робота по IMU, состояниям суставов и лидару. Он публикует `/odom` и TF `odom -> trunk`, не использует `/odom/ground_truth` и не публикует команды движения.

Локальный запуск T4:

```bash
python3 run_local.py --task T4 --participant-image robohack:t2-t3-t5 --gui --logs
```

Запуск узла выполняется командой `/entrypoint.sh T4`. Пакетные тесты находятся в `tests/`, локальные инструменты оценки — в `tools/`.

## T5. Автономная навигация и ArUco

Пакет `go1_navigation` запускает Nav2, публикует `map -> odom`, принимает action `/navigate_to_pose`, формирует `/cmd_vel` и распознаёт ArUco-метки из пяти RGB-камер. Результаты публикуются в `/aruco/detections` как JSON с текущими ID.

Локальный запуск T5 для трассы шириной 1 м:

```bash
python3 run_local.py --task T5 --course-width 1.0 --participant-image robohack:t2-t3-t5 --gui --logs
```

Запуск узла выполняется командой `/entrypoint.sh T5`. Конфигурация Nav2 и дерево поведения находятся в `src/go1_navigation/`.
