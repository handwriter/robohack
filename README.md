# Решение T1

Сборка:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t robohack:t1 .
```

Проверка:

```bash
docker run --rm \
  --platform linux/amd64 \
  --network none \
  --entrypoint /entrypoint.sh \
  robohack:t1 ping
```

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
python3 run_local.py --task T2 --participant-image robohack:t1 --gui
```

Тесты:

```bash
pip install "mujoco==3.3.7" numpy pytest
cd go1_safety
PYTHONPATH=. python3 -m pytest test
```
