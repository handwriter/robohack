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
