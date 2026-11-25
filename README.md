# 9router-cli

Utilities for operating a local [9Router](http://localhost:20128) gateway.

## `free_sync.py`

Syncs live-verified free chat routes into 9Router's `free-burst` combo:

- Harvests free/chat candidates from `GET /v1/models` and OpenRouter's free catalog (`openrouter/<id>`).
- Probes every candidate live against `POST /v1/chat/completions` with `stream=True`, `tools`, and `reasoning_effort="medium"` (8s timeout).
- Keeps up to 2 fastest passing routes per provider prefix in `combos.models` (`free-burst`) and records failures in `excluded_models`.

```bash
python3 free_sync.py
```

Environment overrides:
- `NINEROUTER_URL` (default `http://localhost:20128`)
- `NINEROUTER_KEY` (falls back to `~/.hermes/.env`)
- `NINEROUTER_DB_PATH` (default `~/.9router/db/data.sqlite`)
- `NINEROUTER_FREE_COMBO` (default `free-burst`)
