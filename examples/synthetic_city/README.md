# Synthetic example city ("Testburg")

A four-district synthetic city that exercises the complete workflow without any Münster data. It has a 7 × 7 footway grid, 16 residential buildings, one existing Kita, a census grid with one fallback cell, a forecast and the four vulnerability indicators.

```bash
uv run python examples/synthetic_city/make_city.py              # raw files -> examples/synthetic_city/data/ (git-ignored)
uv run childcare-access check   --city synthetic_city
uv run childcare-access build   --city synthetic_city            # -> workspace/synthetic_city/inputs
uv run childcare-access solve   --city synthetic_city --p 3 --run-name P3      # Gurobi (small; any licence)
uv run python examples/synthetic_city/verify_enumeration.py      # every run vs exhaustive enumeration
```

The raw files are written through `childcare_access.adapters`, as a new city's files would be. The configuration is `config/synthetic_city.toml`, filled in from `config/city_template_schema1.toml`.
