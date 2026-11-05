# Changelog

## Unreleased

- Add an opt-in `--region WEST,SOUTH,EAST,NORTH` option to `rts anomalies` (and `region` on `AnomalyRun`) that checks inputs and outputs for every cell of a quarter-degree rectangle instead of the global grid. Runs without it still require the full 1440 by 721 grid.
- Use the full README as the package description on PyPI.
