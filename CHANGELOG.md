# Changelog

## Unreleased

- Assign storm names by birth date and birth ordering key, with the next unused
  name position stored in tracking checkpoints.
- Generate permanent storm IDs from the birth date and first occupied cell on
  the fixed 0.25° grid, independent of polygon text and ring order.
- Resolve equal overlaps by birth date and daily footprint ordering.
- Require complete daily native-cell coverage when preparing footprint ordering.
- Pin Shapely to 2.1.2 and require GEOS 3.13.1 for footprint ordering.
- Store `birth_cell_id`, `order_digest`, `birth_order_digest` and the next unused
  name position in checkpoints.
- Generate names from the packaged library and seed, independently of the bundled
  storm catalog.
- Use the full README as the package description on PyPI.
