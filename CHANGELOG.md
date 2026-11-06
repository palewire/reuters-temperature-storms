# Changelog

## Unreleased

- Preserve storm naming order across macOS and Linux by retaining each storm's
  birth ordering key and the next unused name position through tracking and
  checkpoint restart.
- Generate permanent storm IDs from the birth date and first occupied cell on
  the fixed 0.25° grid, independent of polygon text and ring order.
- Preserve ordering when overlaps are equal and when storms move, split or merge.
- Require complete daily native-cell coverage when preparing footprint ordering.
- Pin Shapely to 2.1.2 and require GEOS 3.13.1 for footprint ordering.
- **Breaking change:** catalogs and checkpoints using earlier shape-based IDs
  must be rebuilt. Checkpoints now require `birth_cell_id`, `order_digest` and
  `birth_order_digest` in addition to the saved name sequence position.
- Keep the existing name library, seed and bundled published storm table unchanged.
- Use the full README as the package description on PyPI.
