# Storm names and IDs

Storm names come from the fixed Reuters name library and seed. The naming
module can generate any position in that sequence without weather data or a
published storm table:

```python
from reuters_temperature_storms.naming import name_at_ordinal

first_name = name_at_ordinal(0)
next_name = name_at_ordinal(1)
```

Keeping that sequence is only part of reproducing the published names. When
several storms first qualify on one day, they must receive those names in the
same order as the original analysis: birth date, then the original birth
ordering key. RTS computes those keys from the complete day of occupied native
cells,
using the original rectangle, polygon-joining and text order. Keeping the
complete daily batch matters: the order of disconnected dateline pieces can
change if a footprint is calculated alone. The package specifies the sorting and joining steps explicitly so macOS and Linux do not
choose different orders for equal coordinates. Original cell coverage and
qualifying, grace and gap-fill roles supply this calculation.

Naming keeps the birth ordering key through movement, continuation and merge
survival. A new split identity gets its own birth key. Daily ordering keys also
preserve the original tie decisions when overlaps are equal. Public IDs are
separate from both ordering keys, so refactoring their spelling cannot reorder
names. Checkpoints retain these facts and the next unused sequence position.
Given identical complete physical inputs and saved state, the tracker assigns
names in the same order. No issued-name lookup table or published storm file
supplies runtime assignments.

The tracker identifies a new storm from its birth date and the first occupied
cell in its complete footprint on the fixed 0.25° grid. Cells are ordered by
latitude, then longitude; the two map halves of a cell at +180° count once.
Polygon text, ring order and other shape details do not participate in the ID.
Overlap still determines which storm continues through growth, splits and
mergers, and the surviving storm retains its birth ID. Prepared footprint-file
labels remain available for joining associated cell records.

These IDs use the complete 32-character date-and-cell digest after the birth
date. Existing catalogs and checkpoints made with the earlier shape-based IDs
must be rebuilt before using this tracker. Checkpoints also require the
retained `birth_cell_id`, daily `order_digest`, and
`birth_order_digest`; a later moving footprint cannot supply original birth
facts. The bundled story table remains the record of names used in the
published Reuters analysis.
