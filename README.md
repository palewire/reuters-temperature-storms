How to calculate the supersized heat waves tracked by the [Reuters Climate Monitor](https://www.reuters.com/graphics/CLIMATE-AUTOMATED/MONITOR/akpeykqqapr/heat-storms/).

The code in this repository generated the analysis published in an October 20, 2026 Reuters story headlined ["Giant ‘heat storms’ are scorching the Earth. This is the story of the biggest one on record."](https://www.reuters.com/investigates/special-report/climate-heatstorms-science/). The same logic is used by the [Reuters Climate Monitor](https://www.reuters.com/graphics/CLIMATE-AUTOMATED/MONITOR/akpeykqqapr/heat-storms/) to track heat storms in real time.

The inspiration and methodology are explained in ["How (and why) we developed ‘The Age of Heat Storms’"](https://www.reuters.com/investigates/special-report/climate-heatstorms-methodology/).

![A screenshot of the methodology story explaining how we did it](https://raw.githubusercontent.com/palewire/reuters-temperature-storms/refs/heads/main/.github/climate-method-story.png)

For the Reuters analysis, temperature readings were drawn from the [ERA5 reanalysis dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) published by the [European Centre for Medium-Range Weather Forecasts](https://www.ecmwf.int/)’ [Copernicus](https://climate.copernicus.eu/) program. That data is not included here.

Using different temperature readings and baseline climatologies will lead to differing results. Additionally, small differences in computer architecture may affect temperature readings and percentile scores by very small amounts.

RTS includes every one of the 2,527 storm names issued in the published Reuters
table through September 30, 2026. Each label is bound to the corresponding
verified physical storm's birth date and original occupied grid cell. The
tracker uses these packaged assignments when replaying that history. A missing
or conflicting historical match stops the calculation rather than issuing a
different name. The story names and the published `all_storms.parquet` file
remain unchanged.

Future storms receive names from the same fixed Reuters name sequence. When
several storms become eligible on one day, RTS orders them by birth date and
the first cell in their original birth footprints. That cell address is saved
through movement, splits and mergers and checkpoint restarts. Name priority
therefore does not depend on the hexadecimal ID or polygon formatting. The
same complete physical input history, naming table and saved sequence position
produce the same names in RTS and in Climate Monitor.

The assignments, lookup API and naming rule are included in the installed RTS
package and work offline. Climate Monitor reads its temperature, anomaly,
footprint and cell inputs from its own production system; issued labels do not
replace any of those scientific inputs. The historical name table covers the
maximum-temperature warm series; other temperature/direction series keep their
independent name sequences.

```python
from reuters_temperature_storms import load_published_storm_names
from reuters_temperature_storms.series import WaveSeries

names = load_published_storm_names()
assignment = names.lookup(
    WaveSeries("tasmax", "warm"),
    "19870324-ed909ccd8f5f0433953dfe60175b30b7",
)
assert assignment is not None
assert assignment.name == "Wendy 1987"
```

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
retained `birth_cell_id`; a later moving footprint cannot supply that original
cell. The bundled story table remains the record of names used in the published
Reuters analysis.

If you have questions about the code or data, you're encouraged to contact [Ben Welsh](mailto:ben.welsh@thomsonreuters.com) and [Maurice Tamman](mailto:maurice.tamman@reuters.com) at Reuters.
