How to calculate the supersized heat waves tracked by the [Reuters Climate Monitor](https://www.reuters.com/graphics/CLIMATE-AUTOMATED/MONITOR/akpeykqqapr/heat-storms/).

The code in this repository generated the analysis published in an October 20, 2026 Reuters story headlined ["Giant ‘heat storms’ are scorching the Earth. This is the story of the biggest one on record."](https://www.reuters.com/investigates/special-report/climate-heatstorms-science/). The same logic is used by the [Reuters Climate Monitor](https://www.reuters.com/graphics/CLIMATE-AUTOMATED/MONITOR/akpeykqqapr/heat-storms/) to track heat storms in real time.

The inspiration and methodology are explained in ["How (and why) we developed ‘The Age of Heat Storms’"](https://www.reuters.com/investigates/special-report/climate-heatstorms-methodology/).

![A screenshot of the methodology story explaining how we did it](https://raw.githubusercontent.com/palewire/reuters-temperature-storms/refs/heads/main/.github/climate-method-story.png)

For the Reuters analysis, temperature readings were drawn from the [ERA5 reanalysis dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) published by the [European Centre for Medium-Range Weather Forecasts](https://www.ecmwf.int/)’ [Copernicus](https://climate.copernicus.eu/) program. That data is not included here.

Using different temperature readings and baseline climatologies will lead to differing results. Additionally, small differences in computer architecture may affect temperature readings and percentile scores by very small amounts.

The storm names assigned by Reuters are generated using a lottery system with a seeded value so that the results are reproducible. On some computer systems, the seed may not reliably return the same name list. In such a circumstance, you can refer to the table of storms and the names we use in our stories and visualizations, which is included in the repository.

The tracker identifies a new storm from its birth date and the first occupied
cell in its complete footprint on the fixed 0.25° grid. Cells are ordered by
latitude, then longitude; the two map halves of a cell at +180° count once.
Polygon text, ring order and other shape details do not participate in the ID.
Overlap still determines which storm continues through growth, splits and
mergers, and the surviving storm retains its birth ID. Prepared footprint-file
labels remain available for joining associated cell records.

These IDs use the complete 32-character date-and-cell digest after the birth
date. Existing catalogs and checkpoints made with the earlier shape-based IDs
must be rebuilt before using this tracker. The bundled story table remains the
record of names used in the published Reuters analysis.

If you have questions about the code or data, you're encouraged to contact [Ben Welsh](mailto:ben.welsh@thomsonreuters.com) and [Maurice Tamman](mailto:maurice.tamman@reuters.com) at Reuters.
