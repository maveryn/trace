# Chart Map Assets

This directory stores preprocessed vector assets used by chart-domain map
renderers. Runtime task generation must read these bundled assets only; it must
not download map data.

## License

All currently bundled chart map assets are derived from Natural Earth vector
data, which is public domain. The repo-local license note is stored at
`licenses/NATURAL_EARTH_PUBLIC_DOMAIN.md`.

## Assets

| Asset | Source layer | Scope |
|---|---|---|
| `natural_earth_admin0_world_110m_v0.json` | Natural Earth Admin 0 Countries, 1:110m | world countries excluding Antarctica |
| `natural_earth_admin0_eu_110m_v0.json` | Natural Earth Admin 0 Countries, 1:110m | EU member countries, cropped to the European map extent |
| `natural_earth_admin1_usa_contiguous_110m_v0.json` | Natural Earth Admin 1 States/Provinces, 1:110m | contiguous USA states plus DC; Alaska and Hawaii excluded for compact chart readability |
| `natural_earth_admin1_china_50m_v0.json` | Natural Earth Admin 1 States/Provinces, 1:50m | China province-level regions |

Common rendering contract:

- Source URL: <https://github.com/nvkelso/natural-earth-vector>
- License: public domain.
- Projection: equirectangular longitude/latitude normalized at render time.
- Scope: simplified/rounded polygon rings for compact deterministic rendering.
- Rendering: chart tasks fit the projected map into an aspect-preserving panel
  and may vary ocean/land/outline/graticule styling through task config.

The generated chart tasks assign synthetic values/categories to these regions.
The Natural Earth geometry is only the visual scaffold; answers are verified
from Trace metadata, not from pixels or external geography facts.
