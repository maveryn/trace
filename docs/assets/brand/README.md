# Trace Brand Assets

The Trace mark is a compact `3x3` atlas with a Trace monogram at its center.
Eight surrounding motifs summarize the eleven visual domains: games and
puzzles share the space-shooter tile; icons and symbolic tasks share the clock
tile; and illustrations and 3D scenes share an isometric-world tile. Charts,
geometry, graphs, pages, and physics retain dedicated motifs. This keeps the
mark legible while showing recognizable visual grammars rather than abstract
nodes. The game, isometric-world, and physics motifs derive from actual Trace
scene families.

## Files

- `trace-mark.svg`: standalone square mark for the README title, avatars,
  figures, and compact navigation.
- `trace-wordmark.svg`: tightly cropped Trace wordmark paired with the square
  mark in the README title.
- `trace-logo.svg`: horizontal mark and wordmark for repository and
  documentation headers.
- `trace-mark-light.png`: light-theme raster used by static documentation
  composites.

The SVG files are transparent and self-contained. Their structural ink adapts
to the viewer's light or dark color preference. The standalone mark has no font
dependency. The horizontal logo and standalone wordmark use a conventional
sans-serif fallback stack so they remain portable across browsers and document
renderers.

The square mark and the mark within the horizontal logo use the same rounded
outer frame. The horizontal logo places that framed mark beside the Trace
wordmark.

The SVGs in this directory are the source assets for repository branding.
Static generators consume the committed raster derivative rather than
rasterizing the adaptive SVG during each build, because native SVG renderers
can produce host-dependent pixels. The raster derivative is transparent and
intended for light-background composites; do not use it as the general logo.

## Palette

| Role | Hex |
|---|---|
| Structural ink | `#17242d` |
| Tile surface | `#f7f9f9` |
| Tile border | `#cad6da` |
| Secondary ink | `#667983` |
| Coral | `#e85d4a` |
| Gold | `#e5a62b` |
| Blue | `#3974d7` |
| Teal | `#159b91` |
| Space-scene lime | `#b9d63b` |
| Terrain green | `#56a45b` |
| Terrain edge | `#b9864f` |

Preserve the outer frame, tile order, icon geometry, spacing, and palette in the
primary mark. For monochrome applications, all motifs may use one ink color,
but the framed atlas arrangement should remain unchanged.
