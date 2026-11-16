# `elt_grid_add` — add a z-displacement grid to a grid surface

Perturb an optic through its grid-data surface by *adding* an N×N
z-displacement map to whatever grid the element already carries — a
measured figure-error map, a DM influence-function sum, a fabrication
map, etc.

## Call forms

```matlab
macos.elt_grid_add(srf, grid_dz)     % package function
m.elt_grid_add(srf, grid_dz)         % method on a macos.Session m
```

Both forms behave identically; the Session method just forwards to the
package function.

## Arguments

| name      | type            | meaning |
|-----------|-----------------|---------|
| `srf`     | positive int    | Element id. The element **must already carry a grid surface** (`SrfType` Grid / MonGrData / FreeForm-with-grid term). This is *not* a way to add a grid to a non-grid optic. |
| `grid_dz` | real `N×N`      | Displacement map. `N` must equal the element's **current grid sampling size**. Indexed `grid_dz(iy, ix)`: rows run −Y→+Y, columns −X→+X (the same orientation as `zrn_freeform(srf).grid.mat`). **No transpose needed.** Values are displacements along the surface's local z-axis, in the prescription's length units. |

## Semantics

It **accumulates** — the map is *added* onto the existing grid in place,
so calling it twice adds twice. It returns nothing. After the call,
trace / evaluate as usual.

## Typical use

```matlab
m   = macos.Session(256);
m.load_rx('design.in');
srf = 2;                                       % a grid-bearing element

N   = size(m.zrn_freeform(srf).grid.mat, 1);   % query the live grid size
dz  = my_figure_error(N);                      % your N×N displacement map
m.elt_grid_add(srf, dz);                       % stack it onto the surface

s = m.trace();                                 % evaluate the perturbed system
```

For a FreeForm element you can read the grid back (and confirm the add)
with `m.zrn_freeform(srf).grid.mat`.

## Errors

It never silently corrupts — bad input raises:

| identifier                       | condition |
|----------------------------------|-----------|
| `macos:elt_grid_add:noGrid`      | `srf` has no grid surface (size < 3). |
| `macos:elt_grid_add:notSquare`   | `grid_dz` is not square. |
| `macos:elt_grid_add:sizeMismatch`| `grid_dz` size ≠ the element's live grid size. |

## Notes

- **vs. pymacos.** Same behaviour as pymacos's `elt_grid_add`, except
  mmacos takes a **positive** element id (no negative "from-the-end"
  indexing — the mmacos package convention).
- **Backed by** the `macos_api_mod` wrappers `elt_srf_grid_size` and
  `elt_srf_grid_data_add`.

## See also

`macos.zrn_freeform` (read/write the composite FreeForm grid + Zernike
description).
