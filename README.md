# prime-spiral

An interactive Sacks prime spiral that fills the browser window. Zoom, pan, and
at close range the integers themselves appear. Polynomials in `n` are drawn as
connected curves beneath the points, wherever they happen to pass.

## Run

```sh
open index.html
```

Static files only — no build step, no server, no dependencies. It runs straight
off the filesystem, and drops onto any static host as-is.

## The spiral

Integer *n* sits at radius √n and angle 2π√n, so the perfect squares run out
along the +x axis — the same orientation `main.py` writes to `prime.png`, so
the two line up. Point density is uniform everywhere: consecutive integers are
always about π units apart and consecutive turns exactly 1 apart. Primes are
bright, composites dim.

## Controls

| | |
|---|---|
| drag | pan |
| scroll / pinch | zoom |
| double-click | zoom in |
| <kbd>+</kbd> <kbd>−</kbd> | zoom |
| arrows | pan |
| <kbd>0</kbd> | reset view |

Hovering a point names it and factors it. The whole state — camera,
polynomials, colours, visibility — lives in the URL hash, so sharing a view is
just copying the address.

## Polynomials

Type any expression in `n`: `n^2 + n + 41`, `n**3 - n`, `4n^2+2n+41`,
`n(n+1)/2`. `^` and `**` both work, and multiplication can be implicit.
Expressions are parsed, never `eval`'d. Each one is sampled at integer *n* and
its values are plotted on the spiral, joined in order — so a curve wanders off
toward wherever its values land, and Euler's `n^2 + n + 41` visibly threads
through primes.

Only the *n* whose values are actually on screen get evaluated: each curve is
split once into monotone pieces, and "which n are visible" is then a binary
search. A curve dense enough to cost more than 120k points in one frame is cut
short, and its row shows a ⚠.

## Display

Numerals appear automatically once they fit between neighbouring points and
turns. Composites fade out below ~7 px between integers and are dropped under
5 px, where they would be a solid wash. The sieve runs to 16M — a faint ring
marks where the data ends — and zooming out stops just past the full disc.

## How it draws

Zoomed out, the visible integers are found turn by turn: for a given turn only
one contiguous run of *n* falls inside the viewport's angular window, so the
work is proportional to what is on screen rather than to the whole annulus.
Each prime splats a small tent of ink into a float coverage buffer — every
prime in view is plotted, none are subsampled — and coverage maps to alpha, so
dot size, position and brightness all vary continuously with zoom. Zoomed in it
switches to circles, then to numerals. Curves go down first, under the points.

Far enough out, primes outnumber pixels, and drawing each at full brightness
flares the whole disc to white. So each point's brightness is scaled down by
the square root of how far the prime "ink" exceeds a target share of the
screen, and points landing on the same pixel add up — the disc stays evenly lit
at any zoom, and denser stretches still read as brighter. (Scaling by the full
ratio instead holds the average steady but flattens the disc to featureless
grey; the contrast is the picture.)

Worst case (the whole 16M spiral, retina 1400×900) is a few tens of ms of
JavaScript per frame; the 16M sieve builds in about 30 ms.

```
index.html
css/style.css
js/expr.js     expression parser (no eval)
js/sieve.js    odd-only bit sieve, 1 bit per odd number
js/spiral.js   geometry, camera, visible-range solver
js/poly.js     monotone split + binary search for visible n
js/render.js   the two drawing paths
js/app.js      state, input, panel, persistence
```

## PNG renderer

`main.py` still renders a high-resolution still with JAX:

```sh
uv run python main.py
```

or via the console script:

```sh
uv run prime-spiral
```

`uv` creates the virtualenv and installs dependencies on first run. On Linux,
`jax[cuda12]` is installed so the render runs on GPU; on macOS and Windows plain
CPU `jax` is used — the full 6000×6000 render still takes only a few seconds.
Output resolution is set by `WIDTH` / `HEIGHT` at the top of `main.py`.
