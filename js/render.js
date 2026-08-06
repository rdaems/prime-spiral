/* Canvas renderer.
 *
 * The polynomial curves go down first, straight onto the canvas, so the
 * spiral's points and numerals always sit on top of them. Then one of two
 * point passes, picked from how many pixels apart consecutive integers are:
 *   - dense  (< ~22 px): each prime splats a small tent of "ink" into a float
 *     coverage buffer, which is then mapped to alpha and composited over the
 *     curves. Every prime in view is drawn — no subsampling — because that is
 *     the picture. Position, dot size and brightness are all continuous, so
 *     nothing steps or pops while zooming.
 *   - sparse (>= ~22 px): real circles, and numerals once they fit.
 */
(function (root) {
  'use strict';

  var Spiral = root.Spiral;
  var TAU = Spiral.TAU;

  var VECTOR_MIN = 22;    // css px between integers: switch to circles
  var ALL_MIN = 5;        // below this, composites are dropped (they'd be mush)
  var VECTOR_MAX_PTS = 60000;
  var TARGET_INK = 0.03;  // share of the screen primes may light up before dimming
  var CURVE_ALPHA = 0.72;
  var MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

  var C = {
    bg: '#0e0e12',
    composite: '#3d4254',
    compositeText: '#6a7085',
    prime: '#f2f5ff',
    limitRing: 'rgba(255, 255, 255, 0.07)'
  };

  var LE = (function () {
    var b = new ArrayBuffer(4);
    new Uint32Array(b)[0] = 1;
    return new Uint8Array(b)[0] === 1;
  })();

  function rgb(hex) {
    return [parseInt(hex.slice(1, 3), 16),
            parseInt(hex.slice(3, 5), 16),
            parseInt(hex.slice(5, 7), 16)];
  }

  function pack(r, g, b, a) {
    return (LE ? (a << 24) | (b << 16) | (g << 8) | r
               : (r << 24) | (g << 16) | (b << 8) | a) >>> 0;
  }

  var PRIME_RGB = rgb(C.prime), COMP_RGB = rgb(C.composite);

  /* Prime pixels at every alpha, so coverage -> colour is one lookup. */
  var PRIME_A = new Uint32Array(256);
  for (var ai = 0; ai < 256; ai++) {
    PRIME_A[ai] = pack(PRIME_RGB[0], PRIME_RGB[1], PRIME_RGB[2], ai);
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* ------------------------------------------------------------- buffers */

  var img = null, u32 = null, acc = null, bufW = 0, bufH = 0;
  var layer = null, lctx = null;   // offscreen canvas the point layer blits from

  function ensureBuf(ctx, W, H) {
    if (img && bufW === W && bufH === H) return;
    img = ctx.createImageData(W, H);
    u32 = new Uint32Array(img.data.buffer);
    acc = new Float32Array(W * H);
    if (!layer) layer = document.createElement('canvas');
    layer.width = W;
    layer.height = H;
    lctx = layer.getContext('2d');
    bufW = W;
    bufH = H;
  }

  function makeBuf() {
    return { x: new Float32Array(4096), y: new Float32Array(4096), v: new Int32Array(4096), n: 0 };
  }

  function push(b, x, y, v) {
    if (b.n === b.x.length) {
      var cap = b.n * 2;
      var nx = new Float32Array(cap); nx.set(b.x); b.x = nx;
      var ny = new Float32Array(cap); ny.set(b.y); b.y = ny;
      var nv = new Int32Array(cap); nv.set(b.v); b.v = nv;
    }
    b.x[b.n] = x; b.y[b.n] = y; b.v[b.n] = v; b.n++;
  }

  var bufP = makeBuf();     // primes
  var bufC = makeBuf();     // composites
  var pts = new Float32Array(8192);   // polynomial vertices

  /* --------------------------------------------------------------- frame */

  function frame(ctx, view, opts) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, view.W, view.H);
    limitRing(ctx, view);
    curves(ctx, view, opts);

    if (view.spacing >= VECTOR_MIN && view.count <= VECTOR_MAX_PTS) {
      vectorPass(ctx, view, opts);
    } else {
      rasterPass(ctx, view, opts);
    }
  }

  /* --------------------------------------------------------- dense path */

  /* One prime's ink around (x, y): a tent of radius r with unit peak, or a
   * plain bilinear spread (total weight 1) once r is sub-pixel — so dots
   * shrink, move and dim continuously instead of snapping to the grid. */
  function splat(x, y, r, W, H) {
    var xx, yy, row;
    if (r <= 1) {
      var x0 = Math.floor(x), y0 = Math.floor(y);
      var fx = x - x0, fy = y - y0;
      for (yy = y0; yy <= y0 + 1; yy++) {
        if (yy < 0 || yy >= H) continue;
        var wy = yy === y0 ? 1 - fy : fy;
        row = yy * W;
        for (xx = x0; xx <= x0 + 1; xx++) {
          if (xx < 0 || xx >= W) continue;
          acc[row + xx] += (xx === x0 ? 1 - fx : fx) * wy;
        }
      }
    } else {
      var xa = Math.ceil(x - r), xb = Math.floor(x + r);
      var ya = Math.ceil(y - r), yb = Math.floor(y + r);
      for (yy = ya; yy <= yb; yy++) {
        if (yy < 0 || yy >= H) continue;
        var dy = yy - y;
        row = yy * W;
        for (xx = xa; xx <= xb; xx++) {
          if (xx < 0 || xx >= W) continue;
          var dx = xx - x;
          var w = 1 - Math.sqrt(dx * dx + dy * dy) / r;
          if (w > 0) acc[row + xx] += w;
        }
      }
    }
  }

  function rasterPass(ctx, view, opts) {
    ensureBuf(ctx, view.W, view.H);
    u32.fill(0);
    acc.fill(0);

    var W = view.W, H = view.H, k = view.k, ox = view.ox, oy = view.oy;
    var runs = view.runs, sieve = opts.sieve;
    var withComposites = view.spacing >= ALL_MIN;

    // Dot radius in device px, growing smoothly with the zoom.
    var r = clamp(view.spacing * 0.28, 0.75, 3);
    var m = r + 1;
    // Ink one splat deposits: 1 for a bilinear spread, ~pi r^2 / 3 for a tent.
    var area = r <= 1 ? 1 : Math.PI * r * r / 3;

    // Share of the screen the primes would light up at full brightness: one
    // point per pi*k^2 pixels, of which about 1/ln(n) are prime. Past the
    // target we dim by the square root of the excess — dimming by the full
    // ratio holds the average steady but flattens the disc to featureless
    // grey, and it is the contrast between denser and sparser stretches that
    // carries the picture.
    var ink = area / (Math.PI * k * k * Math.log(Math.max(3, view.nHi)));
    var gain = Math.min(1, Math.sqrt(TARGET_INK / ink));
    // Fade composites in over the two px above their cutoff rather than popping.
    var comp32 = pack(COMP_RGB[0], COMP_RGB[1], COMP_RGB[2],
                      (255 * clamp((view.spacing - ALL_MIN) / 2, 0, 1)) | 0);

    var t, th, sx, sy, ri, v, end;

    if (withComposites) {
      for (ri = 0; ri < runs.length; ri += 2) {
        end = runs[ri + 1];
        for (v = runs[ri]; v <= end; v++) {
          t = Math.sqrt(v); th = TAU * t;
          sx = t * Math.cos(th) * k + ox;
          if (sx < -m || sx >= W + m) continue;
          sy = oy - t * Math.sin(th) * k;
          if (sy < -m || sy >= H + m) continue;
          if (sieve.isPrime(v)) {
            splat(sx, sy, r, W, H);
          } else if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
            var idx = (sy | 0) * W + (sx | 0);
            if (u32[idx] === 0) u32[idx] = comp32;
          }
        }
      }
    } else {
      var primes = sieve.primes, np = primes.length;
      var p = runs.length ? sieve.lowerBound(runs[0]) : 0;
      for (ri = 0; ri < runs.length; ri += 2) {
        var a = runs[ri], b = runs[ri + 1];
        while (p < np && primes[p] < a) p++;
        for (; p < np && primes[p] <= b; p++) {
          v = primes[p];
          t = Math.sqrt(v); th = TAU * t;
          sx = t * Math.cos(th) * k + ox;
          if (sx < -m || sx >= W + m) continue;
          sy = oy - t * Math.sin(th) * k;
          if (sy < -m || sy >= H + m) continue;
          splat(sx, sy, r, W, H);
        }
      }
    }

    // Coverage -> alpha. Primes overwrite composites on their pixels, as the
    // brighter of the two is the one worth seeing.
    var npx = W * H;
    for (var i = 0; i < npx; i++) {
      var cov = acc[i];
      if (cov > 0) {
        var bright = cov * gain;
        u32[i] = PRIME_A[bright >= 1 ? 255 : (bright * 255) | 0];
      }
    }

    lctx.putImageData(img, 0, 0);
    ctx.drawImage(layer, 0, 0);
  }

  /* -------------------------------------------------------- sparse path */

  function vectorPass(ctx, view, opts) {
    var W = view.W, H = view.H, k = view.k, ox = view.ox, oy = view.oy, dpr = view.dpr;
    var runs = view.runs, sieve = opts.sieve;
    bufP.n = 0;
    bufC.n = 0;

    for (var ri = 0; ri < runs.length; ri += 2) {
      var end = runs[ri + 1];
      for (var v = runs[ri]; v <= end; v++) {
        var t = Math.sqrt(v), th = TAU * t;
        var sx = t * Math.cos(th) * k + ox;
        if (sx < -40 || sx > W + 40) continue;
        var sy = oy - t * Math.sin(th) * k;
        if (sy < -40 || sy > H + 40) continue;
        if (sieve.isPrime(v)) push(bufP, sx, sy, v);
        else push(bufC, sx, sy, v);
      }
    }

    // Numerals, once they fit: cap the height well under the gap between turns
    // and the width under the gap between neighbouring integers, so labels on
    // adjacent turns cannot collide.
    var digits = String(view.nHi).length;
    var fs = Math.min(0.42 * view.pitch, view.spacing / (0.75 * digits), 24);
    var labels = fs >= 10 && bufP.n + bufC.n <= 40000;

    if (labels) {
      var fpx = fs * dpr;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      if (bufC.n && bufC.n <= 8000) {
        ctx.font = fpx + 'px ' + MONO;
        ctx.fillStyle = C.compositeText;
        for (var i = 0; i < bufC.n; i++) ctx.fillText(String(bufC.v[i]), bufC.x[i], bufC.y[i]);
      } else if (bufC.n) {
        dots(ctx, bufC, clamp(view.spacing * 0.055, 1, 3) * dpr, C.composite);
      }
      ctx.font = '600 ' + fpx + 'px ' + MONO;
      ctx.fillStyle = C.prime;
      for (var j = 0; j < bufP.n; j++) ctx.fillText(String(bufP.v[j]), bufP.x[j], bufP.y[j]);
    } else {
      var rp = clamp(view.spacing * 0.09, 1.5, 6) * dpr;
      if (bufC.n) dots(ctx, bufC, rp * 0.62, C.composite);
      dots(ctx, bufP, rp, C.prime);
    }
  }

  function dots(ctx, buf, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    for (var i = 0; i < buf.n; i++) {
      ctx.moveTo(buf.x[i] + r, buf.y[i]);
      ctx.arc(buf.x[i], buf.y[i], r, 0, TAU);
    }
    ctx.fill();
  }

  /* ------------------------------------------------------------ overlays */

  function limitRing(ctx, view) {
    var r = Math.sqrt(view.maxN);
    if (r < view.rMin || r > view.rMax) return;
    ctx.strokeStyle = C.limitRing;
    ctx.lineWidth = view.dpr;
    ctx.beginPath();
    ctx.arc(view.ox, view.oy, r * view.k, 0, TAU);
    ctx.stroke();
  }

  function curves(ctx, view, opts) {
    var polys = opts.polys;
    if (!polys.length) return;

    var k = view.k, ox = view.ox, oy = view.oy, dpr = view.dpr;
    var W = view.W, H = view.H;
    var dash = [7 * dpr, 5 * dpr];

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (var pi = 0; pi < polys.length; pi++) {
      var poly = polys[pi];
      if (!poly.visible || !poly.compiled) continue;

      var ranges = root.Poly.ranges(poly.compiled, view.nLo, view.nHi);
      poly.truncated = poly.compiled.truncated;
      if (!ranges.length) continue;

      var fn = poly.compiled.fn;
      var segStart = [], segLen = [];
      var m = 0;                       // write cursor into pts
      var lastVisX = 0, lastVisY = 0, sawVisible = false;

      for (var ri = 0; ri < ranges.length; ri += 2) {
        var a = ranges[ri], z = ranges[ri + 1];
        if ((m + (z - a + 1)) * 2 > pts.length) {
          var grown = new Float32Array(Math.max(pts.length * 2, (m + (z - a + 1)) * 2));
          grown.set(pts);
          pts = grown;
        }
        var start = m;
        for (var n = a; n <= z; n++) {
          var val = fn(n);
          if (!(val >= 0) || !isFinite(val)) {
            if (m > start) { segStart.push(start); segLen.push(m - start); }
            start = m;
            continue;
          }
          var t = Math.sqrt(val), th = TAU * t;
          var x = t * Math.cos(th) * k + ox;
          var y = oy - t * Math.sin(th) * k;
          pts[m * 2] = x;
          pts[m * 2 + 1] = y;
          m++;
          if (x >= 0 && x <= W && y >= 0 && y <= H) {
            lastVisX = x; lastVisY = y; sawVisible = true;
          }
        }
        if (m > start) { segStart.push(start); segLen.push(m - start); }
      }
      if (!m) continue;

      ctx.globalAlpha = CURVE_ALPHA;   // let the background read through the curves
      ctx.strokeStyle = poly.color;
      ctx.lineWidth = 2 * dpr;
      ctx.setLineDash(poly.dashed ? dash : []);
      ctx.beginPath();
      for (var s = 0; s < segStart.length; s++) {
        var i0 = segStart[s], len = segLen[s];
        if (len < 2) continue;
        ctx.moveTo(pts[i0 * 2], pts[i0 * 2 + 1]);
        for (var q = 1; q < len; q++) ctx.lineTo(pts[(i0 + q) * 2], pts[(i0 + q) * 2 + 1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.globalAlpha = 1;
      if (sawVisible) label(ctx, poly, lastVisX, lastVisY, dpr, W, H);
    }
  }

  function label(ctx, poly, x, y, dpr, W, H) {
    var fpx = 12 * dpr;
    ctx.font = fpx + 'px ' + MONO;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    var tx = clamp(x + 9 * dpr, 4 * dpr, W - ctx.measureText(poly.src).width - 4 * dpr);
    var ty = clamp(y - 9 * dpr, fpx, H - fpx);
    ctx.lineWidth = 3.5 * dpr;
    ctx.strokeStyle = 'rgba(14, 14, 18, 0.85)';
    ctx.lineJoin = 'round';
    ctx.strokeText(poly.src, tx, ty);
    ctx.fillStyle = poly.color;
    ctx.fillText(poly.src, tx, ty);
  }

  root.Render = { frame: frame, colors: C };
})(window.PS = window.PS || {});
