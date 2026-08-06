/* Sacks spiral geometry.
 *
 * Integer v sits at radius sqrt(v) and angle 2*pi*sqrt(v), so the squares run
 * out along the +x axis — the same orientation main.py writes to prime.png —
 * and the point density is uniform: consecutive integers are always ~pi world
 * units apart and consecutive turns always exactly 1 apart.
 *
 * Because angle = 2*pi*frac(sqrt(v)), every point on a given turn k with
 * polar angle in [a0, a1] has sqrt(v) in [k+u0, k+u1] — which turns "what is on
 * screen" into one contiguous run of integers per turn. That is what runs()
 * returns, and it is why panning far from the origin stays cheap: the work is
 * proportional to the points actually visible, not to the whole annulus.
 */
(function (root) {
  'use strict';

  var TAU = Math.PI * 2;

  function worldX(t) { return t * Math.cos(TAU * t); }
  function worldY(t) { return t * Math.sin(TAU * t); }

  /* Camera -> screen mapping plus everything the renderer needs about what is
   * currently visible. All screen values are device pixels.
   *   sx = x * k + ox        sy = -y * k + oy
   */
  function makeView(st, W, H, dpr, marginCss) {
    var k = st.scale * dpr;
    var ox = W / 2 - st.cx * k;
    var oy = H / 2 + st.cy * k;
    var m = marginCss * dpr;

    var x0 = (-m - ox) / k, x1 = (W + m - ox) / k;
    var y0 = (oy - (H + m)) / k, y1 = (oy + m) / k;

    // Distance from the origin to the viewport rect (0 when it is inside).
    var dx = Math.max(x0, -x1, 0);
    var dy = Math.max(y0, -y1, 0);
    var rMin = Math.sqrt(dx * dx + dy * dy);
    var ax = Math.max(Math.abs(x0), Math.abs(x1));
    var ay = Math.max(Math.abs(y0), Math.abs(y1));
    var rMax = Math.sqrt(ax * ax + ay * ay);

    // Angular window, expressed as the fractional part u of sqrt(v).
    var u0 = 0, u1 = 1;
    if (dx !== 0 || dy !== 0) {
      var mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      var base = Math.atan2(my, mx);
      var lo = 0, hi = 0;
      var cx4 = [x0, x1, x0, x1], cy4 = [y0, y0, y1, y1];
      for (var i = 0; i < 4; i++) {
        var d = Math.atan2(cy4[i], cx4[i]) - base;
        d -= TAU * Math.floor((d + Math.PI) / TAU); // wrap to (-pi, pi]
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
      u0 = (base + lo) / TAU;
      u1 = (base + hi) / TAU;
      var shift = Math.floor(u0);
      u0 -= shift;
      u1 -= shift;
    }

    var nHi = Math.min(st.maxN, Math.ceil(rMax * rMax));
    var nLo = Math.max(0, Math.floor(rMin * rMin));

    var rs = runs(rMin, rMax, u0, u1, nLo, nHi);
    var count = 0;
    for (var j = 0; j < rs.length; j += 2) count += rs[j + 1] - rs[j] + 1;

    return {
      k: k, ox: ox, oy: oy, W: W, H: H, dpr: dpr,
      scale: st.scale,
      spacing: Math.PI * st.scale,   // css px between consecutive integers
      pitch: st.scale,               // css px between consecutive turns
      rMin: rMin, rMax: rMax, u0: u0, u1: u1,
      nLo: nLo, nHi: nHi, maxN: st.maxN,
      runs: rs,
      count: count       // integers inside the visible sector
    };
  }

  /* Flat [start, end, start, end, ...] integer runs, in increasing order. */
  function runs(rMin, rMax, u0, u1, nLo, nHi) {
    var out = [];
    var kA = Math.floor(rMin - u1);
    var kB = Math.ceil(rMax - u0);
    if (kB - kA > 300000) kB = kA + 300000; // paranoia; unreachable in practice
    for (var kk = kA; kk <= kB; kk++) {
      var tA = kk + u0, tB = kk + u1;
      if (tB < rMin || tA > rMax) continue;
      if (tA < rMin) tA = rMin;
      if (tB > rMax) tB = rMax;
      if (tA < 0) tA = 0;
      if (tB < tA) continue;
      var a = Math.ceil(tA * tA), b = Math.floor(tB * tB);
      if (a < nLo) a = nLo;
      if (b > nHi) b = nHi;
      if (a <= b) { out.push(a, b); }
    }
    return out;
  }

  /* Integer nearest to a screen point, for the hover readout. */
  function nearest(view, sx, sy) {
    var x = (sx - view.ox) / view.k;
    var y = (view.oy - sy) / view.k;
    var r = Math.sqrt(x * x + y * y);
    var u = Math.atan2(y, x) / TAU;
    u -= Math.floor(u);

    var best = -1, bestD = Infinity;
    var k0 = Math.floor(r - u);
    for (var kk = k0 - 1; kk <= k0 + 1; kk++) {
      var t = kk + u;
      if (t < 0) continue;
      var centre = Math.round(t * t);
      for (var d = -2; d <= 2; d++) {
        var v = centre + d;
        if (v < 0 || v > view.maxN) continue;
        var tt = Math.sqrt(v);
        var px = worldX(tt) - x, py = worldY(tt) - y;
        var dd = px * px + py * py;
        if (dd < bestD) { bestD = dd; best = v; }
      }
    }
    return { n: best, dist: Math.sqrt(bestD) * view.k };
  }

  root.Spiral = {
    TAU: TAU,
    worldX: worldX,
    worldY: worldY,
    makeView: makeView,
    nearest: nearest
  };
})(window.PS = window.PS || {});
