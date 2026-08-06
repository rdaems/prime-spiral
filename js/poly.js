/* Turning a polynomial into "which n are on screen right now".
 *
 * Rather than precomputing a big table of f(n), each curve is split once into
 * monotone segments (at most degree-1 of them). Within a monotone segment
 * f(n) in [lo, hi] is a binary search, so every frame only touches the n that
 * are actually visible — a curve like 3n+1 stays cheap however far you zoom out.
 */
(function (root) {
  'use strict';

  var HARD_MAX_N = 1e9;      // largest n we will ever feed a polynomial
  var DENSE = 512;           // turning-point scan: every n up to here,
  var RATIO = 1.02;          // then 2% steps out to nEnd
  var MAX_POINTS = 120000;   // per curve, per frame

  function create(src, maxValue) {
    var fn = root.Expr.compile(src); // throws on bad input
    var constant = fn(0) === fn(1) && fn(1) === fn(2);
    var a = analyze(fn, maxValue, constant);
    return {
      fn: fn,
      constant: constant,
      nEnd: a.nEnd,
      bounds: a.bounds,
      truncated: false
    };
  }

  /* How far out is worth looking: keep doubling until the curve has left the
   * [-maxValue, maxValue] band *and* is growing in magnitude for a few steps
   * running, which for a polynomial means every turning point is behind us.
   * (A plain "first n above maxValue" stops too early for something like
   * (n-1000)^2 - 500000, which starts high, dives through the band and back.) */
  function domainEnd(fn, maxValue) {
    var n = 64;
    var settled = 0;
    var mag = Math.abs(fn(n));
    while (n < HARD_MAX_N && settled < 3) {
      var next = n * 2;
      var nextMag = Math.abs(fn(next));
      if (mag > maxValue && nextMag > mag) settled++; else settled = 0;
      n = next;
      mag = nextMag;
    }
    return Math.min(n, HARD_MAX_N);
  }

  /* Every integer up to DENSE, then 2% steps — so a turning point near the
   * origin is caught exactly and a distant one lands inside one short bracket. */
  function scanGrid(nEnd) {
    var pts = [];
    var dense = Math.min(nEnd, DENSE);
    for (var i = 0; i <= dense; i++) pts.push(i);
    var n = dense;
    while (n < nEnd) {
      var nx = Math.max(n + 1, Math.floor(n * RATIO));
      if (nx > nEnd) nx = nEnd;
      pts.push(nx);
      n = nx;
    }
    return pts;
  }

  function analyze(fn, maxValue, constant) {
    if (constant) return { nEnd: 0, bounds: [0, 0] };

    var nEnd = domainEnd(fn, maxValue);
    var grid = scanGrid(nEnd);
    var bounds = [0];
    var prev2 = 0, prev = 0, vPrev = fn(0), dir = 0;

    for (var i = 1; i < grid.length; i++) {
      var n = grid[i];
      var v = fn(n);
      var d = v > vPrev ? 1 : (v < vPrev ? -1 : 0);
      if (d !== 0 && dir !== 0 && d !== dir) {
        var tp = extremum(fn, prev2, n, dir);
        if (tp > bounds[bounds.length - 1] && tp < nEnd) bounds.push(tp);
      }
      if (d !== 0) dir = d;
      prev2 = prev;
      prev = n;
      vPrev = v;
    }
    bounds.push(nEnd);
    return { nEnd: nEnd, bounds: bounds };
  }

  /* Ternary search for the extremum bracketed by [a, b]; dir > 0 means f was
   * rising into it, so we are looking for a maximum. */
  function extremum(fn, a, b, dir) {
    while (b - a > 2) {
      var third = Math.floor((b - a) / 3) || 1;
      var m1 = a + third, m2 = b - third;
      if (m1 >= m2) break;
      var f1 = fn(m1), f2 = fn(m2);
      if (dir > 0 ? f1 < f2 : f1 > f2) a = m1; else b = m2;
    }
    return Math.round((a + b) / 2);
  }

  /* Smallest n in [lo, hi] where pred(n) holds; hi+1 if it never does.
   * pred must be monotone false -> true over the range. */
  function firstTrue(lo, hi, pred) {
    var a = lo, b = hi + 1;
    while (a < b) {
      var mid = a + Math.floor((b - a) / 2);
      if (pred(mid)) b = mid; else a = mid + 1;
    }
    return a;
  }

  /* Flat [start, end, ...] runs of n whose value is inside [lo, hi]. */
  function ranges(poly, lo, hi) {
    poly.truncated = false;
    var fn = poly.fn;
    var out = [];

    if (poly.constant) {
      var c = fn(0);
      if (c >= lo && c <= hi) out.push(0, 0);
      return out;
    }

    var budget = MAX_POINTS;
    var b = poly.bounds;
    for (var i = 0; i + 1 < b.length && budget > 0; i++) {
      var s0 = b[i], s1 = b[i + 1];
      if (s1 <= s0) continue;
      var v0 = fn(s0), v1 = fn(s1);
      var a, z;

      if (v0 <= v1) {                                   // non-decreasing
        if (v1 < lo || v0 > hi) continue;
        a = firstTrue(s0, s1, function (n) { return fn(n) >= lo; });
        z = firstTrue(s0, s1, function (n) { return fn(n) > hi; }) - 1;
      } else {                                          // decreasing
        if (v0 < lo || v1 > hi) continue;
        a = firstTrue(s0, s1, function (n) { return fn(n) <= hi; });
        z = firstTrue(s0, s1, function (n) { return fn(n) < lo; }) - 1;
      }

      if (z < a) continue;
      a = Math.max(0, a - 1);       // one extra point each side so the line
      z = Math.min(poly.nEnd, z + 1); // continues off the edge of the screen

      var len = z - a + 1;
      if (len > budget) { z = a + budget - 1; poly.truncated = true; }
      budget -= (z - a + 1);
      out.push(a, z);
    }
    return out;
  }

  root.Poly = { create: create, ranges: ranges, MAX_POINTS: MAX_POINTS };
})(window.PS = window.PS || {});
