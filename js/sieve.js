/* Odd-only bit sieve: 1 bit per odd number, so 16 M costs 1 MB. */
(function (root) {
  'use strict';

  function build(N) {
    var half = (N >> 1) + 1;            // slot i holds the odd number 2i+1
    var comp = new Uint8Array((half >> 3) + 1);
    var lim = Math.floor(Math.sqrt(N));

    for (var i = 1; 2 * i + 1 <= lim; i++) {
      if (comp[i >> 3] & (1 << (i & 7))) continue;
      var p = 2 * i + 1;
      for (var j = (p * p - 1) >> 1; j < half; j += p) comp[j >> 3] |= 1 << (j & 7);
    }

    var count = N >= 2 ? 1 : 0;
    for (i = 1; i < half; i++) {
      if (2 * i + 1 <= N && !(comp[i >> 3] & (1 << (i & 7)))) count++;
    }

    var primes = new Uint32Array(count);
    var w = 0;
    if (N >= 2) primes[w++] = 2;
    for (i = 1; i < half; i++) {
      if (2 * i + 1 <= N && !(comp[i >> 3] & (1 << (i & 7)))) primes[w++] = 2 * i + 1;
    }

    function isPrime(n) {
      if (n < 2 || n > N) return false;
      if (n === 2) return true;
      if ((n & 1) === 0) return false;
      var k = n >> 1;
      return (comp[k >> 3] & (1 << (k & 7))) === 0;
    }

    /* Prime factors with multiplicity, e.g. 84 -> [2, 2, 3, 7]. */
    function factor(n) {
      var out = [];
      if (n < 2) return out;
      var m = n;
      for (var idx = 0; idx < primes.length; idx++) {
        var q = primes[idx];
        if (q * q > m) break;
        while (m % q === 0) { out.push(q); m /= q; }
      }
      if (m > 1) out.push(m);
      return out;
    }

    /* First index into primes[] whose value is >= target. */
    function lowerBound(target) {
      var lo = 0, hi = primes.length;
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (primes[mid] < target) lo = mid + 1; else hi = mid;
      }
      return lo;
    }

    return { N: N, primes: primes, isPrime: isPrime, factor: factor, lowerBound: lowerBound };
  }

  root.Sieve = { build: build };
})(window.PS = window.PS || {});
