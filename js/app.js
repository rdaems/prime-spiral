/* Wiring: state, input, panel, persistence. */
(function (root) {
  'use strict';

  var Spiral = root.Spiral, Render = root.Render, Poly = root.Poly;

  // Categorical slots, in fixed order, stepped for a dark surface.
  var PALETTE = ['#3987e5', '#d95926', '#199e70', '#c98500',
                 '#d55181', '#008300', '#9085e9', '#e66767'];
  var MAX_SCALE = 200;
  var MAX_N = 16000000;

  var canvas = document.getElementById('stage');
  var ctx = canvas.getContext('2d', { alpha: false });
  var tooltip = document.getElementById('tooltip');
  var loading = document.getElementById('loading');
  var list = document.getElementById('poly-list');
  var errPop = document.getElementById('err-pop');

  var state = { cx: 0, cy: 0, scale: 3, maxN: MAX_N };

  var polys = [];
  var sieve = null;
  var view = null;
  var W = 0, H = 0, dpr = 1;
  var queued = false;
  var hover = null;

  /* --------------------------------------------------------------- utils */

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function compact(n) {
    if (n < 1000) return String(n);
    if (n < 1e6) return (n / 1e3).toFixed(n < 1e5 ? 1 : 0) + 'k';
    return (n / 1e6).toFixed(n < 1e7 ? 2 : 1) + 'M';
  }

  function group(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }

  /* -------------------------------------------------------------- canvas */

  function resize() {
    dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    var cw = window.innerWidth, ch = window.innerHeight;
    W = Math.floor(cw * dpr);
    H = Math.floor(ch * dpr);
    canvas.width = W;
    canvas.height = H;
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    ctx.setTransform(1, 0, 0, 1, 0, 0);   // we work in device pixels throughout
    schedule();
  }

  function cssMin() { return Math.min(W, H) / dpr; }
  // Zoomed all the way out, the full disc plus a little margin fills the screen.
  function minScale() { return cssMin() / 2 / (Math.sqrt(MAX_N) * 1.08); }

  function clampState() {
    state.scale = clamp(state.scale, minScale(), MAX_SCALE);
    var lim = Math.sqrt(MAX_N) * 1.2;
    state.cx = clamp(state.cx, -lim, lim);
    state.cy = clamp(state.cy, -lim, lim);
  }

  function resetView() {
    state.cx = 0;
    state.cy = 0;
    state.scale = clamp(40, minScale(), MAX_SCALE);
    schedule();
    save();
  }

  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(draw);
  }

  function draw() {
    queued = false;
    if (!sieve) return;
    clampState();
    view = Spiral.makeView(state, W, H, dpr, 30);
    Render.frame(ctx, view, { sieve: sieve, polys: polys });
    for (var i = 0; i < polys.length; i++) {
      if (polys[i].warnEl) polys[i].warnEl.hidden = !polys[i].truncated;
    }
  }

  /* ------------------------------------------------------------ the maths */

  function buildSieve(next) {
    loading.textContent = 'sieving to ' + compact(MAX_N) + '…';
    loading.classList.remove('done');
    setTimeout(function () {
      sieve = root.Sieve.build(MAX_N);
      loading.classList.add('done');
      clampState();
      schedule();
      if (next) next();
    }, 30);
  }

  function compile(p) {
    p.error = '';
    p.truncated = false;
    try {
      p.compiled = Poly.create(p.src, MAX_N);
    } catch (e) {
      p.compiled = null;
      p.error = (e && e.message) || 'invalid expression';
    }
    if (p.row) {
      p.row.classList.toggle('bad', !!p.error);
      p.input.title = p.error || p.src;
    }
  }

  /* ---------------------------------------------------------------- panel */

  function addPoly(src, opts) {
    opts = opts || {};
    var idx = opts.colorIndex !== undefined ? opts.colorIndex : polys.length % PALETTE.length;
    var p = {
      src: src,
      colorIndex: idx,
      color: PALETTE[idx % PALETTE.length],
      dashed: opts.dashed !== undefined ? opts.dashed : polys.length >= PALETTE.length,
      visible: opts.visible !== undefined ? opts.visible : true,
      compiled: null,
      error: '',
      truncated: false
    };
    polys.push(p);
    compile(p);
    renderRow(p);
    schedule();
    save();
    return p;
  }

  function renderRow(p) {
    var li = document.createElement('li');

    var swatch = document.createElement('button');
    swatch.type = 'button';
    swatch.className = 'swatch';
    swatch.style.background = p.color;
    swatch.title = 'Next colour';
    swatch.addEventListener('click', function () {
      p.colorIndex = (p.colorIndex + 1) % PALETTE.length;
      p.color = PALETTE[p.colorIndex];
      swatch.style.background = p.color;
      schedule();
      save();
    });

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'expr';
    input.value = p.src;
    input.spellcheck = false;
    input.setAttribute('aria-label', 'Polynomial');
    input.addEventListener('change', function () {
      p.src = input.value.trim();
      compile(p);
      schedule();
      save();
    });

    var warn = document.createElement('span');
    warn.className = 'warn';
    warn.textContent = '⚠';
    warn.title = 'Too many points at this zoom — the curve is cut short';
    warn.hidden = true;

    var eye = document.createElement('button');
    eye.type = 'button';
    eye.className = 'icon';
    eye.textContent = p.visible ? '●' : '○';
    eye.title = 'Show / hide';
    eye.addEventListener('click', function () {
      p.visible = !p.visible;
      eye.textContent = p.visible ? '●' : '○';
      li.classList.toggle('off', !p.visible);
      schedule();
      save();
    });

    var del = document.createElement('button');
    del.type = 'button';
    del.className = 'icon';
    del.textContent = '×';
    del.title = 'Remove';
    del.addEventListener('click', function () {
      polys.splice(polys.indexOf(p), 1);
      li.remove();
      schedule();
      save();
    });

    li.className = (p.visible ? '' : 'off ') + (p.error ? 'bad' : '');
    li.append(swatch, input, warn, eye, del);
    list.append(li);

    p.row = li;
    p.input = input;
    p.warnEl = warn;
    input.title = p.error || p.src;
  }

  /* ---------------------------------------------------------- persistence */

  /* The whole state — camera and polynomials — lives in the URL hash, so
   * sharing a view is just copying the address:
   *   #cx,cy,scale;<colour><~|!><encoded src>;...      (! marks a hidden one)
   */
  function readHash() {
    var parts = location.hash.replace(/^#/, '').split(';');
    var out = { view: null, polys: [] };

    var m = /^(-?[\d.]+),(-?[\d.]+),([\d.]+)$/.exec(parts[0]);
    if (m) {
      var o = { cx: parseFloat(m[1]), cy: parseFloat(m[2]), scale: parseFloat(m[3]) };
      if (isFinite(o.cx) && isFinite(o.cy) && o.scale > 0) out.view = o;
    }

    for (var i = 1; i < parts.length; i++) {
      var pm = /^(\d+)([~!])(.+)$/.exec(parts[i]);
      if (!pm) continue;
      var src;
      try { src = decodeURIComponent(pm[3]); } catch (e) { continue; }
      out.polys.push({
        src: src,
        colorIndex: (+pm[1]) % PALETTE.length,
        visible: pm[2] === '~'
      });
    }
    return out;
  }

  function writeHash() {
    var h = '#' + state.cx.toFixed(3) + ',' + state.cy.toFixed(3) + ',' +
            Number(state.scale.toFixed(4));
    for (var i = 0; i < polys.length; i++) {
      var p = polys[i];
      h += ';' + p.colorIndex + (p.visible ? '~' : '!') + encodeURIComponent(p.src);
    }
    if (h !== location.hash) {
      try { history.replaceState(null, '', h); } catch (e) { location.hash = h; }
    }
  }

  var saveTimer = null;
  function save() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(writeHash, 400);
  }

  /* ---------------------------------------------------------------- input */

  var pointers = new Map();
  var pinch = null;

  canvas.addEventListener('pointerdown', function (e) {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX * dpr, y: e.clientY * dpr });
    if (pointers.size === 2) pinch = pinchState();
    canvas.classList.add('dragging');
    hideTip();
  });

  canvas.addEventListener('pointermove', function (e) {
    var px = e.clientX * dpr, py = e.clientY * dpr;
    var prev = pointers.get(e.pointerId);

    if (!prev) { showTip(px, py, e.clientX, e.clientY); return; }
    pointers.set(e.pointerId, { x: px, y: py });

    if (pointers.size >= 2) {
      var now = pinchState();
      if (pinch && now) {
        var k = state.scale * dpr;
        state.cx += (pinch.x - now.x) / k;
        state.cy -= (pinch.y - now.y) / k;
        if (now.d > 0 && pinch.d > 0) zoomAt(now.x, now.y, now.d / pinch.d);
        pinch = pinchState();
      }
      schedule();
      return;
    }

    var k2 = state.scale * dpr;
    state.cx -= (px - prev.x) / k2;
    state.cy += (py - prev.y) / k2;
    schedule();
  });

  function endPointer(e) {
    pointers.delete(e.pointerId);
    pinch = pointers.size === 2 ? pinchState() : null;
    if (!pointers.size) { canvas.classList.remove('dragging'); save(); }
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  function pinchState() {
    var it = pointers.values();
    var a = it.next().value, b = it.next().value;
    if (!a || !b) return null;
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      d: Math.hypot(a.x - b.x, a.y - b.y)
    };
  }

  function zoomAt(sx, sy, factor) {
    var k0 = state.scale * dpr;
    var wx = state.cx + (sx - W / 2) / k0;
    var wy = state.cy + (H / 2 - sy) / k0;
    state.scale = clamp(state.scale * factor, minScale(), MAX_SCALE);
    var k1 = state.scale * dpr;
    state.cx = wx - (sx - W / 2) / k1;
    state.cy = wy - (H / 2 - sy) / k1;
    schedule();
  }

  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    var d = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    zoomAt(e.clientX * dpr, e.clientY * dpr, Math.exp(-clamp(d, -400, 400) * 0.0015));
    hideTip();
    save();
  }, { passive: false });

  canvas.addEventListener('dblclick', function (e) {
    zoomAt(e.clientX * dpr, e.clientY * dpr, 2);
    save();
  });

  canvas.addEventListener('pointerleave', hideTip);

  window.addEventListener('keydown', function (e) {
    var tag = document.activeElement && document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    var step = 80 * dpr / (state.scale * dpr);
    switch (e.key) {
      case '+': case '=': zoomAt(W / 2, H / 2, 1.3); break;
      case '-': case '_': zoomAt(W / 2, H / 2, 1 / 1.3); break;
      case '0': resetView(); break;
      case 'ArrowLeft': state.cx -= step; schedule(); break;
      case 'ArrowRight': state.cx += step; schedule(); break;
      case 'ArrowUp': state.cy += step; schedule(); break;
      case 'ArrowDown': state.cy -= step; schedule(); break;
      default: return;
    }
    e.preventDefault();
    save();
  });

  window.addEventListener('resize', resize);

  /* -------------------------------------------------------------- tooltip */

  function showTip(px, py, clientX, clientY) {
    if (!view || !sieve) return;
    var hit = Spiral.nearest(view, px, py);
    var reach = Math.max(7, view.spacing * 0.45) * dpr;
    if (hit.n < 0 || hit.dist > reach) { hideTip(); return; }
    if (hover !== hit.n) {
      hover = hit.n;
      var html = '<span class="n">' + group(hit.n) + '</span> ';
      if (hit.n < 2) html += 'neither';
      else if (sieve.isPrime(hit.n)) html += '<span class="p">prime</span>';
      else html += '= ' + sieve.factor(hit.n).join('·');
      tooltip.innerHTML = html;
    }
    tooltip.hidden = false;
    tooltip.style.left = clientX + 'px';
    tooltip.style.top = clientY + 'px';
  }

  function hideTip() {
    hover = null;
    tooltip.hidden = true;
  }

  /* ------------------------------------------------------------- controls */

  var errTimer = null;
  function complain(msg) {
    errPop.textContent = msg;
    errPop.hidden = false;
    clearTimeout(errTimer);
    errTimer = setTimeout(function () { errPop.hidden = true; }, 3000);
  }

  document.getElementById('poly-add').addEventListener('submit', function (e) {
    e.preventDefault();
    var input = document.getElementById('poly-input');
    var src = input.value.trim() || input.placeholder;
    try {
      root.Expr.compile(src);
    } catch (err) {
      complain((err && err.message) || 'invalid expression');
      return;
    }
    errPop.hidden = true;
    addPoly(src);
    input.value = '';
  });

  /* ----------------------------------------------------------------- boot */

  function boot() {
    var linked = readHash();
    resize();
    if (linked.view) {
      state.cx = linked.view.cx;
      state.cy = linked.view.cy;
      state.scale = linked.view.scale;
    } else {
      resetView();
    }
    buildSieve(function () {
      linked.polys.forEach(function (p) {
        addPoly(p.src, { colorIndex: p.colorIndex, visible: p.visible });
      });
    });
  }

  boot();
})(window.PS = window.PS || {});
