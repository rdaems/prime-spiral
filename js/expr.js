/* Tiny expression parser for polynomials in n.
 *
 * Grammar (implicit multiplication allowed between adjacent atoms):
 *   expr  := term (('+' | '-') term)*
 *   term  := unary (('*' | '/')? unary)*
 *   unary := ('+' | '-') unary | power
 *   power := atom (('^' | '**') unary)?
 *   atom  := number | 'n' | '(' expr ')'
 *
 * compile() returns a plain function of n. No eval(), so a pasted expression
 * can do nothing but arithmetic.
 */
(function (root) {
  'use strict';

  var VARS = 'nNx';

  function tokenize(src) {
    var s = src.replace(/\s+/g, '');
    var out = [];
    var i = 0;
    while (i < s.length) {
      var c = s[i];
      if ((c >= '0' && c <= '9') || c === '.') {
        var j = i + 1;
        while (j < s.length && ((s[j] >= '0' && s[j] <= '9') || s[j] === '.')) j++;
        var v = Number(s.slice(i, j));
        if (!isFinite(v)) throw new Error('bad number "' + s.slice(i, j) + '"');
        out.push({ k: 'num', v: v });
        i = j;
      } else if (VARS.indexOf(c) >= 0) {
        out.push({ k: 'var' });
        i++;
      } else if (c === '*' && s[i + 1] === '*') {
        out.push({ k: 'op', v: '^' });
        i += 2;
      } else if ('+-*/^'.indexOf(c) >= 0) {
        out.push({ k: 'op', v: c });
        i++;
      } else if (c === '(' || c === ')') {
        out.push({ k: c });
        i++;
      } else {
        throw new Error('unexpected "' + c + '"');
      }
    }
    return out;
  }

  function compile(src) {
    var t = tokenize(src);
    if (!t.length) throw new Error('empty expression');
    var p = 0;

    function peek() { return t[p]; }
    function isOp(v) { var x = t[p]; return x && x.k === 'op' && x.v === v; }
    function startsAtom() {
      var x = t[p];
      return !!x && (x.k === 'num' || x.k === 'var' || x.k === '(');
    }

    function parseExpr() {
      var node = parseTerm();
      while (isOp('+') || isOp('-')) {
        var op = t[p++].v;
        var rhs = parseTerm();
        node = op === '+' ? add(node, rhs) : sub(node, rhs);
      }
      return node;
    }

    function parseTerm() {
      var node = parseUnary();
      for (;;) {
        if (isOp('*') || isOp('/')) {
          var op = t[p++].v;
          var rhs = parseUnary();
          node = op === '*' ? mul(node, rhs) : div(node, rhs);
        } else if (startsAtom()) {
          node = mul(node, parseUnary()); // implicit: 4n, 2(n+1), n(n+1)
        } else {
          return node;
        }
      }
    }

    function parseUnary() {
      if (isOp('-')) { p++; return neg(parseUnary()); }
      if (isOp('+')) { p++; return parseUnary(); }
      return parsePower();
    }

    function parsePower() {
      var base = parseAtom();
      if (isOp('^')) { p++; return pow(base, parseUnary()); }
      return base;
    }

    function parseAtom() {
      var x = peek();
      if (!x) throw new Error('unexpected end of expression');
      if (x.k === 'num') { p++; return konst(x.v); }
      if (x.k === 'var') { p++; return ident; }
      if (x.k === '(') {
        p++;
        var inner = parseExpr();
        if (!peek() || peek().k !== ')') throw new Error('missing ")"');
        p++;
        return inner;
      }
      throw new Error('unexpected "' + (x.v !== undefined ? x.v : x.k) + '"');
    }

    var fn = parseExpr();
    if (p < t.length) throw new Error('trailing input');

    // Reject anything that never produces a usable value.
    var ok = false;
    for (var probe = 0; probe < 6; probe++) {
      if (isFinite(fn(probe))) { ok = true; break; }
    }
    if (!ok) throw new Error('not a number');
    return fn;
  }

  function konst(v) { return function () { return v; }; }
  function ident(n) { return n; }
  function neg(a) { return function (n) { return -a(n); }; }
  function add(a, b) { return function (n) { return a(n) + b(n); }; }
  function sub(a, b) { return function (n) { return a(n) - b(n); }; }
  function mul(a, b) { return function (n) { return a(n) * b(n); }; }
  function div(a, b) { return function (n) { return a(n) / b(n); }; }
  function pow(a, b) { return function (n) { return Math.pow(a(n), b(n)); }; }

  root.Expr = { compile: compile };
})(window.PS = window.PS || {});
