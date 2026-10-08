// Tiny SVG plotting helper shared by the ZENN video widgets (no dependencies).
const NS = "http://www.w3.org/2000/svg";
const COLORS = {
  truth: "#e8e6df", zenn: "#d95926", mlp: "#3987e5",
  L: "#199e70", R: "#c98500", D: "#9085e9", other: "#6b6a66",
};

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function nearest(arr, v) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (Math.abs(arr[i] - v) < Math.abs(arr[best] - v)) best = i;
  return best;
}

class Plot {
  constructor(svg, o) {
    this.svg = svg;
    this.o = Object.assign({ m: { l: 64, r: 20, t: 16, b: 52 }, xticks: 5, yticks: 5 }, o);
    const { w, h, m } = this.o;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    this.pw = w - m.l - m.r;
    this.ph = h - m.t - m.b;
    this.axes = el("g", { class: "axes" }, svg);
    this.g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
    el("clipPath", { id: o.id + "clip" }, this.g).appendChild(
      el("rect", { x: 0, y: -2, width: this.pw, height: this.ph + 4 }));
    this.data = el("g", { "clip-path": `url(#${o.id}clip)` }, this.g);
    this.over = el("g", {}, this.g);
    this.items = {};
    this.setLim(o.xlim, o.ylim);
  }
  sx(v) { const [a, b] = this.xlim; return ((v - a) / (b - a)) * this.pw; }
  sy(v) { const [a, b] = this.ylim; return this.ph - ((v - a) / (b - a)) * this.ph; }
  ix(px) { const [a, b] = this.xlim; return a + (px / this.pw) * (b - a); }
  setLim(xlim, ylim) {
    this.xlim = xlim; this.ylim = ylim;
    const { m, xlabel, ylabel } = this.o;
    this.axes.innerHTML = "";
    const g = el("g", { transform: `translate(${m.l},${m.t})` }, this.axes);
    const tick = (lo, hi, n) => {
      const step = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(step)));
      const s = [1, 2, 2.5, 5, 10].map(k => k * p).find(k => k >= step);
      const out = [];
      for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(6));
      return out;
    };
    for (const v of tick(...ylim, this.o.yticks)) {
      el("line", { x1: 0, x2: this.pw, y1: this.sy(v), y2: this.sy(v), class: "grid" }, g);
      el("text", { x: -10, y: this.sy(v) + 5, class: "tick", "text-anchor": "end" }, g).textContent = v;
    }
    for (const v of tick(...xlim, this.o.xticks)) {
      el("text", { x: this.sx(v), y: this.ph + 22, class: "tick", "text-anchor": "middle" }, g).textContent = v;
    }
    el("line", { x1: 0, x2: this.pw, y1: this.ph, y2: this.ph, class: "axis" }, g);
    if (xlabel) el("text", { x: this.pw / 2, y: this.ph + 46, class: "label", "text-anchor": "middle" }, g).innerHTML = xlabel;
    if (ylabel) el("text", { x: -this.ph / 2, y: -48, class: "label", "text-anchor": "middle", transform: "rotate(-90)" }, g).innerHTML = ylabel;
  }
  _get(id, tag, parent = this.data) {
    if (!this.items[id]) this.items[id] = el(tag, {}, parent);
    return this.items[id];
  }
  line(id, xs, ys, { color = "#fff", width = 2, dash = "", opacity = 1 } = {}) {
    const d = xs.map((x, i) => `${i ? "L" : "M"}${this.sx(x).toFixed(1)},${this.sy(ys[i]).toFixed(1)}`).join("");
    const p = this._get(id, "path");
    p.setAttribute("d", d);
    p.setAttribute("style", `fill:none;stroke:${color};stroke-width:${width};stroke-dasharray:${dash};opacity:${opacity};stroke-linejoin:round;stroke-linecap:round`);
    return p;
  }
  points(id, xs, ys, { color = "#fff", r = 4.5 } = {}) {
    const g = this._get(id, "g");
    g.innerHTML = "";
    xs.forEach((x, i) => el("circle", { cx: this.sx(x), cy: this.sy(ys[i]), r, fill: color, stroke: "#1a1a19", "stroke-width": 2 }, g));
    return g;
  }
  stack(id, xs, layers, colors) {
    const g = this._get(id, "g");
    g.innerHTML = "";
    let base = xs.map(() => 0);
    layers.forEach((ys, k) => {
      const top = ys.map((y, i) => base[i] + y);
      const up = xs.map((x, i) => `${this.sx(x).toFixed(1)},${this.sy(top[i]).toFixed(1)}`);
      const dn = xs.map((x, i) => `${this.sx(x).toFixed(1)},${this.sy(base[i]).toFixed(1)}`).reverse();
      el("polygon", { points: up.concat(dn).join(" "), fill: colors[k], stroke: "#1a1a19", "stroke-width": 2 }, g);
      base = top;
    });
    return g;
  }
  vline(id, x, { color = "#9a988f", dash = "4 4", label = "", width = 1.5 } = {}) {
    const g = this._get(id, "g", this.over);
    g.innerHTML = "";
    if (x == null || isNaN(x)) return g;
    el("line", { x1: this.sx(x), x2: this.sx(x), y1: 0, y2: this.ph, stroke: color, "stroke-width": width, "stroke-dasharray": dash }, g);
    if (label) el("text", { x: this.sx(x) + 6, y: 16, class: "note", fill: color }, g).textContent = label;
    return g;
  }
  band(id, x0, x1, { color = "#ffffff", opacity = 0.05 } = {}) {
    const r = this._get(id, "rect");
    for (const [k, v] of Object.entries({ x: this.sx(x0), y: 0, width: this.sx(x1) - this.sx(x0), height: this.ph, fill: color, opacity }))
      r.setAttribute(k, v);
    this.data.insertBefore(r, this.data.firstChild);
    return r;
  }
  // Crosshair + tooltip. rowsAt(xValue) -> [{name, color, value}]
  hover(tip, rowsAt) {
    const cross = el("line", { y1: 0, y2: this.ph, class: "cross", visibility: "hidden" }, this.over);
    const hit = el("rect", { x: 0, y: 0, width: this.pw, height: this.ph, fill: "transparent" }, this.g);
    const move = (ev) => {
      const pt = this.svg.createSVGPoint();
      pt.x = ev.clientX; pt.y = ev.clientY;
      const loc = pt.matrixTransform(this.g.getScreenCTM().inverse());
      const xv = this.ix(Math.max(0, Math.min(this.pw, loc.x)));
      const { x, rows, title } = rowsAt(xv);
      cross.setAttribute("x1", this.sx(x)); cross.setAttribute("x2", this.sx(x));
      cross.setAttribute("visibility", "visible");
      tip.innerHTML = `<div class="tt">${title}</div>` + rows.map(r =>
        `<div class="tr"><span class="sw" style="background:${r.color}"></span>${r.name}<b>${r.value}</b></div>`).join("");
      tip.style.display = "block";
      const box = this.svg.getBoundingClientRect();
      const left = ev.clientX - box.left;
      tip.style.left = (left > box.width * 0.6 ? left - tip.offsetWidth - 16 : left + 16) + "px";
      tip.style.top = Math.max(8, ev.clientY - box.top - 30) + "px";
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  }
}

// Play/pause a slider through its range. Returns a toggle function.
function player(slider, button, onStep, { fps = 30, step = 1 } = {}) {
  let timer = null;
  const stop = () => { clearInterval(timer); timer = null; button.textContent = "▶ Play"; };
  const toggle = () => {
    if (timer) return stop();
    if (+slider.value >= +slider.max) slider.value = slider.min;
    button.textContent = "❚❚ Pause";
    timer = setInterval(() => {
      const v = +slider.value + step;
      if (v > +slider.max) return stop();
      slider.value = v;
      onStep();
    }, 1000 / fps);
  };
  button.addEventListener("click", toggle);
  return toggle;
}

// Space toggles play, arrow keys nudge the slider: handy while recording.
function keys(slider, toggle, onStep) {
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); toggle(); }
    if (e.code === "ArrowUp" || e.code === "ArrowDown") {
      e.preventDefault();
      slider.value = +slider.value + (e.code === "ArrowUp" ? 1 : -1);
      onStep();
    }
  });
}
