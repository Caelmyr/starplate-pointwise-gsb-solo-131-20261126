const colors = {
  point: "#ffd98e", box: "#7df4d4", trace: "#9fc5ff", note: "#f0a4ee",
  source: "#73eed0", aperture: "#ffcd84", sky: "#9ec4fd",
};

export function plotImage(overlay, state) {
  const view = state.view;
  if (!view) return;
  const rect = overlay.getBoundingClientRect();
  const density = Math.max(1, devicePixelRatio || 1);
  overlay.width = Math.max(1, Math.round(rect.width * density));
  overlay.height = Math.max(1, Math.round(rect.height * density));
  const ctx = overlay.getContext("2d");
  ctx.scale(density, density);
  const sx = rect.width / view.width;
  const sy = rect.height / view.height;
  const screen = (x, y) => [(x + .5 - view.x) * sx, (y + .5 - view.y) * sy];
  const stroke = (color, width = 1.35, dash = []) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  };
  function line(points, color, width = 1.35, dash = []) {
    if (points.length < 2) return;
    stroke(color, width, dash);
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      const [a, b] = screen(x, y);
      if (index) ctx.lineTo(a, b); else ctx.moveTo(a, b);
    });
    ctx.stroke();
  }
  function box(region, color, dash = [], fill = null) {
    const [a, b] = screen(region.x - .5, region.y - .5);
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fillRect(a + .5, b + .5, sx * region.width - 1, sy * region.height - 1);
    }
    stroke(color, 1.4, dash);
    ctx.strokeRect(a + .5, b + .5, sx * region.width - 1, sy * region.height - 1);
  }
  function mark(x, y, color, radius = 4, label = "") {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const [a, b] = screen(x, y);
    stroke(color);
    ctx.beginPath(); ctx.arc(a, b, radius, 0, 2 * Math.PI); ctx.stroke();
    if (label) {
      ctx.fillStyle = color;
      ctx.font = "11px ui-monospace, monospace";
      ctx.shadowColor = "#070f1b"; ctx.shadowBlur = 5;
      ctx.fillText(label, a + radius + 3, b + 3);
      ctx.shadowBlur = 0;
    }
  }
  function ellipse(x, y, ra, rb, angle, color, dash = []) {
    const [cx, cy] = screen(x, y);
    if (![cx, cy, ra, rb, angle].every(Number.isFinite) || ra < 0 || rb < 0) return;
    ctx.save();
    ctx.translate(cx, cy); ctx.scale(sx, sy);
    stroke(color, 1.4 / Math.max(sx, sy), dash.map((n) => n / sx));
    ctx.beginPath();
    ctx.ellipse(0, 0, ra, rb, angle * Math.PI / 180, 0, Math.PI * 2);
    ctx.stroke(); ctx.restore();
  }
  if (state.skyGrid) {
    for (const item of state.skyGrid.lines) {
      for (const segment of item.segments) line(segment, "#afc8ef85", 1, [4, 3]);
    }
  }
  if (state.box && (state.box.width !== state.frame.width || state.box.height !== state.frame.height)) {
    box(state.box, colors.box, [5, 4]);
  }
  if (state.line) line(state.line, colors.trace, 2);
  if (state.vertices.length > 1) {
    line(state.vertices, colors.box, 1.4);
    if (state.vertices.length >= 3) line([state.vertices[0], state.vertices.at(-1)], colors.box, 1, [3, 3]);
  }
  state.vertices.forEach(([x, y]) => mark(x, y, colors.box, 2.8));
  if (state.temporaryBox) box(state.temporaryBox, "#fff2c0");
  if (state.temporaryLine) line(state.temporaryLine, "#fff2c0", 2);
  for (const note of state.notes) mark(note.x, note.y, colors.note, 4.5, note.title);
  const candidate = state.output;
  if (candidate) {
    const result = candidate.result;
    const {operation} = candidate;
    const sources = result.sources || result.stars || result.peaks || result.marks || [];
    const apertureOperation = operation === "aperture" ? result : state.recentAperture;
    for (const source of sources) {
      const label = source.id && sources.length < 35 ? String(source.id) : "";
      mark(source.x, source.y, colors.source, 4, label);
    }
    if (result.levels) {
      result.levels.forEach((level, index) => {
        const tint = ["#aaffdb", "#bfd8ff", "#ffc58a", "#df9ef9"][index % 4];
        level.paths.forEach((path) => line(path.points, tint, 1.3));
      });
    }
    if (result.cells && ["quality", "contrast-cells", "background", "pair"].includes(operation)) {
      for (const cell of result.cells) {
        if (!cell.box) continue;
        const shade = cell.usable ? "#2df6cf0b" : "#f4988755";
        box(cell.box, cell.usable ? "#72a6a680" : "#e79279a0", [], shade);
      }
    }
    if (operation === "mask") result.edges?.forEach(([x, y]) => mark(x, y, "#ffac99", 1.8));
    if (operation === "histogram-bin") result.samples?.forEach(({x, y}) => mark(x, y, "#ffddac", 2));
    if (operation === "world" && result.in_frame) mark(result.x, result.y, "#90dcff", 7, "ICRS");
    if (operation === "region-flux" && result.shape === "ellipse") {
      const [[x, y], [a, b], angle] = result.points;
      ellipse(x, y, a, b, angle, colors.aperture);
    }
    if (result.psf) {
      const {x, y, fwhm_major, fwhm_minor, angle} = result.psf;
      ellipse(x, y, fwhm_major / 2, fwhm_minor / 2, angle, colors.sky);
    }
    if (operation === "fit-source" && result.center) {
      ellipse(...result.center, result.major_px / 2, result.minor_px / 2, result.angle_deg, colors.sky);
    }
    if (apertureOperation?.point) {
      const {radius, sky_inner, sky_outer} = apertureOperation.settings;
      const [x, y] = apertureOperation.point;
      ellipse(x, y, radius, radius, 0, colors.aperture);
      ellipse(x, y, sky_inner, sky_inner, 0, colors.sky, [3, 3]);
      ellipse(x, y, sky_outer, sky_outer, 0, colors.sky, [3, 3]);
    }
  }
  if (state.point) {
    const [a, b] = screen(...state.point);
    stroke(colors.point);
    ctx.beginPath();
    ctx.moveTo(a - 10, b); ctx.lineTo(a - 4, b);
    ctx.moveTo(a + 4, b); ctx.lineTo(a + 10, b);
    ctx.moveTo(a, b - 10); ctx.lineTo(a, b - 4);
    ctx.moveTo(a, b + 4); ctx.lineTo(a, b + 10);
    ctx.stroke();
  }
}

export function makeChart(rows, keys, label) {
  const canvas = document.createElement("canvas");
  canvas.width = 640; canvas.height = 230;
  canvas.className = "chart";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", label);
  const ctx = canvas.getContext("2d");
  const traces = keys.map((key) => rows.map((row) => row[key]));
  const values = traces.flat().filter(Number.isFinite);
  if (!values.length) return canvas;
  const low = Math.min(...values);
  const high = Math.max(...values);
  const spread = high - low || 1;
  const width = canvas.width - 113;
  const height = canvas.height - 43;
  const xAt = (index) => 76 + width * index / Math.max(1, rows.length - 1);
  const yAt = (value) => 19 + height * (high - value) / spread;
  ctx.font = "17px ui-monospace, monospace"; ctx.textBaseline = "middle";
  for (let i = 0; i < 5; ++i) {
    const level = low + spread * i / 4;
    const y = yAt(level);
    ctx.strokeStyle = "#304259"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(76, y); ctx.lineTo(canvas.width - 19, y); ctx.stroke();
    ctx.fillStyle = "#a9bbd1"; ctx.fillText(level.toPrecision(3), 5, y);
  }
  const palette = ["#77efd0", "#aac9ff", "#ffcf92"];
  traces.forEach((trace, t) => {
    ctx.strokeStyle = palette[t % palette.length];
    ctx.lineWidth = 2.1;
    ctx.beginPath();
    let missing = true;
    for (let i = 0; i < trace.length; ++i) {
      const value = trace[i];
      if (!Number.isFinite(value)) { missing = true; continue; }
      if (missing) ctx.moveTo(xAt(i), yAt(value));
      else ctx.lineTo(xAt(i), yAt(value));
      missing = false;
    }
    ctx.stroke();
  });
  canvas.indexAt = (event) => {
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * canvas.width / bounds.width;
    return Math.round(Math.max(0, Math.min(rows.length - 1, (x - 76) * (rows.length - 1) / width)));
  };
  return canvas;
}
