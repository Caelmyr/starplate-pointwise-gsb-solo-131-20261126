import {groups, tools} from "./tools.js";
import {makeChart, plotImage} from "./view.js";

const el = (query) => document.querySelector(query);
const all = (query) => Array.from(document.querySelectorAll(query));
const newNode = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const makeButton = (text, handler, className = "") => {
  const button = newNode("button", text, className);
  button.type = "button";
  button.addEventListener("click", handler);
  return button;
};
const full = (frame) => ({x: 0, y: 0, width: frame.width, height: frame.height});
const ff = (value, digits = 5) => {
  if (value === null || value === undefined) return "—";
  if (typeof value !== "number") return String(value);
  if (!Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toPrecision(digits)));
};
const pretty = (key) => key.replaceAll("_", " ").replaceAll("-", " ");
const pos = ([x, y]) => `${ff(x, 7)}, ${ff(y, 7)}`;
const framePath = (tail = "", key = state.frame?.key) => `/api/frames/${encodeURIComponent(key)}${tail}`;
const action = (data, method = "POST") => ({method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
const coordinateFields = ["x", "y", "width", "height"];

const state = {
  frames: [], frame: null, meta: null, box: null, view: null,
  point: null, line: null, vertices: [], mode: "point", tool: "aperture",
  output: null, catalog: [], notes: [], saved: [], skyGrid: null,
  showGrid: false, recentAperture: null, drag: null,
  temporaryBox: null, temporaryLine: null, measured: null,
  stretch: {mode: "asinh", palette: "gray", low_pct: 1, high_pct: 99.8, strength: 6, invert: false},
};

let frameTicket = 0;
let pointTicket = 0;
let toolTicket = 0;
let toastTimer;
let colorUrl = null;

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Could not complete this action (HTTP ${response.status}).`;
    try {
      const error = await response.json();
      if (error.error) message = error.field ? `${error.error} (${error.field})` : error.error;
    } catch {}
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function tell(message, bad = false) {
  const node = el("#toast");
  node.hidden = false;
  node.textContent = message;
  node.classList.toggle("error", bad);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 4400);
}
function toolStatus(message, bad = false) {
  const node = el("#operation-status");
  node.textContent = message;
  node.classList.toggle("error", bad);
}
function paint() {
  if (state.frame) plotImage(el("#overlay"), state);
}
function inputNumber(id, value) {
  el(id).value = value === null || value === undefined ? "" : String(value);
}
function setTab(name) {
  all("[data-tab]").forEach((button) => button.setAttribute("aria-selected", String(button.dataset.tab === name)));
  ["measure", "display", "frame"].forEach((id) => { el(`#pane-${id}`).hidden = name !== id; });
}
function showDialog(id) {
  const dialog = el(`#${id}`);
  if (!dialog.open) dialog.showModal();
}

function replaceList(holder, nodes, emptyMessage) {
  el(holder).replaceChildren(...(nodes.length ? nodes : [newNode("span", emptyMessage, "quiet")]));
}
function renderFrameList() {
  el("#library-count").textContent = state.frames.length;
  const buttons = state.frames.map((frame) => {
    const card = makeButton("", () => openFrame(frame.key), "frame-card");
    card.setAttribute("aria-current", String(frame.key === state.frame?.key));
    const icon = newNode("span", frame.origin === "archive" ? "✦" : frame.origin === "derived" ? "◈" : "□", "frame-icon");
    icon.setAttribute("aria-hidden", "true");
    const detail = newNode("span");
    detail.append(newNode("strong", frame.title), newNode("small", `${frame.width} × ${frame.height} · ${frame.origin}`));
    card.append(icon, detail);
    return card;
  });
  replaceList("#frame-list", buttons, "No images.");
}
async function refreshLibrary() {
  state.frames = (await api("/api/frames")).frames;
  renderFrameList();
}
async function loadCollections(key = state.frame?.key) {
  if (!key) return;
  const [notes, saved] = await Promise.all([api(framePath("/notes", key)), api(framePath("/saved", key))]);
  if (key !== state.frame?.key) return;
  state.notes = notes.notes;
  state.saved = saved.measurements;
  renderCollections();
  paint();
}
function renderCollections() {
  const notes = state.notes.map((note) => {
    const row = newNode("div", undefined, "rail-row");
    const title = makeButton(note.title, () => selectPoint([note.x, note.y]));
    title.title = note.text || `Original ${pos([note.x, note.y])}`;
    const remove = makeButton("×", () => deleteCollection("notes", note.key), "small");
    remove.setAttribute("aria-label", `Remove mark ${note.title}`);
    row.append(title, remove);
    return row;
  });
  const measurements = state.saved.map((measure) => {
    const row = newNode("div", undefined, "rail-row");
    const title = makeButton(measure.title, () => loadSaved(measure.key));
    const remove = makeButton("×", () => deleteCollection("saved", measure.key), "small");
    remove.setAttribute("aria-label", `Remove measurement ${measure.title}`);
    row.append(title, remove);
    return row;
  });
  replaceList("#note-list", notes, "No marks on this image.");
  replaceList("#saved-list", measurements, "No saved measurements.");
}
async function deleteCollection(type, key) {
  try {
    await api(framePath(`/${type}/${encodeURIComponent(key)}`), {method: "DELETE"});
    await loadCollections();
  } catch (error) { tell(error.message, true); }
}

function updateBoxForm() {
  coordinateFields.forEach((name) => inputNumber(`#box-${name}`, state.box?.[name]));
  const box = state.box;
  el("#selection-readout").textContent = box ? `Box ${box.width} × ${box.height} px` : "";
}
function setBox(box, clearResults = true) {
  state.box = box;
  if (clearResults) {
    toolTicket++;
    state.output = state.measured = state.recentAperture = null;
    state.catalog = [];
    el("#retain-output").hidden = true;
    el("#result").replaceChildren();
    toolStatus("");
  }
  updateBoxForm();
  paint();
}
function setImageDimensions() {
  const stage = el("#view-stage");
  const plane = el("#image-plane");
  if (!state.view || plane.hidden) return;
  const W = Math.max(1, stage.clientWidth - 24);
  const H = Math.max(1, stage.clientHeight - 20);
  const ratio = Math.min(W / state.view.width, H / state.view.height);
  plane.style.width = `${Math.max(1, Math.round(state.view.width * ratio))}px`;
  plane.style.height = `${Math.max(1, Math.round(state.view.height * ratio))}px`;
  paint();
}
function resetColor() {
  if (colorUrl) URL.revokeObjectURL(colorUrl);
  colorUrl = null;
}
function refreshPreview() {
  if (!state.frame || !state.view) return;
  resetColor();
  const input = new URLSearchParams({...state.view, ...state.stretch, size: 1450, lock_range: true});
  el("#science-image").src = framePath(`/image?${input}`);
  el("#image-plane").hidden = false;
  el("#view-empty").hidden = true;
  setImageDimensions();
}
function changeView(zoomed) {
  if (!state.frame) return;
  state.view = zoomed ? {...state.box} : full(state.frame);
  refreshPreview();
}
function renderMetadata() {
  const meta = state.meta;
  const frame = state.frame;
  const node = el("#frame-metadata");
  if (!meta || !frame) return node.replaceChildren();
  const metric = metricGrid([
    ["Dimensions", `${frame.width} × ${frame.height}`], ["Source", frame.origin],
    ["Units", frame.unit || "pixel values"], ["FITS HDU", frame.hdu],
    ["WCS", frame.has_wcs ? "Celestial" : "Unavailable"],
    ["File", frame.original_name],
  ]);
  const assets = [];
  if (frame.origin === "archive" && frame.source_url) {
    const asset = newNode("a", "Download original archive file");
    asset.href = `/api/archive/${encodeURIComponent(frame.key)}`;
    const real = newNode("a", "SDSS archive record");
    real.href = frame.source_url;
    real.target = "_blank";
    real.rel = "noopener noreferrer";
    assets.push(asset, real);
  } else if (frame.origin !== "archive") {
    assets.push(makeButton("Rename image", async () => {
      const title = prompt("Image display name", state.frame.title);
      if (!title?.trim()) return;
      try {
        await api(framePath(), action({title}, "PATCH"));
        await refreshLibrary();
        await openFrame(frame.key);
      } catch (error) { tell(error.message, true); }
    }), makeButton("Remove image", async () => {
      if (!confirm(`Remove “${frame.title}” and its saved measurements from this workspace?`)) return;
      try {
        await api(framePath(), {method: "DELETE"});
        await refreshLibrary();
        await openFrame(state.frames[0]?.key);
      } catch (error) { tell(error.message, true); }
    }));
  }
  node.replaceChildren(metric, ...assets);
  el("#header-cards").replaceChildren(dataTable(meta.header, ["keyword", "value", "comment"]));
}
async function openFrame(key) {
  const found = state.frames.find((row) => row.key === key);
  if (!found) {
    state.frame = state.meta = state.view = state.box = null;
    el("#image-plane").hidden = true;
    el("#view-empty").hidden = false;
    el("#frame-heading").textContent = "No active frame";
    renderFrameList();
    return;
  }
  const ticket = ++frameTicket;
  ++pointTicket;
  ++toolTicket;
  state.frame = found;
  state.view = full(found);
  state.line = state.point = state.recentAperture = state.output = state.measured = null;
  state.vertices = []; state.catalog = []; state.notes = []; state.saved = []; state.skyGrid = null;
  state.showGrid = false;
  state.meta = null;
  el("#wcs-grid").setAttribute("aria-pressed", "false");
  el("#frame-heading").replaceChildren(newNode("span", found.title), newNode("small", `${found.width} × ${found.height}`));
  el("#display-result").replaceChildren();
  el("#frame-result").replaceChildren();
  el("#quick-probe").textContent = "Pixel readings stay in the original frame units.";
  el("#pointer-readout").textContent = "Move over the image to see original coordinates";
  inputNumber("#point-x", null); inputNumber("#point-y", null);
  renderFrameList(); renderCollections(); renderMetadata();
  setBox(full(found)); configureTool(); refreshPreview();
  try {
    const [metadata] = await Promise.all([api(framePath("", key)), loadCollections(key)]);
    if (ticket !== frameTicket) return;
    state.meta = metadata;
    renderMetadata();
  } catch (error) {
    if (ticket === frameTicket) tell(error.message, true);
  }
}

function selectMode(mode) {
  state.mode = mode;
  all("[data-mode]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
}
function viewPosition(event) {
  const bounds = el("#overlay").getBoundingClientRect();
  const view = state.view;
  const x = Math.max(0, Math.min(1 - 1e-9, (event.clientX - bounds.left) / bounds.width));
  const y = Math.max(0, Math.min(1 - 1e-9, (event.clientY - bounds.top) / bounds.height));
  return [x * view.width + view.x - .5, y * view.height + view.y - .5];
}
function clickedPixel(point) {
  return point.map((axis) => Math.floor(axis + .5));
}
function boxBetween(first, second) {
  const [x1, y1] = clickedPixel(first), [x2, y2] = clickedPixel(second);
  return {x: Math.min(x1, x2), y: Math.min(y1, y2), width: Math.abs(x1 - x2) + 1, height: Math.abs(y1 - y2) + 1};
}
function pointerStart(event) {
  if (!state.frame || event.button !== 0) return;
  const position = viewPosition(event);
  if (state.mode === "point") { selectPoint(clickedPixel(position)); return; }
  if (state.mode === "polygon") {
    if (state.vertices.length >= 30) return tell("A polygon can contain up to 30 marks.", true);
    state.vertices.push(clickedPixel(position));
    paint();
    return;
  }
  state.drag = {start: position, mode: state.mode};
  event.currentTarget.setPointerCapture(event.pointerId);
}
function pointerMove(event) {
  if (!state.frame) return;
  const mark = viewPosition(event);
  const [x, y] = clickedPixel(mark);
  el("#pointer-readout").textContent = `Detector X ${x} · Y ${y}`;
  if (!state.drag) return;
  if (state.drag.mode === "box") state.temporaryBox = boxBetween(state.drag.start, mark);
  else state.temporaryLine = [state.drag.start, mark];
  paint();
}
function pointerEnd(event) {
  if (!state.drag || !state.frame) return;
  const {start, mode} = state.drag;
  const end = viewPosition(event);
  state.drag = state.temporaryBox = state.temporaryLine = null;
  if (mode === "box") setBox(boxBetween(start, end));
  else {
    state.line = [start, end].map((point) => point.map((v) => Number(v.toFixed(2))));
    paint();
    tell(`Line set from ${pos(state.line[0])} to ${pos(state.line[1])}.`);
  }
}
function pointFits(point, frame) {
  return point && point.every(Number.isFinite) && point[0] >= -.5 && point[1] >= -.5
    && point[0] < frame.width - .5 && point[1] < frame.height - .5;
}
async function selectPoint(point, auto = true) {
  if (!state.frame || !pointFits(point, state.frame)) return tell("The point must be inside the original image.", true);
  const ticket = ++pointTicket;
  state.point = [...point];
  inputNumber("#point-x", Number(point[0].toFixed(4)));
  inputNumber("#point-y", Number(point[1].toFixed(4)));
  paint();
  const currentTool = state.tool;
  if (auto && ["aperture", "probe"].includes(currentTool)) runTool();
  try {
    const payload = await api(framePath("/measure/probe"), action({point, radius: 1, patch: 1}));
    if (ticket !== pointTicket) return;
    const result = payload.result;
    const {original_pixel: original, world} = result;
    const celestial = world ? ` · RA ${ff(world.ra_deg, 7)}° Dec ${ff(world.dec_deg, 7)}°` : "";
    el("#quick-probe").textContent = `X ${original.x} Y ${original.y} · ${ff(original.value)} ${result.unit}${celestial}`;
  } catch (error) { if (ticket === pointTicket) tell(error.message, true); }
}
function optionsForField(field) {
  const source = state.frames.filter((frame) => field.optional || frame.key !== state.frame?.key);
  if (field.type === "frames") return source.map((frame) => [frame.key, frame.title]);
  if (field.type === "frame") {
    return [["", field.optional ? "None / active" : "Select image"], ...source.map((frame) => [frame.key, frame.title])];
  }
  return field.options.map((item) => typeof item === "string" ? [item, pretty(item)] : item);
}
function fieldElement(field) {
  const wrap = newNode("label", field.type === "checkbox" ? undefined : field.label);
  const type = field.type;
  const node = ["frame", "frames", "select"].includes(type) ? newNode("select") : newNode("input");
  node.name = field.key;
  if (node.tagName === "SELECT") {
    if (type === "frames") node.multiple = true;
    for (const [value, label] of optionsForField(field)) {
      const option = newNode("option", label);
      option.value = value;
      node.append(option);
    }
    if (field.value !== undefined) node.value = Array.isArray(field.value) ? field.value[0] : field.value;
  } else {
    node.type = type === "point" || type === "list" ? "text" : type;
    if (type === "checkbox") {
      wrap.className = "check";
      node.checked = Boolean(field.value);
      wrap.append(node, newNode("span", field.label));
      return wrap;
    }
    if (type === "number") node.step = field.step;
    if (type === "point") node.placeholder = "x, y";
    node.value = field.value;
  }
  if (type === "frames" || field.key === "levels") wrap.classList.add("wide");
  wrap.append(node);
  return wrap;
}
function configureTool(preserved = null, preserveResult = false) {
  const [, description, fields, , route] = tools[state.tool];
  el("#tool-description").textContent = description;
  el("#tool-fields").replaceChildren(...fields.map(fieldElement));
  el("#tool-select").value = state.tool;
  el("#run-tool").textContent = route === "derive" ? "Preview result" : route === "color" ? "Preview composite" : "Run measurement";
  el("#retain-output").hidden = route !== "derive" || state.measured?.operation !== state.tool;
  if (preserved) {
    for (const field of fields) {
      const value = field.key.split(".").reduce((map, key) => map?.[key], preserved);
      if (value === null || value === undefined) continue;
      const input = el("#tool-form").elements.namedItem(field.key);
      if (field.type === "checkbox") input.checked = value;
      else if (field.type === "frames") Array.from(input.options).forEach((option) => { option.selected = value.includes(option.value); });
      else input.value = Array.isArray(value) ? value.join(",") : String(value);
    }
  }
  toolStatus("");
  if (!preserveResult) el("#result").replaceChildren();
}
function nestedSet(target, path, value) {
  const names = path.split(".");
  let out = target;
  for (const name of names.slice(0, -1)) {
    if (!out[name]) out[name] = {};
    out = out[name];
  }
  out[names.at(-1)] = value;
}
function toolPayload() {
  if (!state.frame) throw new Error("Choose an image first.");
  const [, , fields, needs] = tools[state.tool];
  const parameters = {box: {...state.box}};
  if (state.point) parameters.point = [...state.point];
  if (state.line) [parameters.start, parameters.end] = state.line.map((p) => [...p]);
  if (state.vertices.length) parameters.vertices = state.vertices.map((p) => [...p]);
  if (needs === "point" && !state.point) throw new Error("Choose an original point on the image.");
  if (needs === "line" && !state.line) throw new Error("Use Line and drag a measurement line on the image.");
  for (const field of fields) {
    const input = el("#tool-form").elements.namedItem(field.key);
    let value;
    if (field.type === "checkbox") value = input.checked;
    else if (field.type === "frames") value = Array.from(input.selectedOptions).map((option) => option.value);
    else if (!input.value.trim()) continue;
    else if (field.type === "number") value = input.valueAsNumber;
    else if (field.type === "list" || field.type === "point") {
      const parts = input.value.split(",").map((piece) => piece.trim());
      if (parts.some((part) => !part || !Number.isFinite(Number(part)))) throw new Error(`${field.label} needs comma-separated numbers.`);
      value = parts.map(Number);
    } else value = input.value;
    nestedSet(parameters, field.key, value);
  }
  return parameters;
}
function metricGrid(entries) {
  const node = newNode("dl", undefined, "metrics");
  entries.forEach(([label, value]) => {
    const pair = newNode("div", undefined, "metric");
    pair.append(newNode("dt", label), newNode("dd", ff(value)));
    node.append(pair);
  });
  return node;
}
function dataTable(rows, columns, selectRow = null) {
  if (!rows.length) return newNode("p", "No rows to display.", "quiet");
  const wrap = newNode("div", undefined, "table-wrap");
  const table = newNode("table");
  const header = table.createTHead().insertRow();
  columns.forEach((column) => {
    const field = Array.isArray(column) ? column[1] : pretty(column);
    header.append(newNode("th", field));
  });
  const body = table.createTBody();
  for (const row of rows.slice(0, 240)) {
    const line = body.insertRow();
    columns.forEach((column) => {
      const key = Array.isArray(column) ? column[0] : column;
      line.append(newNode("td", ff(row[key])));
    });
    if (selectRow) {
      line.dataset.select = "true";
      line.tabIndex = 0;
      line.addEventListener("click", () => selectRow(row));
      line.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectRow(row); }
      });
    }
  }
  wrap.append(table);
  return wrap;
}
function detailsResult(result) {
  const details = newNode("details");
  details.append(newNode("summary", "Result details"), newNode("pre", JSON.stringify(result, null, 2)));
  return details;
}
function sourceRow(row) {
  if (Number.isFinite(row.x) && Number.isFinite(row.y)) selectPoint([row.x, row.y], false);
}
function apertureResult(result) {
  const fields = [
    ["Net flux", result.flux], ["Uncertainty", result.uncertainty], ["Signal / noise", result.snr],
    ["Sky level", result.sky?.level], ["Aperture coverage", `${ff(result.aperture_coverage * 100)}%`],
    ["Annulus coverage", `${ff(result.sky_coverage * 100)}%`],
    ["Original x, y", pos(result.point)], ["Input peak", result.input_peak],
  ];
  return [metricGrid(fields), newNode("p", result.reasons.length ? result.reasons.map(pretty).join(" · ") : `Flux in ${result.unit || "image units"}`, "quiet")];
}
async function pickHistogram(bin, histogram, measuredParams) {
  if (!state.frame) return;
  try {
    const isLast = bin === histogram.bins.at(-1);
    const args = {box: measuredParams.box, left: bin.left, right: bin.right, include_upper: isLast};
    const response = await api(framePath("/measure/histogram-bin"), action(args));
    toolStatus(`${response.result.count} original pixels in this bin. Showing up to 300.`);
    state.output = {operation: "histogram-bin", result: response.result};
    paint();
  } catch (error) { toolStatus(error.message, true); }
}
function renderResult(operation, result, readonlyKey = null) {
  const container = el("#result");
  const items = [newNode("h3", `${tools[operation]?.[0] || pretty(operation)} result`)];
  if (operation === "stars") {
    items.push(metricGrid([["Detected islands", result.positive_components], ["Displayed sources", result.sources.length],
      ["Original threshold", result.threshold_value], ["Sky median", result.background.median]]));
    items.push(dataTable(result.sources, ["id", "x", "y", "peak", ["isophotal_flux", "Island flux"], "area"], sourceRow));
    items.push(makeButton("Measure selected source", () => {
      if (!state.point) return tell("Select a source in the table first.", true);
      setTool("aperture");
      runTool();
    }));
  } else if (operation === "aperture") {
    items.push(...apertureResult(result));
  } else if (operation === "probe") {
    const original = result.original_pixel;
    items.push(metricGrid([["Original pixel", `${original.x}, ${original.y}`], ["Pixel value", original.value],
      ["Neighbor median", result.neighborhood.median], ["Finite neighbors", result.neighborhood.finite],
      ["RA", result.world?.ra_deg], ["DEC", result.world?.dec_deg]]));
    const rows = result.patch.y.map((y, index) => Object.assign({y}, Object.fromEntries(result.patch.x.map((x, col) => [x, result.patch.values[index][col]]))));
    items.push(dataTable(rows, ["y", ...result.patch.x.map(String)]));
  } else if (operation === "profile") {
    items.push(metricGrid([["Original length", `${ff(result.length)} px`], ["Samples", result.samples.length]]));
    const chart = makeChart(result.samples, ["value"], "Original intensity along the line");
    chart.addEventListener("click", (event) => sourceRow(result.samples[chart.indexAt(event)]));
    items.push(chart);
    items.push(dataTable(result.samples, ["distance", "x", "y", "value", "valid"], sourceRow));
  } else if (operation === "aperture-curve") {
    items.push(makeChart(result.measurements, ["flux"], "Flux across aperture radii"));
    items.push(dataTable(result.measurements, ["radius", "flux", "relative_flux", "coverage"]));
  } else if (operation === "histogram") {
    const chart = makeChart(result.bins, ["count"], "Intensity histogram");
    const parameters = state.measured?.parameters || {box: state.box};
    chart.addEventListener("click", (event) => pickHistogram(result.bins[chart.indexAt(event)], result, parameters));
    items.push(metricGrid([["Finite pixels", result.finite], ["Unavailable", result.missing],
      ["Below range", result.clip.below], ["Above range", result.clip.above]]));
    items.push(chart, dataTable(result.bins, ["left", "right", "count"], (bin) => pickHistogram(bin, result, parameters)));
  } else if (operation === "strip") {
    items.push(makeChart(result.along, ["mean", "median"], "Strip intensity along the line"));
    items.push(dataTable(result.across, ["at", "median", "mean", "coverage"]));
  } else if (operation === "fit-source") {
    items.push(metricGrid([["Converged", result.converged], ["Major FWHM", result.major_px], ["Minor FWHM", result.minor_px],
      ["Analytic flux", result.analytic_flux], ["Angle", result.angle_deg], ["RMS residual", result.rms_residual]]));
    items.push(makeChart(result.fitted_pixels, ["original", "model"], "Original pixels and fitted profile"));
  } else if (operation === "rings") {
    items.push(dataTable(result.bands, ["inner", "outer", "support", "pixels", "median", "net_sum"]));
  } else if (operation === "seam") {
    items.push(metricGrid([
      ["Max deviation from drafted line", `${ff(result.deviation.max_px)} px`],
      ["Mean deviation", `${ff(result.deviation.mean_px)} px`],
      ["Search band", `±${result.search_band_radius} px`],
      ["Mean switch jump", result.jump.mean], ["Max switch jump", result.jump.max],
      ["Reference pixels", result.samples.reference_pixels],
      ["Companion pixels", result.samples.partner_pixels],
      ["Empty pixels", result.samples.empty_pixels],
      ["Brightness limit", result.brightness_limit],
    ]));
    items.push(newNode("p",
      `Seam fixed at ${result.seam.length} columns; each column differs from its neighbour by at most one row. Retain it to open the mosaic and trace its source pixels.`,
      "quiet"));
    if (result.probe) {
      items.push(metricGrid([
        ["Probe point", pos(result.probe.point)],
        ["Adopted source", result.probe.source],
        ["Reference reading", result.probe.reference_value],
        ["Companion reading", result.probe.partner_value],
        ["Seam row at column", result.probe.seam_row],
      ]));
    }
  } else {
    items.push(...generalOutput(result));
  }
  if (readonlyKey) {
    const downloads = newNode("div", undefined, "button-group");
    ["csv", "fits", "json"].forEach((format) => {
      const link = newNode("a", format.toUpperCase());
      link.href = framePath(`/saved/${encodeURIComponent(readonlyKey)}/export/${format}`);
      downloads.append(link);
    });
    items.push(downloads);
  }
  items.push(detailsResult(result));
  container.replaceChildren(...items);
}
function generalOutput(result) {
  const scalars = Object.entries(result).filter(([, val]) => val === null || ["number", "boolean", "string"].includes(typeof val));
  const report = [];
  if (scalars.length) report.push(metricGrid(scalars.slice(0, 12).map(([key, value]) => [pretty(key), value])));
  const list = result.sources || result.stars || result.cells || result.marks || result.peaks || result.channels || result.samples;
  if (list?.length && typeof list[0] === "object") {
    const cells = new Set();
    for (const row of list.slice(0, 20)) Object.entries(row).forEach(([key, val]) => {
      if (val === null || ["number", "boolean", "string"].includes(typeof val)) cells.add(key);
    });
    if (cells.size) report.push(dataTable(list, Array.from(cells).slice(0, 10), "x" in list[0] ? sourceRow : null));
  }
  if (result.result_statistics) {
    const v = result.result_statistics;
    report.push(metricGrid([["Output median", v.median], ["Output sigma", v.sigma], ["Finite", v.count], ["Range", `${ff(v.min)} … ${ff(v.max)}`]]));
  }
  return report;
}

async function runTool() {
  const tool = state.tool;
  const route = tools[tool][4] || "measure";
  try {
    const parameters = toolPayload();
    const ticket = ++toolTicket;
    const key = state.frame.key;
    const button = el("#run-tool");
    button.disabled = true;
    el("#retain-output").hidden = true;
    toolStatus("Reading original pixels…");
    if (route === "color") {
      await runColor(parameters, key, ticket);
      return;
    }
    const path = route === "direct" ? `/${tool}` : `/${route}/${tool}`;
    const payload = await api(framePath(path, key), action(parameters));
    if (ticket !== toolTicket) return;
    const result = payload.result;
    state.measured = {operation: tool, parameters, key, route};
    state.output = {operation: tool, result};
    if (tool === "stars") state.catalog = result.sources;
    if (tool === "aperture") state.recentAperture = result;
    el("#retain-output").hidden = route !== "derive";
    toolStatus(route === "derive" ? "Result prepared. Keep it to open a separate image." : "");
    renderResult(tool, result);
    paint();
  } catch (error) {
    toolStatus(error.message, true);
  } finally {
    el("#run-tool").disabled = false;
  }
}
async function runColor(parameters, key, ticket) {
  const [response, data] = await Promise.all([
    fetch(framePath("/color", key), action(parameters)),
    api(framePath("/color", key), action({...parameters, point: null, format: "report"})),
  ]);
  if (!response.ok) throw new Error("Color preview could not be rendered.");
  const image = await response.blob();
  if (ticket !== toolTicket) return;
  resetColor();
  colorUrl = URL.createObjectURL(image);
  state.view = {...parameters.box};
  el("#science-image").src = colorUrl;
  setImageDimensions();
  const link = newNode("a", "Download RGB PNG");
  link.download = "starplate-composite.png";
  link.href = colorUrl;
  state.measured = null;
  state.output = null;
  renderResult("color", data.result);
  el("#result").append(link);
  toolStatus("");
}
async function retainOutput() {
  const selected = state.measured;
  if (!selected || selected.route !== "derive" || selected.key !== state.frame?.key || selected.operation !== state.tool) return;
  const button = el("#retain-output");
  button.disabled = true;
  try {
    const path = framePath(`/derive/${selected.operation}`, selected.key);
    const payload = await api(path, action({...selected.parameters, retain: true}));
    await refreshLibrary();
    await openFrame(payload.frame.key);
    tell("Derived frame opened. The input remains in the library.");
  } catch (error) { toolStatus(error.message, true); }
  finally { button.disabled = false; }
}
function setTool(name) {
  if (!(name in tools)) return;
  ++toolTicket;
  state.tool = name;
  state.measured = null;
  configureTool();
  if (["profile", "seam"].includes(name) && !state.line) selectMode("line");
  setTab("measure");
}
async function loadSaved(key) {
  try {
    const data = await api(framePath(`/saved/${encodeURIComponent(key)}`));
    const item = data.measurement;
    if (item.frame_key !== state.frame?.key) return;
    if (!(item.operation in tools)) throw new Error("This saved tool is no longer registered.");
    ++toolTicket;
    state.tool = item.operation;
    state.measured = {operation: item.operation, parameters: item.parameters, key: item.frame_key, route: "measure"};
    state.output = {operation: item.operation, result: item.result};
    if (item.parameters.box) setBox(item.parameters.box, false);
    if (item.parameters.start && item.parameters.end) state.line = [item.parameters.start, item.parameters.end];
    if (item.parameters.point) selectPoint(item.parameters.point, false);
    configureTool(item.parameters, true);
    renderResult(item.operation, item.result, item.key);
    toolStatus(`Saved measurement · ${item.title}`);
    setTab("measure");
    paint();
  } catch (error) { tell(error.message, true); }
}
function displayPayload() {
  const form = el("#display-form");
  const data = new FormData(form);
  const output = {};
  for (const key of ["mode", "palette", "low_pct", "high_pct", "low", "high", "strength"]) {
    const val = data.get(key);
    if (val !== "") output[key] = key === "mode" || key === "palette" ? val : Number(val);
  }
  output.invert = form.elements.invert.checked;
  return output;
}
async function setDisplay(event) {
  event.preventDefault();
  if (!state.frame) return;
  const stretch = displayPayload();
  const key = state.frame.key;
  try {
    const response = await api(framePath("/measure/display"), action({stretch, box: full(state.frame)}));
    if (key !== state.frame?.key) return;
    state.stretch = stretch;
    refreshPreview();
    const result = response.result;
    el("#display-result").replaceChildren(metricGrid([
      ["Display floor", result.low], ["Display ceiling", result.high],
      ["Low-clipped pixels", result.clipped_low], ["High-clipped pixels", result.clipped_high],
    ]), dataTable(result.anchors, ["display", "original"]));
  } catch (error) { tell(error.message, true); }
}
async function toggleGrid() {
  if (!state.frame) return;
  if (state.showGrid) {
    state.showGrid = false;
    state.skyGrid = null;
    el("#wcs-grid").setAttribute("aria-pressed", "false");
    paint();
    return;
  }
  const key = state.frame.key;
  try {
    const grid = await api(framePath("/grid"));
    if (key !== state.frame?.key) return;
    if (!grid.available) return tell("No usable celestial WCS on this image.");
    state.showGrid = true;
    state.skyGrid = grid;
    el("#wcs-grid").setAttribute("aria-pressed", "true");
    paint();
  } catch (error) { tell(error.message, true); }
}
function exportImage(kind) {
  if (!state.frame) return;
  const query = new URLSearchParams({...state.box, ...state.stretch});
  if (kind === "bundle") query.set("measurements", state.saved.slice(0, 8).map((row) => row.key).join(","));
  const link = newNode("a");
  link.href = framePath(`/export/${kind}?${query}`);
  link.download = "";
  link.click();
}
async function frameReport(type) {
  if (!state.frame) return;
  try {
    if (type === "source-samples" && !state.point) return tell("Choose an original point first.", true);
    const params = type === "fingerprint" ? {box: state.box} : {point: state.point};
    const data = await api(framePath(`/${type}`), action(params));
    const target = el("#frame-result");
    if (type === "fingerprint") target.replaceChildren(...generalOutput(data.result), detailsResult(data.result));
    else if (data.operation === "seam") target.replaceChildren(...seamSourceNodes(data), detailsResult(data));
    else target.replaceChildren(dataTable([data.result, ...data.direct_parent_samples], ["key", "point", "value", "available"]), detailsResult(data));
  } catch (error) { tell(error.message, true); }
}
function seamSourceNodes(data) {
  const adopted = data.adopted_source;
  const switchRow = data.switch_row;
  const nodes = [
    metricGrid([
      ["Adopted frame", adopted.key],
      ["Role", adopted.role],
      ["Original frame pixel", adopted.pixel ? pos(adopted.pixel) : "—"],
      ["Mapped coordinate", adopted.point ? pos(adopted.point) : "—"],
      ["Value actually read", adopted.value],
      ["Sample available", adopted.available],
      ["Interpolation support", adopted.interpolation_support ?? "—"],
      ["Seam switch [column, row]", switchRow ? pos(switchRow) : "—"],
      ["Unit", adopted.unit],
    ]),
  ];
  if (!adopted.available) {
    nodes.push(newNode("p", "The adopted source has no valid sample here, so the mosaic pixel is left empty.", "quiet"));
  }
  const rows = [data.result, ...data.direct_parent_samples];
  nodes.push(dataTable(rows, ["key", "role", "adopted", "point", "pixel", "value", "available"]));
  return nodes;
}
async function upload(verify) {
  const form = el("#upload-form");
  if (!form.reportValidity()) return;
  const input = new FormData(form);
  const feedback = el("#upload-feedback");
  feedback.textContent = verify ? "Reading image planes…" : "Importing original image…";
  try {
    const data = await api(verify ? "/api/inspect-import" : "/api/frames", {method: "POST", body: input});
    if (verify) {
      feedback.textContent = data.candidate_planes.length
        ? data.candidate_planes.map((plane) => `HDU ${plane.hdu} · ${plane.name} · ${plane.width} × ${plane.height}`).join(" | ")
        : `Raster · ${data.width} × ${data.height}`;
      return;
    }
    await refreshLibrary();
    await openFrame(data.frame.key);
    form.reset();
    feedback.textContent = "";
    el("#upload-dialog").close();
    tell(data.created ? "Image imported." : "This image plane was already in the workspace.");
  } catch (error) { feedback.textContent = error.message; }
}
function prepareTools() {
  const selector = el("#tool-select");
  for (const [group, found] of Object.entries(groups)) {
    const heading = newNode("optgroup");
    heading.label = group;
    for (const [key, fields] of Object.entries(found)) {
      const option = newNode("option", fields[0]);
      option.value = key;
      heading.append(option);
    }
    selector.append(heading);
  }
  selector.addEventListener("change", () => setTool(selector.value));
}
function hook() {
  prepareTools();
  configureTool();
  all("[data-tab]").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));
  all("[data-mode]").forEach((button) => button.addEventListener("click", () => selectMode(button.dataset.mode)));
  all("[data-close]").forEach((button) => button.addEventListener("click", () => el(`#${button.dataset.close}`).close()));
  all("[data-export]").forEach((button) => button.addEventListener("click", () => exportImage(button.dataset.export)));
  el("#open-upload").addEventListener("click", () => showDialog("upload-dialog"));
  el("#help-toggle").addEventListener("click", () => showDialog("help-dialog"));
  el("#zoom-in").addEventListener("click", () => changeView(true));
  el("#zoom-out").addEventListener("click", () => changeView(false));
  el("#wcs-grid").addEventListener("click", toggleGrid);
  el("#full-box").addEventListener("click", () => { if (state.frame) setBox(full(state.frame)); });
  el("#clear-overlay").addEventListener("click", () => {
    state.point = state.line = state.output = state.recentAperture = state.measured = null;
    state.catalog = []; state.vertices = [];
    inputNumber("#point-x", null); inputNumber("#point-y", null);
    el("#retain-output").hidden = true;
    el("#result").replaceChildren();
    toolStatus("");
    paint();
  });
  el("#box-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!state.frame) return;
    const form = event.currentTarget;
    const data = Object.fromEntries(coordinateFields.map((name) => [name, form.elements[name].valueAsNumber]));
    if (data.x + data.width > state.frame.width || data.y + data.height > state.frame.height) return tell("The box must stay inside the original detector.", true);
    setBox(data);
  });
  el("#point-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    selectPoint([form.elements.x.valueAsNumber, form.elements.y.valueAsNumber]);
  });
  el("#tool-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runTool();
  });
  el("#tool-fields").addEventListener("input", () => {
    ++toolTicket;
    state.measured = null;
    el("#retain-output").hidden = true;
  });
  el("#retain-output").addEventListener("click", retainOutput);
  el("#display-form").addEventListener("submit", setDisplay);
  el("#trace-source").addEventListener("click", () => frameReport("source-samples"));
  el("#fingerprint").addEventListener("click", () => frameReport("fingerprint"));
  el("#upload-form").addEventListener("submit", (event) => { event.preventDefault(); upload(false); });
  el("#inspect-upload").addEventListener("click", () => upload(true));
  el("#new-note").addEventListener("click", () => {
    if (!state.point) return tell("Choose an original point to mark.", true);
    showDialog("note-dialog");
  });
  el("#note-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    try {
      await api(framePath("/notes"), action({title: data.get("title"), text: data.get("text"), point: state.point}));
      form.reset(); el("#note-dialog").close();
      await loadCollections();
    } catch (error) { tell(error.message, true); }
  });
  el("#save-result").addEventListener("click", () => {
    if (!state.measured || state.measured.route !== "measure") return tell("Run a measurement before saving it.", true);
    el("#save-form").elements.title.value = tools[state.measured.operation][0];
    showDialog("save-dialog");
  });
  el("#save-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const selected = state.measured;
    if (!selected || selected.route !== "measure") return;
    try {
      const {operation, parameters} = selected;
      await api(framePath("/saved"), action({operation, parameters, title: form.elements.title.value}));
      el("#save-dialog").close();
      await loadCollections();
    } catch (error) { tell(error.message, true); }
  });
  const overlay = el("#overlay");
  overlay.addEventListener("pointerdown", pointerStart);
  overlay.addEventListener("pointermove", pointerMove);
  overlay.addEventListener("pointerup", pointerEnd);
  overlay.addEventListener("pointercancel", () => {
    state.drag = state.temporaryBox = state.temporaryLine = null;
    paint();
  });
  el("#science-image").addEventListener("load", setImageDimensions);
  el("#science-image").addEventListener("error", () => tell("Could not render the selected original image.", true));
  new ResizeObserver(setImageDimensions).observe(el("#view-stage"));
}
hook();
refreshLibrary().then(() => {
  const requested = new URLSearchParams(location.search).get("frame");
  return openFrame(state.frames.some((f) => f.key === requested) ? requested : state.frames[0]?.key);
}).catch((error) => tell(error.message, true));
