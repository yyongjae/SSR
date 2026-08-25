"use strict";

const state = {
  manifest: null,
  filtered: [],
  current: 0,
  overlay: null,
  cards: new Map(),
  cursor: null,
};

const $ = id => document.getElementById(id);
const DET_COLORS = ["#ff6b6b", "#ffa94d", "#e599f7", "#ff8787", "#ffd8a8", "#ced4da", "#cc5de8", "#4dabf7", "#f783ac", "#dee2e6"];
const MAP_COLORS = ["#ffd43b", "#22d3ee", "#69db7c"];
const MAP_CLASSES = ["divider", "ped_crossing", "boundary"];
const PLAN_COLORS = ["#fb5607", "#ff9f1c", "#ffcf56"];
const MODE_LABELS = {
  pca: "PCA RGB",
  spatial_contrast: "spatial contrast",
  ego_cosine: "ego-prototype cosine",
  rms: "RMS activation",
  channel_std: "channel std",
  abs_mean: "mean |activation|",
  token_max: "planner token max",
  token_entropy: "per-cell token assignment entropy",
};

function optionValues(select, values) {
  [...new Set(values)].sort().forEach(value => {
    const option = document.createElement("option");
    option.value = value; option.textContent = value; select.appendChild(option);
  });
}

function filterSamples() {
  const query = $("search").value.trim().toLowerCase();
  const location = $("location").value;
  const condition = $("condition").value;
  const command = $("command").value;
  state.filtered = state.manifest.samples.filter(sample => {
    const haystack = `${sample.token} ${sample.scene_name} ${sample.description}`.toLowerCase();
    return (!query || haystack.includes(query)) &&
      (!location || sample.location === location) &&
      (!condition || sample.conditions.includes(condition)) &&
      (!command || sample.command === command);
  });
  state.current = Math.min(state.current, Math.max(0, state.filtered.length - 1));
  const select = $("sample"); select.innerHTML = "";
  state.filtered.forEach((sample, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `${String(index + 1).padStart(2, "0")} · ${sample.scene_name} f${sample.frame_idx} · ${sample.command}`;
    select.appendChild(option);
  });
  showSample();
}

function createCards() {
  const container = $("cards");
  const template = $("cardTemplate");
  state.manifest.sources.forEach(source => {
    const card = template.content.firstElementChild.cloneNode(true);
    card.style.setProperty("--source-color", source.color);
    card.querySelector(".source-name").textContent = source.label;
    const bits = [];
    if (source.epoch != null) bits.push(`epoch ${source.epoch}`);
    if (source.weight_type) bits.push(source.weight_type);
    bits.push(source.kind.replaceAll("_", " "));
    card.querySelector(".source-meta").textContent = bits.join(" · ");
    card.querySelector(".capabilities").innerHTML = source.capabilities.map(x => `<span class="cap">${x}</span>`).join("");
    card.querySelector(".source-warning").textContent = source.warning || "";
    const stage = card.querySelector(".bev-stage");
    stage.addEventListener("pointermove", event => {
      const rect = stage.getBoundingClientRect();
      const x = -15 + (event.clientX - rect.left) / rect.width * 30;
      const y = 30 - (event.clientY - rect.top) / rect.height * 60;
      state.cursor = {x, y};
      $("coordinate").textContent = `x/right ${x.toFixed(2)}m, y/forward ${y.toFixed(2)}m`;
      drawAll();
    });
    stage.addEventListener("pointerleave", () => { state.cursor = null; drawAll(); });
    container.appendChild(card);
    state.cards.set(source.key, {source, card});
  });
}

function imagePath(source, token) {
  const requested = $("mode").value;
  let mode = requested;
  let fallback = false;
  if (!source.modes.includes(mode)) {
    mode = source.modes.includes("spatial_contrast") ? "spatial_contrast" : source.modes[0];
    fallback = true;
  }
  if (mode === "pca") return {path: source.pca_template.replace("{token}", token), mode, fallback};
  const normalization = $("normalization").value;
  return {
    path: source.asset_template.replace("{token}", token).replace("{mode}", mode).replace("{normalization}", normalization),
    mode, fallback,
  };
}

async function showSample() {
  if (!state.filtered.length) {
    state.overlay = null;
    $("position").textContent = "0 / 0";
    $("sceneTitle").textContent = "조건에 맞는 sample이 없습니다.";
    $("description").textContent = "필터를 완화하거나 검색어를 지워주세요.";
    $("tags").replaceChildren();
    $("sampleStats").replaceChildren();
    $("cameras").removeAttribute("src");
    $("cameras").classList.add("empty");
    state.cards.forEach(({card}) => {
      card.classList.add("empty");
      card.querySelector(".bev-image").removeAttribute("src");
    });
    drawAll();
    return;
  }
  const sample = state.filtered[state.current];
  // Never combine a newly selected image with the previous sample's overlay
  // while the small overlay JSON is still in flight.
  state.overlay = null;
  drawAll();
  $("sample").value = state.current;
  $("position").textContent = `${state.current + 1} / ${state.filtered.length}`;
  $("sceneTitle").textContent = `${sample.scene_name} · frame ${sample.frame_idx}/${sample.scene_length - 1}`;
  $("description").textContent = sample.description || "No scene description";
  $("tags").innerHTML = [sample.location, sample.command, ...sample.conditions].map(x => `<span class="tag">${x}</span>`).join("");
  const stats = [
    ["token", sample.token], ["val index", sample.index],
    ["objects in ROI", sample.objects_in_bev], ["moving objects", sample.moving_objects],
    ["pedestrians", sample.pedestrians], ["vehicles", sample.vehicles],
    ["ego 3s distance", `${sample.ego_3s_distance_m.toFixed(2)} m`],
    ["lateral excursion", `${sample.ego_lateral_excursion_m.toFixed(2)} m`],
  ];
  $("sampleStats").innerHTML = stats.map(([k,v]) => `<div class="stat"><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  $("cameras").classList.remove("empty");
  $("cameras").src = sample.camera_mosaic;
  state.cards.forEach(({source, card}) => {
    card.classList.remove("empty");
    const image = imagePath(source, sample.token);
    card.classList.toggle("fallback", image.fallback);
    card.querySelector(".bev-image").src = image.path;
    card.querySelector(".view-label").textContent = `${MODE_LABELS[image.mode] || image.mode.replaceAll("_", " ")} · ${image.mode === "pca" ? "source-specific basis" : $("normalization").value + " contrast"}`;
  });
  const requestToken = sample.token;
  try {
    const response = await fetch(sample.overlay);
    if (!response.ok) throw new Error(`overlay fetch failed: ${response.status}`);
    const overlay = await response.json();
    if (state.filtered[state.current]?.token !== requestToken) return;
    state.overlay = overlay;
    requestAnimationFrame(drawAll);
  } catch (error) {
    if (state.filtered[state.current]?.token !== requestToken) return;
    state.overlay = null;
    drawAll();
    $("description").textContent = `Overlay를 불러오지 못했습니다: ${error.message || error}`;
  }
}

function xy(point, width, height) {
  return [(point[0] + 15) / 30 * width, (30 - point[1]) / 60 * height];
}

function path(ctx, points, width, height, color, lineWidth=1, alpha=1, dash=[]) {
  if (!points || points.length < 2) return;
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.globalAlpha = alpha; ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((point, i) => { const [x,y] = xy(point, width, height); i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); });
  ctx.stroke(); ctx.restore();
}

function polygon(ctx, points, width, height, color, lineWidth=1, alpha=1, dash=[]) {
  if (!points?.length) return;
  path(ctx, [...points, points[0]], width, height, color, lineWidth, alpha, dash);
}

function drawMap(ctx, items, width, height, predicted=false) {
  if (!items) return;
  if (predicted) {
    const threshold = Number($("mapThreshold").value);
    items.points.forEach((points, i) => {
      if (items.scores[i] < threshold) return;
      path(ctx, points, width, height, MAP_COLORS[items.labels[i] % 3], 1.4, .88);
    });
  } else {
    items.forEach(item => {
      const index = Math.max(0, MAP_CLASSES.indexOf(item.label));
      path(ctx, item.points, width, height, MAP_COLORS[index], 1, .36, [4,3]);
    });
  }
}

function drawDetection(ctx, det, width, height) {
  if (!det) return;
  const threshold = Number($("detThreshold").value);
  const keep = det.scores.map((score, i) => ({score, i})).filter(x => x.score >= threshold).sort((a,b) => b.score-a.score);
  keep.forEach(({i, score}) => {
    const color = DET_COLORS[det.labels[i] % DET_COLORS.length];
    polygon(ctx, det.corners[i], width, height, color, 1.2, Math.min(.95, .35 + score));
    if ($("motion").checked && det.trajectories?.[i]) {
      const corners = det.corners[i] || [];
      const centre = corners.length ? [
        corners.reduce((sum, p) => sum + p[0], 0) / corners.length,
        corners.reduce((sum, p) => sum + p[1], 0) / corners.length,
      ] : null;
      det.trajectories[i].slice(0, 6).forEach(traj =>
        path(ctx, centre ? [centre, ...traj] : traj, width, height, color, .7, .28));
    }
  });
}

function drawGroundTruthPlanning(ctx, gt, width, height) {
  if (!gt?.ego_trajectory) return;
  const valid = gt.ego_trajectory.filter((_, i) => gt.ego_mask?.[i] ?? true);
  path(ctx, [[0, 0], ...valid], width, height, "#ffffff", 2.2, .92, [6, 4]);
}

function drawPlanning(ctx, planning, width, height, sourceKey) {
  if (!planning?.trajectories || sourceKey === "para_stage1") return;
  const selected = planning.command_index ?? -1;
  planning.trajectories.forEach((traj, i) => {
    path(ctx, [[0,0], ...traj], width, height, PLAN_COLORS[i % 3], i === selected ? 2.8 : 1.2, i === selected ? .98 : .38);
  });
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  return {width, height};
}

function drawCard(sourceKey, card) {
  const canvas = card.querySelector("canvas");
  const {width, height} = resizeCanvas(canvas);
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, width, height);
  if (!state.overlay) return;
  const gt = state.overlay.ground_truth;
  if ($("gtMap").checked) drawMap(ctx, gt.map, width, height, false);
  if ($("gtBoxes").checked) gt.boxes.forEach(box => polygon(ctx, box, width, height, "#ffffff", 1.1, .75, [4,3]));
  if ($("planning").checked) drawGroundTruthPlanning(ctx, gt, width, height);
  if ($("prediction").checked) {
    const pred = state.overlay.sources[sourceKey] || {};
    drawMap(ctx, pred.map, width, height, true);
    drawDetection(ctx, pred.detection, width, height);
    if ($("planning").checked) drawPlanning(ctx, pred.planning, width, height, sourceKey);
  }
  const [ox, oy] = xy([0,0], width, height);
  ctx.save(); ctx.fillStyle = "#fff"; ctx.strokeStyle = "#07101b"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(ox, oy-7); ctx.lineTo(ox-5, oy+6); ctx.lineTo(ox+5, oy+6); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
  if (state.cursor) {
    const [cx, cy] = xy([state.cursor.x, state.cursor.y], width, height);
    ctx.save(); ctx.strokeStyle = "rgba(255,255,255,.65)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,height); ctx.moveTo(0,cy); ctx.lineTo(width,cy); ctx.stroke(); ctx.restore();
  }
}

function drawAll() { state.cards.forEach(({card}, key) => drawCard(key, card)); }

function move(delta) {
  if (!state.filtered.length) return;
  state.current = (state.current + delta + state.filtered.length) % state.filtered.length;
  showSample();
}

async function init() {
  try {
    const response = await fetch("data/manifest.json");
    if (!response.ok) throw new Error(`manifest fetch failed: ${response.status}`);
    state.manifest = await response.json();
    state.filtered = [...state.manifest.samples];
    $("warning").textContent = `${state.manifest.qualitative_warning} ${state.manifest.feature_warning}`;
    optionValues($("location"), state.manifest.samples.map(x => x.location));
    optionValues($("condition"), state.manifest.samples.flatMap(x => x.conditions));
    optionValues($("command"), state.manifest.samples.map(x => x.command));
    $("mode").value = state.manifest.defaults.mode;
    $("normalization").value = state.manifest.defaults.normalization;
    $("detThreshold").value = state.manifest.defaults.det_threshold;
    $("mapThreshold").value = state.manifest.defaults.map_threshold;
    $("detValue").textContent = Number(state.manifest.defaults.det_threshold).toFixed(2);
    $("mapValue").textContent = Number(state.manifest.defaults.map_threshold).toFixed(2);
    createCards(); filterSamples();
    ["search", "location", "condition", "command"].forEach(id => $(id).addEventListener("input", filterSamples));
    $("sample").addEventListener("change", () => { state.current = Number($("sample").value); showSample(); });
    ["mode", "normalization"].forEach(id => $(id).addEventListener("change", showSample));
    ["gtMap", "gtBoxes", "prediction", "planning", "motion"].forEach(id => $(id).addEventListener("change", drawAll));
    ["detThreshold", "mapThreshold"].forEach(id => $(id).addEventListener("input", () => {
      $(id === "detThreshold" ? "detValue" : "mapValue").textContent = Number($(id).value).toFixed(2); drawAll();
    }));
    $("prev").addEventListener("click", () => move(-1)); $("next").addEventListener("click", () => move(1));
    window.addEventListener("resize", drawAll);
    window.addEventListener("keydown", event => {
      if (event.target.matches?.("input, select, textarea") || event.target.isContentEditable) return;
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
    });
  } catch (error) {
    document.body.innerHTML = `<div class="error"><h1>사이트를 불러오지 못했습니다.</h1><p>${error.stack || error}</p><p>file:// 대신 README의 http.server 명령으로 여세요.</p></div>`;
  }
}

init();
