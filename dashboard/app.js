// VayuDrishti: Indo-Gangetic PM2.5 Spatiotemporal Smog Digital Twin Client Controller
let globalData = null;
let activeStep = 6;
let isPlaying = false;
let playInterval = null;
let selectedStationId = 0; // Delhi - Anand Vihar default
let stubbleActive = true;
let windDirectionMode = "nw"; // 'nw' or 'easterly'

// Canvas & Particle Simulation State
let canvas, ctx;
let width = 0, height = 0;
let smokeParticles = [];
let animFrameId = null;
let hoveredStation = null;

// Initialize on window load
window.addEventListener("DOMContentLoaded", async () => {
  canvas = document.getElementById("airQualityCanvas");
  ctx = canvas.getContext("2d");
  resizeCanvas();
  window.addEventListener("resize", () => {
    resizeCanvas();
    updateUI();
  });

  await loadData();
  setupEventListeners();
  renderBenchmarkTable();
  initSmokeParticles();
  startAnimationLoop();
});

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  width = rect.width;
  height = rect.height;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
}

async function loadData() {
  try {
    const res = await fetch("data.json");
    globalData = await res.json();

    // Default select Anand Vihar (Station 0)
    selectedStationId = 0;
    updateUI();
  } catch (err) {
    console.error("Failed to load data.json:", err);
  }
}

function getAQICategory(pm25) {
  if (pm25 <= 30) return { cat: "GOOD", color: "#10b981", class: "aqi-good" };
  if (pm25 <= 60) return { cat: "SATISFACTORY", color: "#84cc16", class: "aqi-sat" };
  if (pm25 <= 90) return { cat: "MODERATE", color: "#f59e0b", class: "aqi-mod" };
  if (pm25 <= 120) return { cat: "POOR", color: "#f97316", class: "aqi-poor" };
  if (pm25 <= 250) return { cat: "VERY POOR", color: "#ef4444", class: "aqi-vpoor" };
  return { cat: "SEVERE", color: "#9333ea", class: "aqi-severe" };
}

function setupEventListeners() {
  const timeSlider = document.getElementById("timeSlider");
  const playBtn = document.getElementById("playBtn");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const counterfactualBtn = document.getElementById("counterfactualBtn");
  const windSelect = document.getElementById("windSelect");

  timeSlider.addEventListener("input", (e) => {
    activeStep = parseInt(e.target.value, 10);
    updateUI();
  });

  playBtn.addEventListener("click", () => {
    if (isPlaying) {
      clearInterval(playInterval);
      isPlaying = false;
      playBtn.innerText = "▶ Play Simulation";
      playBtn.classList.remove("btn-playing");
    } else {
      isPlaying = true;
      playBtn.innerText = "⏸ Pause Simulation";
      playBtn.classList.add("btn-playing");
      playInterval = setInterval(() => {
        activeStep = (activeStep + 1) % globalData.timeline.length;
        timeSlider.value = activeStep;
        updateUI();
      }, 1400);
    }
  });

  prevBtn.addEventListener("click", () => {
    if (activeStep > 0) {
      activeStep--;
      timeSlider.value = activeStep;
      updateUI();
    }
  });

  nextBtn.addEventListener("click", () => {
    if (activeStep < globalData.timeline.length - 1) {
      activeStep++;
      timeSlider.value = activeStep;
      updateUI();
    }
  });

  counterfactualBtn.addEventListener("click", () => {
    stubbleActive = !stubbleActive;
    if (stubbleActive) {
      counterfactualBtn.innerText = "🔥 Stubble Burning: ACTIVE";
      counterfactualBtn.classList.remove("btn-toggle-inactive");
      counterfactualBtn.classList.add("btn-toggle-active");
    } else {
      counterfactualBtn.innerText = "🌱 Counterfactual: ZERO Farm Fires";
      counterfactualBtn.classList.remove("btn-toggle-active");
      counterfactualBtn.classList.add("btn-toggle-inactive");
    }
    updateUI();
  });

  windSelect.addEventListener("change", (e) => {
    windDirectionMode = e.target.value;
    updateUI();
  });

  // Canvas Mouse Click & Hover
  canvas.addEventListener("click", handleCanvasClick);
  canvas.addEventListener("mousemove", handleCanvasHover);
}

function getCanvasCoords(node) {
  // Normalize scale and center
  const scale = Math.min(width / 720, height / 520) * 1.15;
  const centerX = width * 0.48;
  const centerY = height * 0.52;
  return {
    x: centerX + node.canvas_x * scale,
    y: centerY + node.canvas_y * scale
  };
}

function handleCanvasClick(e) {
  if (!globalData) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  let clickedStation = null;
  let minDist = 24;

  globalData.stations.forEach(station => {
    const pt = getCanvasCoords(station);
    const dist = Math.hypot(mouseX - pt.x, mouseY - pt.y);
    if (dist < minDist) {
      minDist = dist;
      clickedStation = station;
    }
  });

  if (clickedStation) {
    selectedStationId = clickedStation.id;
    updateUI();
  }
}

function handleCanvasHover(e) {
  if (!globalData) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  let hovered = null;
  globalData.stations.forEach(station => {
    const pt = getCanvasCoords(station);
    const dist = Math.hypot(mouseX - pt.x, mouseY - pt.y);
    if (dist < 18) {
      hovered = station;
    }
  });

  hoveredStation = hovered;
  canvas.style.cursor = hovered ? "pointer" : "default";
}

function initSmokeParticles() {
  smokeParticles = [];
  const numParticles = 120;
  for (let i = 0; i < numParticles; i++) {
    smokeParticles.push(createSmokeParticle(true));
  }
}

function createSmokeParticle(randomProgress = false) {
  // Emit from Punjab/Haryana fire clusters
  const clusters = [
    { x: -180, y: -140 },
    { x: -220, y: -170 },
    { x: -150, y: -120 },
    { x: -140, y: -50 }
  ];
  const cluster = clusters[Math.floor(Math.random() * clusters.length)];

  // Scale to screen
  const scale = Math.min(width / 720, height / 520) * 1.15;
  const startX = width * 0.48 + (cluster.x + (Math.random() - 0.5) * 40) * scale;
  const startY = height * 0.52 + (cluster.y + (Math.random() - 0.5) * 40) * scale;

  let progress = randomProgress ? Math.random() : 0.0;
  let speed = 0.003 + Math.random() * 0.004;

  return {
    progress: progress,
    speed: speed,
    originX: startX,
    originY: startY,
    size: 2.5 + Math.random() * 5.0,
    opacity: 0.2 + Math.random() * 0.6,
    offsetWobble: Math.random() * Math.PI * 2
  };
}

function updateUI() {
  if (!globalData) return;

  const currentTimeline = globalData.timeline[activeStep];
  const activeDateBadge = document.getElementById("activeDateBadge");
  activeDateBadge.innerText = currentTimeline.date_label;

  // Calculate Delhi NCR PM2.5 with Stubble Counterfactual
  let rawDelhiPm25 = currentTimeline.regional_delhi_ncr_pm25;
  let meanStubblePct = 0;
  let delhiCount = 0;

  currentTimeline.station_data.forEach(sData => {
    const st = globalData.stations[sData.station_id];
    if (st.is_delhi_ncr) {
      meanStubblePct += sData.stubble_attribution_pct;
      delhiCount++;
    }
  });
  meanStubblePct = delhiCount > 0 ? (meanStubblePct / delhiCount) : 40.0;

  let effectiveDelhiPm25 = rawDelhiPm25;
  if (!stubbleActive) {
    // Under zero stubble burning, remove agricultural contribution
    effectiveDelhiPm25 = rawDelhiPm25 * (1.0 - (meanStubblePct / 100.0) * 0.88);
  }

  // Update HUD Metrics
  const mDelhi = document.getElementById("m-delhi-pm25");
  const mAqi = document.getElementById("m-aqi-tag");
  const mFires = document.getElementById("m-fires");
  const mStubble = document.getElementById("m-stubble-share");
  const mCoverage = document.getElementById("m-coverage");

  mDelhi.innerText = `${effectiveDelhiPm25.toFixed(1)} µg/m³`;

  const delhiAqiCat = getAQICategory(effectiveDelhiPm25);
  mAqi.innerText = `▲ CPCB Category: ${delhiAqiCat.cat}`;
  mAqi.className = `metric-delta ${delhiAqiCat.cat === "SEVERE" ? "delta-danger" : (delhiAqiCat.cat === "VERY POOR" ? "delta-danger" : "delta-good")}`;

  if (stubbleActive) {
    mFires.innerText = `${currentTimeline.total_satellite_fires.toLocaleString()} Fires`;
    mStubble.innerText = `${meanStubblePct.toFixed(1)}%`;
  } else {
    mFires.innerText = `0 Fires (Zero-Burn Policy)`;
    mStubble.innerText = `0.0% (Suppressed)`;
  }

  mCoverage.innerText = `${globalData.metadata.metrics.picp_coverage_pct_24h.toFixed(1)}%`;

  // Update Inspector Card
  const selStation = globalData.stations[selectedStationId];
  const selStationData = currentTimeline.station_data.find(s => s.station_id === selectedStationId) || currentTimeline.station_data[0];

  document.getElementById("inspectStationName").innerText = selStation.name;
  document.getElementById("inspectLocation").innerText = `State: ${selStation.state} | Lat: ${selStation.lat.toFixed(2)}°N, Lon: ${selStation.lon.toFixed(2)}°E`;

  let stationPm25 = selStationData.current_observed_pm25;
  let stationStubblePct = selStationData.stubble_attribution_pct;
  let stationUrbanPct = selStationData.urban_baseline_pct;

  if (!stubbleActive) {
    stationPm25 = stationPm25 * (1.0 - (stationStubblePct / 100.0) * 0.88);
    stationStubblePct = 0;
    stationUrbanPct = 100;
  }

  const stAqi = getAQICategory(stationPm25);
  const inspectAqiBadge = document.getElementById("inspectAqiBadge");
  inspectAqiBadge.innerText = stAqi.cat;
  inspectAqiBadge.className = `aqi-badge ${stAqi.class}`;

  // Multi-horizon quantiles
  const h24Median = stubbleActive ? selStationData.forecast_24h.median : selStationData.forecast_24h.median * (1 - selStationData.stubble_attribution_pct/100 * 0.88);
  const h24Q10 = stubbleActive ? selStationData.forecast_24h.q10 : selStationData.forecast_24h.q10 * (1 - selStationData.stubble_attribution_pct/100 * 0.88);
  const h24Q90 = stubbleActive ? selStationData.forecast_24h.q90 : selStationData.forecast_24h.q90 * (1 - selStationData.stubble_attribution_pct/100 * 0.88);

  const h48Median = stubbleActive ? selStationData.forecast_48h.median : selStationData.forecast_48h.median * (1 - selStationData.stubble_attribution_pct/100 * 0.88);
  const h48Q10 = stubbleActive ? selStationData.forecast_48h.q10 : selStationData.forecast_48h.q10 * (1 - selStationData.stubble_attribution_pct/100 * 0.88);
  const h48Q90 = stubbleActive ? selStationData.forecast_48h.q90 : selStationData.forecast_48h.q90 * (1 - selStationData.stubble_attribution_pct/100 * 0.88);

  const h72Median = stubbleActive ? selStationData.forecast_72h.median : selStationData.forecast_72h.median * (1 - selStationData.stubble_attribution_pct/100 * 0.88);
  const h72Q10 = stubbleActive ? selStationData.forecast_72h.q10 : selStationData.forecast_72h.q10 * (1 - selStationData.stubble_attribution_pct/100 * 0.88);
  const h72Q90 = stubbleActive ? selStationData.forecast_72h.q90 : selStationData.forecast_72h.q90 * (1 - selStationData.stubble_attribution_pct/100 * 0.88);

  document.getElementById("h24-val").innerText = h24Median.toFixed(1);
  document.getElementById("h24-band").innerText = `[${h24Q10.toFixed(0)} - ${h24Q90.toFixed(0)}]`;

  document.getElementById("h48-val").innerText = h48Median.toFixed(1);
  document.getElementById("h48-band").innerText = `[${h48Q10.toFixed(0)} - ${h48Q90.toFixed(0)}]`;

  document.getElementById("h72-val").innerText = h72Median.toFixed(1);
  document.getElementById("h72-band").innerText = `[${h72Q10.toFixed(0)} - ${h72Q90.toFixed(0)}]`;

  // Attribution Bar
  document.getElementById("attrRatioText").innerText = `${stationStubblePct.toFixed(0)}% Stubble / ${stationUrbanPct.toFixed(0)}% Urban`;
  document.getElementById("attrStubbleBar").style.width = `${stationStubblePct}%`;
  document.getElementById("attrUrbanBar").style.width = `${stationUrbanPct}%`;

  // Narrative explanation
  const inspectNarrative = document.getElementById("inspectNarrative");
  if (stubbleActive) {
    if (stationPm25 > 250) {
      inspectNarrative.className = "narrative-box alert";
      inspectNarrative.innerHTML = `<strong>Severe Atmospheric Inversion Alert:</strong> Stubble burning plume transported via ${windDirectionMode === 'nw' ? 'northwesterly corridor winds' : 'diffuse winds'} accounts for <strong>${stationStubblePct.toFixed(1)}%</strong> of local PM2.5 loading at ${selStation.name}. Emergency Graded Response Action Plan (GRAP Stage IV) threshold exceeded.`;
    } else {
      inspectNarrative.className = "narrative-box";
      inspectNarrative.innerHTML = `<strong>Regional Transboundary Influx:</strong> Station ${selStation.name} exhibits moderate atmospheric loading with <strong>${stationStubblePct.toFixed(1)}%</strong> attributable to upstream biomass combustion. Boundary layer compression is active.`;
    }
  } else {
    inspectNarrative.className = "narrative-box";
    inspectNarrative.innerHTML = `<strong>Counterfactual Impact Demonstration:</strong> Eliminating agricultural stubble burning reduces PM2.5 at ${selStation.name} from baseline down to <strong>${stationPm25.toFixed(1)} µg/m³</strong> (a ${selStationData.stubble_attribution_pct.toFixed(0)}% direct reduction), bringing air quality from Severe down to ${stAqi.cat}.`;
  }

  // Wind field text
  document.getElementById("windFieldText").innerText = `Wind Speed: ${currentTimeline.wind_speed_mps} m/s | Corridor: ${windDirectionMode === 'nw' ? '315° (North-West Inversion Influx)' : '090° (Easterly Dispersion)'} | Inversion: High`;
}

function renderBenchmarkTable() {
  if (!globalData || !globalData.benchmarks) return;
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";

  globalData.benchmarks.forEach(item => {
    const tr = document.createElement("tr");
    const isOurModel = item.model.includes("VayuDrishti");
    if (isOurModel) {
      tr.className = "highlight-row";
    }

    tr.innerHTML = `
      <td>${isOurModel ? "⭐ " : ""}${item.model}</td>
      <td><strong>${item.mae_24h}</strong></td>
      <td><strong>${item.mae_72h}</strong></td>
      <td><span style="color: ${isOurModel ? '#10b981' : '#94a3b8'};">${item.picp_coverage}</span></td>
      <td>${item.stubble_attribution}</td>
    `;
    tbody.appendChild(tr);
  });
}

function startAnimationLoop() {
  function loop() {
    drawCanvas();
    animFrameId = requestAnimationFrame(loop);
  }
  loop();
}

function drawCanvas() {
  ctx.clearRect(0, 0, width, height);

  // 1. Draw Atmospheric Basin Background Grid & Relief
  drawBasinBackground();

  // 2. Draw Spatial Network Connectivity Edges
  drawStationConnections();

  // 3. Draw Stubble Fire Clusters (Punjab & Haryana Belt)
  drawFireClusters();

  // 4. Update & Draw Wind Smoke Transport Particles
  drawSmokeFlow();

  // 5. Draw CPCB Stations & Measurements
  drawStations();
}

function drawBasinBackground() {
  // Indo-Gangetic Basin boundary subtle gradient
  const grad = ctx.createLinearGradient(0, 0, width, height);
  grad.addColorStop(0, "rgba(15, 23, 42, 0.4)");
  grad.addColorStop(0.5, "rgba(8, 14, 26, 0.6)");
  grad.addColorStop(1, "rgba(23, 15, 30, 0.4)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // Subtle coordinate grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
  ctx.lineWidth = 1;
  const step = 60;
  for (let x = 0; x < width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Himalayan Foothills Barrier Arc (North-East natural boundary)
  ctx.save();
  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.setLineDash([6, 6]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(width * 0.1, height * 0.08);
  ctx.bezierCurveTo(width * 0.4, height * 0.04, width * 0.7, height * 0.12, width * 0.95, height * 0.35);
  ctx.stroke();

  ctx.fillStyle = "rgba(148, 163, 184, 0.25)";
  ctx.font = "10px Inter, sans-serif";
  ctx.fillText("▲ Himalayan Mountain Barrier (Thermal Inversion Trapping Ridge)", width * 0.32, height * 0.06);

  ctx.fillStyle = "rgba(245, 158, 11, 0.35)";
  ctx.fillText("🌾 Punjab & Haryana Upstream Agricultural Belt", width * 0.08, height * 0.22);

  ctx.fillStyle = "rgba(6, 182, 212, 0.35)";
  ctx.fillText("🏙️ Delhi-NCR Inversion Basin", width * 0.42, height * 0.65);
  ctx.restore();
}

function drawStationConnections() {
  if (!globalData || !globalData.stations) return;

  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.12)";
  ctx.lineWidth = 1.2;

  // Draw spatial graph edges between proximal stations
  for (let i = 0; i < globalData.stations.length; i++) {
    const s1 = globalData.stations[i];
    const pt1 = getCanvasCoords(s1);

    for (let j = i + 1; j < globalData.stations.length; j++) {
      const s2 = globalData.stations[j];
      const pt2 = getCanvasCoords(s2);
      const dist = Math.hypot(pt1.x - pt2.x, pt1.y - pt2.y);

      // Connect if within spatial correlation radius
      if (dist < 110) {
        ctx.beginPath();
        ctx.moveTo(pt1.x, pt1.y);
        ctx.lineTo(pt2.x, pt2.y);
        ctx.stroke();
      }
    }
  }
  ctx.restore();
}

function drawFireClusters() {
  if (!globalData || !globalData.fire_clusters) return;

  const timeSec = Date.now() * 0.003;

  globalData.fire_clusters.forEach(cluster => {
    const scale = Math.min(width / 720, height / 520) * 1.15;
    const cx = width * 0.48 + cluster.canvas_x * scale;
    const cy = height * 0.52 + cluster.canvas_y * scale;

    ctx.save();
    if (stubbleActive) {
      const pulse = 1.0 + 0.2 * Math.sin(timeSec + cluster.id);
      const radius = 10 * pulse;

      // Radiant glow
      const fireGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, radius * 2.2);
      fireGrad.addColorStop(0, "rgba(239, 68, 68, 0.8)");
      fireGrad.addColorStop(0.5, "rgba(249, 115, 22, 0.4)");
      fireGrad.addColorStop(1, "rgba(239, 68, 68, 0)");
      ctx.fillStyle = fireGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 2.2, 0, Math.PI * 2);
      ctx.fill();

      // Core fire marker
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = "#fed7aa";
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.fillText(`🔥 ${cluster.name} (${cluster.active_fires})`, cx + 12, cy + 3);
    } else {
      // Extinguished state
      ctx.fillStyle = "#10b981";
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#6ee7b7";
      ctx.font = "9px Inter, sans-serif";
      ctx.fillText(`🌱 ${cluster.name} (Zero Fires)`, cx + 8, cy + 3);
    }
    ctx.restore();
  });
}

function drawSmokeFlow() {
  if (!stubbleActive) return; // Suppress smoke particles when zero fires

  ctx.save();

  // Vector direction based on windMode
  let driftDx = windDirectionMode === "nw" ? width * 0.48 : -width * 0.25;
  let driftDy = windDirectionMode === "nw" ? height * 0.45 : height * 0.15;

  smokeParticles.forEach(p => {
    p.progress += p.speed;
    if (p.progress >= 1.0) {
      // Respawn
      const reset = createSmokeParticle(false);
      p.progress = 0.0;
      p.originX = reset.originX;
      p.originY = reset.originY;
      p.size = reset.size;
    }

    // Curvature advection path into the Delhi corridor
    const t = p.progress;
    const wobble = Math.sin(t * 8 + p.offsetWobble) * 12;
    const currentX = p.originX + driftDx * t + wobble;
    const currentY = p.originY + driftDy * t;

    const currentAlpha = Math.sin(t * Math.PI) * p.opacity * 0.45;
    const currentRadius = p.size * (1 + t * 2.5);

    const smokeGrad = ctx.createRadialGradient(currentX, currentY, 0, currentX, currentY, currentRadius);
    smokeGrad.addColorStop(0, `rgba(249, 115, 22, ${currentAlpha})`);
    smokeGrad.addColorStop(0.6, `rgba(239, 68, 68, ${currentAlpha * 0.5})`);
    smokeGrad.addColorStop(1, "rgba(239, 68, 68, 0)");

    ctx.fillStyle = smokeGrad;
    ctx.beginPath();
    ctx.arc(currentX, currentY, currentRadius, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

function drawStations() {
  if (!globalData || !globalData.stations) return;

  const currentTimeline = globalData.timeline[activeStep];
  const timeSec = Date.now() * 0.003;

  globalData.stations.forEach(station => {
    const pt = getCanvasCoords(station);
    const sData = currentTimeline.station_data.find(s => s.station_id === station.id) || currentTimeline.station_data[0];

    let pm25 = sData.current_observed_pm25;
    if (!stubbleActive) {
      pm25 = pm25 * (1.0 - (sData.stubble_attribution_pct / 100.0) * 0.88);
    }

    const aqi = getAQICategory(pm25);
    const isSelected = (station.id === selectedStationId);
    const isHovered = (hoveredStation && hoveredStation.id === station.id);

    ctx.save();

    // Pulse halo for selected or severe station
    if (isSelected || (aqi.cat === "SEVERE" && stubbleActive)) {
      const pulse = 1.0 + 0.25 * Math.sin(timeSec * 2 + station.id);
      const ringRadius = 14 * pulse;

      ctx.strokeStyle = isSelected ? "rgba(245, 158, 11, 0.8)" : "rgba(147, 51, 234, 0.6)";
      ctx.lineWidth = isSelected ? 2 : 1.2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, ringRadius, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Outer Station Circle
    ctx.fillStyle = aqi.color;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isSelected ? 9 : 7, 0, Math.PI * 2);
    ctx.fill();

    // Inner White Pin
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isSelected ? 3.5 : 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Station Name & PM2.5 Label
    ctx.fillStyle = isSelected ? "#ffffff" : "#cbd5e1";
    ctx.font = isSelected ? "bold 11px Inter, sans-serif" : "10px Inter, sans-serif";
    const labelX = pt.x + 12;
    const labelY = pt.y + 3;

    // Short name
    const shortName = station.name.replace("Delhi - ", "").replace(" - Sector 62", "").replace(" - Vasundhara", "").replace(" - Vikas Sadan", "").replace(" - Sector 16A", "");
    ctx.fillText(`${shortName}: ${pm25.toFixed(0)}`, labelX, labelY);

    if (isSelected) {
      // Glow pointer under label
      ctx.fillStyle = "rgba(245, 158, 11, 0.3)";
      ctx.fillRect(labelX - 2, labelY - 11, ctx.measureText(`${shortName}: ${pm25.toFixed(0)}`).width + 4, 14);
    }

    ctx.restore();
  });
}

