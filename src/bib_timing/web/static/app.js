(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    videoId: null,
    width: 0,
    height: 0,
    points: [],
    zonePoints: [],
    calibMode: "line", // line | zone
    image: null,
    pollTimer: null,
    lastLiveSeq: -1,
    currentJobId: null,
    reviewJobId: null,
    reviewEvents: [],
    reviewIndex: 0,
    reviewLabels: {},
    mode: "ops",
    trainRuntimeId: null,
    trainSessionId: null,
    trainKind: "detect",
    trainPoll: null,
    currentCandidate: null,
    models: [],
    modelTrainPoll: null,
    trainTargetModelId: null,
    trainDetectModelId: null,
    trainModalMode: null, // refine | create
    autoTrainAfterReview: false,
  };

  const canvas = $("stage");
  const ctx = canvas.getContext("2d");

  async function init() {
    // Enlazar UI primero: si se espera a /api/device o modelos, los tabs/video
    // no responden al primer clic / primera selección.
    bind();
    bindTrain();
    bindBibs();
    $("opsRecognizeModel")?.addEventListener("change", onOpsRecognizeChange);
    try {
      await refreshDevice();
    } catch (err) {
      console.warn(err);
    }
    try {
      await refreshVideos();
    } catch (err) {
      console.warn(err);
      if ($("videoMeta")) {
        $("videoMeta").textContent = `No se pudo listar videos: ${err}`;
      }
    }
    try {
      await refreshModels();
    } catch (err) {
      console.warn(err);
    }
  }

  function onOpsRecognizeChange() {
    const id = $("opsRecognizeModel")?.value;
    if (!id) return;
    const m = state.models.find((x) => x.id === id);
    const linked = m?.metrics?.detect_model_id;
    if (linked && $("opsDetectModel") && !$("opsDetectModel").value) {
      const has = [...$("opsDetectModel").options].some((o) => o.value === linked);
      if (has) $("opsDetectModel").value = linked;
    }
  }

  function bind() {
    const drop = $("drop");
    const file = $("file");

    // El <label> ya abre el input nativamente; no llamar file.click()
    // (evita doble diálogo / change que no dispara).
    drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      drop.classList.add("drag");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", async (e) => {
      e.preventDefault();
      drop.classList.remove("drag");
      const f = e.dataTransfer.files?.[0];
      if (f) await upload(f);
    });
    file.addEventListener("change", async () => {
      const f = file.files?.[0];
      if (!f) {
        $("videoMeta").textContent = "No se recibió ningún archivo";
        return;
      }
      try {
        await upload(f);
      } catch (err) {
        $("videoMeta").textContent = `Error al subir: ${err}`;
      } finally {
        // Permite volver a elegir el mismo archivo
        file.value = "";
      }
    });

    $("refreshVideos").addEventListener("click", () => {
      refreshVideos().catch((err) => {
        if ($("videoMeta")) $("videoMeta").textContent = String(err);
      });
    });
    $("videoSelect").addEventListener("change", async (e) => {
      const id = e.target.value;
      if (!id) return;
      try {
        await selectVideo(id);
      } catch (err) {
        setVideoLoading(false);
        if ($("videoMeta")) {
          $("videoMeta").textContent = `Error al cargar video: ${err}`;
        }
      }
    });
    $("loadFrame").addEventListener("click", loadFrame);
    $("resetLine").addEventListener("click", () => {
      state.points = [];
      draw();
      updateLineOut();
      updateRunEnabled();
    });
    $("defaultLine").addEventListener("click", () => {
      if (!state.width) return;
      const y = state.height * 0.55;
      state.points = [
        { x: state.width * 0.05, y },
        { x: state.width * 0.95, y },
      ];
      draw();
      updateLineOut();
      updateRunEnabled();
    });
    $("resetZone")?.addEventListener("click", () => {
      state.zonePoints = [];
      draw();
      updateZoneOut();
      updateRunEnabled();
    });
    $("defaultZone")?.addEventListener("click", () => {
      if (!state.width) return;
      state.zonePoints = [
        { x: state.width * 0.05, y: state.height * 0.05 },
        { x: state.width * 0.95, y: state.height * 0.95 },
      ];
      draw();
      updateZoneOut();
      updateRunEnabled();
    });
    $("calibModeLine")?.addEventListener("click", () => setCalibMode("line"));
    $("calibModeZone")?.addEventListener("click", () => setCalibMode("zone"));
    canvas.addEventListener("click", onCanvasClick);
    $("run").addEventListener("click", startProcess);
    $("stop").addEventListener("click", stopProcess);
    $("backToCalib")?.addEventListener("click", () => showOpsPanel("calib"));
    $("startReview")?.addEventListener("click", startOpsReview);
    $("skipReview")?.addEventListener("click", () => {
      $("reviewOffer")?.classList.add("hidden");
    });
    $("opsReviewOk")?.addEventListener("click", () => labelOpsReview("correct"));
    $("opsReviewBad")?.addEventListener("click", () => labelOpsReview("incorrect"));
    $("opsReviewSkip")?.addEventListener("click", () => labelOpsReview("skip"));
    $("opsReviewPrev")?.addEventListener("click", () => {
      if (state.reviewIndex > 0) {
        state.reviewIndex -= 1;
        showOpsReviewItem();
      }
    });

    $("tabOps").addEventListener("click", () => setMode("ops"));
    $("tabTrain").addEventListener("click", () => setMode("train"));
    $("tabBibs").addEventListener("click", () => setMode("bibs"));
  }

  function setCalibMode(mode) {
    state.calibMode = mode;
    $("calibModeLine")?.classList.toggle("active", mode === "line");
    $("calibModeZone")?.classList.toggle("active", mode === "zone");
    if ($("calibHint")) {
      $("calibHint").textContent =
        mode === "line"
          ? "Hacé clic en dos puntos para definir la línea de meta."
          : "Hacé clic en dos esquinas opuestas del área donde se leerán dorsales.";
    }
  }

  function showOpsPanel(which) {
    $("opsCalibPanel")?.classList.toggle("hidden", which !== "calib");
    $("opsLivePanel")?.classList.toggle("hidden", which !== "live");
    $("opsReviewPanel")?.classList.toggle("hidden", which !== "review");
    $("backToCalib")?.classList.toggle("hidden", which === "calib");
    if ($("opsProcessTitle")) {
      $("opsProcessTitle").textContent =
        which === "review" ? "Modelos / refinar" : "3. Procesar";
    }
  }

  function updateRunEnabled() {
    const ok =
      state.points.length === 2 &&
      state.zonePoints.length === 2 &&
      Boolean(state.videoId);
    if ($("run")) $("run").disabled = !ok;
  }

  function setMode(mode) {
    state.mode = mode;
    $("tabOps")?.classList.toggle("active", mode === "ops");
    $("tabTrain")?.classList.toggle("active", mode === "train");
    $("tabBibs")?.classList.toggle("active", mode === "bibs");
    $("modeOps")?.classList.toggle("hidden", mode !== "ops");
    $("modeTrain")?.classList.toggle("hidden", mode !== "train");
    $("modeBibs")?.classList.toggle("hidden", mode !== "bibs");
    const titles = {
      ops: "Línea de meta",
      train: "Entrenamiento",
      bibs: "Generación de dorsales",
    };
    if ($("pageTitle")) {
      $("pageTitle").textContent = titles[mode] || "AI Checkpoint";
    }
    if (mode === "train") {
      refreshVideos().then(() => syncTrainVideoFromOps());
      refreshModels().catch((err) => console.warn(err));
    }
    if (mode === "bibs") {
      refreshBibPreview().catch((err) => console.warn(err));
    }
  }

  function setVideoLoading(on, msg) {
    const el = $("videoLoadSpinner");
    if (!el) return;
    el.classList.toggle("hidden", !on);
    if (msg && $("videoLoadMsg")) $("videoLoadMsg").textContent = msg;
    if ($("videoSelect")) $("videoSelect").disabled = on;
    // No deshabilitar #file: rompe el diálogo nativo / el change del label.
    if ($("refreshVideos")) $("refreshVideos").disabled = on;
    if ($("drop")) $("drop").classList.toggle("loading", on);
  }

  function syncTrainVideoFromOps() {
    const opsId = state.videoId || $("videoSelect").value;
    const trainSel = $("trainVideoSelect");
    if (!trainSel || !opsId) return;
    const has = [...trainSel.options].some((o) => o.value === opsId);
    if (!has) return;
    trainSel.value = opsId;
    if ($("trainStart")) $("trainStart").disabled = !opsId;
  }

  async function refreshDevice() {
    try {
      const d = await fetchJSON("/api/device");
      const el = $("device");
      el.textContent = `${d.backend} · ${d.name} · ${d.device}`;
      el.classList.toggle("ok", d.backend === "rocm" || d.backend === "cuda");
    } catch {
      $("device").textContent = "Dispositivo no disponible";
    }
  }

  async function refreshVideos() {
    const data = await fetchJSON("/api/videos");
    const preferred =
      state.videoId || $("videoSelect")?.value || $("trainVideoSelect")?.value || "";
    for (const selId of ["videoSelect", "trainVideoSelect"]) {
      const sel = $(selId);
      if (!sel) continue;
      const current = sel.value || preferred;
      sel.innerHTML = '<option value="">— elegir —</option>';
      for (const v of data.videos) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = `${v.name} (${v.size_mb} MB)`;
        sel.appendChild(opt);
      }
      if (current && [...sel.options].some((o) => o.value === current)) {
        sel.value = current;
      }
    }
    if (preferred && $("videoSelect") && !$("videoSelect").value) {
      const has = [...$("videoSelect").options].some((o) => o.value === preferred);
      if (has) $("videoSelect").value = preferred;
    }
    syncTrainVideoFromOps();
    if ($("trainStart") && $("trainVideoSelect")) {
      $("trainStart").disabled = !$("trainVideoSelect").value;
    }
  }

  async function upload(file) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    const dropLabel = $("dropLabel");
    if (dropLabel) dropLabel.textContent = file.name;
    setVideoLoading(true, `Subiendo ${file.name} (${sizeMb} MB)…`);
    $("videoMeta").textContent =
      `Subiendo ${file.name} (${sizeMb} MB)… Si es AV1, la conversión puede demorar.`;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("convert_av1", "true");
    fd.append("clip_seconds", "120");
    let res;
    try {
      res = await fetch("/api/upload", { method: "POST", body: fd });
    } catch (err) {
      setVideoLoading(false);
      $("videoMeta").textContent = `Error de red al subir: ${err}`;
      throw err;
    }
    if (!res.ok) {
      setVideoLoading(false);
      const detail = await res.text();
      $("videoMeta").textContent = `Error al subir (${res.status}): ${detail}`;
      throw new Error(detail);
    }
    const data = await res.json();
    setVideoLoading(true, "Cargando video…");
    $("videoMeta").textContent = data.converted
      ? `Convertido desde ${data.codec || "AV1/VP9"} → ${data.id}`
      : `Cargado: ${data.name || data.id}`;
    await refreshVideos();
    $("videoSelect").value = data.id;
    if ($("trainVideoSelect")) {
      $("trainVideoSelect").value = data.id;
      $("trainStart").disabled = false;
    }
    try {
      await selectVideo(data.id, data);
    } finally {
      setVideoLoading(false);
    }
    if (data.converted) {
      $("videoMeta").textContent += ` · ${data.width}×${data.height}`;
    }
    if (data.warning) {
      $("videoMeta").textContent += ` · aviso: ${data.warning}`;
    }
  }

  async function selectVideo(id, info) {
    setVideoLoading(true, "Cargando video…");
    try {
      state.videoId = id;
      state.points = [];
      if ($("videoSelect")) $("videoSelect").value = id;
      if ($("trainVideoSelect")) {
        $("trainVideoSelect").value = id;
        $("trainStart").disabled = !id;
      }
      const meta = info || (await fetchJSON(`/api/videos/${encodeURIComponent(id)}/info`));
      state.width = meta.width;
      state.height = meta.height;
      $("videoMeta").textContent =
        `${meta.width}×${meta.height} · ${meta.fps?.toFixed?.(1) || meta.fps} fps · ` +
        `${meta.frames || "?"} frames` +
        (meta.codec ? ` · ${meta.codec}` : "");
      $("loadFrame").disabled = false;
      $("defaultLine").disabled = false;
      $("resetLine").disabled = false;
      if ($("defaultZone")) $("defaultZone").disabled = false;
      if ($("resetZone")) $("resetZone").disabled = false;
      if (!state.zonePoints.length && state.width && state.height) {
        state.zonePoints = [
          { x: state.width * 0.05, y: state.height * 0.05 },
          { x: state.width * 0.95, y: state.height * 0.95 },
        ];
      }
      updateLineOut();
      updateZoneOut();
      updateRunEnabled();
      await loadFrame();
    } catch (err) {
      $("videoMeta").textContent = `Error al cargar video: ${err}`;
      throw err;
    } finally {
      setVideoLoading(false);
    }
  }

  async function loadFrame() {
    if (!state.videoId) return;
    const t = Number($("frameTime").value || 0);
    const url = `/api/videos/${encodeURIComponent(state.videoId)}/frame?t=${t}&_=${Date.now()}`;
    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });
    state.image = img;
    state.width = img.naturalWidth;
    state.height = img.naturalHeight;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    $("stageEmpty").classList.add("hidden");
    draw();
  }

  function onCanvasClick(e) {
    if (!state.image) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    if (state.calibMode === "zone") {
      if (state.zonePoints.length >= 2) state.zonePoints = [];
      state.zonePoints.push({ x, y });
      updateZoneOut();
    } else {
      if (state.points.length >= 2) state.points = [];
      state.points.push({ x, y });
      updateLineOut();
    }
    draw();
    updateRunEnabled();
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.image) ctx.drawImage(state.image, 0, 0);
    if (state.zonePoints.length) {
      ctx.fillStyle = "rgba(255, 180, 0, 0.15)";
      ctx.strokeStyle = "#ffb400";
      ctx.lineWidth = Math.max(2, canvas.width / 600);
      if (state.zonePoints.length === 1) {
        const p = state.zonePoints[0];
        ctx.beginPath();
        ctx.arc(p.x, p.y, Math.max(5, canvas.width / 250), 0, Math.PI * 2);
        ctx.fill();
      } else {
        const [a, b] = state.zonePoints;
        const x0 = Math.min(a.x, b.x);
        const y0 = Math.min(a.y, b.y);
        const w = Math.abs(a.x - b.x);
        const h = Math.abs(a.y - b.y);
        ctx.fillRect(x0, y0, w, h);
        ctx.strokeRect(x0, y0, w, h);
      }
    }
    if (state.points.length) {
      ctx.fillStyle = "#d4ff4f";
      ctx.strokeStyle = "#d4ff4f";
      ctx.lineWidth = Math.max(2, canvas.width / 600);
      for (const p of state.points) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, Math.max(5, canvas.width / 250), 0, Math.PI * 2);
        ctx.fill();
      }
      if (state.points.length === 2) {
        const [a, b] = state.points;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  function updateLineOut() {
    const el = $("lineOut");
    if (!el) return;
    if (state.points.length !== 2) {
      el.textContent = "Línea: —";
      return;
    }
    const [a, b] = state.points;
    el.textContent =
      `Línea: ${a.x.toFixed(0)},${a.y.toFixed(0)} → ${b.x.toFixed(0)},${b.y.toFixed(0)}`;
  }

  function updateZoneOut() {
    const el = $("zoneOut");
    if (!el) return;
    if (state.zonePoints.length !== 2) {
      el.textContent = "Área: —";
      return;
    }
    const [a, b] = state.zonePoints;
    const x0 = Math.min(a.x, b.x);
    const y0 = Math.min(a.y, b.y);
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    el.textContent =
      `Área: ${x0.toFixed(0)},${y0.toFixed(0)} – ${x1.toFixed(0)},${y1.toFixed(0)}`;
  }

  async function startProcess() {
    if (state.points.length !== 2 || state.zonePoints.length !== 2 || !state.videoId) {
      return;
    }
    const [a, b] = state.points;
    const [za, zb] = state.zonePoints;
    const maxFramesRaw = $("maxFrames").value;
    const body = {
      video_id: state.videoId,
      x0: a.x,
      y0: a.y,
      x1: b.x,
      y1: b.y,
      zx0: Math.min(za.x, zb.x),
      zy0: Math.min(za.y, zb.y),
      zx1: Math.max(za.x, zb.x),
      zy1: Math.max(za.y, zb.y),
      conf: Number($("conf").value || 0.35),
      max_frames: maxFramesRaw ? Number(maxFramesRaw) : null,
      cpu: $("cpu").checked,
      enable_ocr: $("ocr").checked,
      detect_model_id: $("opsDetectModel").value || null,
      recognize_model_id: $("opsRecognizeModel").value || null,
    };
    $("run").disabled = true;
    $("stop").disabled = false;
    showOpsPanel("live");
    $("reviewOffer")?.classList.add("hidden");
    $("liveFrame")?.classList.add("hidden");
    $("liveEmpty")?.classList.remove("hidden");
    if ($("snapGallery")) $("snapGallery").innerHTML = "";
    if ($("eventsTable")) {
      const tb = $("eventsTable").querySelector("tbody");
      if (tb) tb.innerHTML = "";
    }
    state.lastLiveSeq = -1;
    state.currentJobId = null;
    setJobStatus("running", "Enviando trabajo…");
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      setJobStatus("error", await res.text());
      $("run").disabled = false;
      $("stop").disabled = true;
      showOpsPanel("calib");
      return;
    }
    const { job_id } = await res.json();
    state.currentJobId = job_id;
    state.reviewJobId = job_id;
    pollJob(job_id);
  }

  async function stopProcess() {
    if (!state.currentJobId) return;
    $("stop").disabled = true;
    setJobStatus("running", "Deteniendo…");
    try {
      await fetch(`/api/jobs/${state.currentJobId}/stop`, { method: "POST" });
    } catch (err) {
      setJobStatus("error", String(err));
      $("stop").disabled = false;
    }
  }

  function pollJob(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);
    const tick = async () => {
      try {
        const job = await fetchJSON(`/api/jobs/${jobId}`);
        const progress =
          job.total_frames > 0
            ? ` · ${job.frame}/${job.total_frames}`
            : job.frame
              ? ` · frame ${job.frame}`
              : "";
        setJobStatus(job.status, (job.message || job.status) + progress);
        renderEvents(job.events || []);
        renderPending(job.pending || []);
        renderSnapshots(job.snapshots || []);
        if (job.live_url && job.live_seq !== state.lastLiveSeq) {
          state.lastLiveSeq = job.live_seq;
          const img = $("liveFrame");
          img.src = job.live_url;
          img.classList.remove("hidden");
          $("liveEmpty").classList.add("hidden");
        }
        const links = $("jobLinks");
        if (links) {
          links.innerHTML = "";
          if (job.csv_url) {
            links.innerHTML += `<a href="${job.csv_url}" download>Descargar CSV</a>`;
          }
          if (job.preview_url) {
            links.innerHTML += `<a href="${job.preview_url}" target="_blank">Ver preview video</a>`;
          }
        }
        const finished = ["done", "error", "cancelled"].includes(job.status);
        $("stop").disabled = finished;
        if (finished) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          state.currentJobId = null;
          updateRunEnabled();
          if (job.review_available) {
            $("reviewOffer")?.classList.remove("hidden");
            state.reviewEvents = job.events || [];
            state.reviewJobId = jobId;
          }
        }
      } catch (err) {
        setJobStatus("error", String(err));
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.currentJobId = null;
        updateRunEnabled();
        $("stop").disabled = true;
      }
    };
    tick();
    state.pollTimer = setInterval(tick, 800);
  }

  function setJobStatus(kind, text) {
    const el = $("jobStatus");
    if (!el) return;
    el.className = `job-status ${kind}`;
    el.textContent = text;
  }

  function renderEvents(events) {
    const table = $("eventsTable");
    if (!table) return;
    const tbody = table.querySelector("tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    for (const ev of events) {
      const conf =
        ev.bib_confidence != null ? `${(Number(ev.bib_confidence) * 100).toFixed(0)}%` : "—";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${ev.track_id}</td>
        <td>${ev.bib ?? "—"}</td>
        <td>${conf}</td>
        <td>${ev.bib_votes ?? "—"}</td>
        <td>${Number(ev.time_sec).toFixed(3)}</td>
        <td>${ev.ocr_samples ?? 0}</td>`;
      tbody.appendChild(tr);
    }
  }

  function renderPending(pending) {
    const el = $("pendingBox");
    if (!pending.length) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    const parts = pending.map((p) => {
      const votes = p.votes
        ? Object.entries(p.votes)
            .map(([b, c]) => `${b}×${c}`)
            .join(" ")
        : "";
      return `#${p.track_id} → ?${p.provisional_bib || "…"} (${p.ocr_samples} OCR) ${votes}`;
    });
    el.textContent =
      "Esperando salida del área para confirmar dorsal: " + parts.join(" · ");
    el.classList.remove("hidden");
  }

  function renderSnapshots(urls) {
    const gal = $("snapGallery");
    if (!gal) return;
    const existing = new Set(
      [...gal.querySelectorAll("a")].map((a) => a.getAttribute("href"))
    );
    for (const url of urls) {
      if (existing.has(url)) continue;
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.title = "Abrir captura";
      const img = document.createElement("img");
      img.src = url;
      img.alt = "Cruce detectado";
      a.appendChild(img);
      gal.appendChild(a);
    }
  }

  function startOpsReview() {
    if (!state.reviewEvents.length) {
      state.reviewEvents = [];
    }
    $("reviewOffer")?.classList.add("hidden");
    state.reviewIndex = 0;
    state.reviewLabels = {};
    showOpsPanel("review");
    showOpsReviewItem();
  }

  function showOpsReviewItem() {
    const events = state.reviewEvents;
    const i = state.reviewIndex;
    if (!events.length) {
      if ($("opsReviewMeta")) {
        $("opsReviewMeta").textContent = "No hay detecciones para revisar.";
      }
      return;
    }
    if (i >= events.length) {
      finishOpsReview();
      return;
    }
    const ev = events[i];
    const prev = state.reviewLabels[ev.track_id];
    if ($("opsReviewMeta")) {
      $("opsReviewMeta").textContent =
        `Detección ${i + 1} / ${events.length} · track #${ev.track_id} · t=${Number(ev.time_sec).toFixed(3)}s`;
    }
    if ($("opsReviewBib")) {
      $("opsReviewBib").value = prev?.bib || ev.bib || "";
    }
    if ($("opsReviewPrev")) $("opsReviewPrev").disabled = i <= 0;
    const jobId = state.reviewJobId;
    const snap = $("opsReviewSnap");
    const crop = $("opsReviewCrop");
    if (snap) {
      if (ev.snapshot) {
        const name = String(ev.snapshot).split("/").pop();
        snap.src = `/api/jobs/${jobId}/snapshot/${name}?t=${Date.now()}`;
        snap.classList.remove("hidden");
      } else {
        snap.removeAttribute("src");
      }
    }
    if (crop) {
      if (ev.crop) {
        const name = String(ev.crop).split("/").pop();
        crop.src = `/api/jobs/${jobId}/crop/${name}?t=${Date.now()}`;
      } else if (ev.snapshot) {
        const name = String(ev.snapshot).split("/").pop();
        crop.src = `/api/jobs/${jobId}/snapshot/${name}?t=${Date.now()}`;
      } else {
        crop.removeAttribute("src");
      }
    }
    if ($("opsReviewStatus")) {
      $("opsReviewStatus").textContent = prev
        ? `Marcado: ${prev.verdict}${prev.bib ? ` · ${prev.bib}` : ""}`
        : "";
    }
  }

  async function labelOpsReview(verdict) {
    const events = state.reviewEvents;
    if (!events.length || state.reviewIndex >= events.length) return;
    const ev = events[state.reviewIndex];
    const bib = ($("opsReviewBib")?.value || "").trim() || null;
    state.reviewLabels[ev.track_id] = { track_id: ev.track_id, verdict, bib };
    state.reviewIndex += 1;
    if (state.reviewIndex >= events.length) {
      await finishOpsReview();
    } else {
      showOpsReviewItem();
    }
  }

  async function finishOpsReview() {
    const reviews = state.reviewEvents.map((ev) => {
      return (
        state.reviewLabels[ev.track_id] || {
          track_id: ev.track_id,
          verdict: "skip",
          bib: ev.bib || null,
        }
      );
    });
    if ($("opsReviewStatus")) {
      $("opsReviewStatus").textContent = "Aplicando revisión y refinando modelos…";
    }
    const jobId = state.reviewJobId;
    try {
      const res = await fetch(`/api/jobs/${jobId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reviews,
          refine: true,
          cpu: $("cpu")?.checked || false,
        }),
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      if ($("opsReviewStatus")) {
        $("opsReviewStatus").textContent =
          `Listo · correctos=${data.correct} incorrectos=${data.incorrect} ` +
          `saltados=${data.skipped}` +
          (data.train_started?.length
            ? ` · entrenando: ${data.train_started.join(", ")}`
            : "");
      }
      await refreshModels().catch(() => {});
    } catch (err) {
      if ($("opsReviewStatus")) {
        $("opsReviewStatus").textContent = `Error: ${err}`;
      }
    }
  }

  async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  function bindTrain() {
    $("trainVideoSelect").addEventListener("change", () => {
      $("trainStart").disabled = !$("trainVideoSelect").value;
    });
    $("trainKind").addEventListener("change", () => {
      state.trainKind = $("trainKind").value;
      syncTrainActions();
    });
    $("trainStart").addEventListener("click", openTrainScanModal);
    $("trainStopScan").addEventListener("click", stopTrainScan);
    $("trainYes").addEventListener("click", () => labelDetect("yes"));
    $("trainNo").addEventListener("click", () => labelDetect("no"));
    $("trainSkip").addEventListener("click", () => labelDetect("skip"));
    $("trainSaveBib").addEventListener("click", labelRecognize);
    $("trainSkipRec").addEventListener("click", () =>
      labelDetect("skip", true)
    );
    $("trainPrev").addEventListener("click", goPrevCandidate);
    $("trainPrevRec").addEventListener("click", goPrevCandidate);

    $("trainModalChoiceRefine").addEventListener("click", () =>
      selectModalMode("refine")
    );
    $("trainModalChoiceCreate").addEventListener("click", () =>
      selectModalMode("create")
    );
    $("trainModalCancel").addEventListener("click", closeTrainScanModal);
    $("trainModalBack").addEventListener("click", modalGoBack);
    $("trainModalNext").addEventListener("click", modalGoNext);
    $("trainModalRefineSelect").addEventListener("change", updateModalNextEnabled);
    $("trainModalNewName").addEventListener("input", updateModalNextEnabled);
    $("trainModalDetectSelect")?.addEventListener("change", updateModalNextEnabled);
    $("trainScanModal")
      .querySelector("[data-close-modal]")
      ?.addEventListener("click", closeTrainScanModal);
    $("trainFitConfirmYes").addEventListener("click", confirmFitYes);
    $("trainFitConfirmNo").addEventListener("click", confirmFitNo);
    $("trainFitConfirmBack").addEventListener("click", async () => {
      $("trainFitConfirmModal").classList.add("hidden");
      await goPrevCandidate();
    });
    $("trainFitConfirmModal")
      .querySelector("[data-close-fit-confirm]")
      ?.addEventListener("click", confirmFitNo);
  }

  function syncTrainActions() {
    const detect = state.trainKind === "detect";
    $("trainDetectActions").classList.toggle("hidden", !detect);
    $("trainRecognizeActions").classList.toggle("hidden", detect);
    $("trainRecognizeBox").classList.toggle("hidden", detect);
  }

  function modelsForKind(kind) {
    return state.models.filter((m) => m.kind === kind);
  }

  function readyDetectModels() {
    return state.models.filter((m) => m.kind === "detect" && m.ready);
  }

  async function openTrainScanModal() {
    if (!$("trainVideoSelect").value) return;
    await refreshModels().catch(() => {});
    state.trainKind = $("trainKind").value;
    state.trainModalMode = null;
    $("trainModalError").textContent = "";
    $("trainModalNewName").value = "";

    const existing = modelsForKind(state.trainKind);
    const refineBtn = $("trainModalChoiceRefine");
    refineBtn.classList.toggle("hidden", existing.length === 0);
    refineBtn.classList.remove("selected");
    $("trainModalChoiceCreate").classList.remove("selected");

    showModalStep("choice");
    $("trainModalBack").disabled = true;
    $("trainModalNext").disabled = true;
    $("trainModalNext").textContent = "Siguiente";
    $("trainScanModalTitle").textContent = "Antes de escanear";
    if (state.trainKind === "recognize") {
      $("trainScanModalHint").textContent = existing.length
        ? "Elegí el modelo de reconocimiento (crear/refinar) y el detector de dorsal para las propuestas."
        : "Creá un modelo de reconocimiento. Si tenés un detector listo, elegilo para filtrar dorsales.";
    } else {
      $("trainScanModalHint").textContent = existing.length
        ? "¿Querés refinar un modelo existente o crear uno nuevo?"
        : "No hay modelos previos de este tipo. Creá uno nuevo para continuar.";
    }
    if (existing.length === 0) {
      selectModalMode("create");
    }
    $("trainScanModal").classList.remove("hidden");
  }

  function closeTrainScanModal() {
    $("trainScanModal").classList.add("hidden");
  }

  function showModalStep(step) {
    $("trainModalStepChoice").classList.toggle("hidden", step !== "choice");
    $("trainModalStepRefine").classList.toggle("hidden", step !== "refine");
    $("trainModalStepCreate").classList.toggle("hidden", step !== "create");
    // Detector: visible en refine/create solo si el tipo de sesión es reconocimiento
    const showDetect =
      state.trainKind === "recognize" && (step === "refine" || step === "create");
    $("trainModalDetectPick").classList.toggle("hidden", !showDetect);
    if (showDetect) fillModalDetectSelect();
  }

  function fillModalDetectSelect() {
    const sel = $("trainModalDetectSelect");
    if (!sel) return;
    const prev = sel.value;
    const ready = readyDetectModels();
    sel.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = ready.length
      ? "— elegí un detector —"
      : "HSV (no hay detectores listos)";
    sel.appendChild(empty);
    for (const m of ready) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.name} (listo)`;
      sel.appendChild(opt);
    }
    if (prev && [...sel.options].some((o) => o.value === prev)) {
      sel.value = prev;
    } else if (ready.length === 1) {
      sel.value = ready[0].id;
    }
    const hint = $("trainModalDetectHint");
    if (hint) {
      hint.textContent = ready.length
        ? "El detector filtra las propuestas: solo se te muestran recortes que el modelo cree que son dorsales."
        : "No hay detectores listos; se usará la heurística HSV. Entrená antes un modelo de detección.";
    }
  }

  function selectModalMode(mode) {
    state.trainModalMode = mode;
    $("trainModalChoiceRefine").classList.toggle("selected", mode === "refine");
    $("trainModalChoiceCreate").classList.toggle("selected", mode === "create");
    $("trainModalError").textContent = "";
    if (mode === "refine") {
      const sel = $("trainModalRefineSelect");
      sel.innerHTML = "";
      for (const m of modelsForKind(state.trainKind)) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = `${m.name} (${m.ready ? "listo" : m.status})`;
        sel.appendChild(opt);
      }
      showModalStep("refine");
      $("trainModalBack").disabled = false;
      $("trainModalNext").textContent = "Siguiente";
    } else if (mode === "create") {
      showModalStep("create");
      $("trainModalBack").disabled = modelsForKind(state.trainKind).length === 0;
      $("trainModalNext").textContent = "Siguiente";
      $("trainModalNewName").focus();
    }
    updateModalNextEnabled();
  }

  function modalGoBack() {
    state.trainModalMode = null;
    $("trainModalChoiceRefine").classList.remove("selected");
    $("trainModalChoiceCreate").classList.remove("selected");
    showModalStep("choice");
    $("trainModalBack").disabled = true;
    $("trainModalNext").disabled = true;
    $("trainModalNext").textContent = "Siguiente";
    $("trainModalError").textContent = "";
  }

  function updateModalNextEnabled() {
    let ok = false;
    if (state.trainModalMode === "refine") {
      ok = Boolean($("trainModalRefineSelect").value);
    } else if (state.trainModalMode === "create") {
      ok = Boolean(($("trainModalNewName").value || "").trim());
    }
    // En reconocimiento, si hay detectores listos, exigir uno
    if (ok && state.trainKind === "recognize" && readyDetectModels().length > 0) {
      ok = Boolean($("trainModalDetectSelect")?.value);
    }
    $("trainModalNext").disabled = !ok;
  }

  async function modalGoNext() {
    $("trainModalError").textContent = "";
    $("trainModalNext").disabled = true;
    try {
      let modelId = null;
      let detectModelId = null;

      if (state.trainKind === "recognize") {
        detectModelId = $("trainModalDetectSelect")?.value || null;
        if (readyDetectModels().length > 0 && !detectModelId) {
          throw new Error("Seleccioná un modelo de detección de dorsal");
        }
      }

      if (state.trainModalMode === "refine") {
        modelId = $("trainModalRefineSelect").value;
        if (!modelId) throw new Error("Seleccioná un modelo");
        const m = state.models.find((x) => x.id === modelId);
        if (m?.kind === "detect" && m.ready) {
          // Refinar detección: el mismo modelo propone candidatos
          detectModelId = modelId;
        }
      } else if (state.trainModalMode === "create") {
        const name = ($("trainModalNewName").value || "").trim();
        if (!name) throw new Error("Indicá un nombre");
        const res = await fetch("/api/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            kind: state.trainKind,
            parent_id: null,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const info = await res.json();
        modelId = info.id;
        await refreshModels();
      } else {
        throw new Error("Elegí refinar o crear");
      }

      state.trainTargetModelId = modelId;
      state.trainDetectModelId = detectModelId;
      state.autoTrainAfterReview = true;
      closeTrainScanModal();
      await startTrainScan(detectModelId);
    } catch (err) {
      $("trainModalError").textContent = String(err.message || err);
      updateModalNextEnabled();
    }
  }

  async function startTrainScan(detectModelId) {
    const videoId = $("trainVideoSelect").value;
    if (!videoId) return;
    state.trainKind = $("trainKind").value;
    syncTrainActions();
    const body = {
      video_id: videoId,
      kind: state.trainKind,
      frame_stride: Number($("trainStride").value || 5),
      max_candidates: Number($("trainMaxCand").value || 200),
      min_score: Number($("trainMinScore").value || 0.14),
      max_frames: $("trainMaxFrames").value
        ? Number($("trainMaxFrames").value)
        : null,
      cpu: $("cpu")?.checked || false,
      detect_model_id: detectModelId || null,
    };
    $("trainStart").disabled = true;
    $("trainStopScan").disabled = false;
    $("trainScanStatus").textContent = "Iniciando escaneo…";
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      $("trainScanStatus").textContent = await res.text();
      $("trainStart").disabled = false;
      $("trainStopScan").disabled = true;
      return;
    }
    const data = await res.json();
    state.trainRuntimeId = data.session_id;
    state.trainSessionId = data.session_id;
    pollTrainScan();
  }

  async function stopTrainScan() {
    const id = state.trainRuntimeId || state.trainSessionId;
    if (!id) return;
    await fetch(`/api/train/${id}/stop`, { method: "POST" });
    $("trainScanStatus").textContent = "Deteniendo…";
  }

  function pollTrainScan() {
    if (state.trainPoll) clearInterval(state.trainPoll);
    const tick = async () => {
      const id = state.trainSessionId || state.trainRuntimeId;
      if (!id) return;
      try {
        const st = await fetchJSON(`/api/train/${id}`);
        const sid = st.real_id || st.session_id || id;
        state.trainSessionId = sid;
        state.trainRuntimeId = sid;
        $("trainScanStatus").textContent =
          `${st.message || st.status} · candidatos=${st.candidates}` +
          (st.total ? ` · frame ${st.frame}/${st.total}` : "");
        if (st.status === "review" || st.status === "done") {
          clearInterval(state.trainPoll);
          state.trainPoll = null;
          $("trainStart").disabled = !$("trainVideoSelect").value;
          $("trainStopScan").disabled = true;
          await loadNextCandidate();
        }
        if (st.status === "error") {
          clearInterval(state.trainPoll);
          state.trainPoll = null;
          $("trainStart").disabled = false;
          $("trainStopScan").disabled = true;
        }
      } catch (err) {
        // Evitar ensuciar la UI con un 404 transitorio al arrancar
        console.warn("pollTrainScan", err);
      }
    };
    tick();
    state.trainPoll = setInterval(tick, 1000);
  }

  async function loadNextCandidate() {
    if (!state.trainSessionId) return;
    let data;
    try {
      data = await fetchJSON(`/api/train/${state.trainSessionId}/next`);
    } catch (err) {
      $("trainReviewMeta").textContent = String(err);
      return;
    }
    if (data.done) {
      $("trainReviewMeta").textContent = data.message || "Revisión completa";
      state.currentCandidate = null;
      setReviewEnabled(false);
      $("trainPrev").disabled = !(data.can_prev);
      $("trainPrevRec").disabled = !(data.can_prev);
      if (state.autoTrainAfterReview && state.trainTargetModelId) {
        openFitConfirmModal();
      }
      return;
    }
    showCandidate(data);
  }

  function showCandidate(data) {
    const c = data.candidate;
    state.currentCandidate = c;
    state.trainKind = c.kind;
    syncTrainActions();
    $("trainReviewMeta").textContent =
      `Candidato ${data.index + 1}/${data.total} · pendientes ${data.pending} · ` +
      `frame ${c.frame} · score ${Number(c.score).toFixed(2)}` +
      (c.method ? ` · ${c.method}` : "");
    $("trainCrop").src = c.image_url + "?t=" + Date.now();
    if ($("trainContext")) {
      $("trainContext").src = (c.context_url || c.image_url) + "?t=" + Date.now();
    }
    if (c.kind === "recognize") {
      $("trainBibInput").value = c.label_bib || "";
    }
    setReviewEnabled(true);
    const canPrev = Boolean(data.can_prev);
    $("trainPrev").disabled = !canPrev;
    $("trainPrevRec").disabled = !canPrev;
  }

  async function goPrevCandidate() {
    if (!state.trainSessionId) return;
    $("trainFitConfirmModal").classList.add("hidden");
    setReviewEnabled(false);
    $("trainPrev").disabled = true;
    $("trainPrevRec").disabled = true;
    try {
      const res = await fetch(`/api/train/${state.trainSessionId}/prev`, {
        method: "POST",
      });
      if (!res.ok) {
        $("trainReviewMeta").textContent = await res.text();
        setReviewEnabled(Boolean(state.currentCandidate));
        return;
      }
      const data = await res.json();
      if (data.done) return;
      showCandidate(data);
    } catch (err) {
      $("trainReviewMeta").textContent = String(err);
    }
  }

  function openFitConfirmModal() {
    const m = state.models.find((x) => x.id === state.trainTargetModelId);
    const name = m?.name || state.trainTargetModelId;
    const kind = m?.kind || state.trainKind;
    if (kind === "recognize") {
      $("trainFitConfirmHint").textContent =
        `¿Guardar el dataset de reconocimiento de «${name}» con los números etiquetados en esta sesión?`;
      $("trainFitConfirmYes").textContent = "Guardar dataset";
    } else {
      $("trainFitConfirmHint").textContent =
        `¿Iniciar el entrenamiento o refinamiento de «${name}» con las etiquetas de esta sesión?`;
      $("trainFitConfirmYes").textContent = "Entrenar modelo";
    }
    $("trainFitConfirmModal").classList.remove("hidden");
  }

  function confirmFitNo() {
    $("trainFitConfirmModal").classList.add("hidden");
    state.autoTrainAfterReview = false;
    $("trainScanStatus").textContent =
      "Revisión lista. Entrenamiento omitido; podés escanear de nuevo más tarde.";
  }

  async function confirmFitYes() {
    $("trainFitConfirmModal").classList.add("hidden");
    state.autoTrainAfterReview = false;
    const modelId = state.trainTargetModelId;
    if (!modelId) return;
    await startAutoModelTrain(modelId);
  }

  function setReviewEnabled(on) {
    for (const id of [
      "trainYes",
      "trainNo",
      "trainSkip",
      "trainSaveBib",
      "trainSkipRec",
    ]) {
      $(id).disabled = !on;
    }
  }

  async function labelDetect(verdict) {
    if (!state.trainSessionId || !state.currentCandidate) return;
    setReviewEnabled(false);
    await fetch(`/api/train/${state.trainSessionId}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate_id: state.currentCandidate.id,
        verdict,
      }),
    });
    await loadNextCandidate();
  }

  async function labelRecognize() {
    if (!state.trainSessionId || !state.currentCandidate) return;
    const bib = $("trainBibInput").value.trim();
    if (!bib) {
      $("trainReviewMeta").textContent = "Ingresá el número del dorsal";
      return;
    }
    setReviewEnabled(false);
    await fetch(`/api/train/${state.trainSessionId}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate_id: state.currentCandidate.id,
        verdict: "yes",
        bib,
      }),
    });
    await loadNextCandidate();
  }

  function showFitOverlay(msg, title) {
    if ($("trainFitTitle")) {
      $("trainFitTitle").textContent = title || "Entrenando modelo";
    }
    $("trainFitMessage").textContent = msg || "Entrenando…";
    $("trainFitOverlay").classList.remove("hidden");
  }

  function hideFitOverlay() {
    $("trainFitOverlay").classList.add("hidden");
  }

  async function startAutoModelTrain(modelId) {
    const m = state.models.find((x) => x.id === modelId);
    const isRec = (m?.kind || state.trainKind) === "recognize";
    showFitOverlay(
      isRec
        ? "Consolidando números etiquetados en el dataset…"
        : "Exportando dataset y entrenando YOLO…",
      isRec ? "Guardando reconocimiento" : "Entrenando detector"
    );
    const startedAt = Date.now();
    let res;
    try {
      res = await fetch(`/api/models/${modelId}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          epochs: 40,
          cpu: $("cpu")?.checked || false,
          detect_model_id:
          state.trainDetectModelId || readyDetectModels()[0]?.id || null,
        }),
      });
    } catch (err) {
      $("trainScanStatus").textContent = `Error al iniciar: ${err}`;
      hideFitOverlay();
      return;
    }
    if (!res.ok) {
      const detail = await res.text();
      $("trainScanStatus").textContent = `No se pudo entrenar (${res.status}): ${detail}`;
      $("trainReviewMeta").textContent = `No se pudo entrenar: ${detail}`;
      hideFitOverlay();
      return;
    }
    const body = await res.json().catch(() => ({}));
    pollModelTrain(modelId, {
      kind: body.kind || m?.kind || state.trainKind,
      startedAt,
    });
  }

  async function refreshModels() {
    const data = await fetchJSON("/api/models");
    state.models = data.models || [];
    fillModelSelect(
      $("opsDetectModel"),
      state.models.filter((m) => m.kind === "detect" && m.ready),
      "HSV / sin modelo entrenado",
      $("opsDetectModel")?.value
    );
    fillModelSelect(
      $("opsRecognizeModel"),
      state.models.filter((m) => m.kind === "recognize" && m.ready),
      "EasyOCR genérico",
      $("opsRecognizeModel")?.value
    );
  }

  function fillModelSelect(sel, models, emptyLabel, keep) {
    if (!sel) return;
    const prev = keep || sel.value;
    sel.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = emptyLabel;
    sel.appendChild(empty);
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      const tag = m.ready ? "listo" : m.status;
      opt.textContent = `${m.name} (${tag})`;
      sel.appendChild(opt);
    }
    if (prev && [...sel.options].some((o) => o.value === prev)) {
      sel.value = prev;
    }
  }

  function pollModelTrain(modelId, opts = {}) {
    if (state.modelTrainPoll) clearInterval(state.modelTrainPoll);
    const kind = opts.kind || "detect";
    const startedAt = opts.startedAt || Date.now();
    const minMs = kind === "recognize" ? 1200 : 600;
    let finishing = false;

    const finish = async (st) => {
      if (finishing) return;
      finishing = true;
      if (state.modelTrainPoll) {
        clearInterval(state.modelTrainPoll);
        state.modelTrainPoll = null;
      }
      const wait = Math.max(0, minMs - (Date.now() - startedAt));
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      await refreshModels();
      hideFitOverlay();
      const m = state.models.find((x) => x.id === modelId);
      if (st.status === "ready") {
        if (kind === "recognize" || m?.kind === "recognize") {
          const n = st.metrics?.labeled ?? m?.metrics?.labeled;
          const msg =
            `Dataset de reconocimiento listo: ${m?.name || modelId}` +
            (n != null ? ` · ${n} dorsales etiquetados` : "");
          $("trainScanStatus").textContent = msg;
          $("trainReviewMeta").textContent = msg;
        } else {
          if (m && m.kind === "detect" && $("opsDetectModel")) {
            $("opsDetectModel").value = modelId;
          }
          $("trainScanStatus").textContent =
            `Modelo detector listo: ${m?.name || modelId}`;
          $("trainReviewMeta").textContent =
            `Modelo detector listo: ${m?.name || modelId}`;
        }
      } else {
        const err = st.message || "error";
        $("trainScanStatus").textContent = `Entrenamiento falló: ${err}`;
        $("trainReviewMeta").textContent = `Entrenamiento falló: ${err}`;
      }
    };

    const tick = async () => {
      if (finishing) return;
      try {
        const st = await fetchJSON(`/api/models/${modelId}/train-status`);
        $("trainFitMessage").textContent = st.message || st.status;
        $("trainScanStatus").textContent = st.message || st.status;
        if (st.status === "ready" || st.status === "error") {
          await finish(st);
        }
      } catch (err) {
        $("trainFitMessage").textContent = String(err);
        $("trainScanStatus").textContent = String(err);
      }
    };
    tick();
    state.modelTrainPoll = setInterval(tick, 800);
  }

  function bindBibs() {
    $("bibRefreshPreview")?.addEventListener("click", () => refreshBibPreview());
    $("bibDownloadOne")?.addEventListener("click", () => downloadBibs(false));
    $("bibDownloadRange")?.addEventListener("click", () => downloadBibs(true));
    const liveIds = [
      "bibPreviewNumber",
      "bibRaceText",
      "bibBg",
      "bibNumberColor",
      "bibMarkerColor",
      "bibBorderColor",
      "bibRaceColor",
      "bibPageBg",
    ];
    let timer = null;
    for (const id of liveIds) {
      $(id)?.addEventListener("input", () => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => refreshBibPreview().catch(() => {}), 280);
      });
    }
  }

  function bibPayload(extra) {
    return {
      number: ($("bibPreviewNumber")?.value || "0").trim() || "0",
      bib_bg: $("bibBg")?.value || "#ffffff",
      number_color: $("bibNumberColor")?.value || "#111111",
      marker_color: $("bibMarkerColor")?.value || "#111111",
      border_color: $("bibBorderColor")?.value || "#111111",
      race_text: $("bibRaceText")?.value || "",
      race_color: $("bibRaceColor")?.value || "#444444",
      page_bg: $("bibPageBg")?.value || "#f0f0f0",
      width: 900,
      height: 1100,
      ...extra,
    };
  }

  async function refreshBibPreview() {
    const img = $("bibPreviewImg");
    const empty = $("bibPreviewEmpty");
    if (!img) return;
    if ($("bibStatus")) $("bibStatus").textContent = "Generando vista previa…";
    const res = await fetch("/api/bibs/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bibPayload({})),
    });
    if (!res.ok) {
      const detail = await res.text();
      if ($("bibStatus")) $("bibStatus").textContent = `Error: ${detail}`;
      return;
    }
    const blob = await res.blob();
    if (img._objectUrl) URL.revokeObjectURL(img._objectUrl);
    img._objectUrl = URL.createObjectURL(blob);
    img.src = img._objectUrl;
    img.classList.add("visible");
    empty?.classList.add("hidden");
    if ($("bibStatus")) $("bibStatus").textContent = "Vista previa lista";
  }

  async function downloadBibs(range) {
    const payload = bibPayload(
      range
        ? {
            start: Number($("bibStart")?.value || 1),
            end: Number($("bibEnd")?.value || 1),
            pad_width: Number($("bibPad")?.value || 0),
          }
        : {}
    );
    if ($("bibStatus")) {
      $("bibStatus").textContent = range
        ? "Generando lote ZIP…"
        : "Generando PNG…";
    }
    const res = await fetch("/api/bibs/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.text();
      if ($("bibStatus")) $("bibStatus").textContent = `Error: ${detail}`;
      return;
    }
    const blob = await res.blob();
    const cd = res.headers.get("Content-Disposition") || "";
    const m = /filename="?([^"]+)"?/.exec(cd);
    const name =
      m?.[1] ||
      (range ? "dorsales.zip" : `dorsal_${payload.number}.png`);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    if ($("bibStatus")) {
      $("bibStatus").textContent = range
        ? `Lote descargado (${payload.start}–${payload.end})`
        : `PNG descargado: ${name}`;
    }
  }

  init();
})();
