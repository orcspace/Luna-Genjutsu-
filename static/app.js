const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

let config = null;
let currentJob = null;
let jobTimer = null;
const STAGES = ["normalize", "mask", "upload", "generate", "finish", "done"];
const STATUS_TEXT = { queued: "В очереди", running: "Подготовка", review: "Ждёт проверки", generating: "Генерация", done: "Готово", failed: "Ошибка", canceled: "Отменено" };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

// ---------- config & settings ----------

async function loadConfig() {
  config = await api("/api/config");
  const sel = $("#providerSelect");
  sel.innerHTML = config.providers.map(p => `<option value="${p.name}">${p.label}${p.ready ? "" : " — нет ключа"}</option>`).join("");
  sel.value = config.provider;
  syncModels();
  const p = config.providers.find(x => x.name === config.provider);
  const badge = $("#providerBadge");
  badge.textContent = p ? p.label + (p.ready ? "" : " · нет ключа") : "";
  badge.className = "badge " + (p && p.ready ? "ok" : "bad");
  const localOpt = $('select[name=masking] option[value=local]');
  localOpt.disabled = !config.local_masks;
  $('select[name=masking] option[value=cloud]').disabled = !config.replicate;
  $('select[name=audio] option[value=vocals]').disabled = !config.replicate;
  if (!config.ffmpeg) hint("ffmpeg не найден — установите его, без него обработка видео невозможна.", true);
  const hw = config.hardware || {};
  $("#hwLine").textContent = [
    hw.gpu ? `Видеокарта: ${hw.gpu.name}${hw.gpu.vram_gb ? ` ${hw.gpu.vram_gb} ГБ` : ""}` : "Видеокарта NVIDIA не найдена",
    `кодирование: ${hw.encoder === "nvenc" ? "NVENC (видеокарта)" : "процессор"}`,
    hw.cpu_threads ? `потоков CPU: ${hw.cpu_threads}` : "",
    hw.ram_gb ? `ОЗУ: ${hw.ram_gb} ГБ` : "",
  ].filter(Boolean).join(" · ");
}

// Model choice for providers that offer several (Dreamina).
function syncModels() {
  const p = config.providers.find(x => x.name === $("#providerSelect").value);
  const models = (p && p.models) || {};
  const names = Object.keys(models);
  $("#modelField").hidden = !names.length;
  const ms = $("#modelSelect");
  ms.innerHTML = names.map(n => `<option value="${n}">${models[n]}</option>`).join("");
  if (names.includes(config.settings.DREAMINA_MODEL)) ms.value = config.settings.DREAMINA_MODEL;
}
$("#providerSelect").addEventListener("change", syncModels);

$("#openSettings").onclick = () => {
  const f = $("#settingsForm");
  for (const [k, v] of Object.entries(config.settings)) if (f.elements[k]) f.elements[k].value = v;
  $("#settings").showModal();
};
$("#settings").addEventListener("close", async () => {
  if ($("#settings").returnValue !== "save") return;
  const body = Object.fromEntries(new FormData($("#settingsForm")));
  try {
    await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    await loadConfig();
    hint("Настройки сохранены.");
  } catch (e) { hint(e.message, true); }
});

// ---------- form ----------

function hint(text, err = false) {
  const h = $("#formHint");
  h.textContent = text;
  h.className = "hint" + (err ? " err" : "");
}

function setupDrop(label, onFile) {
  const input = $("input", label);
  label.addEventListener("dragover", e => { e.preventDefault(); label.classList.add("over"); });
  label.addEventListener("dragleave", () => label.classList.remove("over"));
  label.addEventListener("drop", e => {
    e.preventDefault();
    label.classList.remove("over");
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      input.dispatchEvent(new Event("change"));
    }
  });
  input.addEventListener("change", () => input.files[0] && onFile(input.files[0], input.files));
}

setupDrop($("#videoDrop"), file => {
  const drop = $("#videoDrop");
  const video = $(".drop-preview", drop);
  const meta = $(".drop-meta", drop);
  video.src = URL.createObjectURL(file);
  video.hidden = false;
  $(".drop-empty", drop).hidden = true;
  drop.classList.add("filled");
  video.onloadedmetadata = () => {
    video.play().catch(() => {});
    const d = video.duration;
    meta.hidden = false;
    meta.className = "drop-meta";
    meta.textContent = `${d.toFixed(1)} с · ${video.videoWidth}×${video.videoHeight}`;
    if (d > 30) { meta.textContent += " · будут взяты первые 30 с"; meta.classList.add("warn"); }
    if (d < 4) { meta.textContent += " · коротко, лучше от 4 с"; meta.classList.add("warn"); }
  };
});

$$(".photo-drop").forEach(drop => setupDrop(drop, file => {
  const img = $(".drop-preview", drop);
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $(".drop-empty", drop).hidden = true;
  drop.classList.add("filled");
  checkPhoto(drop, file);
}));

// Instant photo check: face found, size, sharpness, light — before anything is paid for.
async function checkPhoto(drop, file) {
  const box = $(".photo-check", drop);
  box.hidden = false;
  box.className = "photo-check busy";
  box.textContent = "Проверяю фото…";
  const data = new FormData();
  data.append("photo", file);
  data.append("role", drop.dataset.role);
  try {
    const r = await api("/api/check-photo", { method: "POST", body: data });
    if (!r.available) { box.hidden = !r.warnings.length; box.className = "photo-check warn"; box.textContent = r.warnings.join(" "); return; }
    if (r.warnings.length) {
      box.className = "photo-check warn";
      box.textContent = "⚠ " + r.warnings.join(" ");
    } else {
      box.className = "photo-check ok";
      box.textContent = drop.dataset.role === "face" ? `✓ Лицо ${r.face_px}px, резкое, свет хороший` : "✓ Человек и лицо найдены";
    }
  } catch (e) {
    box.className = "photo-check warn";
    box.textContent = e.message;
  }
}

$(".extras input").addEventListener("change", e => {
  const n = e.target.files.length;
  $("#extrasLabel").textContent = n ? `Дополнительных ракурсов: ${n} (нажмите, чтобы заменить)` : "+ Добавить ракурсы (сбоку, сзади) — необязательно";
});

// ---------- templates ----------

const GLYPHS = {
  person: "M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0 2c-4 0-8 2-8 5v1h16v-1c0-3-4-5-8-5z",
  box: "M12 2 3 7v10l9 5 9-5V7l-9-5zm0 2.3L18.7 8 12 11.7 5.3 8 12 4.3z",
  shirt: "M8 3 3 6l2 4 2-1v12h10V9l2 1 2-4-5-3c0 2-1.8 3-4 3S8 5 8 3z",
  pin: "M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5z",
  spark: "M12 2l2.4 6.6L21 11l-6.6 2.4L12 20l-2.4-6.6L3 11l6.6-2.4z",
  blob: "M17 4c3 2 4 6 2 10s-6 6-10 5-6-5-5-9 4-6 7-7 4-1 6 1z",
  cube: "M12 2 3 7v10l9 5 9-5V7zM12 12 5 8m7 4 7-4m-7 4v9",
  film: "M4 4h16v16H4zM8 4v16M16 4v16M4 8h4m8 0h4M4 12h4m8 0h4M4 16h4m8 0h4",
  bolt: "M13 2 4 14h7l-1 8 9-12h-7z",
  eye: "M12 5C6 5 2 12 2 12s4 7 10 7 10-7 10-7-4-7-10-7zm0 11a4 4 0 1 1 0-8 4 4 0 0 1 0 8z",
};
let templateList = [];

async function loadTemplates() {
  templateList = await api("/api/templates");
  $("#templates").innerHTML = templateList.map(t => `
    <button type="button" class="tpl" data-id="${t.id}" style="--g: linear-gradient(135deg, ${t.colors[0]}, ${t.colors[1]})" title="${escapeHtml(t.subtitle)}">
      <svg viewBox="0 0 24 24" width="22" height="22" fill="${["cube", "film"].includes(t.glyph) ? "none" : "currentColor"}" stroke="${["cube", "film"].includes(t.glyph) ? "currentColor" : "none"}" stroke-width="1.6"><path d="${GLYPHS[t.glyph] || GLYPHS.spark}"/></svg>
      <b>${escapeHtml(t.title)}</b><small>${escapeHtml(t.subtitle)}</small>
    </button>`).join("");
  $$(".tpl").forEach(b => b.onclick = () => pickTemplate(b.dataset.id));
  pickTemplate(sessionTemplate() || "hero");
}

function sessionTemplate() {
  try { return localStorage.getItem("genjutsu.template"); } catch { return null; }
}

function pickTemplate(id) {
  const t = templateList.find(x => x.id === id) || templateList[0];
  if (!t) return;
  $("#form").template.value = t.id;
  try { localStorage.setItem("genjutsu.template", t.id); } catch {}
  $$(".tpl").forEach(b => b.classList.toggle("on", b.dataset.id === t.id));
  const field = $("#tplField");
  field.hidden = !t.input;
  if (t.input) {
    $("#tplLabel").textContent = t.input.label;
    $("#tplInput").placeholder = t.input.placeholder;
    $("#tplInput").required = !!t.input.required;
  } else {
    $("#tplInput").required = false;
  }
  const object = t.photos === "object";
  $("#objectPhotos").hidden = !object;
  $("#personPhotos").hidden = object;
  $("#extrasPick").hidden = object;
  $("#photosTitle").textContent = object ? "Фото объекта" : "Ваши фото";
  $(".consent span").textContent = object
    ? "У меня есть права на это видео и фото объекта."
    : "На фото я или человек, который дал согласие на такое видео.";
}

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target;
  if (!form.video.files[0]) return hint("Загрузите видео-референс.", true);
  const tpl = templateList.find(x => x.id === form.template.value) || {};
  if (tpl.photos === "object") {
    if (!form.object.files[0]) return hint("Загрузите фото объекта.", true);
  } else if (!form.face.files[0] && !form.body.files[0]) return hint("Загрузите хотя бы одно фото: лицо или в полный рост.", true);
  if (tpl.input && tpl.input.required && !form.tpl_input.value.trim()) return hint(`Заполните «${tpl.input.label}».`, true);
  const data = new FormData(form);
  // Only the photos the chosen template uses are sent.
  for (const k of ["face", "body", "object"]) {
    const used = tpl.photos === "object" ? k === "object" : k !== "object";
    if (!form[k].files[0] || !used) data.delete(k);
  }
  if (tpl.photos === "object") data.delete("extras");
  if (!form.extras.files.length) data.delete("extras");
  if ($("#modelField").hidden) data.delete("model");
  const btn = $("#submitBtn");
  btn.disabled = true;
  hint("Загружаю файлы…");
  try {
    const job = await api("/api/jobs", { method: "POST", body: data });
    hint("");
    openJob(job.id);
    loadHistory();
  } catch (err) {
    hint(err.message, true);
  } finally {
    btn.disabled = false;
  }
});

// ---------- job view ----------

function openJob(id) {
  currentJob = id;
  $("#createCard").hidden = true;
  $("#jobCard").hidden = false;
  $("#resultVideo").removeAttribute("src");
  $("#resultVideo").dataset.src = "";
  refreshJob();
  window.scrollTo({ top: 0, behavior: "smooth" });
  history.replaceState(null, "", "#" + id);
}

$("#backBtn").onclick = () => {
  currentJob = null;
  clearTimeout(jobTimer);
  $("#jobCard").hidden = true;
  $("#createCard").hidden = false;
  history.replaceState(null, "", location.pathname);
  $$(".hitem").forEach(x => x.classList.remove("active"));
};

const file = (job, name) => `/files/${job.id}/${name}?download=1`;

async function refreshJob() {
  clearTimeout(jobTimer);
  if (!currentJob) return;
  let job;
  try { job = await api(`/api/jobs/${currentJob}`); } catch (e) { jobTimer = setTimeout(refreshJob, 4000); return; }
  if (job.id !== currentJob) return;
  render(job);
  const hdRunning = job.hd && job.hd.status === "running";
  if (["queued", "running", "generating"].includes(job.status) || hdRunning) jobTimer = setTimeout(refreshJob, 2000);
}

function render(job) {
  $("#jobId").textContent = job.id;
  $("#jobStage").textContent = job.stage || STATUS_TEXT[job.status];
  $("#jobPct").textContent = job.status === "done" ? "100%" : `${job.progress || 0}%`;
  $("#jobBar").style.width = `${job.progress || 0}%`;
  $(".bar").classList.toggle("live", ["running", "generating"].includes(job.status));
  $$(".hitem").forEach(x => x.classList.toggle("active", x.dataset.id === job.id));

  const nowIdx = job.status === "done" ? STAGES.length : job.status === "review" ? STAGES.indexOf("upload") : STAGES.indexOf(job.stage_key);
  $$("#jobSteps li").forEach((li, i) => {
    li.classList.toggle("done", i < nowIdx || job.status === "done");
    li.classList.toggle("now", i === nowIdx && job.status !== "done");
  });

  const err = $("#jobError");
  err.hidden = !job.error;
  err.textContent = job.error || "";

  const notices = [...(job.notices || []), ...((job.qa && job.qa.issues) || [])];
  if (job.used) notices.push(`Маска: ${({ cloud: "SAM 3 + глубина (облако)", local: "MediaPipe (локально)", none: "без маски" })[job.used.masking]} · звук для модели: ${({ vocals: "голос", full: "весь звук", none: "нет" })[job.used.audio]}${job.used.face_mesh ? " · сетка лица: да" : ""}`);
  const n = $("#jobNotices");
  n.hidden = !notices.length;
  n.textContent = notices.join("\n");

  $("#reviewBox").hidden = job.status !== "review";
  $("#retryBtn").hidden = !(["failed", "canceled"].includes(job.status) || (job.hd && job.hd.status === "failed"));
  $("#stopBtn").hidden = !["queued", "running"].includes(job.status);

  $("#previewFig").hidden = !job.guide;
  // Only touch src when the file really changed, so images don't reload on every poll.
  if (job.guide) setSrc($("#previewImg"), `/files/${job.id}/preview.jpg?v=${encodeURIComponent(job.guide + (job.qa ? job.qa.mean_area : ""))}`);
  $("#sheetFig").hidden = false;
  setSrc($("#sheetImg"), `/files/${job.id}/sheet.jpg?v=${(job.refs || []).map(r => r.send).join(",")}`);

  const box = $("#resultBox");
  box.hidden = !job.result;
  renderTakes(job);
  if (job.result) {
    const hdDone = job.hd && job.hd.status === "done";
    const src = `/files/${job.id}/${hdDone ? job.hd.result : job.result}`;
    const v = $("#resultVideo");
    if (v.dataset.src !== src) { v.src = src; v.dataset.src = src; }
    $("#downloadBtn").href = file(job, job.result);
    $("#downloadBtn").textContent = job.options.quality === "hd" ? "Скачать 1080p" : "Скачать черновик";
    const canHd = job.options.quality === "draft" && !job.hd;
    $("#hdBtn").hidden = !canHd;
    $("#downloadHdBtn").hidden = !hdDone;
    if (hdDone) $("#downloadHdBtn").href = file(job, job.hd.result);
    const hh = $("#hdHint");
    hh.className = "hint";
    hh.textContent = job.hd ? ({ running: "1080p генерируется…", done: "1080p готово", failed: "1080p: " + (job.hd.error || "ошибка") })[job.hd.status] || "" : "";
  }

  $("#jobPrompt").textContent = job.prompt || "Промпт появится перед отправкой.";
  $("#jobLog").innerHTML = (job.log || []).slice().reverse().map(l => `<li>${l.t.slice(11, 19)} · ${escapeHtml(l.msg)}</li>`).join("");
}

function setSrc(el, src) {
  if (el.dataset.src !== src) { el.src = src; el.dataset.src = src; }
}

function pctClass(p) { return p >= 60 ? "hi" : p >= 40 ? "mid" : "lo"; }

function renderTakes(job) {
  const takes = job.takes || [];
  const chosen = takes.find(t => t.index === job.chosen);
  const line = $("#scoreLine");
  const sc = chosen && chosen.score;
  line.hidden = !sc;
  if (sc) {
    line.innerHTML = `Сходство лица с вашими фото: <b class="pct ${pctClass(sc.score)}">${sc.score}%</b>` +
      (sc.coverage < 0.6 ? " · лицо видно не на всех кадрах" : "") +
      (sc.score < 40 ? " · попробуйте другое фото лица или ещё дубль" : "");
  }
  const box = $("#takesBox");
  box.hidden = takes.length < 2;
  if (takes.length < 2) return;
  box.innerHTML = takes.map(t => {
    const p = t.score ? `<span class="pct ${pctClass(t.score.score)}">${t.score.score}%</span>` : "";
    const state = t.state === "done" ? (t.index === job.chosen ? "Выбран" : "Выбрать") : t.state === "failed" ? "Ошибка" : "Генерируется…";
    const img = t.thumb ? `<img src="/files/${job.id}/${t.thumb}" alt="">` : `<img alt="">`;
    return `<button type="button" class="take ${t.index === job.chosen ? "chosen" : ""} ${t.state}" data-i="${t.index}" ${t.state !== "done" ? "disabled" : ""}>${img}<span class="tl"><span>Дубль ${t.index + 1}</span>${p}</span><span class="tl muted">${state}</span></button>`;
  }).join("");
  $$(".take", box).forEach(b => b.onclick = async () => {
    const i = +b.dataset.i;
    if (i === job.chosen || job.status !== "done") return;
    if (job.hd && job.hd.status === "done" && job.hd.take !== i && !confirm("Готовое 1080p относится к другому дублю. Выбрать этот?")) return;
    try { render(await api(`/api/jobs/${job.id}/choose/${i}`, { method: "POST" })); loadHistory(); } catch (e) { alert(e.message); }
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

async function jobAction(action, confirmText) {
  if (confirmText && !confirm(confirmText)) return;
  try {
    render(await api(`/api/jobs/${currentJob}/${action}`, { method: "POST" }));
    refreshJob();
    loadHistory();
  } catch (e) { alert(e.message); }
}
$("#approveBtn").onclick = () => jobAction("approve");
$("#cancelBtn").onclick = () => jobAction("cancel");
$("#stopBtn").onclick = () => jobAction("cancel");
$("#hdBtn").onclick = () => jobAction("hd", "Сделать финальную версию в 1080p? Это отдельная платная генерация.");
$("#retryBtn").onclick = () => jobAction("retry", "Повторить? Если генерация уже была отправлена и завершилась ошибкой, новая попытка оплачивается заново.");

// ---------- history ----------

async function loadHistory() {
  let list;
  try { list = await api("/api/jobs"); } catch { return; }
  const box = $("#historyList");
  if (!list.length) { box.innerHTML = '<p class="muted">Пока пусто.</p>'; return; }
  box.innerHTML = list.map(j => {
    const st = j.hd && j.hd.status === "running" ? "1080p…" : STATUS_TEXT[j.status] || j.status;
    const thumb = j.result ? `<img src="/files/${j.id}/thumb.jpg?c=${j.chosen}" alt="" loading="lazy">` : `<img src="/files/${j.id}/sheet.jpg" alt="" loading="lazy">`;
    const time = new Date(j.created).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
    const bar = ["running", "generating", "queued"].includes(j.status) ? `<div class="mini"><i style="width:${j.progress || 0}%"></i></div>` : "";
    return `<div class="hitem${j.id === currentJob ? " active" : ""}" data-id="${j.id}">${thumb}<div><div class="t">${time} · ${escapeHtml((templateList.find(t => t.id === j.options.template) || {}).title || (j.options.mode === "motion" ? "новая сцена" : "замена"))}</div><div class="s ${j.status}">${st}</div>${bar}</div></div>`;
  }).join("");
  $$(".hitem", box).forEach(el => el.onclick = () => openJob(el.dataset.id));
}

// ---------- boot ----------

loadConfig().catch(e => hint("Сервер недоступен: " + e.message, true));
loadTemplates().catch(e => hint("Шаблоны не загрузились: " + e.message, true));
loadHistory();
setInterval(loadHistory, 5000);
if (location.hash.length > 1) openJob(location.hash.slice(1));
