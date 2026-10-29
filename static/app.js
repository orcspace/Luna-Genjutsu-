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
  const p = config.providers.find(x => x.name === config.provider);
  const badge = $("#providerBadge");
  badge.textContent = p ? p.label + (p.ready ? "" : " · нет ключа") : "";
  badge.className = "badge " + (p && p.ready ? "ok" : "bad");
  const localOpt = $('select[name=masking] option[value=local]');
  localOpt.disabled = !config.local_masks;
  $('select[name=masking] option[value=cloud]').disabled = !config.replicate;
  $('select[name=audio] option[value=vocals]').disabled = !config.replicate;
  if (!config.ffmpeg) hint("ffmpeg не найден — установите его, без него обработка видео невозможна.", true);
}

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

$$(".seg").forEach(seg => seg.addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  $$("button", seg).forEach(x => x.classList.toggle("on", x === b));
  if (seg.dataset.name === "mode") $("#sceneField").hidden = b.dataset.value !== "motion";
}));

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target;
  if (!form.video.files[0]) return hint("Загрузите видео-референс.", true);
  if (!form.face.files[0] && !form.body.files[0]) return hint("Загрузите хотя бы одно фото: лицо или в полный рост.", true);
  const data = new FormData(form);
  data.set("mode", $('.seg[data-name=mode] .on').dataset.value);
  for (const k of ["face", "body"]) if (!form[k].files[0]) data.delete(k);
  if (!form.extras.files.length) data.delete("extras");
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

const file = (job, name, dl = false) => `/files/${job.id}/${name}${dl ? "?download=1" : `?v=${encodeURIComponent(job.progress)}`}`;

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

  const nowIdx = job.status === "done" ? STAGES.length : STAGES.indexOf(job.stage_key);
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
  if (job.guide) $("#previewImg").src = file(job, "preview.jpg");
  $("#sheetFig").hidden = false;
  $("#sheetImg").src = `/files/${job.id}/sheet.jpg?v=${(job.refs || []).length}`;

  const box = $("#resultBox");
  box.hidden = !job.result;
  renderTakes(job);
  if (job.result) {
    const hdDone = job.hd && job.hd.status === "done";
    const src = `/files/${job.id}/${hdDone ? job.hd.result : job.result}`;
    const v = $("#resultVideo");
    if (v.dataset.src !== src) { v.src = src; v.dataset.src = src; }
    $("#downloadBtn").href = file(job, job.result, true);
    $("#downloadBtn").textContent = job.options.quality === "hd" ? "Скачать 1080p" : "Скачать черновик";
    const canHd = job.options.quality === "draft" && !job.hd;
    $("#hdBtn").hidden = !canHd;
    $("#downloadHdBtn").hidden = !hdDone;
    if (hdDone) $("#downloadHdBtn").href = file(job, job.hd.result, true);
    const hh = $("#hdHint");
    hh.className = "hint";
    hh.textContent = job.hd ? ({ running: "1080p генерируется…", done: "1080p готово", failed: "1080p: " + (job.hd.error || "ошибка") })[job.hd.status] || "" : "";
  }

  $("#jobPrompt").textContent = job.prompt || "Промпт появится перед отправкой.";
  $("#jobLog").innerHTML = (job.log || []).slice().reverse().map(l => `<li>${l.t.slice(11, 19)} · ${escapeHtml(l.msg)}</li>`).join("");
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
    return `<div class="hitem${j.id === currentJob ? " active" : ""}" data-id="${j.id}">${thumb}<div><div class="t">${time} · ${j.options.mode === "motion" ? "новая сцена" : "замена"}</div><div class="s ${j.status}">${st}</div>${bar}</div></div>`;
  }).join("");
  $$(".hitem", box).forEach(el => el.onclick = () => openJob(el.dataset.id));
}

// ---------- boot ----------

loadConfig().catch(e => hint("Сервер недоступен: " + e.message, true));
loadHistory();
setInterval(loadHistory, 5000);
if (location.hash.length > 1) openJob(location.hash.slice(1));
