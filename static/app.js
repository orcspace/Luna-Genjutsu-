const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

let config = null;
let currentJob = null;
let jobTimer = null;
let templateList = [];
let videoSeconds = null;
let gallerySignature = "";
let mode = "photo";
const PHOTO_STEPS = [["photos", "Фото"], ["prompt", "Промпт GPT"], ["generate", "Генерация"], ["finish", "Оценка"], ["done", "Готово"]];
const VIDEO_STEPS = [["normalize", "Подготовка"], ["mask", "Маска"], ["upload", "Отправка"], ["generate", "Генерация"], ["finish", "Звук"], ["done", "Готово"]];
const STATUS_TEXT = { queued: "В очереди", running: "Подготовка", review: "Ждёт проверки", generating: "Генерация", done: "Готово", failed: "Ошибка", canceled: "Отменено" };
const SHORT_MODEL = { "seedance-2.5": "Seedance 2.5", "seedance-2.0": "Seedance 2.0", "seedance-2.0-fast": "2.0 Fast", "seedance-2.0-mini": "2.0 Mini" };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function hint(text, err = false) {
  const h = $("#formHint");
  h.textContent = text;
  h.className = "hint" + (err ? " err" : "");
}

// ---------- views: create / gallery / job ----------
// Desktop shows the form and the workspace side by side; on a phone the tabs switch between them.

function setView(view) {
  const app = $(".app");
  app.classList.toggle("view-create", view === "create");
  $("#tabCreate").classList.toggle("on", view === "create");
  $("#tabGallery").classList.toggle("on", view !== "create");
  $("#galleryCard").hidden = view === "job";
  $("#jobCard").hidden = view !== "job";
}

const isPhone = () => window.matchMedia("(max-width: 800px)").matches;

$("#tabCreate").onclick = () => {
  if (isPhone()) setView("create");
  else {
    // The form is always visible on a wide screen: just bring it to the top.
    $("#tabCreate").classList.add("on");
    $("#tabGallery").classList.remove("on");
  }
  $("#createCard").scrollTo({ top: 0, behavior: "smooth" });
  window.scrollTo({ top: 0, behavior: "smooth" });
};
$("#tabGallery").onclick = () => closeJob();

// ---------- config & settings ----------

async function loadConfig() {
  config = await api("/api/config");
  const sel = $("#providerSelect");
  sel.innerHTML = config.providers.map(p => `<option value="${p.name}">${escapeHtml(p.label)}${p.ready ? "" : " — нет ключа"}</option>`).join("");
  sel.value = config.provider;
  syncModels();
  const p = config.providers.find(x => x.name === config.provider);
  const badge = $("#providerBadge");
  badge.textContent = p ? p.label + (p.ready ? "" : " · нет ключа") : "";
  badge.className = "badge " + (p && p.ready ? "ok" : "bad");
  $('select[name=masking] option[value=local]').disabled = !config.local_masks;
  $('select[name=masking] option[value=cloud]').disabled = !config.replicate;
  $('select[name=audio] option[value=vocals]').disabled = !config.replicate;
  if (!config.ffmpeg) hint("ffmpeg не найден — установите его, без него обработка видео невозможна.", true);
  else if (!(config.providers.find(x => x.name === "dreamina") || {}).ready) hint("Вставьте токен useapi.net (Dreamina) в настройках — шестерёнка справа вверху.");
  $("#modeSwitch").hidden = !config.show_video;
  setMode(config.show_video ? savedMode() : "photo");
  renderGpt(config.chatgpt);
  syncImageModels();
  const hw = config.hardware || {};
  $("#hwLine").textContent = [
    hw.gpu ? `Видеокарта: ${hw.gpu.name}${hw.gpu.vram_gb ? ` ${hw.gpu.vram_gb} ГБ` : ""}` : "Видеокарта NVIDIA не найдена",
    `кодирование: ${hw.encoder === "nvenc" ? "NVENC (видеокарта)" : "процессор"}`,
    hw.cpu_threads ? `потоков CPU: ${hw.cpu_threads}` : "",
    hw.ram_gb ? `ОЗУ: ${hw.ram_gb} ГБ` : "",
  ].filter(Boolean).join(" · ");
  updateSummary();
}

// ---------- modes: photo (default) / video ----------

function savedMode() {
  try { return localStorage.getItem("genjutsu.mode") === "video" ? "video" : "photo"; } catch { return "photo"; }
}

function setMode(m) {
  mode = m === "video" ? "video" : "photo";
  try { localStorage.setItem("genjutsu.mode", mode); } catch {}
  $(".app").classList.toggle("mode-photo", mode === "photo");
  $(".app").classList.toggle("mode-video", mode === "video");
  $$("#modeSwitch button").forEach(b => b.classList.toggle("on", b.dataset.mode === mode));
  $("#extraHint").textContent = mode === "photo" ? "можно по-русски — GPT переведёт" : "по-английски, необязательно";
  renderTemplateCards();
  if (templateList.length) pickTemplate($("#form").template.value);
  modelWarning();
  updateSummary();
}
$$("#modeSwitch button").forEach(b => b.onclick = () => setMode(b.dataset.mode));

// ---------- ChatGPT sign-in ----------

function renderGpt(st) {
  st = st || {};
  if (config && st.connected !== undefined) config.chatgpt = st;
  const btn = $("#gptBtn");
  btn.classList.toggle("on", !!st.connected);
  $("#gptLabel").textContent = st.connected ? `ChatGPT${st.email ? " · " + st.email : ""}` : st.login_pending ? "Ждём вход…" : "Войти через ChatGPT";
  btn.title = st.connected ? "Промпты пишет GPT по вашей подписке. Нажмите, чтобы выйти." : "Промпты пишет GPT по вашей подписке ChatGPT";
  $("#gptLine").innerHTML = st.connected
    ? `Промпт пишет <b>GPT</b> по вашей подписке ChatGPT${st.model ? ` (${escapeHtml(st.model)})` : ""}.`
    : "Промпт соберётся по встроенному шаблону. <b>Войдите через ChatGPT</b> вверху — тогда его напишет GPT под ваше фото.";
}

$("#gptBtn").onclick = async () => {
  const st = await api("/api/chatgpt").catch(() => ({}));
  if (st.connected) {
    if (!confirm("Выйти из ChatGPT? Промпты будут собираться по встроенному шаблону.")) return;
    renderGpt(await api("/api/chatgpt/logout", { method: "POST" }));
    return;
  }
  try {
    const { url } = await api("/api/chatgpt/login", { method: "POST" });
    window.open(url, "_blank", "noopener");
    renderGpt({ login_pending: true });
    const started = Date.now();
    const poll = async () => {
      const now = await api("/api/chatgpt").catch(() => ({}));
      renderGpt(now);
      if (now.login_error) return hint(now.login_error, true);
      if (now.connected) return hint("ChatGPT подключён — промпты пишет GPT.");
      if (Date.now() - started < 300000) setTimeout(poll, 2000);
      else renderGpt(now), hint("Вход не завершён. Нажмите «Войти через ChatGPT» ещё раз.", true);
    };
    setTimeout(poll, 2000);
  } catch (e) { hint(e.message, true); }
};

// Image model pills (Dreamina images).
function syncImageModels() {
  const models = config.image_models || {};
  const names = Object.keys(models);
  const sel = $("#imageModelSelect");
  const keep = names.includes(sel.value) ? sel.value : names.includes(config.settings.DREAMINA_IMAGE_MODEL) ? config.settings.DREAMINA_IMAGE_MODEL : names[0];
  sel.innerHTML = names.map(n => `<option value="${n}">${escapeHtml(models[n])}</option>`).join("");
  if (keep) sel.value = keep;
  $("#imageModelPills").innerHTML = names.map(n => `<button type="button" data-value="${n}" class="${n === keep ? "on" : ""}">${escapeHtml(models[n])}</button>`).join("");
  $$("#imageModelPills button").forEach(b => b.onclick = () => {
    sel.value = b.dataset.value;
    $$("#imageModelPills button").forEach(x => x.classList.toggle("on", x === b));
    updateSummary();
  });
}

// Model pills for providers that offer several (Dreamina).
function syncModels() {
  const p = config.providers.find(x => x.name === $("#providerSelect").value);
  const models = (p && p.models) || {};
  const names = Object.keys(models);
  $("#modelField").hidden = !names.length;
  const ms = $("#modelSelect");
  const keep = names.includes(ms.value) ? ms.value : names.includes(config.settings.DREAMINA_MODEL) ? config.settings.DREAMINA_MODEL : names[0];
  ms.innerHTML = names.map(n => `<option value="${n}">${escapeHtml(models[n])}</option>`).join("");
  if (keep) ms.value = keep;
  $("#modelPills").innerHTML = names.map(n => `<button type="button" data-value="${n}" title="${escapeHtml(models[n])}" class="${n === keep ? "on" : ""}">${escapeHtml(SHORT_MODEL[n] || n)}</button>`).join("");
  $$("#modelPills button").forEach(b => b.onclick = () => {
    ms.value = b.dataset.value;
    $$("#modelPills button").forEach(x => x.classList.toggle("on", x === b));
    modelWarning();
    updateSummary();
  });
  modelWarning();
}
$("#providerSelect").addEventListener("change", () => { syncModels(); updateSummary(); });

function currentTemplate() {
  return templateList.find(x => x.id === $("#form").template.value) || {};
}

// Dreamina's Seedance 2.0 family rejects photos of real faces; 2.5 on the same subscription accepts them.
function modelWarning() {
  const warn = $("#modelWarn");
  const model = $("#modelSelect").value;
  const withFace = currentTemplate().photos !== "object";
  const show = mode === "video" && !$("#modelField").hidden && $("#providerSelect").value === "dreamina" && model && model !== "seedance-2.5" && withFace;
  warn.hidden = !show;
  if (show) warn.textContent = "Seedance 2.0 в Dreamina отклоняет фото реальных лиц. Для своего лица выберите Seedance 2.5 — та же подписка. 2.0 подходит для замены объекта.";
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

// ---------- pills bound to hidden selects ----------

$$(".pills[data-for]").forEach(group => group.addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  $(`select[name=${group.dataset.for}]`).value = b.dataset.value;
  $$("button", group).forEach(x => x.classList.toggle("on", x === b));
  updateSummary();
}));

function updateSummary() {
  if (!config) return;
  const f = $("#form");
  if (mode === "photo") {
    const n = +f.count.value;
    const parts = [(config.image_models || {})[f.image_model.value] || "", f.pquality.value === "4k" ? "4K" : "стандарт",
      n === 1 ? "1 вариант" : `${n} варианта`, config.chatgpt && config.chatgpt.connected ? "промпт: GPT" : "промпт: шаблон"];
    $("#summary").textContent = parts.filter(Boolean).join(" · ");
    return;
  }
  const p = config.providers.find(x => x.name === f.provider.value);
  const parts = [];
  if (p) parts.push(!$("#modelField").hidden ? SHORT_MODEL[f.model.value] || f.model.value : p.label.split("·")[0].trim());
  parts.push(f.quality.value === "hd" ? "1080p" : "720p");
  const takes = +f.takes.value;
  parts.push(takes === 1 ? "1 дубль" : `${takes} дубля`);
  if (videoSeconds) parts.push(`${Math.min(videoSeconds, 30).toFixed(0)} с видео`);
  $("#summary").textContent = parts.join(" · ") + (takes > 1 ? " — оплачивается каждый дубль" : "");
}

// ---------- uploads ----------

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
  if (video.src) URL.revokeObjectURL(video.src);
  video.src = URL.createObjectURL(file);
  video.hidden = false;
  $(".drop-empty", drop).hidden = true;
  drop.classList.add("filled");
  video.onloadedmetadata = () => {
    video.play().catch(() => {});
    const d = video.duration;
    videoSeconds = isFinite(d) ? d : null;
    meta.hidden = false;
    meta.className = "drop-meta";
    meta.textContent = `${d.toFixed(1)} с · ${video.videoWidth}×${video.videoHeight}`;
    if (d > 30) { meta.textContent += " · возьму первые 30 с"; meta.classList.add("warn"); }
    if (d < 4) { meta.textContent += " · коротко, лучше от 4 с"; meta.classList.add("warn"); }
    updateSummary();
  };
});

setupDrop($("#sceneDrop"), file => {
  const drop = $("#sceneDrop");
  const img = $(".drop-preview", drop);
  if (img.src) URL.revokeObjectURL(img.src);
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $(".drop-empty", drop).hidden = true;
  drop.classList.add("filled");
});

$$(".photo-drop").forEach(drop => setupDrop(drop, file => {
  const img = $(".drop-preview", drop);
  if (img.src) URL.revokeObjectURL(img.src);
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
  $("#extrasLabel").textContent = n ? `Доп. ракурсов: ${n} (нажмите, чтобы заменить)` : "+ Доп. ракурсы (сбоку, сзади)";
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
const OUTLINE = ["cube", "film"];

async function loadTemplates() {
  templateList = await api("/api/templates");
  renderTemplateCards();
  pickTemplate(savedTemplate() || "hero");
  gallerySignature = "";
  loadHistory();
}

function renderTemplateCards() {
  if (!templateList.length) return;
  const current = $("#form").template.value;
  $("#templates").innerHTML = templateList.map(t => {
    const sub = mode === "photo" ? t.photo_subtitle || t.subtitle : t.subtitle;
    return `<button type="button" class="tpl${t.id === current ? " on" : ""}" data-id="${t.id}" style="--g: linear-gradient(150deg, ${t.colors[0]}, ${t.colors[1]})" title="${escapeHtml(sub)}">
      <svg viewBox="0 0 24 24" width="22" height="22" fill="${OUTLINE.includes(t.glyph) ? "none" : "currentColor"}" stroke="${OUTLINE.includes(t.glyph) ? "currentColor" : "none"}" stroke-width="1.6"><path d="${GLYPHS[t.glyph] || GLYPHS.spark}"/></svg>
      <b>${escapeHtml(t.title)}</b><small>${escapeHtml(sub)}</small>
    </button>`;
  }).join("");
  $$(".tpl").forEach(b => b.onclick = () => pickTemplate(b.dataset.id));
}

function savedTemplate() {
  try { return localStorage.getItem("genjutsu.template"); } catch { return null; }
}

function pickTemplate(id) {
  const t = templateList.find(x => x.id === id) || templateList[0];
  if (!t) return;
  $("#form").template.value = t.id;
  try { localStorage.setItem("genjutsu.template", t.id); } catch {}
  $$(".tpl").forEach(b => b.classList.toggle("on", b.dataset.id === t.id));
  $("#tplSubtitle").textContent = t.title;
  $("#tplField").hidden = !t.input;
  $("#tplInput").required = !!(t.input && t.input.required);
  if (t.input) {
    $("#tplLabel").textContent = t.input.label;
    $("#tplInput").placeholder = t.input.placeholder;
  }
  const object = t.photos === "object";
  $("#objectPhotos").hidden = !object;
  $("#personPhotos").hidden = object;
  $("#extrasPick").hidden = object;
  $("#photosTitle").textContent = object ? "Фото объекта" : "Ваши фото";
  $("#photosHint").textContent = object ? "предмет, которым заменить" : "лицо и полный рост — лучше оба";
  const what = mode === "photo" ? "фото" : "видео";
  $(".consent span").textContent = object
    ? `У меня есть права на этот материал и фото объекта.`
    : `На фото я или человек, который дал согласие на такое ${what}.`;
  modelWarning();
}

// ---------- submit ----------

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target;
  if (mode === "photo" && !form.scene.files[0]) return hint("Загрузите фото-референс.", true);
  if (mode === "video" && !form.video.files[0]) return hint("Загрузите видео-референс.", true);
  const tpl = currentTemplate();
  if (tpl.photos === "object") {
    if (!form.object.files[0]) return hint("Загрузите фото объекта.", true);
  } else if (!form.face.files[0] && !form.body.files[0]) return hint("Загрузите хотя бы одно фото: лицо или в полный рост.", true);
  if (tpl.input && tpl.input.required && !form.tpl_input.value.trim()) return hint(`Заполните «${tpl.input.label}».`, true);
  if (!form.consent.checked) return hint("Отметьте согласие внизу.", true);
  const data = new FormData(form);
  // Only the photos the chosen template uses are sent.
  for (const k of ["face", "body", "object"]) {
    const used = tpl.photos === "object" ? k === "object" : k !== "object";
    if (!form[k].files[0] || !used) data.delete(k);
  }
  if (tpl.photos === "object" || !form.extras.files.length) data.delete("extras");
  if ($("#modelField").hidden) data.delete("model");
  if (!tpl.input) data.delete("tpl_input");
  let endpoint = "/api/jobs";
  if (mode === "photo") {
    endpoint = "/api/photo-jobs";
    data.set("quality", form.pquality.value);
    for (const k of ["video", "model", "takes", "masking", "audio", "review", "face_mesh", "provider", "pquality"]) data.delete(k);
  } else {
    for (const k of ["scene", "image_model", "count", "pquality", "hide"]) data.delete(k);
  }
  const btn = $("#submitBtn");
  btn.disabled = true;
  hint("Загружаю файлы…");
  try {
    const job = await api(endpoint, { method: "POST", body: data });
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
  setView("job");
  $("#resultVideo").removeAttribute("src");
  $("#resultVideo").dataset.src = "";
  refreshJob();
  window.scrollTo({ top: 0, behavior: "smooth" });
  history.replaceState(null, "", "#" + id);
  $$(".card").forEach(x => x.classList.toggle("active", x.dataset.id === id));
}

function closeJob() {
  currentJob = null;
  clearTimeout(jobTimer);
  $("#resultVideo").pause();
  setView("gallery");
  history.replaceState(null, "", location.pathname);
  $$(".card").forEach(x => x.classList.remove("active"));
}
$("#backBtn").onclick = closeJob;

const download = (job, name) => `/files/${job.id}/${name}?download=1`;

async function refreshJob() {
  clearTimeout(jobTimer);
  if (!currentJob) return;
  const id = currentJob;
  let job;
  try { job = await api(`/api/jobs/${id}`); } catch (e) {
    if (/не найдена/.test(e.message)) return closeJob();
    jobTimer = setTimeout(refreshJob, 4000);
    return;
  }
  if (id !== currentJob) return;
  render(job);
  const hdRunning = job.hd && job.hd.status === "running";
  if (["queued", "running", "generating"].includes(job.status) || hdRunning) jobTimer = setTimeout(refreshJob, 2000);
}

function render(job) {
  const photoJob = job.kind === "photo";
  const steps = photoJob ? PHOTO_STEPS : VIDEO_STEPS;
  const stepsBox = $("#jobSteps");
  if (stepsBox.dataset.kind !== (job.kind || "video")) {
    stepsBox.innerHTML = steps.map(([k, t]) => `<li data-k="${k}">${t}</li>`).join("");
    stepsBox.dataset.kind = job.kind || "video";
  }
  $("#jobId").textContent = job.id;
  $("#jobStage").textContent = job.stage || STATUS_TEXT[job.status];
  $("#jobPct").textContent = job.status === "done" ? "100%" : `${job.progress || 0}%`;
  $("#jobBar").style.width = `${job.status === "done" ? 100 : job.progress || 0}%`;
  $(".bar").classList.toggle("live", ["running", "generating"].includes(job.status));

  const keys = steps.map(([k]) => k);
  const nowIdx = job.status === "done" ? keys.length : job.status === "review" ? keys.indexOf("upload") : keys.indexOf(job.stage_key);
  $$("#jobSteps li").forEach((li, i) => {
    li.classList.toggle("done", i < nowIdx || job.status === "done");
    li.classList.toggle("now", i === nowIdx && job.status !== "done" && !["failed", "canceled"].includes(job.status));
  });

  const err = $("#jobError");
  err.hidden = !job.error;
  err.textContent = job.error || "";

  const notices = [...(job.notices || []), ...((job.qa && job.qa.issues) || [])];
  if (job.used && !photoJob) notices.push(`Маска: ${({ cloud: "SAM 3 + глубина (облако)", local: "MediaPipe (локально)", none: "без маски" })[job.used.masking]} · звук для модели: ${({ vocals: "голос", full: "весь звук", none: "нет" })[job.used.audio]}${job.used.face_mesh ? " · сетка лица: да" : ""}`);
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
  setSrc($("#sheetImg"), `/files/${job.id}/sheet.jpg?v=${encodeURIComponent((job.refs || []).map(r => r.send).join(","))}`);

  $("#resultBox").hidden = photoJob || !job.result;
  renderPhoto(job);
  if (photoJob) {
    $("#jobPrompt").textContent = job.prompt || "Промпт появится после подготовки фото.";
    $("#jobLog").innerHTML = (job.log || []).slice().reverse().map(l => `<li>${l.t.slice(11, 19)} · ${escapeHtml(l.msg)}</li>`).join("");
    if (job.guide) setSrc($("#previewImg"), `/files/${job.id}/${job.guide}`);
    $("#previewFig").hidden = !job.masked;
    $("#previewFig figcaption").textContent = "Фото-референс: человек скрыт силуэтом, поза и черты лица — линиями";
    return;
  }
  $("#previewFig figcaption").textContent = "Оригинал / что видит модель";
  renderTakes(job);
  if (job.result) {
    const hdDone = job.hd && job.hd.status === "done";
    setSrc($("#resultVideo"), `/files/${job.id}/${hdDone ? job.hd.result : job.result}`);
    $("#downloadBtn").href = download(job, job.result);
    $("#downloadBtn").textContent = job.options.quality === "hd" ? "Скачать 1080p" : "Скачать черновик";
    $("#hdBtn").hidden = !(job.options.quality === "draft" && !job.hd);
    $("#downloadHdBtn").hidden = !hdDone;
    if (hdDone) $("#downloadHdBtn").href = download(job, job.hd.result);
    $("#hdHint").textContent = job.hd ? ({ running: "1080p генерируется…", done: "1080p готово", failed: "1080p: " + (job.hd.error || "ошибка") })[job.hd.status] || "" : "";
  }

  $("#jobPrompt").textContent = job.prompt || "Промпт появится перед отправкой.";
  $("#jobLog").innerHTML = (job.log || []).slice().reverse().map(l => `<li>${l.t.slice(11, 19)} · ${escapeHtml(l.msg)}</li>`).join("");
}

let photoShown = null;

function renderPhoto(job) {
  const box = $("#photoResult");
  const images = job.kind === "photo" ? job.images || [] : [];
  box.hidden = !images.length;
  if (!images.length) return;
  if (!images.some(im => im.file === photoShown)) photoShown = images[0].file;
  const shown = images.find(im => im.file === photoShown);
  setSrc($("#photoMain"), `/files/${job.id}/${shown.file}`);
  $("#photoDownload").href = `/files/${job.id}/${shown.file}?download=1`;
  const sc = shown.score;
  $("#photoScore").hidden = !(sc && sc.face !== undefined);
  if (sc) $("#photoScore").innerHTML = sc.face === false ? "Лицо на этом варианте не найдено" : `Сходство лица с вашими фото: <b class="pct ${pctClass(sc.score)}">${sc.score}%</b>`;
  $("#photoSummary").hidden = !job.summary;
  $("#photoSummary").textContent = job.summary || "";
  const idle = job.status === "done";
  $("#moreBtn").disabled = !idle;
  $("#refineBtn").disabled = !idle || !(config && config.chatgpt && config.chatgpt.connected);
  $("#refineBtn").title = $("#refineBtn").disabled && idle ? "Войдите через ChatGPT, чтобы GPT мог сравнить результат" : "GPT сравнит этот вариант с вашими фото и исправит промпт";
  $("#editPromptBtn").disabled = !idle;
  if (!idle) $("#promptEdit").hidden = true;
  $("#promptBy").textContent = ({ gpt: "Промпт написал GPT", "gpt-review": "Промпт исправил GPT", user: "Ваш промпт", builtin: "Промпт по шаблону" })[job.prompt_by] || "";
  $("#photoThumbs").innerHTML = images.length < 2 ? "" : images.map(im => {
    const p = im.score && im.score.face ? `<span class="pct ${pctClass(im.score.score)}">${im.score.score}%</span>` : "";
    return `<button type="button" class="${im.file === photoShown ? "on" : ""}" data-f="${escapeHtml(im.file)}"><img src="/files/${job.id}/${escapeHtml(im.file)}" alt="" loading="lazy">${p}</button>`;
  }).join("");
  $$("#photoThumbs button").forEach(b => b.onclick = () => { photoShown = b.dataset.f; renderPhoto(job); });
}

$("#moreBtn").onclick = () => jobAction("more", "Сгенерировать ещё вариант с тем же промптом? Это спишет кредиты Dreamina.");
$("#refineBtn").onclick = () => jobAction(`refine?image=${encodeURIComponent(photoShown || "")}`,
  "GPT сравнит показанный вариант с вашими фото, исправит промпт и нарисует новый вариант. Это спишет кредиты Dreamina.");
$("#editPromptBtn").onclick = async () => {
  const job = await api(`/api/jobs/${currentJob}`);
  $("#promptText").value = job.prompt || "";
  $("#promptEdit").hidden = false;
  $("#promptText").focus();
};
$("#promptCancel").onclick = () => { $("#promptEdit").hidden = true; };
$("#regenBtn").onclick = async () => {
  if (!confirm("Нарисовать новый вариант с этим промптом? Это спишет кредиты Dreamina.")) return;
  try {
    render(await api(`/api/jobs/${currentJob}/regenerate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: $("#promptText").value, count: 1 }) }));
    $("#promptEdit").hidden = true;
    refreshJob();
  } catch (e) { alert(e.message); }
};

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
    try { render(await api(`/api/jobs/${job.id}/choose/${i}`, { method: "POST" })); gallerySignature = ""; loadHistory(); } catch (e) { alert(e.message); }
  });
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

// ---------- gallery ----------

async function loadHistory() {
  let list;
  try { list = await api("/api/jobs"); } catch { return; }
  $("#galleryCount").textContent = list.length || "";
  // Rebuild only when something changed, so a hover preview isn't cut off every 5 s.
  const signature = JSON.stringify(list.map(j => [j.id, j.status, j.progress, j.chosen, j.result, j.hd && j.hd.status]));
  if (signature === gallerySignature) return;
  gallerySignature = signature;
  $("#emptyState").hidden = list.length > 0;
  const running = list.filter(j => ["queued", "running", "generating"].includes(j.status)).length;
  $("#galleryHint").textContent = running ? `в работе: ${running}` : list.length ? `всего: ${list.length}` : "";
  $("#historyList").innerHTML = list.map(j => {
    const live = ["queued", "running", "generating"].includes(j.status) || (j.hd && j.hd.status === "running");
    const st = j.hd && j.hd.status === "running" ? "1080p…" : STATUS_TEXT[j.status] || j.status;
    const cls = live ? "run" : j.status;
    const photoJob = j.kind === "photo";
    const img = photoJob ? (j.result ? `thumb.jpg?r=${encodeURIComponent(j.result)}` : "scene.jpg") : j.result ? `thumb.jpg?c=${j.chosen}` : "sheet.jpg";
    const time = new Date(j.created).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
    const title = (templateList.find(t => t.id === j.options.template) || {}).title || "Замена";
    const model = photoJob ? ` · фото` : j.options.model ? ` · ${SHORT_MODEL[j.options.model] || j.options.model}` : "";
    return `<button type="button" class="card${j.id === currentJob ? " active" : ""}" data-id="${j.id}" data-video="${j.result && !photoJob ? `/files/${j.id}/${j.result}` : ""}">
      <div class="thumb"><img src="/files/${j.id}/${img}" alt="" loading="lazy"><span class="status ${cls}">${st}</span>
      ${live ? `<div class="mini"><i style="width:${j.progress || 0}%"></i></div>` : ""}</div>
      <div class="meta"><b>${escapeHtml(title)}</b><span>${time}${escapeHtml(model)}</span></div></button>`;
  }).join("");
  $$(".card").forEach(card => {
    card.onclick = () => openJob(card.dataset.id);
    if (!card.dataset.video) return;
    // Hover plays the clip, like a preset gallery.
    card.addEventListener("mouseenter", () => {
      const v = document.createElement("video");
      Object.assign(v, { src: card.dataset.video, muted: true, loop: true, playsInline: true, autoplay: true });
      $(".thumb", card).appendChild(v);
    });
    card.addEventListener("mouseleave", () => $$("video", card).forEach(v => v.remove()));
  });
}

// ---------- boot ----------

setView(location.hash.length > 1 ? "job" : "create");
loadConfig().catch(e => hint("Сервер недоступен: " + e.message, true));
loadTemplates().catch(e => hint("Шаблоны не загрузились: " + e.message, true));
loadHistory();
setInterval(loadHistory, 5000);
if (location.hash.length > 1) openJob(location.hash.slice(1));
