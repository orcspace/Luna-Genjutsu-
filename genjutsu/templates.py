"""Ten one-click templates modelled on what Higgsfield Genjutsu offers (character swap, object
swap, outfit and location changes, restyles, cartoon to live action). The recipes are our own:
Higgsfield does not publish its preset list.

A template fixes the mode and the parts of the prompt that make the look; the user only adds
the photos and, for some templates, one line of text.
"""

TEMPLATES = [
    {
        "id": "hero",
        "photo_subtitle": "Вы вместо человека на фото. Поза, фон и свет как в оригинале.",
        "title": "Я в главной роли",
        "subtitle": "Вы вместо человека в кадре. Движение, фон и звук как в оригинале.",
        "mode": "swap",
        "glyph": "person",
        "colors": ["#7c5cf0", "#c084fc"],
    },
    {
        "id": "object",
        "photo_subtitle": "Предмет на фото меняется на ваш. Люди и фон остаются.",
        "title": "Замена объекта",
        "subtitle": "Товар, предмет или машина в кадре меняется на ваш. Люди и движение остаются.",
        "mode": "object",
        "masking": "none",
        "photos": "object",
        "input": {"name": "who", "label": "Что заменить (англ.)", "placeholder": "the phone in her hand", "required": True},
        "glyph": "box",
        "colors": ["#0ea5e9", "#22d3ee"],
    },
    {
        "id": "outfit",
        "photo_subtitle": "Вы на фото, но в другой одежде — опишите наряд.",
        "title": "Новый образ",
        "subtitle": "Вы в кадре, но в другой одежде — опишите наряд.",
        "mode": "swap",
        "input": {"name": "extra", "label": "Наряд (англ.)", "placeholder": "a white linen suit and sneakers", "required": True,
                  "wrap": "Dress the person in {value} instead of the outfit in the reference photos; keep the face and hair from the references."},
        "glyph": "shirt",
        "colors": ["#ec4899", "#f9a8d4"],
    },
    {
        "id": "location",
        "photo_subtitle": "Та же поза и ракурс, но в другом месте — опишите где.",
        "title": "Новая локация",
        "subtitle": "То же движение и камера, но в другом месте — опишите где.",
        "mode": "motion",
        "input": {"name": "scene", "label": "Где (англ.)", "placeholder": "a rooftop in Tokyo at night, light rain", "required": True},
        "glyph": "pin",
        "colors": ["#10b981", "#6ee7b7"],
    },
    {
        "id": "anime",
        "photo_subtitle": "Фото в стиле рисованного аниме, вы — герой.",
        "title": "Аниме 90-х",
        "subtitle": "Всё видео в стиле рисованного аниме, вы — его герой.",
        "mode": "swap",
        "style": "1990s Japanese cel anime: clean ink outlines, flat cel shading, hand-painted backgrounds, soft film grain",
        "glyph": "spark",
        "colors": ["#f43f5e", "#fb923c"],
    },
    {
        "id": "clay",
        "photo_subtitle": "Вы и весь кадр — сцена из пластилина.",
        "title": "Пластилин",
        "subtitle": "Стоп-моушен из пластилина: вы и весь мир вылеплены.",
        "mode": "swap",
        "style": "stop-motion claymation: hand-sculpted plasticine figures and sets, visible fingerprints, soft studio lighting, slightly stepped motion",
        "glyph": "blob",
        "colors": ["#f59e0b", "#fde047"],
    },
    {
        "id": "toon3d",
        "photo_subtitle": "Как кадр из полнометражного 3D-мультфильма.",
        "title": "3D-мультфильм",
        "subtitle": "Как кадр из полнометражного 3D-мультфильма.",
        "mode": "swap",
        "style": "feature-film 3D animation: stylised proportions, expressive large eyes, soft subsurface skin, rich global illumination",
        "glyph": "cube",
        "colors": ["#3b82f6", "#a5b4fc"],
    },
    {
        "id": "cinema",
        "photo_subtitle": "Киношный кадр: цветокор, анаморф, зерно плёнки.",
        "title": "Блокбастер",
        "subtitle": "Киношная картинка: цветокор, анаморф, зерно плёнки.",
        "mode": "swap",
        "style": "big-budget cinema look: anamorphic lens, teal-and-orange grade, shallow depth of field, 35mm film grain, photorealistic",
        "photoreal": True,
        "glyph": "film",
        "colors": ["#334155", "#f97316"],
    },
    {
        "id": "cyberpunk",
        "photo_subtitle": "Ночной неоновый город, дождь и отражения.",
        "title": "Неон-киберпанк",
        "subtitle": "Ночной город будущего в неоне, дождь и отражения.",
        "mode": "motion",
        "scene": "a rain-soaked cyberpunk city street at night, neon signs, holograms, reflections on wet asphalt",
        "style": "cyberpunk neon look: magenta and cyan rim light, volumetric haze, photorealistic",
        "photoreal": True,
        "glyph": "bolt",
        "colors": ["#d946ef", "#06b6d4"],
    },
    {
        "id": "live",
        "photo_subtitle": "Рисунок или аниме становится живым фото с вами.",
        "title": "Мульт → живое",
        "subtitle": "Мультяшное или аниме-видео становится живой съёмкой с вами.",
        "mode": "swap",
        "masking": "none",
        "style": "live-action film footage: real locations, real materials, natural light, photorealistic; turn every drawn element into its real-world equivalent",
        "photoreal": True,
        "glyph": "eye",
        "colors": ["#14b8a6", "#a3e635"],
    },
]

BY_ID = {t["id"]: t for t in TEMPLATES}


def public():
    """What the UI needs (no prompt internals)."""
    keys = ("id", "title", "subtitle", "photo_subtitle", "mode", "photos", "input", "glyph", "colors")
    return [{k: t[k] for k in keys if k in t} for t in TEMPLATES]


def apply(template_id, options):
    """Fold a template into the job options. Raises ValueError for a missing required input."""
    t = BY_ID.get(template_id or "hero")
    if t is None:
        raise ValueError(f"Нет шаблона «{template_id}».")
    out = dict(options, template=t["id"], mode=t["mode"])
    if t.get("masking") and options.get("masking", "auto") == "auto":
        out["masking"] = t["masking"]
    if t.get("scene") and not (options.get("scene") or "").strip():
        out["scene"] = t["scene"]
    out["style"] = t.get("style", "")
    out["photoreal"] = t.get("photoreal", not t.get("style"))
    field = t.get("input")
    if field:
        value = (options.get("tpl_input") or "").strip()[:300]
        if field.get("required") and not value:
            raise ValueError(f"Шаблон «{t['title']}»: заполните поле «{field['label']}».")
        if value:
            text = field["wrap"].format(value=value) if field.get("wrap") else value
            if field["name"] == "extra":
                # Keep whatever the user also wrote in "Дополнительно".
                text = f"{text} {options.get('extra') or ''}".strip()
            out[field["name"]] = text
    return out
