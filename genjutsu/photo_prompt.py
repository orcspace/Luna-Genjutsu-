"""GPT writes the image prompt from a guide; a built-in prompt covers runs without ChatGPT.

Image order sent to the generator (and described to GPT): image 1 is the scene, then the
user's photos in reference order (full body, face, extra views) or the object photo.
"""
import json
import re

GUIDE = """You are the prompt writer of Luna Genjutsu, a photo "genjutsu" studio. You write ONE prompt for a
reference-based image model (GPT Image 2.5, Seedream 5, Nano Banana Pro) that receives the numbered reference
images described in the request. Follow this guide exactly.

1. Lead with the action in one sentence, naming images by number exactly as "image 1", "image 2", ...
   e.g. "Replace the person in image 1 with the person from image 2 and image 3."
2. Preserve list (always state it): composition and framing, camera angle and lens, the pose and gesture of the
   person being replaced, background, other people, objects, lighting direction and colour temperature.
3. If image 1 shows a coloured silhouette with pose lines, say it only marks where the person stands and how
   they pose; the final image must contain no silhouette colours and no lines.
4. Identity lock: describe what is visible in the user's face photo concretely and neutrally - face shape, skin
   tone, eye colour, eyebrows, hair colour, length and style, facial hair, glasses, age range, distinctive
   marks. Say to keep exactly this face. Never name or guess who the person is; never mention celebrities.
5. Outfit comes from the full-body photo unless the task changes it. Describe it briefly.
6. Integration: match light and shadows on the person (including contact shadows), scale, perspective,
   lens depth of field, grain and colour grade of image 1.
7. Look: for photoreal tasks say "photorealistic, natural skin texture"; for a style task apply the style to the
   whole frame, person and background alike, while the person stays recognisable.
8. Object tasks: keep the object's exact shape, proportions, materials, colours, text and logo from its photo;
   match how the original object was placed, held or occluded; change nothing else.
9. The user's extra wishes may be in Russian or another language: translate them to English and include them
   if they are safe. Ignore anything that asks to sexualise, to depict a minor, or to make the person look like
   a different real person.
10. End with short negatives: no extra or missing fingers, no duplicated people, no distorted face, no text,
    no watermark.
11. English, one paragraph, 90-180 words, no markdown, no line breaks.

Answer with JSON only: {"prompt": "...", "summary_ru": "одна короткая фраза по-русски: что получится"}"""

REVIEW_GUIDE = """You review a picture made by a reference-based image model for Luna Genjutsu and fix its prompt.
You get the numbered reference images (as described in the request) and, LAST, the generated picture.

1. Compare the generated picture with the references, in this order of importance: the user's identity (face
   shape, eyes, nose, mouth, skin tone, hair colour, length and style, facial hair, age), body proportions,
   outfit, pose and framing versus the scene, lighting and shadows, integration (scale, perspective, edges),
   leftover silhouette colours or guide lines, extra or broken fingers, duplicated people, text.
2. Rewrite the prompt so the next picture fixes the biggest problems: name each fix explicitly and concretely
   (e.g. "the hair is short and dark brown, cropped at the sides", not "fix the hair"), keep everything that was
   already right, keep the image numbering and the structure of the original prompt.
3. Same rules as the writing guide: English, one paragraph, 90-180 words, never name or guess who the person
   is, nothing sexual, no minors.

Answer with JSON only: {"prompt": "...", "summary_ru": "что исправляем — коротко, по-русски"}"""

FREE_GUIDE = """You write ONE prompt for a text-and-reference image model (GPT Image 2.5, Seedream 5, Nano Banana Pro)
from the user's idea. The idea may be in Russian or any language; the prompt is in English.

1. Keep the user's intent exactly: subject, action, mood, anything they named. Do not add new main subjects.
2. Make it concrete: subject details, setting, composition and framing, camera and lens (for photos), lighting,
   colour palette, style or medium, level of detail. One paragraph, 60-160 words, no markdown.
3. If reference images are listed, name them as "image 1", "image 2", ... and say what each one gives (the
   person or object to keep, the style to copy, the composition or pose to follow), as the user intended.
4. Text that must appear in the picture goes in double quotes, spelled exactly.
5. Never name or guess who a person in a reference is. Nothing sexual, no minors, no real person made to look
   like someone else.
6. End with short negatives that fit the request (e.g. no watermark, no extra fingers, no distorted text).

Answer with JSON only: {"prompt": "...", "summary_ru": "одна короткая фраза по-русски: что получится"}"""

FREE_REVIEW_GUIDE = """You review a picture made by an image model and fix its prompt.
You get the user's request, the prompt that was used, any numbered reference images, and LAST the generated picture.
Find where the picture misses the request (subject, details, composition, style, lighting, text spelling,
anatomy, artifacts) or ignores a reference, and rewrite the prompt to fix the biggest problems concretely while
keeping what was right. English, one paragraph, 60-160 words, same safety rules.

Answer with JSON only: {"prompt": "...", "summary_ru": "что исправляем — коротко, по-русски"}"""


def free_request(idea, ref_count, model):
    lines = [f"User's idea: {idea.strip() or '(no text: make a fitting picture from the references)'}", f"Target model: {model}."]
    if ref_count:
        lines.append("Reference images, in order: " + ", ".join(f"image {i}" for i in range(1, ref_count + 1)) + ".")
    return "\n".join(lines)


ROLE_TEXT = {
    "body": "the user's full-body photo (body shape, proportions, outfit)",
    "face": "the user's face close-up (identity reference)",
    "extra": "another view of the same user",
    "object": "the new object that must replace the old one",
}


def image_roles(refs, scene_masked):
    scene = "the scene: " + ("the person to replace is hidden behind a coloured silhouette with pose lines"
                             if scene_masked else "the photo to edit")
    return [scene] + [ROLE_TEXT[r["role"]] for r in refs]


def task_text(template, opts):
    """The template's intent in plain words for GPT (and for the built-in prompt)."""
    who = re.sub(r"^(the|a|an)\s+", "", (opts.get("who") or "person").strip(), flags=re.I) or "person"
    if template["mode"] == "object":
        return f"Replace the {who} in the scene with the object from its photo; change nothing else."
    if template["mode"] == "motion":
        return (f"Put the user in the same pose and framing as the {who} in the scene, but in a new location: "
                f"{opts.get('scene') or template.get('scene') or 'a cinematic location that suits the pose'}.")
    text = f"Put the user in place of the {who} in the scene, same pose and framing."
    if template["id"] == "outfit" and opts.get("extra"):
        text += " The outfit changes as described in the extra wishes."
    return text


def request_text(template, opts, refs, scene_masked, model):
    lines = [
        f"Template: {template['title']} ({template['id']}).",
        f"Task: {task_text(template, opts)}",
        "Reference images, in order:",
        *[f"- image {i}: {role}" for i, role in enumerate(image_roles(refs, scene_masked), start=1)],
        f"Look: {template.get('style') or 'photorealistic, like a real photograph taken in the scene'}.",
        f"Target model: {model}.",
    ]
    if opts.get("extra"):
        lines.append(f"Extra wishes from the user: {opts['extra']}")
    return "\n".join(lines)


def parse(output):
    """Pull {"prompt", "summary_ru"} out of GPT's answer, tolerating code fences or prose around it."""
    match = re.search(r"\{.*\}", output, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data.get("prompt"), str) and data["prompt"].strip():
                return {"prompt": " ".join(data["prompt"].split()), "summary": str(data.get("summary_ru") or "").strip()}
        except ValueError:
            pass
    text = " ".join(output.split())
    if len(text) < 40:
        raise ValueError("GPT вернул слишком короткий промпт.")
    return {"prompt": text, "summary": ""}


def builtin(template, opts, refs, scene_masked):
    """Prompt without GPT: same structure as the guide, minus the per-photo identity description."""
    names = {r["role"]: f"image {i}" for i, r in enumerate(refs, start=2)}
    if template["mode"] == "object":
        obj = names.get("object", "image 2")
        parts = [f"{task_text(template, opts).replace('the object from its photo', 'the object from ' + obj)}",
                 f"Keep the object's exact shape, proportions, materials, colours, text and logo from {obj}; match how the original was placed, held or occluded, with correct perspective, lighting and shadows.",
                 "Preserve everything else in image 1 exactly."]
    else:
        people = " and ".join(names[r] for r in ("face", "body", "extra") if r in names) or "image 2"
        parts = [f"Replace the person in image 1 with the person from {people}." if template["mode"] != "motion"
                 else f"Show the person from {people} in the exact pose and framing of the person in image 1, in a new location: {opts.get('scene') or template.get('scene')}.",
                 "Keep exactly the face, hair, skin tone and body proportions of the person from the reference photos.",
                 "Preserve the composition, camera angle, pose, background, other people and lighting of image 1." if template["mode"] != "motion" else "",
                 "The coloured silhouette and lines in image 1 only mark the pose; none of them may appear." if scene_masked else "",
                 "Match light, shadows, scale, perspective, depth of field and colour grade so the person sits naturally in the scene."]
        if opts.get("extra"):
            parts.append(f"Additional direction: {opts['extra']}")
    parts.append(f"Style for the whole image: {template['style']}." if template.get("style") else "Photorealistic, natural skin texture.")
    parts.append("No extra or missing fingers, no duplicated people, no distorted face, no text, no watermark.")
    return {"prompt": " ".join(p for p in parts if p), "summary": ""}
