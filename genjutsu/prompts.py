"""Prompt assembly.

Order matters for Seedance: what must stay (motion, camera, timing) comes first, then the
replacement, then the negatives. Adapted from the MiniMax H3 Character Swap LoRA card prompt and
the Genjustsu open-source workflow's reference labelling.
"""

ROLES = {
    "body": "shows the full body: body shape, proportions, outfit and shoes",
    "face": "is a face close-up: keep exactly this face, skin tone, eyes, hairstyle and hair colour",
    "extra": "is an additional view of the same person",
}

NEGATIVE = (
    "Do not show the reference photos or their backgrounds. No face morphing, no identity drift, "
    "no extra limbs or fingers, no flicker, no text, no subtitles, no watermark."
)


def build(mode, tags, image_roles, who="person", masked=True, lipsync=False, scene="", extra="", pose_lines=False, face_mesh=False):
    """tags: {'video': '@Video1', 'image': lambda i: f'@Image{i}'} for the provider's syntax."""
    v = tags["video"]
    who = (who or "person").strip()
    refs = " ".join(f"{tags['image'](i)} {ROLES[role]}." for i, role in enumerate(image_roles, start=1))
    sync = f" Sync the lips and facial expressions to the speech in the audio embedded in {v}." if lipsync else ""
    lines = ""
    if masked and pose_lines:
        lines += (" Coloured pose lines inside the silhouette mark the arms, legs and torso; follow them for the limb positions.")
    if masked and face_mesh:
        lines += (" Thin lines on the head trace the eyes, eyebrows and lips (pink) of the original performance: follow them for "
                  "head turns, blinks, expressions and mouth shapes.")
    if lines:
        lines += " All these lines are guidance only and must not appear in the output."

    if mode == "motion":
        parts = [
            f"Use {v} only as the motion reference: copy its choreography, body movement, timing, camera movement and framing exactly."
            + (f" The performer in {v} is hidden behind a coloured silhouette." + lines if masked else ""),
            f"Create a new scene: {scene.strip() or 'a cinematic setting that fits the movement'}.",
            f"The performer is the person from the reference images. {refs}",
            "Keep their identity, face, hair, body proportions and outfit consistent in every frame. Photorealistic, natural lighting that matches the new scene." + sync,
        ]
    elif masked:
        parts = [
            f"Replace only the {who} in {v} with the person from the reference images.",
            f"In {v} the {who} is shown as a coloured depth silhouette. It is motion guidance only: the output must show the real person "
            "from the references in its place, fully photorealistic, with no depth colours, outlines or silhouette left anywhere." + lines,
            refs,
            "Keep the person's identity, face, hair, body proportions and outfit from the references. "
            f"Preserve the camera movement, background, lighting, objects and all other people of {v}. "
            "Match the silhouette's position, scale, pose, movement and timing frame by frame. "
            "Light the person so they sit naturally in the scene." + sync,
        ]
    else:
        parts = [
            f"Replace only the {who} in {v} with the person from the reference images.",
            refs,
            "Keep the replacement person's identity, face, hair, body proportions and outfit from the references. "
            f"Preserve the camera, background, lighting, objects and all other people of {v}. "
            "Match the target person's position, scale, pose, movement and timing." + sync,
        ]
    if extra.strip():
        parts.append("Additional direction: " + extra.strip())
    parts.append(NEGATIVE)
    return "\n".join(p for p in parts if p)
