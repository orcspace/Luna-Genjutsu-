"""Reference photos: upright, RGB, sensible size, plus a character sheet for previews."""
from PIL import Image, ImageOps

try:  # iPhone photos (HEIC/HEIF)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

MAX_SIDE = 2048
MIN_SIDE = 256


class PhotoError(ValueError):
    pass


def prepare(src, dst):
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
    except Exception:
        raise PhotoError("Не удалось открыть фото. Подойдут JPG, PNG, WebP или HEIC.")
    if min(img.size) < MIN_SIDE:
        raise PhotoError(f"Фото слишком маленькое ({img.size[0]}×{img.size[1]}). Нужно хотя бы {MIN_SIDE} px по короткой стороне.")
    if img.mode in ("RGBA", "LA", "P"):
        # Transparent cut-outs go on plain white, which Seedance reads as "no background".
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    img.save(dst, "JPEG", quality=95)
    return {"width": img.size[0], "height": img.size[1]}


def sheet(paths, dst, height=1024):
    """Full body, face and extras side by side on white."""
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = im.resize((max(1, round(im.width * height / im.height)), height), Image.Resampling.LANCZOS)
        imgs.append(im)
    gap = 24
    out = Image.new("RGB", (sum(i.width for i in imgs) + gap * (len(imgs) + 1), height + 2 * gap), (255, 255, 255))
    x = gap
    for im in imgs:
        out.paste(im, (x, gap))
        x += im.width + gap
    out.save(dst, "JPEG", quality=92)
