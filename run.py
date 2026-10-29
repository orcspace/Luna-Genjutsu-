"""Start Luna Genjutsu: python run.py  →  http://127.0.0.1:8770"""
import threading
import webbrowser

import uvicorn

from genjutsu import hosting
from genjutsu.config import setting


def warm_tunnel():
    # Only Enhancor and BytePlus video runs fetch files from us; Dreamina gets them uploaded.
    # Bring their tunnel up early so the first such generation doesn't wait for it.
    if setting("PROVIDER") not in ("enhancor", "byteplus") or setting("MEDIA_HOST") != "tunnel" or setting("PUBLIC_BASE_URL"):
        return
    try:
        url = hosting.start_tunnel()
        print(f"  Туннель для провайдера: {url} (отдаёт только файлы задач и /hook)")
    except Exception as exc:
        print(f"  Туннель не поднялся: {exc}")


if __name__ == "__main__":
    port = int(setting("PORT"))
    print(f"\n  Luna Genjutsu → http://127.0.0.1:{port}\n")
    threading.Thread(target=warm_tunnel, daemon=True).start()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run("genjutsu.server:app", host="127.0.0.1", port=port, log_level="warning")
