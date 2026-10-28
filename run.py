"""Start Luna Genjutsu: python run.py  →  http://127.0.0.1:8770"""
import threading
import webbrowser

import uvicorn

from genjutsu import hosting
from genjutsu.config import setting


def warm_tunnel():
    # Bring the provider-facing tunnel up early so the first generation doesn't wait for it.
    if setting("PROVIDER") == "mock" or setting("PUBLIC_BASE_URL"):
        hosting.start_public_server()
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
