"""Guide audio sent with the motion video. The final file always gets the untouched original track."""
from pathlib import Path

from . import media, replicate

DEMUCS_VERSION = "25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953"  # cjwbw/demucs
SEMITONES = 3


def guide_track(folder, source_wav, mode, status):
    """Return the audio file to embed into the guide video, or None.

    vocals: isolated voice for lip sync, without the music bed (Demucs on Replicate)
    full:   the whole mix
    Both are pitched +3 semitones, so the original speaker's voice is not handed to the generator.
    """
    folder = Path(folder)
    if mode == "none" or source_wav is None:
        return None
    src = source_wav
    if mode == "vocals":
        status("Demucs · отделяю голос от музыки для липсинка")
        output = replicate.run(DEMUCS_VERSION, {"audio": replicate.upload(source_wav), "model_name": "htdemucs", "output_format": "wav"}, folder / "demucs.json")
        vocals = output.get("vocals") if isinstance(output, dict) else None
        if not vocals:
            raise replicate.ReplicateError("Demucs не вернул дорожку с голосом.")
        src = replicate.download(vocals, folder / "vocals.wav")
    status(f"Сдвигаю тон голоса на +{SEMITONES} полутона, длительность не меняется")
    pitched = folder / "guide-audio.wav"
    media.pitch_shift(src, pitched, SEMITONES)
    return pitched
