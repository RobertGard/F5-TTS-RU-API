import os
import re
import time
import uuid
import glob
import shutil
import subprocess
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from ruaccent import RUAccent

# -------------------
# Конфигурация
# -------------------
DEVICE = os.getenv("DEVICE", "cuda")
INPUT_DIR = "/data/input"
OUTPUT_DIR = "/data/output"
REF_AUDIO_ENV = os.getenv("REF_AUDIO_PATH", "")
REF_TEXT_ENV = os.getenv("REF_TEXT_PATH", "")

# Проверка наличия бинарников
if shutil.which("f5-tts_infer-cli") is None:
    raise RuntimeError("f5-tts_infer-cli not found in PATH")
if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg not found in PATH")

# -------------------
# Инициализация FastAPI
# -------------------
app = FastAPI(title="F5-TTS-API (wrapper)")

# -------------------
# Модель запроса
# -------------------
class TTSRequest(BaseModel):
    input: str   # Текст, который модель должна синтезировать в речь
    out_format: str | None = "wav"   # wav или mp3
    ref_audio: str | None = None  # путь к эталонному аудиофайлу
    ref_text: str | None = None   # текст эталонного аудио
    vocoder_name: str | None = None  # название вокодера (например, "hifigan")
    remove_silence: bool | None = None  # удалять ли тишину в начале/конце
    target_rms: float | None = None  # целевой RMS для нормализации громкости
    speed: float | None = None  # скорость воспроизведения (1.0 = стандарт)
    cfg_strength: float | None = None  # сила CFG (контроль генерации)
    nfe_step: int | None = None  # количество шагов NFE (качество/скорость)
    fix_duration: bool | None = None  # фиксировать длительность (True/False)
    cross_fade_duration: float | None = None  # длительность кроссфейда между чанками
    save_chunk: bool | None = None  # сохранять ли промежуточные чанки


# -------------------
# Основной эндпоинт
# -------------------
@app.post("/v1/audio/speech")
async def synthesize(req: TTSRequest):
    """
    Эндпоинт синтеза речи. Принимает текст, опционально референс-аудио и текст, возвращает аудиофайл.

    Args:
        req (TTSRequest): Запрос с параметрами input, out_format, ref_audio, ref_text.

    Returns:
        FileResponse: итоговый аудиофайл (wav или mp3)

    Raises:
        HTTPException: При ошибках валидации, генерации или скачивания файлов.
    """
    if not req.input or req.input.strip() == "":
        raise HTTPException(status_code=400, detail="input text required")

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = generate_output_filename()

    # Обработка текста с помощью RUAccent
    gen_text = process_with_ruaccent(req.input)
    # Получаем пути к ckpt и vocab
    ckpt_path, vocab_path = get_model_paths()
    ref_audio_path = get_ref_audio_path(req.ref_audio)
    ref_text_value = get_ref_text_value(req.ref_text)

    # Формируем команду для f5-tts_infer-cli
    cmd = build_cli_command(
        ckpt_path, vocab_path, gen_text, out_file, 
        ref_audio_path=ref_audio_path, 
        ref_text_value=ref_text_value,
        vocoder_name=req.vocoder_name,
        remove_silence=req.remove_silence,
        target_rms=req.target_rms,
        speed=req.speed,
        cfg_strength=req.cfg_strength,
        nfe_step=req.nfe_step,
        fix_duration=req.fix_duration,
        cross_fade_duration=req.cross_fade_duration,
        save_chunk=req.save_chunk
    )

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"f5-tts failed: {proc.stderr.decode('utf-8')[:1000]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="TTS generation timed out")

    return save_and_return_final_audio_file(out_file, req.out_format)



def save_and_return_final_audio_file(out_file: str, out_format: str | None) -> FileResponse:
    """
    Проверяет существование итогового wav-файла, при необходимости ищет его в OUTPUT_DIR.
    Если требуется mp3 — конвертирует через ffmpeg и возвращает FileResponse с mp3.
    Иначе возвращает FileResponse с wav.

    Args:
        out_file (str): ожидаемый путь к wav-файлу
        out_format (str|None): требуемый формат ('wav' или 'mp3')

    Returns:
        FileResponse: итоговый аудиофайл для отдачи пользователю

    Raises:
        HTTPException: Если не найден итоговый файл или возникла ошибка при генерации.
    """
    if not os.path.exists(out_file):
        # try to find any produced wav in OUTPUT_DIR
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
        if not files:
            raise HTTPException(status_code=500, detail="No output produced")
        out_file = os.path.join(OUTPUT_DIR, sorted(files)[-1])

    # Optionally convert to mp3 if requested (ffmpeg must be present)
    if out_format and out_format.lower() == "mp3":
        mp3_path = out_file.replace(".wav", ".mp3")
        subprocess.run(["ffmpeg", "-y", "-i", out_file, mp3_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return FileResponse(mp3_path, media_type="audio/mpeg", filename=os.path.basename(mp3_path))

    return FileResponse(out_file, media_type="audio/wav", filename=os.path.basename(out_file))


def get_model_paths() -> tuple[str, str]:
    """
    Находит snapshot-директорию модели в HuggingFace cache и возвращает абсолютные пути к ckpt (model_last_inference.safetensors)
    и vocab.txt для F5-TTS. Если что-то не найдено — выбрасывает HTTPException.

    Returns:
        tuple[str, str]: (путь к ckpt, путь к vocab.txt)

    Raises:
        HTTPException: Если не найдены snapshot, ckpt или vocab.txt
    """
    snapshot_glob = "/root/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/*"
    snapshot_dirs = sorted(glob.glob(snapshot_glob), key=os.path.getmtime, reverse=True)
    if not snapshot_dirs:
        raise HTTPException(status_code=500, detail="Model snapshot not found in huggingface cache")
    snapshot_dir = snapshot_dirs[0]
    
    ckpt_path = os.path.join(snapshot_dir, "F5TTS_v1_Base_v2/model_last_inference.safetensors")
    vocab_path = os.path.join(snapshot_dir, "F5TTS_v1_Base/vocab.txt")
    if not os.path.isfile(ckpt_path):
        raise HTTPException(status_code=500, detail=f"model_last_inference.safetensors not found: {ckpt_path}")
    if not os.path.isfile(vocab_path):
        raise HTTPException(status_code=500, detail=f"vocab.txt not found: {vocab_path}")
    
    return ckpt_path, vocab_path



def download_ref_audio(ref_audio_path: str) -> str:
    """
    Если ref_audio_path — это http(s) URL, скачивает аудиофайл-референс и возвращает путь к нему.
    Если это локальный путь — возвращает его без изменений.
    В случае ошибки скачивания выбрасывает HTTPException.

    Args:
        ref_audio_path (str): Ссылка на аудиофайл-референс или локальный путь.

    Returns:
        str: Абсолютный путь к локально сохранённому аудиофайлу или исходный путь.

    Raises:
        HTTPException: Если не удалось скачать файл.
    """
    if ref_audio_path.startswith("http://") or ref_audio_path.startswith("https://"):
        try:
            ext = os.path.splitext(ref_audio_path)[1] or ".wav"
            local_name = f"ref_{int(time.time()*1000)}{ext}"
            local_path = os.path.join(INPUT_DIR, local_name)
            r = requests.get(ref_audio_path, timeout=10)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download ref_audio: {e}")
    else:
        return ref_audio_path


def process_ref_text(ref_text: str) -> str:
    """
    Обрабатывает параметр ref_text для передачи в f5-tts_infer-cli.
    Если это URL — скачивает файл, читает его содержимое и возвращает текст.
    Если это обычный текст — возвращает его как есть.
    Поверхностно фильтрует содержимое (убирает html-теги и опасные символы).

    Args:
        ref_text (str): Значение параметра ref_text (текст или URL).

    Returns:
        str: Готовый к подстановке текст для --ref_text.

    Raises:
        HTTPException: Если не удалось скачать или прочитать файл.
    """
    # Проверка на URL
    if ref_text.startswith("http://") or ref_text.startswith("https://"):
        try:
            ext = os.path.splitext(ref_text)[1] or ".txt"
            local_name = f"ref_text_{int(time.time()*1000)}{ext}"
            local_path = os.path.join(INPUT_DIR, local_name)
            r = requests.get(ref_text, timeout=10)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            with open(local_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download/read ref_text: {e}")
    else:
        text = ref_text
    # Поверхностная фильтрация: убираем html-теги и опасные символы
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\x00", "")
    return text.strip()


def process_with_ruaccent(text: str) -> str:
    """
    Обрабатывает входной текст с помощью RUAccent для постановки ударений.

    Args:
        text (str): Входной текст для обработки.

    Returns:
        str: Текст с расставленными ударениями, готовый для генерации речи.
    """
    accentizer = RUAccent()
    accentizer.load(
        omograph_model_size='turbo3.1',
        use_dictionary=True,
        tiny_mode=False
    )
    return accentizer.process_all(text) + ' '


def get_ref_audio_path(req_ref_audio: str | None) -> str | None:
    """
    Возвращает путь к референс-аудио: сначала из env, затем скачивает или возвращает None.

    Args:
        req_ref_audio (str | None): Путь или URL к аудиофайлу из запроса.

    Returns:
        str | None: Абсолютный путь к аудиофайлу или None, если не задан.
    """
    if REF_AUDIO_ENV and os.path.isfile(REF_AUDIO_ENV):
        return REF_AUDIO_ENV
    if req_ref_audio:
        return download_ref_audio(req_ref_audio)
    return None


def get_ref_text_value(req_ref_text: str | None) -> str | None:
    """
    Возвращает текст для --ref_text: сначала из env, затем скачивает/фильтрует или возвращает None.

    Args:
        req_ref_text (str | None): Текст, путь или URL из запроса.

    Returns:
        str | None: Готовый текст для --ref_text или None, если не задан.
    """
    if REF_TEXT_ENV and os.path.isfile(REF_TEXT_ENV):
        try:
            with open(REF_TEXT_ENV, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read ref_text from env path: {e}")
    if req_ref_text:
        return process_ref_text(req_ref_text)
    return None


def generate_output_filename() -> str:
    """
    Генерирует уникальное имя выходного файла в OUTPUT_DIR.

    Returns:
        str: Абсолютный путь к новому wav-файлу в OUTPUT_DIR.
    """
    return os.path.join(OUTPUT_DIR, f"out_{uuid.uuid4().hex}.wav")


def build_cli_command(
    ckpt_path: str,
    vocab_path: str,
    gen_text: str,
    out_file: str,
    ref_audio_path: str | None,
    ref_text_value: str | None,
    vocoder_name: str | None = None,
    remove_silence: bool | None = None,
    target_rms: float | None = None,
    speed: float | None = None,
    cfg_strength: float | None = None,
    nfe_step: int | None = None,
    fix_duration: bool | None = None,
    cross_fade_duration: float | None = None,
    save_chunk: bool | None = None
) -> list[str]:
    """
    Формирует команду для f5-tts_infer-cli с учётом всех параметров, включая дополнительные опции.

    Args:
        ckpt_path (str): Путь к файлу весов модели.
        vocab_path (str): Путь к словарю.
        gen_text (str): Текст для синтеза.
        out_file (str): Путь к выходному wav-файлу.
        ref_audio_path (str | None): Путь к референс-аудио (или None).
        ref_text_value (str | None): Текст для --ref_text (или None).
        vocoder_name (str | None): название вокодера (например, "hifigan")
        remove_silence (bool | None): удалять ли тишину в начале/конце
        target_rms (float | None): целевой RMS для нормализации громкости
        speed (float | None): скорость воспроизведения (1.0 = стандарт)
        cfg_strength (float | None): сила CFG (контроль генерации)
        nfe_step (int | None): количество шагов NFE (качество/скорость)
        fix_duration (bool | None): фиксировать длительность (True/False)
        cross_fade_duration (float | None): длительность кроссфейда между чанками
        save_chunk (bool | None): сохранять ли промежуточные чанки

    Returns:
        list[str]: Сформированная команда для subprocess.run
    """
    cmd = [
        "f5-tts_infer-cli",
        "--ckpt_file", ckpt_path,
        "--vocab_file", vocab_path,
        "--gen_text", gen_text,
        "--output_dir", OUTPUT_DIR,
        "--output_file", os.path.basename(out_file),
        "--device", DEVICE
    ]
    if ref_audio_path:
        cmd += ["--ref_audio", ref_audio_path]
    if ref_text_value:
        cmd += ["--ref_text", ref_text_value]
    if vocoder_name:
        cmd += ["--vocoder_name", vocoder_name]
    if remove_silence is not None:
        cmd += ["--remove_silence", str(remove_silence).lower()]
    if target_rms is not None:
        cmd += ["--target_rms", str(target_rms)]
    if speed is not None:
        cmd += ["--speed", str(speed)]
    if cfg_strength is not None:
        cmd += ["--cfg_strength", str(cfg_strength)]
    if nfe_step is not None:
        cmd += ["--nfe_step", str(nfe_step)]
    if fix_duration is not None:
        cmd += ["--fix_duration", str(fix_duration).lower()]
    if cross_fade_duration is not None:
        cmd += ["--cross_fade_duration", str(cross_fade_duration)]
    if save_chunk is not None:
        cmd += ["--save_chunk", str(save_chunk).lower()]
    return cmd