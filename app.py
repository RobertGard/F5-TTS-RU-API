import requests
import re
import time
import os
import subprocess
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from ruaccent import RUAccent


DEVICE = os.getenv("DEVICE", "cpu")
INPUT_DIR = "/data/input"
OUTPUT_DIR = "/data/output"

app = FastAPI(title="F5-TTS-API (wrapper)")

class TTSRequest(BaseModel):
    input: str
    out_format: str | None = "wav"   # wav or mp3
    ref_audio: str | None = None  # путь к эталонному аудиофайлу
    ref_text: str | None = None   # текст эталонного аудио


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
    out_file = os.path.join(OUTPUT_DIR, f"out_{int(os.times()[4]*1000)}.wav")

    # Обработка текста с помощью RUAccent
    gen_text = process_with_ruaccent(req.input)

    # Получаем пути к ckpt и vocab
    ckpt_path, vocab_path = get_model_paths()

    # Формируем команду с флагами
    cmd = [
        "f5-tts_infer-cli",
        "--ckpt_file", ckpt_path,
        "--vocab_file", vocab_path,
        "--gen_text", gen_text,
        "--output_dir", OUTPUT_DIR,
        "--output_file", os.path.basename(out_file),
        "--device", DEVICE
    ]
    if req.ref_audio:
        ref_audio_path = download_ref_audio(req.ref_audio)
        cmd += ["--ref_audio", ref_audio_path]
    if req.ref_text:
        ref_text_value = process_ref_text(req.ref_text)
        cmd += ["--ref_text", ref_text_value]

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
        out_file = os.path.join(OUTPUT_DIR, files[-1])

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
    snapshot_dirs = glob.glob(snapshot_glob)
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
    text = re.sub(r'<[^>]+>', '', text)  # убираем html-теги
    text = text.replace('\x00', '')  # убираем null-байты
    text = text.strip()
    return text


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