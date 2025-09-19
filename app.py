import os
import shutil
import subprocess
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse

MODEL_ENV = os.getenv("MODEL_NAME", "hotstone228/F5-TTS-Russian")
PORT = int(os.getenv("PORT", 4123))
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")

app = FastAPI(title="F5-TTS-API (wrapper)")

class TTSRequest(BaseModel):
    input: str
    voice: str | None = None
    out_format: str | None = "wav"   # wav or mp3

@app.on_event("startup")
async def startup_event():
    # ensure model is downloaded to hf cache to avoid runtime hiccups
    # huggingface_hub will read HUGGINGFACE_HUB_TOKEN from env if set
    try:
        from huggingface_hub import snapshot_download
        print("Ensuring model is cached:", MODEL_ENV)
        snapshot_download(repo_id=MODEL_ENV, cache_dir="/root/.cache/huggingface/hub")
    except Exception as e:
        print("Model download/check warning:", e)

@app.post("/v1/audio/speech")
async def synthesize(req: TTSRequest):
    if not req.input or req.input.strip() == "":
        raise HTTPException(status_code=400, detail="input text required")

    # create temp working dir
    td = tempfile.mkdtemp(prefix="f5tts_")
    toml_path = os.path.join(td, "request.toml")
    out_dir = "/data/output"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"out_{int(os.times()[4]*1000)}.wav")

    # render TOML from template (very simple)
    toml_template = """[model]
name = "{model}"

[output]
outdir = "{outdir}"
wav_filename = "{wav_name}"

[generate]
gen_text = "{gen_text}"
"""
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(toml_template.format(model=MODEL_ENV, outdir=out_dir, wav_name=os.path.basename(out_file), gen_text=req.input.replace('"', '\\"')))

    # Run f5-tts_infer-cli with config
    cmd = ["f5-tts_infer-cli", "-c", toml_path]
    env = os.environ.copy()
    env["DEVICE"] = DEVICE
    env["MODEL_SIZE"] = MODEL_SIZE

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=600)
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"f5-tts failed: {proc.stderr.decode('utf-8')[:1000]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="TTS generation timed out")

    # return file
    if not os.path.exists(out_file):
        # try to find any produced wav in out_dir
        files = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
        if not files:
            raise HTTPException(status_code=500, detail="No output produced")
        out_file = os.path.join(out_dir, files[-1])

    # Optionally convert to mp3 if requested (ffmpeg must be present)
    if req.out_format and req.out_format.lower() == "mp3":
        mp3_path = out_file.replace(".wav", ".mp3")
        subprocess.run(["ffmpeg", "-y", "-i", out_file, mp3_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return FileResponse(mp3_path, media_type="audio/mpeg", filename=os.path.basename(mp3_path))

    return FileResponse(out_file, media_type="audio/wav", filename=os.path.basename(out_file))
