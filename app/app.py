import io
import os
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Body,
    BackgroundTasks,
    Request,
    UploadFile
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import requests
import asyncio
import numpy as np
import uuid
from faster_whisper import WhisperModel
from transformers.pipelines.audio_utils import ffmpeg_read

admin_key = os.environ.get(
    "ADMIN_KEY",
)

hf_token = os.environ.get(
    "HF_TOKEN",
)

# fly runtime env https://fly.io/docs/machines/runtime-environment
fly_machine_id = os.environ.get(
    "FLY_MACHINE_ID",
)

# define our torch configuration
device = "cuda"
compute_type = "float16"

# load model on GPU if available, else cpu
model = WhisperModel("distil-large-v3", device=device, compute_type=compute_type)

app = FastAPI()
loop = asyncio.get_event_loop()
running_tasks = {}


class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}


def object_to_dict(obj):
    attributes = ['avg_logprob', 'start', 'end', 'id', 'no_speech_prob', 'text']
    return {attr: getattr(obj, attr) for attr in attributes}

def process(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        generate_kwargs = {
            "task": task,
            "language": None if language == "None" else language,
        }

        segments, info = model.transcribe(url, beam_size=1, **generate_kwargs)
        outputs = segments, info
    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    if task_id is not None:
        del running_tasks[task_id]

    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        if fly_machine_id is not None:
            webhookResp["fly_machine_id"] = fly_machine_id

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs


@app.middleware("http")
async def admin_key_auth_check(request: Request, call_next):
    if admin_key is not None:
        if ("x-admin-api-key" not in request.headers) or (
            request.headers["x-admin-api-key"] != admin_key
        ):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response


@app.post("/")
async def root(
    url: str | None = None,
    file: UploadFile | None = None,
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(
        default=False,
    ),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
    managed_task_id: str | None = Body(default=None),
):

    if url is None and file is None:
        raise HTTPException(status_code=400, detail="Either URL or file must be provided")

    if url and url.lower().startswith("http") is False:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if diarise_audio is True and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async is True and webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    task_id = managed_task_id if managed_task_id is not None else str(uuid.uuid4())
    if file:
        audio_bytes = await file.read()
        url = io.BytesIO(audio_bytes)

    try:
        resp = {}
        if is_async is True:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    url,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    webhook,
                    task_id,
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            segments, info = process(
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
                task_id,
            )
            resp = {
                "segments": [object_to_dict(segment) for segment in segments],
                "language": info.language,
                "language_probability": info.language_probability,
                "status": "completed",
                "task_id": task_id,
            }
        if fly_machine_id is not None:
            resp["fly_machine_id"] = fly_machine_id
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {"tasks": list(running_tasks.keys())}


@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]

    if task is None:
        return {"status": "processing"}
    elif task.done() is False:
        return {"status": "processing"}
    else:
        return {"status": "completed", "output": task.result()}


@app.delete("/cancel/{task_id}")
def cancel(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]
    if task is None:
        return HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}
