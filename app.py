from pathlib import Path
import json
import traceback
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import get_agent
from database import (
    create_or_update_conversation,
    get_chat_history,
    init_db,
    list_conversations,
    save_chat_message,
)
from rag import add_document_to_rag
from tools import set_current_thread_id


init_db()

app = FastAPI(title="LabheshGPT")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ChatStreamRequest(BaseModel):
    message: str
    thread_id: str
    model: str = "gemini-2.5-flash"


def extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        return ""

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()

    return str(content)


def is_visible_assistant_chunk(message_chunk) -> bool:
    message_type = getattr(message_chunk, "type", "")

    if message_type not in {"ai", "AIMessageChunk"}:
        return False

    if getattr(message_chunk, "tool_call_chunks", None):
        return False

    additional_kwargs = getattr(message_chunk, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls"):
        return False

    return True


def serialize_message(message):
    return {
        "id": message.id,
        "thread_id": message.thread_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def serialize_conversation(conversation):
    return {
        "id": conversation.id,
        "thread_id": conversation.thread_id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
        "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
    }


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
def home(request: Request):
    try:
        thread_id = request.query_params.get("thread_id") or f"thread_{uuid4().hex[:10]}"
        create_or_update_conversation(thread_id)
        history = get_chat_history(thread_id)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "thread_id": thread_id,
                "history": history,
            },
        )

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


@app.get("/conversations")
def conversations():
    try:
        return {
            "conversations": [
                serialize_conversation(conversation)
                for conversation in list_conversations()
            ],
        }

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


@app.get("/history/{thread_id}")
def history(thread_id: str):
    try:
        create_or_update_conversation(thread_id)

        return {
            "messages": [
                serialize_message(message)
                for message in get_chat_history(thread_id)
            ],
        }

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


@app.post("/chat")
def chat(message: str = Form(...), thread_id: str = Form(...), model: str = Form("gemini-2.5-flash")):
    try:
        message = message.strip()
        if not message:
            return JSONResponse({"error": "Message cannot be empty."}, status_code=400)

        set_current_thread_id(thread_id)
        create_or_update_conversation(thread_id, message)
        save_chat_message(thread_id, "user", message)

        agent = get_agent(model)
        config = {"configurable": {"thread_id": thread_id}}
        final_text = ""

        for message_chunk, _metadata in agent.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="messages",
        ):
            if not is_visible_assistant_chunk(message_chunk):
                continue

            chunk_text = extract_text(message_chunk.content)
            if chunk_text:
                final_text += chunk_text

        final_text = final_text.strip() or "I did not get a text response."
        save_chat_message(thread_id, "assistant", final_text)

        return {"reply": final_text}

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


@app.post("/chat/stream")
def chat_stream(payload: ChatStreamRequest):
    def stream_reply():
        try:
            user_message = payload.message.strip()

            if not user_message:
                yield sse_event({"error": "Message cannot be empty.", "done": True})
                return

            set_current_thread_id(payload.thread_id)
            create_or_update_conversation(payload.thread_id, user_message)
            save_chat_message(payload.thread_id, "user", user_message)

            agent = get_agent(payload.model)
            config = {"configurable": {"thread_id": payload.thread_id}}
            final_text = ""

            for message_chunk, _metadata in agent.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="messages",
            ):
                if not is_visible_assistant_chunk(message_chunk):
                    continue

                chunk_text = extract_text(message_chunk.content)
                if chunk_text:
                    final_text += chunk_text
                    yield sse_event({"token": chunk_text})

            final_text = final_text.strip() or "I did not get a text response."
            save_chat_message(payload.thread_id, "assistant", final_text)
            yield sse_event({"done": True})

        except Exception as exc:
            traceback.print_exc()
            yield sse_event(
                {
                    "error": str(exc),
                    "done": True,
                }
            )

    return StreamingResponse(stream_reply(), media_type="text/event-stream")


@app.post("/upload")
async def upload_document(thread_id: str = Form(...), file: UploadFile = File(...)):
    try:
        set_current_thread_id(thread_id)

        safe_name = Path(file.filename or "uploaded_file").name
        file_path = UPLOAD_DIR / f"{thread_id}_{safe_name}"

        file_path.write_bytes(await file.read())
        result = add_document_to_rag(str(file_path), thread_id)

        return {
            "success": True,
            "message": f"Uploaded {result['filename']} and indexed {result['chunks']} chunks.",
        }

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {
                "success": False,
                "error": str(exc),
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


if __name__ == "__main__":
    print("Labhesh GPT is running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
