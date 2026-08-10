from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.generator import generate_cartoon

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str
    style: str


@app.get("/")
def root():
    return {"message": "PromptCartoon API running"}


@app.post("/generate")
def generate(req: PromptRequest):
    try:
        image_path = generate_cartoon(
            prompt=req.prompt,
            style=req.style
        )
        return FileResponse(image_path)
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}") # This will show up in your terminal
        return {"error": "Generation failed", "details": str(e)}