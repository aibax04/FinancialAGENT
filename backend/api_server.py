# Import patch first to fix inspect.getargspec issue
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import patch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import uvicorn
import tempfile
from financial_agent import web_search_agent, financial_agent, multimodel_agent, extract_pdf_text


app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(
    message: str = Form(...),
    agent: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    pdf_paths = []
    pdf_text = ""
    for file in files:
        if file.content_type == "application/pdf":
            suffix = os.path.splitext(file.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                pdf_paths.append(tmp.name)

    if pdf_paths:
        pdf_text = extract_pdf_text(pdf_paths)
        for path in pdf_paths:
            try:
                os.remove(path)
            except Exception:
                pass

    prompt_with_pdf = message
    if pdf_text:
        prompt_with_pdf += f"\n\n[PDF Content]\n{pdf_text[:3000]}"

    if agent == "financial":
        response = financial_agent.run(prompt_with_pdf)
    elif agent == "web":
        response = web_search_agent.run(prompt_with_pdf)
    else:
        response = multimodel_agent.run(prompt_with_pdf)

    if hasattr(response, 'content'):
        full_response = response.content
    elif isinstance(response, str):
        full_response = response
    else:
        full_response = str(response)
    return {"response": full_response}

# Serve static files at /static

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root
@app.get("/")
def read_index():
    return FileResponse("frontend/agent.html")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
    