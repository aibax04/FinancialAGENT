from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import PyPDF2
import os
from typing import List
import tempfile

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_pdf_text(pdf_paths: List[str]) -> str:
    text = ""
    for path in pdf_paths:
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            text += f"\n[Error reading {path}: {e}]\n"
    return text

web_search_agent = Agent(
    name="web search agent",
    role="search the web for the information",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGo()],
    instructions=["If PDFs are provided, extract their text and use it to answer the user's question.","always include sources in your response"],
    show_tools_calls=True,
    markdown=True,
)

financial_agent = Agent(
    name="financial agent",
    role="analyze financial data and answer questions, including extracting info from PDFs",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[YFinanceTools(), DuckDuckGo()],
    instructions=[
        "If PDFs are provided, extract their text and use it to answer the user's question.",
        "For financial data, use YFinanceTools to get accurate information.",
        "Always include sources in your response."
    ],
    show_tools_calls=True,
    markdown=True,
)

multimodel_agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    team=[web_search_agent, financial_agent],
    instructions=["if pdf's are provided, extract their text and use it to answer the user's question",
                  "always include sources in your response",
                  "use tables to display stock data"],
    show_tools_calls=True,
    markdown=True,
)

@app.post("/chat")
async def chat_with_agent(
    message: str = Form(...),
    agent: str = Form("multimodel"),
    files: List[UploadFile] = File([])
):
    try:
        # Save uploaded PDFs to temp files
        temp_files = []
        pdf_texts = []
        
        for uploaded_file in files:
            if uploaded_file.content_type == "application/pdf":
                # Create a temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_file.write(await uploaded_file.read())
                temp_file.close()
                temp_files.append(temp_file.name)
        
        # Extract text from PDFs if any were uploaded
        if temp_files:
            pdf_text = extract_pdf_text(temp_files)
            pdf_texts.append(f"Extracted text from {len(temp_files)} PDF file(s):\n{pdf_text}")
            
            # Add PDF text to the message
            message_with_pdfs = f"{message}\n\n{'\n'.join(pdf_texts)}"
        else:
            message_with_pdfs = message
        
        # Select the appropriate agent
        selected_agent = {
            "multimodel": multimodel_agent,
            "financial": financial_agent,
            "web": web_search_agent
        }.get(agent, multimodel_agent)
        
        # Get response from the agent
        response = selected_agent.run(message_with_pdfs, stream=False)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)