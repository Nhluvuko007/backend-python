from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pypdf
import io
import os
import json
import traceback
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize Gemini Client
client = genai.Client()

app = FastAPI(title="Smart Learn Algorithmic Engine")

@app.get("/")
def home():
    return {"status": "online", "message": "Python microservice running smoothly!"}

# 1. ROBUST REVIEW ENDPOINT (Handles string/null/missing inputs safely)
@app.post("/api/algorithm/review")
def process_card_review(data: dict):
    try:
        # Use .get() with defaults to prevent crashes if fields are missing/null
        repetitions = int(data.get("repetitions", 0))
        interval = int(data.get("interval", 0))
        ease_factor = float(data.get("ease_factor", 2.5))
        rating = int(data.get("rating", 1))

        # Core Algorithm
        if rating == 1:
            repetitions = 0
            interval = 1
        else:
            if repetitions == 0:
                interval = 1
            elif repetitions == 1:
                interval = 4
            else:
                interval = int(round(interval * ease_factor))
            repetitions += 1

        ease_factor = ease_factor + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02))
        if ease_factor < 1.3:
            ease_factor = 1.3

        next_review_date = datetime.now() + timedelta(days=interval)
        
        return {
            "repetitions": repetitions,
            "interval": interval,
            "easeFactor": round(ease_factor, 2),
            "nextReviewDate": next_review_date.isoformat()
        }
    except Exception as e:
        print("ALGORITHM CRASHED:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=400, detail=str(e))

# 2. AI GENERATION ENDPOINT
class Flashcard(BaseModel):
    front: str
    back: str

class FlashcardCollection(BaseModel):
    cards: List[Flashcard]

@app.post("/api/parser/generate")
async def generate_cards_from_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        reader = pypdf.PdfReader(pdf_file)
        
        full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        if not full_text.strip():
            return {"cards": [], "warning": "No readable text found."}

        sample_context = full_text[:8000]

        prompt_instruction = f"""
        You are an expert educational assistant specializing in active recall.
        Extract up to 10 high-quality flashcards from this text.
        Make the 'front' an engaging question or identification prompt, and the 'back' a concise, comprehensive answer.
        
        Study Text: {sample_context}
        """

        response = client.models.generate_content(
            model='gemini-2.0-flash', # Ensure you use a valid model name
            contents=prompt_instruction,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FlashcardCollection,
                temperature=0.3
            ),
        )

        return json.loads(response.text)

    except Exception as e:
        print("LLM PARSING CRASHED:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
