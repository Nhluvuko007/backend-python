from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from typing import List
import pypdf
import spacy
import io
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load Python environment file variables
load_dotenv()

# Initialize the Gemini Client (It automatically looks for the GEMINI_API_KEY env variable)
client = genai.Client()

app = FastAPI(title="Smart Learn Algorithmic Engine")

# Load our NLP language model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class CardReviewData(BaseModel):
    repetitions: int
    interval: int
    ease_factor: float
    rating: int

@app.get("/")
def home():
    return {"status": "online", "message": "Python microservice running smoothly!"}

@app.post("/api/algorithm/review")
def process_card_review(data: CardReviewData):
    repetitions = data.repetitions
    interval = data.interval
    ease_factor = data.ease_factor
    rating = data.rating

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

# 1. Define the structural contract for a single flashcard
class Flashcard(BaseModel):
    front: str = Field(description="A clear, concise question or term based on the text.")
    back: str = Field(description="The corresponding answer, definition, or explanation.")

# 2. Define the container contract that forces the LLM to return an array list
class FlashcardCollection(BaseModel):
    cards: List[Flashcard]

# 3. UPGRADED AI GENERATION ENDPOINT VIA GEMINI
@app.post("/api/parser/generate")
async def generate_cards_from_pdf(file: UploadFile = File(...)):
    try:
        # Read uploaded file binary stream
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        # Extract text using PyPDF
        reader = pypdf.PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                
        # Guard clause check if the document is blank or an unreadable image scan
        if not full_text.strip():
            return {"cards": [], "warning": "The uploaded PDF document contains no readable text."}

        # Truncate text context safely if it's excessively massive for an MVP test
        sample_context = full_text[:8000]

        # Formulate the programmatic instructions for Gemini
        prompt_instruction = f"""
        You are an expert educational assistant specializing in active recall.
        Analyze the following text extracted from a study document and extract up to 10 high-quality flashcards.
        Focus on key concepts, technical terms, core definitions, or critical process steps.
        Make the 'front' an engaging question or identification prompt, and the 'back' a concise, comprehensive answer.
        
        Study Text Material:
        \"\"\"{sample_context}\"\"\"
        """

        # Call the model enforcing a strict JSON return schema
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_instruction,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FlashcardCollection,
                temperature=0.3 # Lower temperature ensures more accurate, fact-based extraction
            ),
        )

        # The response.text is guaranteed to be a valid JSON string matching FlashcardCollection
        import json
        structured_data = json.loads(response.text)
        
        return structured_data

    except Exception as e:
        print(f"CRITICAL LLM API PROCESSING ERROR: {str(e)}")
        return {"cards": [], "error": str(e)}