from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import torch
import clip
from starlette.background import BackgroundTasks
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import soundfile as sf
import re
from kokoro import KPipeline
import json
import os
import tempfile
import base64
from googletrans import Translator

# Chatbot-specific imports
import spacy
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import string
import random

app = FastAPI()

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("best.pt")

# Initialize translator
translator = Translator()

# Load hieroglyph stories
file_path = 'hieroglyphs_stories (1).json'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Hieroglyph stories file not found at {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Data loaded successfully")

class QueryRequest(BaseModel):
    query: str

class HieroglyphStoryReader:
    def __init__(self, json_data: dict[str, any], lang_code: str = 'a'):
        self.data = json_data
        self.pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')

    def get_story(self, query: str) -> str:
        for key, value in self.data.items():
            try:
                gardiner, symbol = key.split('-')
                if query == symbol or query == gardiner or query == key:
                    return value['story']
            except ValueError:
                continue
        return "No story for this hieroglyph symbol"

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s.]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def speak_story(self, query: str) -> tuple[str, str]:
        story = self.get_story(query)
        if story.startswith("No story"):
            return story, ""
        cleaned_story = self.clean_text(story)
        try:
            generator = self.pipeline(cleaned_story, voice='af_heart')
            if callable(generator):
                generator = generator()
            audio_data = None
            for output in generator:
                if len(output) >= 3:
                    gs, ps, audio = output[:3]
                    audio_data = audio
                elif isinstance(output, (np.ndarray, torch.Tensor)):
                    audio_data = output
                else:
                    continue
            if audio_data is not None:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_data, 24000)
                    return temp_file.name, cleaned_story
            return "Error generating audio: no audio data received", ""
        except Exception as e:
            return f"Error: {str(e)}", ""

@app.get("/")
async def root():
    return {"status": "API is running"}

@app.post("/detect")
async def detect_hieroglyphs(file: UploadFile = File(...)):
    try:
        # Read and open image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize image to 640x640
        resized_image = image.resize((640, 640))

        # CLIP Classification
        image_input = clip_preprocess(resized_image).unsqueeze(0).to(device)
        text_input = clip.tokenize(["ancient Egyptian hieroglyphs", "other content"]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            similarity = (image_features @ text_features.T).softmax(dim=-1)

        is_hieroglyph = similarity[0][0].item() > 0.5

        # If not hieroglyph, return the message; otherwise, proceed to YOLO detection
        if not is_hieroglyph:
            return JSONResponse(content={"message": "No hieroglyph found"}, status_code=200)

        # YOLO Detection
        img_np = np.array(resized_image)
        results = yolo_model.predict(img_np, imgsz=640)

        # Get the first result
        result = results[0]

        # Get the plotted image with boxes (returns BGR numpy array)
        plotted_img = result.plot()

        # Convert BGR to RGB and then to PIL Image
        plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(plotted_img_rgb)

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-story-audio")
async def get_hieroglyph_story_audio(request: QueryRequest, background_tasks: BackgroundTasks):
    reader = HieroglyphStoryReader(data)
    audio_file_path, transcription = reader.speak_story(request.query)

    if audio_file_path.startswith("No story") or audio_file_path.startswith("Error"):
        raise HTTPException(status_code=404, detail=audio_file_path)

    if not os.path.exists(audio_file_path):
        raise HTTPException(status_code=500, detail="Audio file was not generated properly")

    # Read the audio file and encode it as base64
    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    # Translate the transcription to Arabic
    translated_text = ""
    if transcription:
        try:
            translated = translator.translate(transcription, src='en', dest='ar')
            translated_text = translated.text
        except Exception as e:
            translated_text = f"Translation failed: {str(e)}"

    # Clean up the temporary file
    background_tasks.add_task(os.unlink, audio_file_path)

    # Return JSON response with audio, transcription, and translated text
    return JSONResponse(content={
        "audio": audio_base64,
        "transcription": transcription,
        "translated_text": translated_text
    })

# Chatbot setup
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

try:
    with open('assi bot data.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Error: 'assi bot data.json' file not found. Please ensure the file exists.")
except json.JSONDecodeError:
    raise ValueError("Error: Invalid JSON format in 'assi bot data.json'.")

stop_words = set(stopwords.words('english')) - {'not'}
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[@%s]' % re.escape(string.punctuation), ' ', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

X = []
y = []
intent_responses = {}
for intent in intents['info']:
    intent_title = intent.get('title', '')
    if not intent_title:
        continue
    intent_responses[intent_title] = intent.get('responses', [])
    for request in intent.get('requests', []):
        if not request:
            continue
        processed_request = preprocess_text(request)
        X.append(processed_request)
        y.append(intent_title)

X_with_ngrams = []
for text in X:
    tokens = text.split()
    bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
    trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
    combined = text + ' ' + ' '.join(bigrams) + ' '.join(trigrams)
    X_with_ngrams.append(combined)

if len(X_with_ngrams) == 0 or len(y) == 0:
    raise ValueError("Error: No valid training data found in the dataset.")

X_train, X_test, y_train, y_test = train_test_split(X_with_ngrams, y, test_size=0.2, random_state=42)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

def generate_text(user_input):
    if user_input is None or user_input.strip() == "":
        return "Sorry, I didn't catch that. Please type something again."
    processed_input = preprocess_text(user_input)
    tokens = processed_input.split()
    bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
    trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
    combined_input = processed_input + ' ' + ' '.join(bigrams) + ' '.join(trigrams)
    predicted_intent = model.predict([combined_input])[0]
    responses = intent_responses.get(predicted_intent, [])
    if responses:
        return random.choice(responses)
    else:
        return "I'm not sure how to respond to that."

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_message = request.message
        response = generate_text(user_message)
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)