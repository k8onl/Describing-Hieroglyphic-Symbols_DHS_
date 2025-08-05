from fastapi import FastAPI, UploadFile, File, HTTPException ,BackgroundTasks
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

app = FastAPI()

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("best.pt")

# Load hieroglyph stories
file_path = 'Semantic meaning.json'
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

    def speak_story(self, query: str) -> str:
        story = self.get_story(query)
        if story.startswith("No story"):
            return story
        cleaned_story = self.clean_text(story)
        try:
            # Get the generator object
            generator = self.pipeline(cleaned_story, voice='af_heart')

            # Check if the generator is callable or needs to be invoked
            if callable(generator):
                generator = generator()  # Call if it's a generator function

            # Process the generator output
            audio_data = None
            for output in generator:
                # Handle different possible output formats
                if len(output) >= 3:  # If output has at least 3 elements
                    gs, ps, audio = output[:3]  # Unpack first three elements
                    audio_data = audio
                elif isinstance(output, (np.ndarray, torch.Tensor)):  # If output is directly audio
                    audio_data = output
                else:
                    continue  # Skip unexpected formats

            if audio_data is not None:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_data, 24000)
                    return temp_file.name
            return "Error generating audio: no audio data received"
        except Exception as e:
            return f"Error: {str(e)}"


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

        # If not hieroglyph, return a message
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
    result = reader.speak_story(request.query)

    if isinstance(result, str) and (result.startswith("No story") or result.startswith("Error")):
        raise HTTPException(status_code=404, detail=result)

    if not os.path.exists(result):
        raise HTTPException(status_code=500, detail="Audio file was not generated properly")

    safe_query = re.sub(r'[^\w\-]', '_', request.query.encode('ascii', 'ignore').decode('ascii'))

    background_tasks.add_task(os.unlink, result)
    return FileResponse(
        path=result,
        media_type='audio/wav',
        headers={'Content-Disposition': f'attachment; filename="{safe_query}_story.wav"'}
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
