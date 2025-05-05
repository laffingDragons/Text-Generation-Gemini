from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

genai.configure(api_key="AIzaSyDb15g4bUk_DoexsrU6ztYo8HfY00NjVbY")
model = genai.GenerativeModel("gemini-1.5-flash")

class StoryRequest(BaseModel):
    title: str

@app.post("/generate")
async def generate_story(request: StoryRequest):
    title = request.title
    prompt = f"Generate a hooking story about {title}"  # Use f-string for clarity
    response = model.generate_content(prompt)
    return {"title": title, "story": response.text}