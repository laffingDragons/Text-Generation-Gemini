from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Dict, Any
import logging
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = FastAPI(
    title="Story Generator API",
    description="API for generating and summarizing stories using Google's Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment")
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

class StoryRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=200, description="Title or topic for the story")

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to translate")
    target_language: str = Field(..., min_length=2, max_length=50, description="Target language for translation")

class ErrorResponse(BaseModel):
    detail: str

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred"}
    )

@app.post("/generate", response_model=Dict[str, str], responses={
    400: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def generate_story(request: StoryRequest):
    try:
        title = request.title
        prompt = f"Generate a hooking story about {title}"
        
        try:
            response = model.generate_content(prompt)
            if not response.text:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Generated content was empty"
                )
            return {"title": title, "story": response.text}
        except genai.types.generation_types.BlockedPromptException:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Content generation was blocked due to safety concerns with the prompt"
            )
        except genai.types.generation_types.StopCandidateException:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content generation was stopped unexpectedly"
            )
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower() or "limit" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="API rate limit exceeded. Please try again later."
                )
            logger.error(f"Error generating content: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating content: {str(e)}"
            )
    except Exception as e:
        logger.error(f"Unhandled error in generate_story: {str(e)}")
        raise

@app.post("/summarize", response_model=Dict[str, str], responses={
    400: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def summarize_story(request: StoryRequest):
    try:
        title = request.title
        if len(title) < 50:  # Basic check if there's enough content to summarize
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too short to summarize. Please provide a longer text."
            )
            
        prompt = f"Summarize the following text: \n\n {title}"
        
        try:
            response = model.generate_content(prompt)
            if not response.text:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Generated summary was empty"
                )
            return {"title": title[:50] + "..." if len(title) > 50 else title, "summary": response.text}
        except genai.types.generation_types.BlockedPromptException:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Content generation was blocked due to safety concerns with the input text"
            )
        except genai.types.generation_types.StopCandidateException:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content generation was stopped unexpectedly"
            )
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower() or "limit" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="API rate limit exceeded. Please try again later."
                )
            logger.error(f"Error summarizing content: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error summarizing content: {str(e)}"
            )
    except Exception as e:
        logger.error(f"Unhandled error in summarize_story: {str(e)}")
        raise
    
@app.post("/translate", response_model=Dict[str, str], responses={
    400: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def translate_text(request: TranslationRequest):
    try:
        text = request.text
        target_language = request.target_language
        
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        
        try:
            response = model.generate_content(prompt)
            if not response.text:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Generated translation was empty"
                )
            return {
                "original_text": text[:100] + "..." if len(text) > 100 else text,
                "translated_text": response.text,
                "target_language": target_language
            }
        except genai.types.generation_types.BlockedPromptException:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Content translation was blocked due to safety concerns with the input text"
            )
        except genai.types.generation_types.StopCandidateException:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content translation was stopped unexpectedly"
            )
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower() or "limit" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="API rate limit exceeded. Please try again later."
                )
            logger.error(f"Error translating content: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error translating content: {str(e)}"
            )
    except Exception as e:
        logger.error(f"Unhandled error in translate_text: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Welcome to the Story Generator API! Use /generate or /summarize endpoints."}

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    try:
        # Perform a simple check to verify Gemini connectivity
        response = model.generate_content("Hello")
        return {"status": "healthy", "message": "API is operational and connected to Gemini"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service is currently unavailable: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)