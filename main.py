from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.generate import get_ai_service


class GenerateRequest(BaseModel):
    age: str = Field(..., examples=["20s"])
    mood: str = Field(..., examples=["quiet"])
    purpose: str = Field(..., examples=["healing"])
    interest: str = Field(..., examples=["nature"])


class RecommendRequest(BaseModel):
    query: str = Field(..., examples=["quiet nature culture trip in Jeonju"])


class DiscoverRequest(BaseModel):
    region: str = Field(..., examples=["Jeonju"])


app = FastAPI(title="Curator AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
def generate_story(request: GenerateRequest):
    try:
        service = get_ai_service()
        return service.generate(
            age=request.age,
            mood=request.mood,
            purpose=request.purpose,
            interest=request.interest,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/story")
def story(request: GenerateRequest):
    try:
        service = get_ai_service()
        return service.generate(
            age=request.age,
            mood=request.mood,
            purpose=request.purpose,
            interest=request.interest,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/recommend")
def recommend(request: RecommendRequest):
    try:
        service = get_ai_service()
        return service.recommend(query=request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/travel")
def travel(request: GenerateRequest):
    try:
        service = get_ai_service()
        return service.travel(
            age=request.age,
            mood=request.mood,
            purpose=request.purpose,
            interest=request.interest,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/trend")
def trend():
    try:
        service = get_ai_service()
        return service.trend()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/discover")
def discover(request: DiscoverRequest):
    try:
        service = get_ai_service()
        return service.discover(region=request.region)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
