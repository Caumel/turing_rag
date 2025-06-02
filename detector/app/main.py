from fastapi import FastAPI, File, UploadFile
from .detect import detect_objects
import uvicorn

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    results = detect_objects(image_bytes)
    return results

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
