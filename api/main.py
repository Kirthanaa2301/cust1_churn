# api/main.py
from fastapi import FastAPI, File, UploadFile
from typing import List
import pandas as pd
import io
import requests
import os

BENTO_URL = os.environ.get("BENTO_URL", "http://bento:3000")  # internal name for compose
app = FastAPI()

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = {}
    for f in files:
        content = await f.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_excel(io.BytesIO(content))
        payload = {"rows": df.to_dict(orient="records")}
        r = requests.post(f"{BENTO_URL}/predict", json=payload, timeout=60)
        results[f.filename] = r.json()
    return results

@app.get("/healthz")
def health():
    try:
        r = requests.get(f"{BENTO_URL}/healthz", timeout=2)
        return r.json()
    except Exception as e:
        return {"error": str(e)}
