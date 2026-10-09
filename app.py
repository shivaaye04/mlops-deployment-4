from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def root():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: list[float]):
    prediction = model.predict([data])
    return {"prediction": int(prediction[0])}