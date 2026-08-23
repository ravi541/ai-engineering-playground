from fastapi import FastAPI
import json
app = FastAPI()

@app.get("/")
def hello_world():
    return {"message":"Hello World"}

@app.get("/ravi")
def ravi():
    return {"message":"This is a FastAPI application created by Me"}

@app.get("/patientdata")
def patient_data():
    return {"message":"This is a patient data application created by Me"}

def load_data():
    with open("patients.json","r") as f:
        data = json.load(f)
    return data


@app.get("/view")
def view():
    data = load_data()
    return data