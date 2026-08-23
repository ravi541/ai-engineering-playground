from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def hello_world():
    return {"message":"Hello World"}

@app.get("/ravi")
def ravi():
    return {"message":"This is a FastAPI application created by Me"}