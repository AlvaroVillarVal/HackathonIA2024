from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import datetime, uvicorn
import requests


#Recibir fichero json




#FASTAPI
app = FastAPI(
    root_path="/api/"
)

@app.get("/")
def landingAPI():
    return {"API": "API is working fine"}

@app.post("/inferencia")
async def inferencia(information: dict):
    """
    Receives the data for the pressed button to get the information inserted into the database.
    
    Parameters:
    information: dictionary with all the data to be saved

    Return:
    message: status information message
    """  

    
    try:
        totalTime = information['TotalTime']
        totalScore = information['TotalScore']
        minScore = information['MinScore']
        maxScore = information['MaxScore']
        objectsCaught = information['ObjectsCaught']
        objectsLost = information['ObjectsLost']
        responseTimeAverage = information['ResponseTimeAverage']
        responseTimeMin = information['ResponseTimeMin']
        responseTimeMax = information['ResponseTimeMax']
    except:
        print("Error")
        
    content = {"SpawnRatio": 0.5, "ObjectSize": 0.5, "DistanceToPlayer": 0.5, "RewardRatio": 0}
    
    json_compatible_item_data = jsonable_encoder(content)
    return JSONResponse(content=json_compatible_item_data)

