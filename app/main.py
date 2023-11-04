import cv2
import numpy as np
import librosa
from fastapi import FastAPI,UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "This is my APIS"}


def convert_audio_to_array(audio_file):
    audio_array, sr = librosa.load(audio_file, sr=None)
    test_mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)

    resized_data = cv2.resize(test_mfcc, (216, 40))
    resized_data = resized_data.reshape(1, 40, 216, 1)

    return resized_data

@app.post("/tovector")
async def upload_file(audio_file: UploadFile):
    if audio_file:
        audio_array = convert_audio_to_array(audio_file.file)
        return {"vecsound": audio_array.tolist()} 
        # return {"vecsound": audio_array}  
    else:
        return {"message": "ไม่พบไฟล์เสียงที่อัพโหลด"}
    
