import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from analyzer import inspect_photo_quality

app = FastAPI()


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")

        height, width, channels = img.shape

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        is_blured, blur_score, quality_score, has_error = inspect_photo_quality(img)

        return JSONResponse({
            "is_blured": bool(is_blured),
            "blur_score": blur_score,
            "quality_score": quality_score,
            "has_error": bool(has_error),
            "filename": file.filename,
            "dimensions": f"{width}x{height}",
            "channels": channels,
            "message": "Изображение успешно обработано",
            "processed": "grayscale conversion done"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
