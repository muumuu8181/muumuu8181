from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
import io
from app.services.image_recognition import ImageRecognitionService
from app.utils.logger import logger
import os

router = APIRouter()
recognition_service = ImageRecognitionService()

# 一般的な物体の日本語訳
OBJECT_TRANSLATIONS = {
    "cup": "コップ",
    "pencil": "鉛筆",
    "pen": "ペン",
    "book": "本",
    "chair": "椅子",
    "table": "テーブル",
    "computer": "コンピュータ",
    "phone": "電話",
    "bottle": "ボトル",
    "glass": "グラス",
    "laptop": "ノートパソコン",
    "keyboard": "キーボード",
    "mouse": "マウス",
    "desk": "机",
    "paper": "紙",
    "notebook": "ノート",
    "bag": "バッグ",
    "backpack": "リュックサック",
    "headphones": "ヘッドフォン",
    "camera": "カメラ"
}

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))  # デフォルト10MB

@router.post("/analyze-image")
async def analyze_image(file: UploadFile):
    logger.log("INFO", "image_upload_started", filename=file.filename)
    
    if not file.content_type.startswith('image/'):
        logger.log("ERROR", "invalid_file_type", 
                  filename=file.filename, 
                  content_type=file.content_type)
        raise HTTPException(
            status_code=400,
            detail="画像ファイルのみアップロード可能です"
        )
    
    try:
        contents = await file.read()
        if len(contents) > MAX_IMAGE_SIZE:
            logger.log("ERROR", "file_too_large", 
                      filename=file.filename, 
                      size=len(contents))
            raise HTTPException(
                status_code=400,
                detail=f"画像サイズは{MAX_IMAGE_SIZE/1024/1024}MB以下にしてください"
            )
        
        image = Image.open(io.BytesIO(contents))
        predictions = recognition_service.predict(image)
        
        results = [
            {
                "object": OBJECT_TRANSLATIONS.get(pred[0], pred[0]),
                "confidence": f"{pred[1]:.1%}"
            }
            for pred in predictions[:5]  # 上位5件のみ返す
        ]
        
        logger.log("INFO", "image_analysis_completed", 
                  filename=file.filename, 
                  results=results)
        
        return {
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log("ERROR", "image_analysis_failed",
                  filename=file.filename,
                  error=str(e))
        raise HTTPException(
            status_code=500,
            detail="画像の分析中にエラーが発生しました"
        )
