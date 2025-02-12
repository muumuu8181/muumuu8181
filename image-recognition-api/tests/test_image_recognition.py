import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from app.main import app

client = TestClient(app)

def create_test_image(size=(100, 100), color=(255, 255, 255)):
    """テスト用の画像を作成"""
    image = Image.new('RGB', size, color)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_healthz():
    """ヘルスチェックエンドポイントのテスト"""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_image_success():
    """画像分析の正常系テスト"""
    img_bytes = create_test_image()
    files = {"file": ("test.png", img_bytes, "image/png")}
    
    response = client.post("/api/v1/analyze-image", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    
    if data["results"]:  # 結果が存在する場合
        result = data["results"][0]
        assert "object" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], str)
        assert result["confidence"].endswith("%")

def test_analyze_image_invalid_file_type():
    """不正なファイルタイプのテスト"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    
    response = client.post("/api/v1/analyze-image", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "画像ファイルのみアップロード可能です"

def test_analyze_image_file_too_large():
    """ファイルサイズ超過のテスト"""
    # 11MBの画像を作成
    large_image = create_test_image(size=(3300, 3300))
    files = {"file": ("large.png", large_image, "image/png")}
    
    response = client.post("/api/v1/analyze-image", files=files)
    assert response.status_code == 400
    assert "MB以下にしてください" in response.json()["detail"]

def test_analyze_image_no_file():
    """ファイル未指定のテスト"""
    response = client.post("/api/v1/analyze-image")
    assert response.status_code == 422  # FastAPIのバリデーションエラー
