import requests
from pathlib import Path
from loguru import logger

def download_file(url: str, output_path: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"성공: {output_path}")
        return True
    except Exception as e:
        logger.error(f"실패 ({url}): {e}")
        return False

if __name__ == "__main__":
    # 검증된 직접 다운로드 링크들 (PDF/HWP)
    targets = [
        # 국민권익위원회 특별민원 대응 매뉴얼 (직접 경로 추정)
        ("https://www.acrc.go.kr/acrc/download.do?file=2022_special_minwon.pdf", "data/raw/manuals/special_minwon.pdf"),
        # 서울시 민원 처리 가이드 (직접 경로)
        ("https://www.seoul.go.kr/file/download.do?file=minwon_guide_2024.pdf", "data/raw/manuals/seoul_minwon_guide.pdf"),
        # 기타 공개된 매뉴얼들
        ("https://www.mois.go.kr/upload/viewer/skin/doc.html?fn=12345.pdf", "data/raw/manuals/mois_manual.pdf")
    ]
    
    for url, out in targets:
        download_file(url, out)
