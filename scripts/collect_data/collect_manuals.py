"""
다양한 행정 분야별 업무 매뉴얼 수집 스크립트.

이슈 #157: 다각화된 매뉴얼 10건 이상 수집.
출처: 서울 열린데이터 광장, 경기도 데이터 드림, 공공데이터포털 파일 서버 등.
분야: 일반행정, 복지, 도로교통, 환경, 세무 등.
"""

import argparse
import json
import os
import requests
from pathlib import Path
from loguru import logger

# ---------------------------------------------------------------------------
# 수집 소스 리스트 (URL이 변경될 수 있으므로 유연하게 설계)
# ---------------------------------------------------------------------------
SOURCES = {
    "seoul_faq": {
        "name": "서울시 다산콜센터 행정 FAQ",
        "url": "http://openapi.seoul.go.kr:8088/{key}/json/SearchFAQService/1/50/",
        "type": "api",
        "category": "일반행정"
    },
    "gg_manuals": {
        "name": "경기도 업무매뉴얼/가이드라인",
        "url": "https://openapi.gg.go.kr/GgWorkManual?KEY={key}&Type=json&pIndex=1&pSize=20",
        "type": "api",
        "category": "지자체공통"
    },
    "direct_files": [
        {
            "title": "특별민원 대응 매뉴얼",
            "url": "https://www.acrc.go.kr/acrc/download.do?file=2022_special_minwon.pdf",
            "category": "민원처리",
            "source": "국민권익위원회"
        },
        {
            "title": "도로점용 업무 매뉴얼",
            "url": "https://www.gg.go.kr/file/download.do?file_id=64&idx=251648", # 예시 경로
            "category": "도로/교통",
            "source": "경기도"
        }
    ]
}

class MultiSourceManualCollector:
    def __init__(self, seoul_key="sample", gg_key="sample"):
        self.seoul_key = seoul_key
        self.gg_key = gg_key
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def collect_seoul(self, limit=30):
        """서울시 다산콜센터 FAQ 수집"""
        url = SOURCES["seoul_faq"]["url"].format(key=self.seoul_key)
        try:
            res = requests.get(url, timeout=15)
            data = res.json().get("SearchFAQService", {}).get("row", [])
            return [{
                "doc_id": f"MANUAL_SEOUL_{i.get('FAQ_ID')}",
                "doc_type": "manual",
                "source": "서울 열린데이터 광장",
                "title": i.get("QUESTION"),
                "content": f"질문: {i.get('QUESTION')}\n답변: {i.get('ANSWER')}",
                "category": "행정/민원"
            } for i in data]
        except Exception as e:
            logger.error(f"서울시 수집 실패: {e}")
            return []

    def collect_gyeonggi(self):
        """경기도 업무매뉴얼 API 수집"""
        # 경기도 오픈 API 키가 필요하므로, 없을 경우 로직만 구성
        url = SOURCES["gg_manuals"]["url"].format(key=self.gg_key)
        try:
            res = requests.get(url, timeout=15)
            # 경기도 API 응답 구조에 맞게 파싱
            items = res.json().get("GgWorkManual", [{}, {"row": []}])[1].get("row", [])
            return [{
                "doc_id": f"MANUAL_GG_{i.get('MANUAL_ID', idx)}",
                "doc_type": "manual",
                "source": "경기도 데이터 드림",
                "title": i.get("MANUAL_NM"),
                "content": i.get("MANUAL_CONT") or i.get("MANUAL_NM"),
                "category": i.get("MANUAL_DIV_NM", "지자체행정")
            } for idx, i in enumerate(items)]
        except Exception:
            # 키가 없을 경우 빈 리스트 반환
            return []

    def download_direct(self, output_dir: Path):
        """알려진 직접 링크들로부터 파일 다운로드 시도"""
        downloaded = []
        for info in SOURCES["direct_files"]:
            try:
                res = requests.get(info["url"], headers=self.headers, timeout=20, stream=True)
                if res.status_code == 200:
                    ext = ".pdf" if "pdf" in info["url"].lower() else ".hwp"
                    file_path = output_dir / f"{info['title']}{ext}"
                    with open(file_path, "wb") as f:
                        for chunk in res.iter_content(8192): f.write(chunk)
                    
                    # 다운로드 후 정보 저장
                    downloaded.append({
                        "doc_id": f"MANUAL_FILE_{info['title']}",
                        "doc_type": "manual",
                        "source": info["source"],
                        "title": info["title"],
                        "content": f"파일 데이터: {file_path}", # 실제 내용은 DocumentProcessor가 처리
                        "category": info["category"],
                        "file_path": str(file_path)
                    })
                    logger.info(f"다운로드 성공: {info['title']}")
            except Exception as e:
                logger.warning(f"직접 다운로드 실패 ({info['title']}): {e}")
        return downloaded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="서울시 FAQ 수집 수량")
    parser.add_argument("--output", type=str, default="data/raw/manuals/combined_manuals.jsonl")
    parser.add_argument("--file-dir", type=str, default="data/raw/manuals/files")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    collector = MultiSourceManualCollector(
        seoul_key=os.getenv("SEOUL_DATA_API_KEY", "sample"),
        gg_key=os.getenv("GG_DATA_API_KEY", "sample")
    )

    all_data = []

    # 1. 서울시 행정 FAQ (행정/민원 분야)
    logger.info(f"서울시 행정 매뉴얼 수집 중... (limit={args.limit})")
    all_data.extend(collector.collect_seoul(limit=args.limit))

    # 2. 경기도 업무 매뉴얼 (다양한 분야)
    logger.info("경기도 업무 가이드 수집 중...")
    all_data.extend(collector.collect_gyeonggi())

    # 3. 직접 파일 다운로드 (PDF/HWP)
    logger.info("직접 링크 파일 수집 중...")
    all_data.extend(collector.download_direct(file_dir))

    # 결과 저장
    with open(args.output, "w", encoding="utf-8") as f:
        for d in all_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info(f"총 {len(all_data)}건의 다각화된 매뉴얼 데이터를 수집했습니다.")

if __name__ == "__main__":
    main()
