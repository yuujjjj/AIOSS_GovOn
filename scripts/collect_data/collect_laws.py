"""
국가법령정보센터(law.go.kr) 법령 실시간 수집 스크립트.

이슈 #157: 법령 50건 이상 실제 API 수집.
샘플 데이터나 하드코딩 없이 오직 API 결과만을 사용함.

필요 환경변수:
    LAW_GO_KR_OC: 국가법령정보센터에서 발급받은 부처코드(인증키)
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

# API 설정
BASE_URL = "http://www.law.go.kr/DRF/lawSearch.do"
DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"

# 민원 밀접 키워드
TARGET_KEYWORDS = [
    "민원", "개인정보", "도로", "교통", "환경", "위생", "복지", "건축", "지방자치"
]

class LawCollector:
    def __init__(self, oc: str):
        if not oc or oc == "test":
            logger.error("유효한 LAW_GO_KR_OC가 설정되지 않았습니다. API 키를 발급받아 설정해주세요.")
            sys.exit(1)
        self.oc = oc

    def search_laws(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        params = {
            "target": "law",
            "query": query,
            "type": "XML",
            "OC": self.oc,
            "display": limit
        }
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            laws = []
            for item in root.findall(".//law"):
                laws.append({
                    "law_id": item.findtext("법령ID"),
                    "title": item.findtext("법령명한글")
                })
            return laws
        except Exception as e:
            logger.error(f"검색 API 호출 실패 ({query}): {e}")
            return []

    def get_law_detail(self, law_id: str) -> Optional[Dict[str, Any]]:
        params = {
            "target": "law",
            "MST": law_id,
            "type": "XML",
            "OC": self.oc
        }
        try:
            response = requests.get(DETAIL_URL, params=params, timeout=20)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            content_parts = []
            title = root.findtext(".//기본정보/법령명한글")
            
            # 조문 추출
            for jo in root.findall(".//조문단위"):
                jo_text = (jo.findtext("조문내용") or "").strip()
                hangs = [h.findtext("항내용").strip() for h in jo.findall(".//항단위") if h.findtext("항내용")]
                if jo_text:
                    content_parts.append(f"{jo_text}\n" + "\n".join(hangs))
            
            if not content_parts:
                return None

            return {
                "doc_id": f"LAW_{law_id}",
                "doc_type": "law",
                "source": "국가법령정보센터",
                "title": title,
                "content": "\n\n".join(content_parts),
                "law_number": root.findtext(".//기본정보/법령번호"),
                "enforcement_date": root.findtext(".//기본정보/시행일자")
            }
        except Exception as e:
            logger.error(f"상세 API 호출 실패 (ID={law_id}): {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/raw/laws.jsonl")
    args = parser.parse_args()

    oc = os.getenv("LAW_GO_KR_OC")
    if not oc:
        logger.error("환경변수 LAW_GO_KR_OC가 없습니다.")
        return

    collector = LawCollector(oc)
    all_laws = []
    collected_ids = set()

    for kw in TARGET_KEYWORDS:
        if len(all_laws) >= args.limit: break
        
        logger.info(f"법령 검색 중: {kw}")
        for l_info in collector.search_laws(kw):
            if l_info["law_id"] in collected_ids: continue
            
            detail = collector.get_law_detail(l_info["law_id"])
            if detail:
                all_laws.append(detail)
                collected_ids.add(l_info["law_id"])
                logger.info(f"[{len(all_laws)}] 수집: {detail['title']}")
                time.sleep(0.3)
            
            if len(all_laws) >= args.limit: break

    # 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for l in all_laws:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    
    logger.info(f"수집 완료: {len(all_laws)}건")

if __name__ == "__main__":
    main()
