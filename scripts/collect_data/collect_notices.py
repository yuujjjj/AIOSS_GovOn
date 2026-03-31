"""
기획재정부 공공기관 정보(ALIO) 실시간 수집 스크립트.

이슈 #157: 공시정보 20건 이상 실시간 수집.
실제 확인된 API 응답 구조 및 필드명 반영.

엔드포인트: https://apis.data.go.kr/1051000/public_inst
필요 환경변수:
    DATA_GO_KR_API_KEY: 공공데이터포털 API 인증키 (Decoding)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# API 설정
# ---------------------------------------------------------------------------
BASE_URL = "https://apis.data.go.kr/1051000/public_inst"

class AlioNoticeCollector:
    def __init__(self, api_key: str):
        if not api_key:
            logger.error("DATA_GO_KR_API_KEY가 설정되지 않았습니다.")
            sys.exit(1)
        self.api_key = api_key

    def fetch_public_institutions(self, limit: int = 20) -> List[Dict]:
        """공공기관 목록 및 기본 정보를 수집한다."""
        operation = "/list" 
        url = f"{BASE_URL}{operation}"
        
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": limit,
            "resultType": "json"
        }
        
        try:
            logger.info(f"ALIO API 호출 중: {url}")
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            # 실제 확인된 응답 구조: resultCode, resultMsg, totalCount, result
            if data.get("resultCode") == 200:
                items = data.get("result", [])
                logger.info(f"성공: {len(items)}건의 기관 정보 수신")
                return items
            else:
                logger.warning(f"API 응답 오류: {data.get('resultMsg')}")
                return []
        except Exception as e:
            logger.error(f"ALIO API 호출 실패: {e}")
            return []

    def process_item(self, item: Dict, idx: int) -> Dict:
        """수집된 기관 정보를 NOTICE 인덱스 스키마로 변환한다."""
        title = item.get("instNm", f"공공기관 {idx}")
        inst_type_nm = item.get("instTypeNm", "공공기관")
        ministry_nm = item.get("sprvsnInstNm", "관련부처")
        address = item.get("roadNmAddr") or item.get("lotnoAddr", "주소 정보 없음")

        # 인덱싱될 본문 텍스트 구성
        content = (
            f"기관명: {title}\n"
            f"기관유형: {inst_type_nm}\n"
            f"주무부처: {ministry_nm}\n"
            f"주소: {address}\n"
            f"홈페이지: {item.get('siteUrl', '')}\n"
            f"전화번호: {item.get('rprsTelno', '')}\n"
            f"설립일자: {item.get('fndnYmd', '')}"
        )

        return {
            "doc_id": f"NOTICE_ALIO_{item.get('instCd', idx)}",
            "doc_type": "notice",
            "source": "기획재정부 ALIO",
            "title": f"[{inst_type_nm}] {title}",
            "content": content,
            "category": "공공기관/경영",
            "reliability_score": 0.9,
            "extras": {
                "inst_code": item.get("instCd"),
                "ministry": ministry_nm,
                "location": item.get("ctpvNm", ""),
                "homepage": item.get("siteUrl", "")
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="수집할 기관 수 (최대 1000)")
    parser.add_argument("--output", type=str, default="data/raw/notices/alio_info.jsonl")
    args = parser.parse_args()

    api_key = os.getenv("DATA_GO_KR_API_KEY")
    if not api_key:
        logger.error("환경변수 DATA_GO_KR_API_KEY가 설정되지 않았습니다.")
        return

    collector = AlioNoticeCollector(api_key)
    
    logger.info(f"기획재정부 ALIO 데이터 수집 시작 (목표: {args.limit}건)")
    raw_items = collector.fetch_public_institutions(limit=args.limit)
    
    if not raw_items:
        logger.error("데이터 수집에 실패했습니다. API 응답을 확인하세요.")
        return

    # 데이터 변환 및 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(raw_items):
            processed = collector.process_item(item, i)
            f.write(json.dumps(processed, ensure_ascii=False) + "\n")
            
    logger.info(f"수집 및 변환 완료: {len(raw_items)}건 저장됨 -> {args.output}")

if __name__ == "__main__":
    main()
