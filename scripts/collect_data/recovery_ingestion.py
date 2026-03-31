import requests
import json
import os
from loguru import logger

def collect_seoul_manuals():
    """서울시 다산콜센터 FAQ (진짜 텍스트) 수집"""
    logger.info("서울시 행정 매뉴얼 수집 시작...")
    url = "http://openapi.seoul.go.kr:8088/sample/json/SearchFAQService/1/50/"
    try:
        res = requests.get(url, timeout=15)
        data = res.json()
        rows = data.get("SearchFAQService", {}).get("row", [])
        if rows:
            output_path = "data/raw/manuals/seoul_manuals.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for r in rows:
                    item = {
                        "doc_id": f"MANUAL_SEOUL_{r.get('FAQ_ID')}",
                        "doc_type": "manual",
                        "source": "서울 열린데이터 광장",
                        "title": r.get("QUESTION"),
                        "content": f"질문: {r.get('QUESTION')}\n답변: {r.get('ANSWER')}",
                        "category": "일반행정"
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"✅ 서울시 매뉴얼 수집 완료: {len(rows)}건")
            return True
    except Exception as e:
        logger.error(f"서울시 수집 실패: {e}")
    return False

def collect_alio_info():
    """이미 검증된 ALIO API를 통해 기관 정보 재수집 (안전성 확보)"""
    key = "e8cd6e25666c8391c17658e557e65e526051027e52a060548e0d5879a9fe5fc4"
    logger.info("ALIO 공공기관 정보 수집 시작...")
    url = "https://apis.data.go.kr/1051000/public_inst/list"
    params = {"serviceKey": key, "pageNo": 1, "numOfRows": 100, "resultType": "json"}
    try:
        res = requests.get(url, params=params, timeout=15)
        data = res.json()
        if data.get("resultCode") == 200:
            items = data.get("result", [])
            output_path = "data/raw/notices/alio_info.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for it in items:
                    item = {
                        "doc_id": f"NOTICE_ALIO_{it.get('instCd')}",
                        "doc_type": "notice",
                        "source": "기획재정부 ALIO",
                        "title": it.get("instNm"),
                        "content": f"기관명: {it.get('instNm')}\n유형: {it.get('instTypeNm')}\n설립목적: {it.get('instGoal', '')}\n주요사업: {it.get('mainBiz', '')}",
                        "category": "경영/공시"
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"✅ ALIO 데이터 수집 완료: {len(items)}건")
            return True
    except Exception as e:
        logger.error(f"ALIO 수집 실패: {e}")
    return False

if __name__ == "__main__":
    os.makedirs("data/raw/manuals", exist_ok=True)
    os.makedirs("data/raw/notices", exist_ok=True)
    
    s1 = collect_seoul_manuals()
    s2 = collect_alio_info()
    
    if s1 and s2:
        logger.info("🚀 모든 핵심 데이터 수집에 성공했습니다! (손상 없음)")
    else:
        logger.warning("⚠️ 일부 데이터 수집에 실패했습니다.")
