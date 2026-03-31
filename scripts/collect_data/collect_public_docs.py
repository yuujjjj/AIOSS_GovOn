"""
행정안전부 정부 공문서 AI 학습데이터 최종 수집 스크립트.

이슈 #157: 고품질 행정 공문서(보도자료, 보고서 등) 수집 및 인덱싱.
Base URL: apis.data.go.kr/1741000/publicDoc
상세기능: /getDocAll (5종 전체 조회)
"""

import argparse
import json
import os
import zipfile
import io
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

BASE_URL = "https://apis.data.go.kr/1741000/publicDoc/getDocAll"

class PublicDocCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_all_docs(self, limit: int = 50) -> List[Dict]:
        """모든 유형의 공문서 AI 데이터를 일괄 수집한다. ZIP 다운로드 방식을 지원함."""
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": limit,
            "type": "json"
        }
        
        try:
            logger.info(f"정부 공문서 AI 데이터 수집 시작 (URL: {BASE_URL})")
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            body = data.get("response", {}).get("body", {})
            result_list = body.get("resultList", [])

            if not result_list:
                logger.warning("API 응답에 resultList가 없습니다.")
                return []

            all_items = []
            for res in result_list:
                download_url = res.get("url")
                if download_url and res.get("extType") == "zip":
                    logger.info(f"ZIP 데이터 다운로드 중: {download_url}")
                    zip_res = requests.get(download_url, timeout=60)
                    with zipfile.ZipFile(io.BytesIO(zip_res.content)) as z:
                        for filename in z.namelist():
                            if filename.endswith(".json"):
                                with z.open(filename) as f:
                                    content = json.load(f)
                                    if isinstance(content, list):
                                        all_items.extend(content)
                                    else:
                                        all_items.append(content)
            return all_items
        except Exception as e:
            logger.error(f"공문서 API 호출 및 다운로드 실패: {e}")
            return []

    def process_item(self, item: Dict) -> List[Dict]:
        """실제 행안부 JSON 구조(meta, data.text)에 맞춰 데이터를 변환한다."""
        results = []
        
        meta = item.get("meta", {})
        data = item.get("data", {})
        
        doc_id = meta.get("doc_id", "DOC")
        title = meta.get("title") or "정부 공문서"
        content = data.get("text") or ""
        
        if not content:
            return []

        # HTML 태그 제거 및 텍스트 정제 (간단한 정규식 처리)
        import re
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()

        # 1. 요약/재구성 태스크 (데이터셋에 요약이 명시적으로 있을 경우)
        # 실제 데이터에 'summary' 필드가 위치할 가능성이 있는 곳을 모두 탐색
        summary = item.get("summary") or data.get("summary") or meta.get("summary")
        
        if summary:
            results.append({
                "instruction": "다음 정부 공문서 내용을 바탕으로 핵심 내용을 요약하고 공공기관 보고서 형식(개조식)으로 재작성하세요.",
                "input": f"제목: {title}\n본문: {clean_content}",
                "output": summary
            })
        else:
            # 요약 데이터가 없는 경우: 본문에서 제목을 유추하거나 핵심 내용을 추출하는 태스크로 활용
            results.append({
                "instruction": "다음은 정부에서 발행한 공문서 본문입니다. 이 문서의 내용을 한 문장으로 요약하고 적절한 제목을 제안하세요.",
                "input": f"본문: {clean_content[:2000]}", # 너무 길 경우 자름
                "output": f"제목: {title}\n요약: 이 문서는 {title}에 관한 내용을 담고 있습니다."
            })

        # 2. 질의응답 태스크 (qna_data 필드 탐색)
        qna = item.get("qna_data") or data.get("qna_data") or item.get("qna")
        if isinstance(qna, list):
            for qa in qna:
                q = qa.get("question") or qa.get("q")
                a = qa.get("answer") or qa.get("a")
                if q and a:
                    results.append({
                        "instruction": f"제공된 공문서 '{title}'의 내용을 바탕으로 다음 질문에 답하세요.",
                        "input": f"질문: {q}",
                        "output": a
                    })

        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/raw/manuals/public_docs.jsonl")
    args = parser.parse_args()

    api_key = os.getenv("DATA_GO_KR_API_KEY")
    if not api_key:
        logger.error("DATA_GO_KR_API_KEY 환경변수가 없습니다.")
        return

    collector = PublicDocCollector(api_key)
    raw_docs = collector.fetch_all_docs(limit=args.limit)
    
    if not raw_docs:
        logger.warning("수집된 데이터가 없습니다. API 승인 및 키 상태를 확인하세요.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_records = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for d in raw_docs:
            processed_items = collector.process_item(d)
            for item in processed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_records += 1
            
    logger.info(f"✅ 수집 및 변환 완료: {len(raw_docs)}개 문서로부터 {total_records}개 레코드 생성 -> {output_path}")

if __name__ == "__main__":
    main()
