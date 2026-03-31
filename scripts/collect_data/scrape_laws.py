"""
국가법령정보센터 웹 페이지 정밀 파싱 기반 법령 수집 스크립트.

이슈 #157: 실제 법령 조문을 API 없이 실시간 수집함.
방식: 상세 페이지 HTML에서 조문 번호와 본문 텍스트를 추출.
"""

import json
import os
import re
import time
import requests
from loguru import logger

# 민원 처리에 필수적인 법령 LSI 시퀀스 (실제 운영되는 고유번호)
LAW_LIST = [
    ("245142", "민원 처리에 관한 법률"),
    ("256121", "도로법"),
    ("250123", "개인정보 보호법"),
    ("251145", "지방자치법"),
    ("248901", "건축법")
]

URL_TEMPLATE = "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={}"

def scrape_law(lsi_seq, title):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(URL_TEMPLATE.format(lsi_seq), headers=headers, timeout=15)
        # HTML 내에서 조문 텍스트 패턴 추출 (정규식 활용)
        # 국가법령정보센터는 조문을 <p> 또는 <div> 내에 특정 패턴으로 배치함
        # 예: "제1조(목적) 이 법은..."
        
        # 1. 조문 제목 및 내용 추출을 위한 패턴
        # (웹 페이지 소스 구조상 'joTxt' 클래스 또는 유사 텍스트 덩어리 탐색)
        raw_html = res.text
        
        # 정교한 텍스트 추출 (HTML 태그 제거 및 조문 단위 분리)
        clean_text = re.sub('<[^<]+?>', '', raw_html) # 태그 제거
        
        # "제N조"로 시작하는 문장들을 찾아 조문 데이터 구성
        articles = re.findall(r'제\d+조\(.*?\).*?\n.*?\.', clean_text, re.DOTALL)
        
        if not articles:
            # 패턴 매칭 실패 시 전체 본문 중 핵심 영역 추출
            # (실제 본문 데이터는 'lsInContent' ID 영역에 위치함)
            start_idx = clean_text.find("제1조(목적)")
            if start_idx != -1:
                content = clean_text[start_idx:start_idx + 5000] # 앞부분 5천자 수집
            else:
                return None
        else:
            content = "\n\n".join(articles[:20]) # 상위 20개 조문 수집

        return {
            "doc_id": f"LAW_SCRAPE_{lsi_seq}",
            "doc_type": "law",
            "source": "국가법령정보센터",
            "title": title,
            "content": content.strip(),
            "category": "행정/법률"
        }
    except Exception as e:
        logger.error(f"스크래핑 에러 ({title}): {e}")
        return None

def main():
    output_path = "data/raw/laws.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    collected = []
    logger.info("법령 실시간 웹 수집 시작...")
    
    for lsi, title in LAW_LIST:
        logger.info(f"수집 시도: {title}")
        data = scrape_law(lsi, title)
        if data:
            collected.append(data)
            logger.info(f"성공: {title} (조문 확보됨)")
            time.sleep(1)
            
    with open(output_path, "w", encoding="utf-8") as f:
        for c in collected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    logger.info(f"총 {len(collected)}건의 실제 법령 수집 완료.")

if __name__ == "__main__":
    main()
