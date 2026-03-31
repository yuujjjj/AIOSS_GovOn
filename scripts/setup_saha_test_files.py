import os
from pathlib import Path

def setup_saha_test_files():
    # 기본 경로 설정
    dirs = {
        "manuals": "data/raw/manuals",
        "notices": "data/raw/notices",
        "laws": "data/raw/laws"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1. MANUAL - 사하구 도로점용 지침 (PDF 확장자 시뮬레이션)
    # 실제 PDF 바이너리는 아니나, 확장자 인식 및 파서 호출 여부 확인용
    with open(os.path.join(dirs["manuals"], "saha_road_guide.pdf"), "w", encoding="utf-8") as f:
        f.write("%PDF-1.4 (Simulated)\n"
                "[사하구 도로점용 업무 매뉴얼]\n"
                "1. 신청 대상: 도로 구역 내 공작물 신설 시\n"
                "2. 필수 서류: 점용허가 신청서")

    # 2. LAW - 사하구 주차장 조례 (HWP 확장자 시뮬레이션)
    with open(os.path.join(dirs["laws"], "saha_parking_rules.hwp"), "w", encoding="utf-8") as f:
        f.write("HWP Document (Simulated)\n"
                "부산광역시 사하구 주차장 설치 및 관리 조례\n"
                "제10조(감면): 경차 및 장애인 차량 50% 감면")

    # 3. NOTICE - 사하구 주차단속 공고 (TXT)
    with open(os.path.join(dirs["notices"], "saha_parking_notice.txt"), "w", encoding="utf-8") as f:
        f.write("[사하구 공고 제2026-01호]\n"
                "불법 주정차 단속 구역 확대 안내")

    # 4. 기존 JSONL 데이터와도 호환성 유지
    # (이미 성공한 ALIO 데이터 등을 위해 디렉토리 구조 유지)

    print("✅ 사하구청 다형식(PDF, HWP, TXT) 테스트 파일셋 생성 완료")

if __name__ == "__main__":
    setup_saha_test_files()
