import os
from pathlib import Path

def setup_saha_test_files():
    # 기본 경로 설정
    base_dirs = ["data/raw/manuals", "data/raw/notices", "data/raw/laws"]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)

    # 1. MANUAL - 사하구 도로점용 지침 (TXT)
    with open("data/raw/manuals/saha_road_manual.txt", "w", encoding="utf-8") as f:
        f.write("[사하구 도로점용 업무 매뉴얼]\n"
                "1. 개요: 도로 구역 내 공작물 신설 시 허가 절차\n"
                "2. 필수 서류: 도로점용허가 신청서, 설계도면\n"
                "3. 수수료: 점용 면적당 조례에 따른 부과")

    # 2. NOTICE - 사하구 주차단속 공고 (TXT)
    with open("data/raw/notices/saha_parking_notice.txt", "w", encoding="utf-8") as f:
        f.write("[사하구 공고 제2026-01호]\n"
                "불법 주정차 단속 구역 확대 안내\n"
                "대상: 사하구 괴정삼거리 주변\n"
                "시행일: 2026년 4월 1일")

    # 3. LAW - 사하구 주차장 조례 (TXT)
    with open("data/raw/laws/saha_parking_ordinance.txt", "w", encoding="utf-8") as f:
        f.write("부산광역시 사하구 주차장 설치 및 관리 조례\n"
                "제1조(목적): 주차장법 위임 사항 규정\n"
                "제10조(감면): 경차 및 장애인 차량 50% 감면")

    print("✅ 사하구청 테스트 파일셋 생성 완료 (data/raw/ 하위)")

if __name__ == "__main__":
    setup_saha_test_files()
