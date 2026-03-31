import os

def generate_valid_large_txt_files():
    # PDF 생성이 라이브러리 문제로 실패하므로, 
    # DocumentProcessor가 대용량 데이터를 처리하는지 테스트하기 위해
    # 실제 수만 자 분량의 사하구청 업무 텍스트 파일을 생성합니다.
    # (나중에 실제 PDF 파일을 이 자리에 덮어쓰시면 됩니다.)
    
    dirs = ["data/raw/manuals", "data/raw/laws", "data/raw/notices"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 1. MANUAL - 사하구 도로점용 업무 편람 (대용량 텍스트)
    with open("data/raw/manuals/saha_road_manual_large.txt", "w", encoding="utf-8") as f:
        f.write("[부산광역시 사하구 도로점용허가 실무 매뉴얼 전문]\n\n")
        for i in range(1, 501):
            f.write(f"제{i}장 세부 지침 제{i}절 개요: 이 규정은 사하구 관내 도로의 효율적 관리와 "
                    f"민원인의 편의를 위해 제정되었습니다. {i}번째 세부 항목에 따르면...\n")
            f.write("도로점용허가를 받으려는 자는 신청서에 설계도면 및 주요 구조물 도면을 첨부하여 "
                    "사하구청장에게 제출하여야 합니다. 이 경우 담당 공무원은 현장을 확인하여...\n\n")

    # 2. LAW - 사하구 조례 통합본 (대용량 텍스트)
    with open("data/raw/laws/saha_ordinance_combined.txt", "w", encoding="utf-8") as f:
        f.write("[부산광역시 사하구 조례 및 규칙 통합 지식베이스]\n\n")
        for i in range(1, 301):
            f.write(f"제{i}조(조례 명칭 {i}): 사하구의 발전을 위하여 본 조례는 다음과 같이 규정한다.\n"
                    f"1. 해당 사무의 범위는 관내 전역으로 한다. 2. {i}항에 따른 수수료는 별표에 따른다.\n\n")

    print("✅ 사하구청 대용량(수만 자) 실무 테스트 데이터셋 생성 완료.")

if __name__ == "__main__":
    generate_valid_large_txt_files()
