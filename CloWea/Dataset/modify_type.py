import json
import os

# 바지 유형을 결정하는 기준
def determine_pants_type(pants_length, inner_seam_length):
    if pants_length <= 40 or inner_seam_length <= 20:
        return "반바지"
    else:
        return "긴바지"

# 스커트 유형을 결정하는 기준
def determine_skirt_type(skirt_length):
    if skirt_length <= 40:
        return "숏스커트"
    else:
        return "롱스커트"

# JSON 파일이 있는 폴더 경로 설정
folder_path = r"Dataset\bottom"  # JSON 파일들이 있는 폴더의 경로로 변경하세요

# 폴더 내의 모든 JSON 파일 목록 가져오기
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

for file_name in json_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clothes_type = data["metadata.clothes"]["metadata.clothes.type"]

    clothing_type = "알 수 없음"
    
    if "pants" in clothes_type:
        # 바지 유형 결정
        try:
            pants_length = data["metadata.clothes"]["metadata.pants.pants_length"]
            pants_length = float(pants_length)
        except KeyError:
            pants_length = None

        try:
            inner_seam_length = data["metadata.clothes"]["metadata.pants.inner_seam_length"]
            inner_seam_length = float(inner_seam_length)
        except KeyError:
            inner_seam_length = None

        if pants_length is not None and inner_seam_length is not None:
            clothing_type = determine_pants_type(pants_length, inner_seam_length)
        else:
            clothing_type = "null"
    
    elif "skirt" in clothes_type:
        # 스커트 유형 결정
        try:
            skirt_length = data["metadata.clothes"]["metadata.skirt.length"]
            skirt_length = float(skirt_length)
        except KeyError:
            skirt_length = None

        if skirt_length is not None:
            clothing_type = determine_skirt_type(skirt_length)
        else:
            clothing_type = "null"

    # 의류 유형 추가
    data["metadata.clothes"]["clothing_type"] = clothing_type

    # JSON 파일에 변경사항 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

print("모든 JSON 파일에 의류 유형 추가 완료.")
