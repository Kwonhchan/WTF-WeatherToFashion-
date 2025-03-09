import json
import os

def extract_clothes_info_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # metadata 아래 clothes 아래 정보 접근
        weather = data.get('metadata.clothes', {})
        clothes_type = weather.get('metadata.clothes.type')
        clothes_season = weather.get('metadata.clothes.season')
    return clothes_type, clothes_season

def extract_unique_clothes_info_from_directory(directory):
    unique_types = set()
    unique_seasons = set()
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(directory, file_name)
            clothes_type, clothes_season = extract_clothes_info_from_json(json_file_path)
            if clothes_type:
                unique_types.add(clothes_type)
            if clothes_season:
                unique_seasons.add(clothes_season)
    return unique_types, unique_seasons

# JSON 파일이 있는 디렉토리 경로
directory_path = r"Dataset\outer"

# 중복 없는 옷 종류와 계절 추출
unique_types, unique_seasons = extract_unique_clothes_info_from_directory(directory_path)

# 결과 출력
print("Unique Clothes Types:")
for clothes_type in unique_types:
    print(clothes_type)

print("\nUnique Seasons:")
for season in unique_seasons:
    print(season)
