import os

def rename_json_files(directory):
    # 디렉토리 내 모든 파일 목록을 가져옵니다.
    files = os.listdir(directory)
    
    # "front"가 파일명에 들어가는 파일들을 제외하고 삭제합니다.
    json_files = [f for f in files if f.endswith('.json') and 'front' not in f]
    
    for file in json_files:
        file_path = os.path.join(directory, file)
        os.remove(file_path)
    
    # 남아 있는 JSON 파일 목록을 가져옵니다.
    remaining_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    # 파일 이름을 1부터 차례대로 바꿉니다.
    for i, file in enumerate(remaining_files, start=1):
        new_name = f"{i}.json"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
    
    print("파일 정리가 완료되었습니다.")

# 사용할 디렉토리 경로를 지정합니다.
directory_path = "top"

# 스크립트를 실행합니다.
rename_json_files(directory_path)
