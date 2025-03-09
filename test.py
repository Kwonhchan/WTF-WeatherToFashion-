import requests

url = "http://220.69.241.92:8080/predict"

# 테스트 데이터를 정의합니다.
test_data = {
    "gender": "FEMALE",
    "age": "40대",
    "temperature": 2.0
}

# POST 요청을 보내고 응답을 받습니다.
response = requests.post(url, json=test_data)

# 응답을 JSON 형태로 출력합니다.
if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print("Failed to get a valid response. Status code:", response.status_code)
    print("Response content:", response.content)