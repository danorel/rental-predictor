from pathlib import Path
from requests import get
from json import loads, dump, dumps


def send_request():
    params = {
        "apartment_verification": True,
        "price_verification": True,
        "floor": 10,
        "walls": "кирпич",
        "rooms": 1,
        "year": 2015,
        "image_urls": [],
        "latitude": 50.45466,
        "longitude": 30.5238,
        "description": "Продаю квартиру 50кв.м. Район Лук\'яновка",
        "total_area": 60,
        "living_area": 50,
        "kitchen_area": 18,
        "city": "Київ",
        "region": "Київська",
        "heating": "централизованное",
    }
    json_params = dumps(params)
    from project.app import app
    response = get(
        url=f"http://{app.config['HOST']}:{app.config['PORT']}/api/v1/price/predict",
        params={
            "model": "nn",
            "features": json_params
        }
    )
    print(response.text)
    # output = response.text
    # print(output)
    # path = Path('.') / Path('results.json')
    # with open(path, "a", encoding="utf-8") as file:
    #     data = loads(output)
    #     dump(data, file, ensure_ascii=False, indent=4)
    #     file.write(",\n")


if __name__ == "__main__":
    send_request()
