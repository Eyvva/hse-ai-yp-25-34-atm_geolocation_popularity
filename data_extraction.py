import requests

def count_objects_nearby(lat, lon, object_type='school', radius_meters=500):
    """
    Подсчитывает количество объектов определенного типа в радиусе вокруг заданных координат.

    Parameters:
    - lat (float): Широта местоположения.
    - lon (float): Долгота местоположения.
    - object_type (str): Тип объекта для поиска (например, 'school', 'pharmacy', 'shop').
                        Поддерживаются англоязычные обозначения из тегов OpenStreetMap.
    - radius_meters (int): Радиус поиска в метрах.

    Returns:
    - count (int): Число найденных объектов.
    """
    # Создание шаблона запроса Overpass API
    query_template = """
    [out:json][timeout:25];
    (
      node(around:%d,%f,%f)["amenity"="%s"];
      way(around:%d,%f,%f)["amenity"="%s"];
      relation(around:%d,%f,%f)["amenity"="%s"];
    );
    out center;
    """ % (radius_meters, lat, lon, object_type,
           radius_meters, lat, lon, object_type,
           radius_meters, lat, lon, object_type)

    try:
        # Отправляем запрос на сервер Overpass API
        response = requests.post("http://overpass-api.de/api/interpreter",
                                 data={"data": query_template})

        if response.status_code != 200:
            raise Exception("Ошибка выполнения запроса.")
            
        elements = response.json().get("elements", [])
        return len(elements)  # Возвращаем количество найденных объектов
    except Exception as e:
        print(f"Ошибка обработки запроса: {e}")
        return None

# Пример использования
# Список интересующих объектов
# object_types = ['school', 'shop']

# Создаем новые фичи для каждого типа объекта
# for obj_type in object_types:
#     column_name = f'num_{obj_type}s'  # Имя нового столбца
#    df[column_name] = df.apply(lambda row: count_objects_nearby(row['lat'], row['long'], object_type=obj_type), axis=1)пше