from pathlib import Path # современная os
from dataclasses import dataclass # для модификатора класса данных

@dataclass # класс будет хранить данные
class Config:
    # Пути к данным
    DATA_DIR: Path = Path('data') # путь к папке с данными
    SOURCE_A: Path = DATA_DIR / 'cup_it_example_src_A.csv' # путь к первому csv
    SOURCE_B: Path = DATA_DIR / 'cup_it_example_src_B.csv' # путь ко второму csv
    RESULTS_DIR: Path = Path('results') # путь к папке с результатами

    # Параметры геометрии
    CRS_INPUT: str = 'EPSG:4326' # WGS84, входные данные
    CRS_METRIC: str = 'EPSG:3857' # Web Mercator для расчётов в метрах
    MIN_AREA_SQM: float = 1.0 # минимальная площадь здания 1м^2
    MIN_IOU_THRESHOLD: float = 0.25 # порог IoU для сопоставления
    MAX_AREA_RATIO: float = 4.0 # макс. отношение площадей при матчинге

    # Параметры оценки высоты
    DEFAULT_FLOOR_HEIGHT: float = 3.0  # м, если нет данных
    RESIDENTIAL_FLOOR_HEIGHT: float = 2.8 # м, для жилых зданий
    COMMERCIAL_FLOOR_HEIGHT: float = 3.5 # м, для коммерческих

    # Параметры модели (LightGBM)
    M_RADIUS_METERS: float = 50.0 # радиус поиска соседей для признаков
    MIN_NEIGHBORS_FOR_ML: int = 3 # мин. соседей для предсказания
    MODEL_N_ESTIMATORS: int = 100 # количество деревьев в LightGBM

    #Визуализация
    MAP_CENTER: tuple = (59.9343, 30.3351)  # СПб, координаты центра
    MAP_ZOOM: int = 11 # приближение
    SAMPLE_SIZE_FOR_MAP: int = 5000 # сколько зданий показать на карте

    #Производительность
    CHUNK_SIZE: int = 10000 # размер чанка для обработки
    USE_H3_INDEXING: bool = True # использовать ли H3 для ускорения
    H3_RESOLUTION: int = 9  # ~0.1 км² на гексагон

config = Config()