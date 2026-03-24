import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import wkt
import logging

# Импорт внутренних файлов
from config.config import config

logger = logging.getLogger(__name__)

def load_source(filepath: str, geometry_col: str = 'geometry') -> gpd.GeoDataFrame:
    """
    Загружает CSV файл в формат GeoDataFrame.
    С автоматическим определением колонки геометрии.
    :param filepath: путь до файла
    :param geometry_col: колонка с геометрией
    :return: gpd.GeoDataFrame
    """

    logger.info(f"Загрузка {filepath}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f'{Path(filepath).name} загружен')
    except:
        raise ValueError(f'Не найден путь к файлу {Path(filepath).name}')

    # Определяем колонку с геометрией
    geo_col = geometry_col if geometry_col in df.columns else 'wkt'
    if geo_col not in df.columns:
        raise ValueError(f'Не найдена колонка с геометрией: {geo_col} или wkt')

    # парсинг WKT
    df['geometry'] = df[geo_col].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
    df = df[df['geometry'].notna()].copy

    # Создаем GeoDataFrame
    try:
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=config.CRS_INPUT)
        logger.info(f'Загружено {len(gdf)} объектов, CRS {gdf.crs}')
    except Exception as e:
        print(f'Невозможно создать gdf, ошибка {str(e)}')
        return

    return gdf


def add_basic_features(gdf: gpd.GeoDataFrame, source: str) -> gpd.GeoDataFrame:
    """
    Добавлляем признаки для дальнейшего анализа
    :param gdf: таблица данных
    :param source: путь к ней
    :return: gpd.GeoDataFrame с новыми признаками
    """

    # копируем для осторожности
    gdf = gdf.copy()

    # Центроиды для быстрых расчетов (может помочь при кластеризации)
    gdf['centroid'] = gdf.geometry.centroid
    gdf['centroid_x'] = gdf.centroid.x
    gdf['centroid_y'] = gdf.centroid.y

    # Площадь (пересчитываем т.к. может быть неточной)
    gdf_proj = gdf.to_crs(config.CRS_METRIC)
    gdf['area_sqm'] = gdf_proj.geometry.area
    gdf_proj = gdf.to_crs(config.CRS_INPUT)

    # Лог-площадь для моделей
    gdf['log_area'] = (gdf['area_sqm'] + 1).apply(np.log1p)

    #Источник данных
    gdf['data_sources'] = source

    logger.info(f'Добавлены новые колонки (признаки) для источника {source}')
    return gdf




