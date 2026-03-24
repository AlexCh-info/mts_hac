import geopandas as gpd
import pandas as pd
from shapely import make_valid, buffer
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import logging

# Импорт внутренних файлов
from config.config import config

logger = logging.getLogger(__name__)

def clean_geometry(geo: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Получает колонку из gpd.GeoDataFrame
    :param geo: колонка геометрии для отчистки
    :return: отчищенная колонка
    """
    if geo is None or geo.is_empty():
        return None

    try:
        # Исправление топологических ошибок и т.д.
        if not geo.is_valid():
            geo = make_valid(geo)

        # Если после получилась коллекция берем крупнейший полигон
        if geo.geom_type == 'MultiPolygon':
            if len(geo.geoms) == 0:
                return None
            geo = max(geo.geoms, key=lambda g: g.area)

        # Если это не полигон после преобразований, отбрасываем
        if geo.geom_type not in ['Polygon', 'MultiPolygon']:
            return None

        # Проверяем площадь
        geo_proj = gpd.GeoSeries([geo], crs=config.CRS_INPUT).to_crs(config.CRS_METRIC)[0]
        if geo_proj.area < config.MIN_AREA_SQM:
            return None

        # Простая буферизация
        geo = geo.buffer(0)

        return geo
    except Exception as e:
        logger.debug(f'Ошибка при отчистке геометрии {str(e)}')
        return None

def clean_geodataframe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Отчистка геометрий ко всему GeoDataFrame
    :param gdf: GeoDataFrame
    :return: GeoDataFrame отчищенный
    """
    logger.info(f"Отчистка геометрий: {len(gdf)} записей")

    # общее кол-во записей
    initial_count = len(gdf)
    # копируем для безопасности
    gdf = gdf.copy()


    # применяем отчистку
    gdf['geometry'] = gdf['geometry'].apply(clean_geometry)

    # Удаляем строки с невалидными геометриями
    gdf = gdf[gdf['geometry'].notna()].reset_index(drop=True)

    removed = initial_count - len(gdf)
    logger.info(f'Отчистка завершена: удалено {removed} записей в процентах ({removed/initial_count*100:.1f}%)')

    return gdf

def detect_outliers_by_height(gdf: gpd.GeoDataFrame, height_col: str = 'height') -> pd.Series:
    """
    Определяем выбросы по высоте здания
    :param gdf: gpd.GeoDataFrame
    :param height_col: название колонки с высотой
    :return: pd.Series
    """
    if height_col not in gdf.columns or gdf[height_col].isna().all():
        return pd.Series(False, index=gdf.index)

    heights = gdf[height_col].dropna()
    if len(heights) < 10:
        return pd.Series(False, index=gdf.index)

    Q1 = heights.quantile(0.25)
    Q3 = heights.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    # Уберем физически невозможные высоты, например ниже 2м или выше 300м (редкость для СПб)
    lower_bound = max(lower_bound, 2.0)
    upper_bound = min(upper_bound, 300.0)

    return ~gdf[height_col].between(lower_bound, upper_bound) # записи которые не входят в диапазон