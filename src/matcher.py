import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
from config.config import config

# Проверяем можем ли мы ускорить процесс с помощью h3
try:
    import h3
    H3_AVAILABLE = True # импорт удался
except ImportError:
    H3_AVAILABLE = False
    logging.warning('H3 не установлен')

logger = logging.getLogger(__name__)

def calculate_iou(geom1, geom2) -> float:
    """
    IoU для двух геометрий
    :param geom1: первый параметр
    :param geom2: второй параметр
    :return: float
    """

    try:
        intersection = geom1.intersection(geom2).area
        union = geom1.union(geom2).area
        return intersection / union if union > 1e-6 else 0.0
    except:
        return 0.0

def add_h3_index(gdf: gpd.GeoDataFrame, resolution: int=None) -> gpd.GeoDataFrame:
    """Добавляем индекс h3 для ускорения поиска"""
    if not H3_AVAILABLE or not config.USE_H3_INDEXING:
        gdf['h3_index'] = 0 # заглушка
        return gdf

    resolution = resolution or config.H3_RESOLUTION
    gdf = gdf.copy()

    def geom_to_h3(geom):
        centroid = geom.centroid
        return h3.latlng_to_cell(centroid.y, centroid.x, resolution)

    gdf['h3_index'] = gdf.geometry.apply(geom_to_h3)
    return gdf

def find_matches_spatial(
        gdf_a: gpd.GeoDataFrame,
        gdf_b: gpd.GeoDataFrame,
        iou_threshold: float = None,
        area_ratio_max: float = None
) -> pd.DataFrame:
    """
    Находит соответствия между зданиями из источников А и Б.
    Используем пространственный индекс + IoU метрику.
    :param gdf_a: первый датасет
    :param gdf_b: второй датасет
    :param iou_threshold: порог для метрики
    :param area_ratio_max: максимальное отношение площадей
    :return: pd.DataFrame({'id_a', 'id_b', 'iou_score', 'area_a', 'area_b'})
    """

    iou_threshold = iou_threshold or config.MIN_IOU_THRESHOLD
    area_ratio_max = area_ratio_max or config.MAX_AREA_RATIO

    # проекция в метрическую систему координат для корректных расчетов
    gdf_a_proj = gdf_a.to_crs(config.CRS_METRIC)
    gdf_b_proj = gdf_b.to_crs(config.CRS_METRIC)

    # Группировка по h3 для ускорения
    if config.USE_H3_INDEXING and H3_AVAILABLE:
        gdf_a_proj = add_h3_index(gdf_a_proj)
        gdf_b_proj = add_h3_index(gdf_b_proj)
        groups = set(gdf_a_proj['h3_index']).intersection(set(gdf_b_proj['h3_index']))
    else:
        groups = [None]

    matches = []

    for group in tqdm(groups, desc='Матчинг по группам', disable=len(groups)==1):
        # Фильтрация
        subset_a = gdf_a_proj[gdf_a_proj['h3_index'] == group] if group else gdf_a_proj
        subset_b = gdf_b_proj[gdf_b_proj['h3_index'] == group] if group else gdf_b_proj

        if len(subset_a) == 0 or len(subset_b) == 0:
            continue

        # Находим все пространственные пересечения
        candidates = gpd.sjoin(subset_a, subset_b, predicate='intersects', how='inner', lsuffix='_a')

        if len(candidates) == 0:
            continue

        # Для каждой пары считаем IoU и фильтруем
        for idx, row in candidates.iterrows():
            geom_a = row['geometry_a']
            geom_b = row['geometry_b']

            iou = calculate_iou(geom_a, geom_b)
            if iou < iou_threshold:
                continue

            area_a = geom_a.area
            area_b = geom_b.area
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1e-6)

            if area_ratio > area_ratio_max:
                continue

            matches.append({
                'id_a': row['id_a'] if 'id_a' in row else row['id'],
                'id_b': row['id_b'] if 'id_b' in row else row['id_right'],
                'iou_score': iou,
                'area_a_sqm': area_a / 10000,
                'area_b_sqm': area_b / 10000,
                'area_ratio': area_ratio
            })
    df_matches = pd.DataFrame(matches)
    logger.info(f'Найдено {len(df_matches)} потенциальных совпадений')

    if len(df_matches) > 0 and 'id_a' in df_matches.columns:
        df_matches = df_matches.iloc[df_matches.groupby('id_a')['iou_score'].idxmax()].reset_index(drop=True)
        logger.info(f'После удаления дубликатов: {len(df_matches)}')
    return df_matches