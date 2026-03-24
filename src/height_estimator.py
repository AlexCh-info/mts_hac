import pandas as pd
import numpy as np
import geopandas as gpd
import logging

from webcolors import hex_to_rgb_percent

from config.config import config

logger = logging.getLogger(__name__)

def estimate_height_from_source_b(row_b: pd.Series) -> tuple[float, str]:
    """
    :param row_b: параметр из параметра Б
    :return: (height_meters, method_description)
    """

    # Приоритет 1: прямая высота
    if pd.notna(row_b.get('height')) and row_b['height'] > 2:
        return row_b['height'], 'direct_height_B'

    #Приоритет 2: расчет по этажам и средней высоте этажа
    if pd.notna(row_b.get('stairs')) and row_b['stairs'] > 0:
        floor_h = row_b.get('avg_floor_height')
        if pd.notna(floor_h) and floor_h > 2:
            height = row_b['stairs'] * floor_h
            return height, 'stairs_x_avg_floor_B'
        # Если avg_floor_height нет - используем дефолт по типу здания
        purpose = str(row_b.get('purpose_of_building', '')).lower()
        if 'жилое' in purpose or 'residential' in purpose:
            h_per_floor = config.RESIDENTIAL_FLOOR_HEIGHT
        elif any(kw in purpose for kw in ['офис', 'торг', 'бизнес', 'commercial']):
            h_per_floor = config.RESIDENTIAL_FLOOR_HEIGHT
        else:
            h_per_floor = config.DEFAULT_FLOOR_HEIGHT
        return row_b['stairs'] * h_per_floor, 'stairs_x_default_floor'

    return None, 'no_data_B'

def estimate_height_from_source_a(row_a: pd.Series, row_b: pd.Series = None) -> tuple[float, str]:
    if pd.notna(row_a.get('gkh_floor_count_max')) and row_a['gkh_floor_count_max'] > 0:
        floors = row_a['gkh_floor_count_max']
        tags = str(row_a.get('tags', '')).lower()
        purpose = str(row_b.get('purpose_of_building', '') if row_b is not None else '').lower()

        if 'жилое' in tags or 'жилое' in purpose:
            h_per_floor = config.RESIDENTIAL_FLOOR_HEIGHT
        elif any(kw in tags + purpose for kw in ['офис', 'торг', 'бизнес']):
            h_per_floor = config.COMMERCIAL_FLOOR_HEIGHT
        else:
            h_per_floor = config.DEFAULT_FLOOR_HEIGHT

        return floors * h_per_floor, 'estimated_from_A_floors'

    return None, 'no_data_A'

def build_height_priority_pipeline(df_matched: pd.DataFrame, gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame) -> pd.DataFrame:
    results = []

    for _, match in df_matched.iterrows():
        id_a, id_b, = match['id_a'], match['id_b']

        row_a = gdf_a[gdf_a['id'] == id_a].iloc[0] if id_a in gdf_a['id'].values else None
        row_b = gdf_b[gdf_b['id'] == id_b].iloc[0] if id_b in gdf_b['id'].values else None

        height = None
        method = 'unknown'

        if row_b is not None:
            height, method = estimate_height_from_source_b(row_b)

        if height is None and row_a is not None:
            height, method = estimate_height_from_source_a(row_a, row_b)

        results.append({
            'id_a': id_a,
            'id_b': id_b,
            'final_height_m': height,
            'estimation_method': method,
            'iou_score': match['iou_score'],
            'geometry': row_a.geometry if row_a is not None else (row_b.geometry if row_b is not None else None)
        })

    return pd.DataFrame(results)