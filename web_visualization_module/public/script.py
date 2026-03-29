import json

zone_id_map = {
    'Зона A — Лёгкое покрытие': 1,
    'Зона B — Среднее покрытие': 2,
    'Зона C — Сложное покрытие': 3,
}

with open('zones.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

for feature in data['features']:
    zone = feature['properties'].get('coverage_zone', '')
    feature['properties']['zone_id'] = zone_id_map.get(zone, 0)

with open('zones.geojson', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print('Done')