import json
import re

# Загрузка исходного файла аннотаций
with open("all_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = []
annotation_id = 1

# Для каждого изображения назначаем категорию на основе имени файла
for image in data["images"]:
    file_name = image["file_name"]
    # По умолчанию считаем изображение безопасным (category_id = 2)
    category_id = 2
    # Ищем шаблон "IMG_" за которым следует число
    match = re.search(r"IMG_(\d+)", file_name, re.IGNORECASE)
    if match:
        number = int(match.group(1))
        # Если число в диапазоне от 2093 до 2158, то изображение считается опасным
        if 2093 <= number <= 2158:
            category_id = 1
    annotations.append({
        "id": annotation_id,
        "image_id": image["id"],
        "category_id": category_id
    })
    annotation_id += 1

# Определяем категории
categories = [
    {"id": 1, "name": "hazardous_object", "supercategory": "hazard"},
    {"id": 2, "name": "non_hazardous_object", "supercategory": "safe"}
]

# Формируем новый JSON с аннотациями и категориями
new_data = {
    "images": data["images"],
    "annotations": annotations,
    "categories": categories
}

# Сохраняем результат в новый файл
with open("all_annotations_modified.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4)

print("Новый файл аннотаций сохранён как all_annotations_modified.json")
