"""Шаг 1. Обучение демо-модели, которая находит и считает таблетки на фото.

Датасет medical-pills от Ultralytics (92 фото для обучения, 23 для проверки)
скачается сам при первом запуске. На RTX 3060 обучение займёт минут 10–15.

Запуск:
    python train.py
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")
    model.train(data="medical-pills.yaml", epochs=60, imgsz=640, batch=16,
                project="runs", name="pills", exist_ok=True)
    metrics = model.val()
    print(f"\nТочность на проверочных фото: mAP50 = {metrics.box.map50:.3f}, mAP50-95 = {metrics.box.map:.3f}")
    print(f"Веса модели: {model.trainer.best}")


if __name__ == "__main__":
    main()
