import cv2
import numpy as np


def is_blurry(image, threshold=100):
    """Проверка на размытость через Лапласиан."""
    if image is None:
        raise ValueError("Изображение не может быть None")

    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


def has_jpeg_artifacts(image, threshold=0.3):
    """Обнаружение JPEG-артефактов."""
    if image is None:
        raise ValueError("Изображение не может быть None")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    laplacian = cv2.Laplacian(s_channel, cv2.CV_64F)
    artifact_score = np.mean(laplacian ** 2)

    return artifact_score > threshold, artifact_score


def inspect_photo_quality(image):
    try:
        is_blurred, blur_score = is_blurry(image)
        has_artifacts, artifact_score = has_jpeg_artifacts(image)

        print(f"Размытость: {is_blurred} (Оценка: {blur_score:.2f})")
        print(f"JPEG-артефакты: {has_artifacts} (Оценка: {artifact_score:.2f})")

        if is_blurred or has_artifacts:
            print("⚠️ Изображение «мыльное»! Низкое качество.")
        else:
            print("✅ Изображение нормального качества.")

        return is_blurred, f"{blur_score:.2f}", f"{artifact_score:.2f}", False

    except Exception as e:
        print(f"Ошибка: {e}")
        return True, "", "", True
