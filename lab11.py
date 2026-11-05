import os
import time
import torch
from transformers import pipeline
from huggingface_hub import constants
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def main():
    print("="*50)
    print(" ШАГ 2. АНАЛИЗ ТОНАЛЬНОСТИ ТЕКСТА (ruBERT-tiny)")
    print("="*50)
    # Скачивание модели ~47 МБ
    clf = pipeline('sentiment-analysis', model='cointegrated/rubert-tiny-sentiment-balanced')
    
    phrases = [
        'Отличный курс!', 
        'Ничего не понял', 
        'Нормально, но долго',
        'Вроде бы неплохо, но ожидал большего',
        'Ужасно скучно'
    ]
    for t in phrases:
        t0 = time.time()
        res = clf(t)
        dt = (time.time() - t0) * 1000
        print(f"Текст: '{t}'\nОтвет: {res} | Время: {dt:.1f} мс\n")

    print("="*50)
    print(" ШАГ 3. ПЕРЕВОД ТЕКСТА (Transformers v5 Style)")
    print("="*50)
    try:
        # Используем современный пайплайн text-generation
        translator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
        text_to_translate = 'Нейросеть — это цепочка матричных умножений.'
        
        # Формируем обычный текстовый промпт, это самый стабильный способ для pipeline
        prompt = f"System: Translate the following Russian text into English.\nUser: {text_to_translate}\nAssistant:"
        
        t0 = time.time()
        # Генерируем перевод
        outputs = translator(prompt, max_new_tokens=64, return_full_text=False)
        dt = (time.time() - t0) * 1000
        
        # Безопасно извлекаем текст из структуры ответа
        translation = outputs[0]['generated_text']
        
        print(f"Оригинал: {text_to_translate}")
        print(f"Перевод:  {translation.strip()} | Время: {dt:.1f} мс\n")
        
    except Exception as e:
        print(f"Ошибка при генерации перевода: {e}\n")

    print(" ШАГ 4. ПРОВЕРКА КЭША HUGGING FACE")
    print("="*50)
    print(f"Путь к кэшу моделей: {constants.HF_HUB_CACHE}")
    print("Скачанные репозитории в кэше:")
    if os.path.exists(constants.HF_HUB_CACHE):
        for d in os.listdir(constants.HF_HUB_CACHE):
            print(f" - {d}")
    else:
        print("Папка кэша еще не создана.")
    print()


    print("="*50)
    print(" ШАГ 5. КЛАССИФИКАЦИЯ ФОТО (MobileNetV3 Small)")
    print("="*50)
    # Скачивание весов ~10 МБ. На CPU отрабатывает мгновенно (~20 мс)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights).eval()
    transforms = weights.transforms()

    image_path = 'photo.jpg'
    if not os.path.exists(image_path):
        print(f"Внимание: Файл '{image_path}' не найден! Положите любую картинку в папку с кодом.")
    else:
        img = Image.open(image_path).convert('RGB')
        
        t0 = time.time()
        with torch.no_grad():
            # Накатываем трансформации, добавляем размерность батча (unsqueeze) и берем Softmax
            predictions = model(transforms(img).unsqueeze(0)).softmax(1)[0]
        dt = (time.time() - t0) * 1000
        
        print(f"Файл '{image_path}' успешно распознан за {dt:.1f} мс.")
        print("Топ-5 предсказаний по версии ImageNet:")
        for value, idx in zip(*predictions.topk(5)):
            category = weights.meta["categories"][idx.item()]
            print(f" - {category:25s} {value.item():.3f}")
        print()


    print("="*50)
    print(" ШАГ 6. СРАВНЕНИЕ КОЛИЧЕСТВА ПАРАМЕТРОВ")
    print("="*50)
    # Считаем количество обучаемых параметров в MobileNetV3
    mobilenet_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Количество параметров в MobileNetV3 Small: {mobilenet_params:,} (~2.5 млн)")
    print("Для сравнения, наша MLP сеть из задачи 10 имела около 100,000 параметров.")
    print("="*50)

if __name__ == '__main__':
    main()
