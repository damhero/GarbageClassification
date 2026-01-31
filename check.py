from ultralytics import YOLO
import os
from pathlib import Path
import time
import pandas as pd
from datetime import datetime

# Ścieżki do modeli
models_paths = {
    "YOLO11n": "yolo11n run/runs/detect/train/weights/best.onnx",
    "YOLO11s": "yolo11s_results/runs/detect/train/weights/best.onnx",
    "YOLO11m": "yolo11m_results/runs/detect/train/weights/best.onnx"
}

# Ścieżka do zdjęć testowych
test_images_dir = "nasze-zdjecia/images"
classes_file = "nasze-zdjecia/classes.txt"

# Ścieżki do plików wynikowych
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_csv = f"ewaluacja_wyniki_{timestamp}.csv"
results_txt = f"ewaluacja_raport_{timestamp}.txt"
results_images_dir = f"ewaluacja_obrazy_{timestamp}"
val_config_yaml = f"val_config_{timestamp}.yaml"

# Tworzenie katalogu na wyniki
os.makedirs(results_images_dir, exist_ok=True)

# Tworzenie pliku konfiguracyjnego YAML dla walidacji
with open(val_config_yaml, 'w') as f:
    f.write(f"""path: nasze-zdjecia
train: images
val: images
test: images

names:
""")
    # Wczytanie klas
    with open(classes_file, 'r') as cf:
        classes = [line.strip() for line in cf.readlines()]
    for idx, cls in enumerate(classes):
        f.write(f"  {idx}: {cls}\n")

print("=" * 80)
print("EWALUACJA MODELI YOLO11n, YOLO11s, YOLO11m")
print("=" * 80)
print(f"\nKlasy: {', '.join(classes)}")
print(f"Zdjęcia testowe: {test_images_dir}\n")

# Zbieranie zdjęć
image_files = list(Path(test_images_dir).glob("*.JPEG")) + \
              list(Path(test_images_dir).glob("*.jpeg")) + \
              list(Path(test_images_dir).glob("*.png"))

print(f"Znaleziono {len(image_files)} zdjęć testowych\n")

# Słownik do przechowywania wyników
results_summary = {}
val_metrics_summary = {}
detailed_results = []  # Lista do przechowywania szczegółowych wyników

# Ewaluacja każdego modelu
for model_name, model_path in models_paths.items():
    print("=" * 80)
    print(f"Ewaluacja modelu: {model_name}")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"❌ Model nie znaleziony: {model_path}")
        continue
    
    # Wczytanie modelu
    print(f"Ładowanie modelu z: {model_path}")
    model = YOLO(model_path)
    
    # Tworzenie katalogu dla wyników tego modelu
    model_output_dir = os.path.join(results_images_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Statystyki
    total_detections = 0
    total_time = 0
    detections_per_image = []
    confidences = []
    
    # Predykcja na każdym zdjęciu
    for img_path in image_files:
        print(f"\n📷 Przetwarzanie: {img_path.name}")
        
        start_time = time.time()
        results = model(str(img_path), verbose=False)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Analiza wyników
        result = results[0]
        num_detections = len(result.boxes)
        total_detections += num_detections
        detections_per_image.append(num_detections)
        
        # Zapisanie obrazu z detekcjami
        output_image_path = os.path.join(model_output_dir, img_path.name)
        result.save(filename=output_image_path)
        
        print(f"   ⏱️  Czas: {inference_time:.3f}s")
        print(f"   🎯 Detekcje: {num_detections}")
        print(f"   💾 Zapisano: {output_image_path}")
        
        if num_detections > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                confidences.append(confidence)
                class_name = classes[class_id] if class_id < len(classes) else f"Class_{class_id}"
                print(f"      - {class_name}: {confidence:.2%}")
                
                # Zapisz szczegółowy wynik
                detailed_results.append({
                    "Model": model_name,
                    "Obraz": img_path.name,
                    "Klasa": class_name,
                    "Pewność": confidence,
                    "Czas_inferencji_s": inference_time
                })
    
    # Walidacja modelu z wykorzystaniem etykiet
    print(f"\n{'─' * 80}")
    print(f"WALIDACJA MODELU {model_name} Z ETYKIETAMI")
    print(f"{'─' * 80}")
    
    try:
        metrics = model.val(data=val_config_yaml, split='val', verbose=True)
        
        # Wyciąganie metryk
        val_metrics_summary[model_name] = {
            "mAP50": round(float(metrics.box.map50), 4) if hasattr(metrics.box, 'map50') else 0,
            "mAP50-95": round(float(metrics.box.map), 4) if hasattr(metrics.box, 'map') else 0,
            "Precision": round(float(metrics.box.p.mean() if hasattr(metrics.box.p, 'mean') else metrics.box.p), 4) if hasattr(metrics.box, 'p') else 0,
            "Recall": round(float(metrics.box.r.mean() if hasattr(metrics.box.r, 'mean') else metrics.box.r), 4) if hasattr(metrics.box, 'r') else 0,
            "F1": round(float(metrics.box.f1.mean() if hasattr(metrics.box.f1, 'mean') else metrics.box.f1), 4) if hasattr(metrics.box, 'f1') else 0,
        }
        
        print(f"\n  mAP@50:      {val_metrics_summary[model_name]['mAP50']:.4f}")
        print(f"  mAP@50-95:   {val_metrics_summary[model_name]['mAP50-95']:.4f}")
        print(f"  Precision:   {val_metrics_summary[model_name]['Precision']:.4f}")
        print(f"  Recall:      {val_metrics_summary[model_name]['Recall']:.4f}")
        print(f"  F1-score:    {val_metrics_summary[model_name]['F1']:.4f}")
    except Exception as e:
        print(f"⚠️  Błąd podczas walidacji: {e}")
        val_metrics_summary[model_name] = {
            "mAP50": 0,
            "mAP50-95": 0,
            "Precision": 0,
            "Recall": 0,
            "F1": 0,
        }
    
    # Podsumowanie dla modelu
    avg_time = total_time / len(image_files) if image_files else 0
    avg_detections = total_detections / len(image_files) if image_files else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    results_summary[model_name] = {
        "Całkowity czas [s]": round(total_time, 3),
        "Średni czas/obraz [s]": round(avg_time, 3),
        "Całkowita liczba detekcji": total_detections,
        "Średnia liczba detekcji/obraz": round(avg_detections, 2),
        "Średnia pewność": round(avg_confidence, 4),
        "Obrazy przetworzone": len(image_files)
    }
    
    print(f"\n{'─' * 80}")
    print(f"PODSUMOWANIE {model_name}:")
    print(f"{'─' * 80}")
    print(f"  Całkowity czas:              {total_time:.3f}s")
    print(f"  Średni czas na obraz:        {avg_time:.3f}s")
    print(f"  Całkowita liczba detekcji:   {total_detections}")
    print(f"  Średnia liczba detekcji:     {avg_detections:.2f}")
    print(f"  Średnia pewność:             {avg_confidence:.2%}")
    print()

# Porównanie modeli
print("=" * 80)
print("PORÓWNANIE MODELI - WYDAJNOŚĆ")
print("=" * 80)

df = pd.DataFrame(results_summary).T
print(df.to_string())
print("\n")

print("=" * 80)
print("PORÓWNANIE MODELI - METRYKI WALIDACJI")
print("=" * 80)

df_val = pd.DataFrame(val_metrics_summary).T
print(df_val.to_string())
print("\n")

# Ranking modeli
print("📊 RANKING:")
print(f"   Najszybszy model:        {df['Średni czas/obraz [s]'].idxmin()}")
print(f"   Najwięcej detekcji:      {df['Całkowita liczba detekcji'].idxmax()}")
print(f"   Najwyższa pewność:       {df['Średnia pewność'].idxmax()}")
if len(df_val) > 0:
    print(f"   Najwyższy mAP@50:        {df_val['mAP50'].idxmax()}")
    print(f"   Najwyższy mAP@50-95:     {df_val['mAP50-95'].idxmax()}")
    print(f"   Najwyższy F1-score:      {df_val['F1'].idxmax()}")
print("\n" + "=" * 80)

# Zapisywanie wyników do plików
print("\n💾 Zapisywanie wyników...\n")

# 1. Zapis tabeli podsumowującej do CSV
df.to_csv(results_csv, encoding='utf-8-sig', index=True)
print(f"✅ Tabela porównawcza zapisana do: {results_csv}")

# 1b. Zapis metryk walidacji do CSV
if len(df_val) > 0:
    val_metrics_csv = f"ewaluacja_metryki_{timestamp}.csv"
    df_val.to_csv(val_metrics_csv, encoding='utf-8-sig', index=True)
    print(f"✅ Metryki walidacji zapisane do: {val_metrics_csv}")

# 2. Zapis szczegółowych wyników do CSV
if detailed_results:
    df_detailed = pd.DataFrame(detailed_results)
    detailed_csv = f"ewaluacja_szczegoly_{timestamp}.csv"
    df_detailed.to_csv(detailed_csv, encoding='utf-8-sig', index=False)
    print(f"✅ Szczegółowe wyniki zapisane do: {detailed_csv}")

# 3. Zapis raportu tekstowego
with open(results_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RAPORT EWALUACJI MODELI YOLO11n, YOLO11s, YOLO11m\n")
    f.write("=" * 80 + "\n")
    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Klasy: {', '.join(classes)}\n")
    f.write(f"Katalog ze zdjęciami: {test_images_dir}\n")
    f.write(f"Liczba zdjęć: {len(image_files)}\n")
    f.write(f"Obrazy z detekcjami: {results_images_dir}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("PORÓWNANIE MODELI - WYDAJNOŚĆ\n")
    f.write("=" * 80 + "\n")
    f.write(df.to_string() + "\n\n")
    
    if len(df_val) > 0:
        f.write("=" * 80 + "\n")
        f.write("PORÓWNANIE MODELI - METRYKI WALIDACJI\n")
        f.write("=" * 80 + "\n")
        f.write(df_val.to_string() + "\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("RANKING\n")
    f.write("=" * 80 + "\n")
    f.write(f"Najszybszy model:        {df['Średni czas/obraz [s]'].idxmin()}\n")
    f.write(f"Najwięcej detekcji:      {df['Całkowita liczba detekcji'].idxmax()}\n")
    f.write(f"Najwyższa pewność:       {df['Średnia pewność'].idxmax()}\n")
    if len(df_val) > 0:
        f.write(f"Najwyższy mAP@50:        {df_val['mAP50'].idxmax()}\n")
        f.write(f"Najwyższy mAP@50-95:     {df_val['mAP50-95'].idxmax()}\n")
        f.write(f"Najwyższy F1-score:      {df_val['F1'].idxmax()}\n")
    f.write("=" * 80 + "\n")

print(f"✅ Raport tekstowy zapisany do: {results_txt}")
print(f"✅ Obrazy z detekcjami zapisane do: {results_images_dir}/")
print("\n" + "=" * 80)
print("🎉 EWALUACJA ZAKOŃCZONA POMYŚLNIE!")
print("=" * 80)