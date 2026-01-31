# System Detekcji i Klasyfikacji Odpadów (YOLOv11) ♻️

Projekt zaliczeniowy z przedmiotu **Podstawy Reprezentacji i Analizy Danych (2025Z)**.
Celem projektu jest automatyczne wykrywanie i klasyfikacja odpadów na zdjęciach przy użyciu sieci neuronowych YOLO (v11n, v11s, v11m).

## 👥 Autorzy
* Damian Brudkowski
* Cyprian Ciesielski
* Wojciech Ziembowicz

## 🎯 Główne funkcjonalności
* **Analiza Eksploracyjna (EDA):** Badanie struktury i balansu zbioru *Garbage Classification 3*.
* **Trening Modeli:** Porównanie wydajności architektur YOLO11 **Nano, Small i Medium**.
* **Walidacja:** Analiza metryk mAP, Precision, Recall oraz macierzy pomyłek.
* **Testy "In-the-wild":** Weryfikacja działania modelu na autorskim zbiorze zdjęć wykonanych smartfonem w warunkach domowych (uwzględniająca analizę *Domain Shift*).

## 📂 Struktura klas
Model rozpoznaje 5 klas odpadów:
1. `BIODEGRADABLE` (Bio)
2. `GLASS` (Szkło)
3. `METAL` (Metal)
4. `PAPER` (Papier/Karton)
5. `PLASTIC` (Plastik)

## 🚀 Jak uruchomić projekt

### Wymagania
Projekt wymaga środowiska Python (rekomendowane 3.8+) oraz bibliotek:
```bash
pip install ultralytics pandas matplotlib seaborn opencv-python tqdm notebook