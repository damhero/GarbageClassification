Repozytorium dotyczące projektu detekcji odpadów. 

W pliku script.py znajduje się kod skryptu używany do trenowania modeli, należy zmienć w nim ścieżkę do pliku data.yaml

Modele trenowane są na image size 640x640, 100 epochs, patience 10.

Kaggle free cloud compute GPU P100 30h/week
W pliku script.py są naprawione importy bibliotek, ponieważ kaggle ma automatycznie niekompatybilne opencv oraz numpy, wszystkie zdjęcia w katalogu dataset są poprawne.
