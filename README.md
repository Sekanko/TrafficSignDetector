# TrafficSignDetector

Detektor + klasyfikator znaków drogowych (PyTorch → TensorFlow Lite, na telefon).

## Status

Obecny etap: **przygotowanie danych** (zakończone dla wybranych datasetów).
Pipeline pobiera datasety z Kaggle, mapuje ich natywne klasy na jedną wspólną
taksonomię (`configs/taxonomy.yaml`, **50 klas**: 43 z GTSRB/GTSDB + 7
unikalnych dla polskich znaków) i generuje dwa zbiory wyjściowe:

- `data/processed/classification/` - wycięte znaki, pogrupowane po
  unifikowanej klasie (format `ImageFolder` dla PyTorch). **53 771 obrazów**
  (train/val/test).
- `data/processed/detection/` - pełne zdjęcia + etykiety YOLO, do trenowania
  detektora. Detektor jest **jednoklasowy** (`traffic_sign`) - tylko
  lokalizuje znaki; rozpoznanie typu znaku robi klasyfikator. Dzięki temu
  można połączyć źródła o różnej granulacji etykiet (GTSDB ma etykiety per
  typ znaku, polski dataset detekcyjny ma tylko "znak/brak znaku"). **2021
  obrazów** (train/val/test).

## Datasety

| Dataset | Źródło | Rola |
|---|---|---|
| GTSRB | [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) | klasyfikacja (43 klasy), wycinki kadrowane wg `Roi.*` z `Train.csv`/`Test.csv` |
| GTSDB | [Kaggle](https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb) | tylko detekcja (600 pełnych scen ulicznych) - nieużywane do klasyfikacji, bo jego wycinki w ~47% duplikują obrazy z GTSRB (te same nagrania) |
| Polish Traffic Signs Dataset | [Kaggle](https://www.kaggle.com/datasets/chriskjm/polish-traffic-signs-dataset) | klasyfikacja (19 polskich kodów znaków zmapowanych w `configs/datasets/polish.yaml`, w tym 7 nowych klas bez odpowiednika w GTSRB) + własny zbiór detekcyjny (1515 zdjęć, jednoklasowy) |

Pominięte foldery polskiego datasetu: `B33` (ograniczenie prędkości ze
zmienną wartością liczbową na znaku - wymagałoby OCR) i `other` (zbiór
różnych, niejednorodnych znaków).

Inne europejskie/polskie datasety rozważane na później: Mapillary Traffic
Sign Dataset, "Traffic Road Object Detection Polish 12k", DFG Traffic Sign
Dataset (Słowenia).

## Trening klasyfikatora

```powershell
python -m src.classification.train --model mobilenet_v2 --epochs 10 --lr 1e-3
python -m src.classification.train --model efficientnet_lite0 --epochs 10
python -m src.classification.train --model squeezenet --epochs 10
python -m src.classification.train --model custom_cnn --epochs 10 --no-pretrained
```

Dostępne modele (`src/classification/models/`): `mobilenet_v2`,
`efficientnet_lite0`, `squeezenet` (wagi ImageNet) oraz `custom_cnn`
(placeholder - mała sieć od zera, do zastąpienia własną architekturą).

Każdy run zapisuje się pod unikalną nazwą `models/<model>_<timestamp>.{pt,json}`
(katalog `models/` jest gitignored), więc kolejne treningi z innymi
parametrami nie nadpisują poprzednich. Plik `.json` zawiera pełny zestaw
hiperparametrów, historię epok i metryki (accuracy/loss train/val/test, liczba
parametrów, czas treningu) - po treningu te informacje są też wypisywane na
konsolę.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Konfiguracja Kaggle API: [docs/kaggle_setup.md](docs/kaggle_setup.md).

## Uruchomienie pipeline'u

```powershell
# 1. Pobranie datasetów do data/raw/
python -m src.data_prep.download

# 2. Budowa zbioru do klasyfikacji
python -m src.data_prep.build_classification

# 3. Budowa zbioru do detekcji (YOLO, jednoklasowy "traffic_sign")
python -m src.data_prep.build_detection
```

`src/data_prep/inspect_polish.py` to jednorazowy skrypt diagnostyczny użyty
do odkrycia struktury polskiego datasetu (wynik już wpisany do
`configs/datasets/polish.yaml`'s `id_map`) - nie jest częścią normalnego
przebiegu pipeline'u, ale przydatny, jeśli dataset zostanie zaktualizowany.

## Struktura

```
configs/
  taxonomy.yaml           # wspólna taksonomia klas (50 klas: GTSRB/GTSDB + polskie)
  datasets/*.yaml         # konfiguracja per dataset (slug Kaggle, mapowanie klas, splity)
src/data_prep/
  adapters/               # GtsrbAdapter, GtsdbAdapter, PolishAdapter
  download.py             # pobieranie z Kaggle (kagglehub)
  inspect_polish.py       # podgląd struktury polskiego datasetu (jednorazowo)
  build_classification.py # -> data/processed/classification/
  build_detection.py      # -> data/processed/detection/ (YOLO)
src/classification/
  models/                 # rejestr modeli: mobilenet_v2, efficientnet_lite0, squeezenet, custom_cnn
  dataset.py              # ManifestDataset (czyta manifest.csv)
  train.py                # trening + zapis models/<model>_<timestamp>.{pt,json}
data/                      # (gitignored) raw/ + processed/
models/                    # (gitignored) checkpointy + podsumowania treningów
```
