# Konfiguracja Kaggle API (Windows)

Skrypt `src/data_prep/download.py` pobiera datasety przez
[`kagglehub`](https://github.com/Kaggle/kagglehub), który potrzebuje danych
uwierzytelniających Kaggle. Jednorazowa konfiguracja:

1. **Załóż konto na [kaggle.com](https://www.kaggle.com/)** (jeśli jeszcze
   nie masz).

2. **Zaakceptuj regulaminy datasetów** - wejdź na stronę każdego datasetu i
   kliknij "Download" raz przez stronę (wystarczy, że pobieranie się zacznie
   i je przerwiesz) - część datasetów wymaga zaakceptowania reguł przed
   użyciem API:
   - https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
   - https://www.kaggle.com/datasets/chriskjm/polish-traffic-signs-dataset
   - https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb

3. **Stwórz token API**:
   - Kliknij swój awatar (prawy górny róg) → **Settings**.
   - Sekcja **API** → **Create New Token**.
   - Pobierze się plik `kaggle.json` zawierający `{"username": "...", "key": "..."}`.

4. **Umieść `kaggle.json` w odpowiednim miejscu** - wybierz jedną z opcji:

   - **Opcja A (zalecana): plik w `~/.kaggle/`**

     ```powershell
     New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
     Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
     ```

   - **Opcja B: zmienne środowiskowe** (przydatne np. w CI) - ustaw na
     podstawie zawartości `kaggle.json`:

     ```powershell
     $env:KAGGLE_USERNAME = "twoja_nazwa_uzytkownika"
     $env:KAGGLE_KEY = "twoj_token"
     ```

     (to ustawia zmienne tylko dla obecnej sesji PowerShell; aby zapisać je
     na trwałe, użyj `[System.Environment]::SetEnvironmentVariable(...)`).

5. **Sprawdź konfigurację**:

   ```powershell
   python -m src.data_prep.download --dataset gtsrb
   ```

   Jeśli zobaczysz błąd o nieznalezionych danych uwierzytelniających, wróć
   do kroku 4. Jeśli zobaczysz błąd 403 przy konkretnym datasecie, wróć do
   kroku 2 (akceptacja regulaminu).

**Bezpieczeństwo**: `kaggle.json` zawiera Twój prywatny token - nie commituj
go do repozytorium (`.gitignore` już go ignoruje, gdyby trafił do katalogu
projektu).
