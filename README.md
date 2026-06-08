# hf-embedding-catalog

Automatycznie odświeżany katalog modeli embeddingowych z Hugging Face Hub, przeznaczonych do lokalnego wnioskowania przez **transformers.js**.

Dane są aktualizowane codziennie przez GitHub Actions i commitowane bezpośrednio do repozytorium — bez serwera, bez bazy danych. Wystarczy sklonować repo i czytać pliki.

---

## Struktura repozytorium

```
hf-embedding-catalog/
├── fetch_catalog.py          # skrypt pobierający dane z HF API
├── requirements.txt          # requests>=2.31.0, pyyaml>=6.0
├── catalog.json              # indeks wszystkich modeli (JSONL, jeden rekord per linia)
├── catalog.yaml              # indeks wszystkich modeli (YAML, czytelny dla człowieka)
├── data/
│   └── <org>/
│       └── <model>/
│           ├── model.json    # pełne metadane jednego modelu (kompaktowy JSON, jedna linia)
│           └── model.yaml    # to samo co model.json, format YAML
└── .github/
    └── workflows/
        └── update-catalog.yml
```

> **`.gitignore`** — `data/` i `catalog.*` są ignorowane lokalnie (ochrona przed przypadkowym `git add .`). GitHub Actions używa `git add --force`, więc pliki i tak trafiają do repozytorium.

### `catalog.json` / `catalog.yaml`

Płaski indeks wszystkich pobranych modeli, posortowany malejąco wg `downloads`.

`catalog.json` jest w formacie **JSONL** — każda linia to osobny obiekt JSON. `catalog.yaml` zawiera te same dane w czytelnym formacie YAML.

Zawiera skrócony zestaw pól — wystarczający do przeglądania i filtrowania bez ładowania osobnych plików.

| Pole | Opis |
|---|---|
| `id` | identyfikator modelu (`org/name`) |
| `pipeline_tag` | `sentence-similarity` lub `feature-extraction` |
| `library_name` | biblioteka: `sentence-transformers`, `transformers` |
| `author` | organizacja lub użytkownik |
| `model_type` | rodzina architektury: `bert`, `xlm-roberta` |
| `architectures` | klasa PyTorch: `BertModel`, `XLMRobertaModel` |
| `hidden_size` | **wymiar embeddingu** → rozmiar kolumny `vector(N)` |
| `max_position_embeddings` | maksymalna długość sekwencji (tokeny) |
| `num_hidden_layers` | liczba bloków Transformer |
| `torch_dtype` | typ wag: `float32`, `float16`, `bfloat16` |
| `param_count` | łączna liczba parametrów |
| `model_size_mb` | rozmiar wag w MB (safetensors lub szacunek z parametrów) |
| `onnx_size_mb` | rozmiar głównego pliku `onnx/model.onnx` w MB |
| `downloads` | pobrania z ostatnich 30 dni |
| `likes` | polubienia |
| `trending_score` | aktualny wynik trending |
| `has_onnx` | czy repo zawiera pliki `.onnx` |
| `has_safetensors` | czy repo zawiera pliki `.safetensors` |
| `lang` | kody językowe ISO 639-1, przecinkiem (np. `en`, `en,de,fr`) |
| `multilingual` | `true` gdy model oznaczony jako wielojęzyczny |
| `gated` | wymaga akceptacji warunków przed pobraniem |
| `created_at` | data pierwszej publikacji |
| `last_modified` | data ostatniej zmiany w repo |
| `fetched_at` | czas pobrania metadanych |

### `data/<org>/<model>/model.json`

Pełne surowe metadane z HF API, wzbogacone o kilka pól obliczanych lokalnie. Zapisywany jako **kompaktowy JSON w jednej linii**. Przykład (`sentence-transformers/all-MiniLM-L6-v2`) w czytelniejszym formacie YAML (`model.yaml`):

```yaml
id: sentence-transformers/all-MiniLM-L6-v2
pipeline_tag: sentence-similarity
library_name: sentence-transformers
author: sentence-transformers
downloads: 253044030
likes: 4909
trendingScore: 12.4
gated: false
private: false
disabled: false
createdAt: '2022-03-02T23:29:05.000Z'
lastModified: '2026-06-01T06:29:13.000Z'
sha: abc123...
tags:
  - sentence-transformers
  - bert
  - onnx
  - safetensors
  - license:apache-2.0
siblings:
  - rfilename: config.json
  - rfilename: model.safetensors
  - rfilename: onnx/model.onnx
  - rfilename: tokenizer.json
safetensors:
  parameters:
    F32: 22713216
    I64: 512
  total: 22713728
configJson:                        # zawartość config.json z repo
  model_type: bert
  architectures: [BertModel]
  hidden_size: 384
  intermediate_size: 1536
  max_position_embeddings: 512
  num_hidden_layers: 6
  num_attention_heads: 12
  vocab_size: 30522
  hidden_act: gelu
  torch_dtype: float32
onnxFiles:                         # obliczane lokalnie: {ścieżka → rozmiar MB}
  onnx/model.onnx: 90.4
  onnx/model_O4.onnx: 45.2
  onnx/model_qint8_avx512.onnx: 23.0
safetensorsSizeMb: 90.9            # obliczane lokalnie: suma plików .safetensors
approxSizeMb: 90.9                 # obliczane lokalnie: params × bytes/dtype
fetchedAt: '2026-06-08T00:19:21.000Z'
```

#### Pola obliczane lokalnie (nie z HF API)

| Pole | Opis |
|---|---|
| `onnxFiles` | dict `{ścieżka → rozmiar_MB}` dla każdego pliku `.onnx` w repo |
| `safetensorsSizeMb` | suma rozmiarów wszystkich plików `.safetensors` (decimal MB) |
| `approxSizeMb` | szacunek: parametry × bytes/dtype / 1 000 000 |
| `fetchedAt` | ISO timestamp pobrania |

#### Pola z HF API (kluczowe)

| Pole | Źródło | Opis |
|---|---|---|
| `id` | `/api/models` | identyfikator modelu (`org/name`) |
| `pipeline_tag` | `/api/models` | `sentence-similarity` lub `feature-extraction` |
| `library_name` | `/api/models` | biblioteka: `sentence-transformers`, `transformers` |
| `safetensors.total` | `expand=safetensors` | łączna liczba parametrów |
| `safetensors.parameters` | `expand=safetensors` | rozkład parametrów per dtype, np. `{F32: 22M}` |
| `configJson.*` | `config.json` z repo | `hidden_size`, `max_position_embeddings`, `model_type` itp. |
| `cardData` | `expand=cardData` | sparsowany YAML z model card (języki, licencja, datasety) |
| `siblings` | `?full=true` | lista plików w repo |
| `gated` | `/api/models` | wymaga akceptacji warunków przed pobraniem |
| `downloadsAllTime` | `expand=downloadsAllTime` | łączne pobrania wszystkich czasów |

---

## Uruchamianie skryptu

### Wymagania

```bash
pip install -r requirements.txt
```

Token HF jest opcjonalny, ale **silnie zalecany** — bez niego obowiązują niższe limity rate-limit:

```bash
export HF_TOKEN=hf_...
```

### Tryby uruchomienia

```bash
# domyślnie: top 1000 modeli per pipeline_tag, pomija już aktualne
python fetch_catalog.py

# bez limitu — pobiera wszystkie dostępne modele
python fetch_catalog.py --all

# ograniczenie do top 500 per tag
python fetch_catalog.py --max-list 500

# tylko listowanie, bez zapisu plików
python fetch_catalog.py --dry-run

# maksymalnie 50 nowych/stale modeli w tym uruchomieniu
python fetch_catalog.py --limit 50
```

### Odświeżanie

Model jest (ponownie) pobierany gdy `model.json` nie istnieje **lub** pole `fetchedAt` jest starsze niż 180 dni (`REFRESH_DAYS`). Zmień stałą w skrypcie, żeby to dostosować.

---

## GitHub Actions

Workflow [`.github/workflows/update-catalog.yml`](.github/workflows/update-catalog.yml) uruchamia się **codziennie o 03:00 UTC** i commituje zmiany bezpośrednio do gałęzi głównej.

Można go uruchomić ręcznie z poziomu zakładki **Actions** z opcjonalnymi parametrami:
- `max_list` — maksymalna liczba modeli do wylistowania per pipeline_tag (0 = bez limitu)
- `limit` — maksymalna liczba stale modeli do pobrania w tym uruchomieniu (0 = wszystkie)

Wymagane: sekret repozytorium `HF_TOKEN` z tokenem Hugging Face.

Actions używa `git add --force`, żeby ominąć `.gitignore` i wypchnąć `data/` oraz `catalog.*` do repozytorium.

---

## Jakie modele są katalogowane

Skrypt filtruje modele po:

- **library:** `transformers.js` — modele przygotowane do lokalnego wnioskowania w przeglądarce / Node.js
- **pipeline_tag:** `sentence-similarity` i `feature-extraction` — dwa tagi obejmujące modele embeddingowe na HF Hub

Modele są sortowane malejąco wg liczby pobrań. Globalny limit zapytań HTTP wynosi **10 req/s** (wszystkie wątki razem). Każdy model jest pobierany w 4 zapytaniach:
1. `GET /api/models/<id>?full=true` — kompletny rekord bazowy
2. `GET /api/models/<id>?expand=safetensors&expand=cardData&...` — dodatkowe pola
3. `GET /api/models/<id>/tree/main?recursive=true` — drzewo plików z prawdziwymi rozmiarami
4. `GET <id>/resolve/main/config.json` — architektura modelu

---

## Przykład użycia danych

### Znalezienie modeli ONNX z wymiarem 384

```python
import json

# catalog.json jest w formacie JSONL — jedna linia = jeden model
with open("catalog.json") as f:
    catalog = [json.loads(line) for line in f if line.strip()]

results = [
    m for m in catalog
    if m["hidden_size"] == 384 and m["has_onnx"]
]
for m in results[:5]:
    print(m["id"], m["downloads"], m["onnx_size_mb"], "MB")
```

### Odczyt pełnych metadanych jednego modelu

```python
import json
from pathlib import Path

path = Path("data/sentence-transformers/all-MiniLM-L6-v2/model.json")
model = json.loads(path.read_text())

cfg = model.get("configJson") or {}
print(f"Wymiar:     {cfg.get('hidden_size')}")
print(f"Max tokeny: {cfg.get('max_position_embeddings')}")
print(f"Parametry:  {(model.get('safetensors') or {}).get('total', 0):,}")
print(f"Rozmiar:    {model['approxSizeMb']} MB")

# onnxFiles to dict {ścieżka: rozmiar_MB}
for path_onnx, size_mb in model['onnxFiles'].items():
    print(f"  ONNX: {path_onnx} ({size_mb} MB)")
```

### Kolumna pgvector

```sql
-- hidden_size modelu = N w vector(N)
CREATE TABLE documents (
  id        bigserial PRIMARY KEY,
  content   text,
  embedding vector(384)   -- all-MiniLM-L6-v2
);
```
