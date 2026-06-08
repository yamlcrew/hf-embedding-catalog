# hf-embedding-catalog

Automatycznie odświeżany katalog modeli embeddingowych z Hugging Face Hub, przeznaczonych do lokalnego wnioskowania przez **transformers.js**.

Dane są aktualizowane codziennie przez GitHub Actions i commitowane bezpośrednio do repozytorium — bez serwera, bez bazy danych. Wystarczy sklonować repo i czytać pliki JSON.

---

## Struktura repozytorium

```
hf-embedding-catalog/
├── fetch_catalog.py          # skrypt pobierający dane z HF API
├── requirements.txt          # tylko: requests>=2.31.0
├── catalog.yaml              # indeks wszystkich modeli (posortowany wg downloads)
├── data/
│   └── <org>/
│       └── <model>/
│           └── model.yaml    # pełne metadane jednego modelu
└── .github/
    └── workflows/
        └── update-catalog.yml
```

### `catalog.yaml`

Płaski indeks wszystkich pobranych modeli, posortowany malejąco wg `downloads`. Zawiera skrócony zestaw pól — wystarczający do przeglądania i filtrowania bez ładowania osobnych plików.

### `data/<org>/<model>/model.yaml`

Pełne metadane każdego modelu. Przykład (`sentence-transformers/all-MiniLM-L6-v2`):

```json
{
  "id":               "sentence-transformers/all-MiniLM-L6-v2",
  "pipelineTag":      "sentence-similarity",
  "libraryName":      "sentence-transformers",
  "author":           "sentence-transformers",
  "createdAt":        "2022-03-02T23:29:05.000Z",
  "lastModified":     "2026-06-01T06:29:13.000Z",
  "sha":              "abc123...",
  "downloads":        253044030,
  "likes":            4909,
  "trendingScore":    12.4,
  "gated":            false,
  "private":          false,
  "disabled":         false,
  "tags":             ["sentence-transformers", "bert", "onnx", "license:apache-2.0", "..."],
  "onnxFiles":        ["onnx/model.onnx", "onnx/model_quantized.onnx"],
  "hasSafetensors":   true,
  "paramCount":       22713728,
  "paramsByDtype":    { "F32": 22713728 },
  "modelType":        "bert",
  "architectures":    "BertModel",
  "hiddenSize":       384,
  "intermediateSize": 1536,
  "maxSeqLen":        512,
  "numLayers":        6,
  "numHeads":         12,
  "numKvHeads":       0,
  "vocabSize":        30522,
  "torchDtype":       "float32",
  "hiddenAct":        "gelu",
  "fetchedAt":        "2026-06-07T21:47:23.000Z"
}
```

#### Opis pól

| Pole | Źródło | Opis |
|---|---|---|
| `id` | HF API | identyfikator modelu (`org/name`) |
| `pipelineTag` | HF API | `sentence-similarity` lub `feature-extraction` |
| `libraryName` | HF API | biblioteka: `sentence-transformers`, `transformers` |
| `author` | HF API | organizacja lub użytkownik |
| `createdAt` | HF API | data pierwszej publikacji |
| `lastModified` | HF API | data ostatniej zmiany w repo |
| `sha` | HF API | hash commita (pinning wersji) |
| `downloads` | HF API | pobrania z ostatnich 30 dni |
| `likes` | HF API | polubienia |
| `trendingScore` | HF API | aktualny wynik trending |
| `gated` | HF API | wymaga akceptacji warunków przed pobraniem |
| `private` | HF API | prywatne repo (widoczne tylko z tokenem) |
| `disabled` | HF API | wyłączony przez HF |
| `tags` | HF API | wszystkie tagi, w tym `license:*`, `dataset:*`, `arxiv:*` |
| `onnxFiles` | `siblings` | lista plików `.onnx` w repo (pusta = brak ONNX) |
| `hasSafetensors` | `siblings` | czy repo zawiera pliki `.safetensors` |
| `paramCount` | `expand=safetensors` | liczba parametrów (łącznie) |
| `paramsByDtype` | `expand=safetensors` | rozkład parametrów per dtype, np. `{"F32": 22M}` |
| `modelType` | `config.json` | rodzina architektury: `bert`, `xlm-roberta`, `distilbert` |
| `architectures` | `config.json` | klasa PyTorch: `BertModel`, `XLMRobertaModel` |
| `hiddenSize` | `config.json` | **wymiar embeddingu** → rozmiar kolumny `vector(N)` |
| `intermediateSize` | `config.json` | szerokość warstwy FFN |
| `maxSeqLen` | `config.json` | maksymalna długość sekwencji (tokeny) |
| `numLayers` | `config.json` | liczba bloków Transformer |
| `numHeads` | `config.json` | liczba głowic attention |
| `numKvHeads` | `config.json` | głowice KV (>0 = GQA/MQA) |
| `vocabSize` | `config.json` | rozmiar słownika |
| `torchDtype` | `config.json` | typ wag: `float32`, `float16`, `bfloat16` |
| `hiddenAct` | `config.json` | funkcja aktywacji: `gelu`, `silu` |
| `fetchedAt` | lokalnie | czas pobrania metadanych |

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

Model jest (ponownie) pobierany gdy `model.yaml` nie istnieje **lub** pole `fetchedAt` jest starsze niż 60 dni (`REFRESH_DAYS`). Zmień stałą w skrypcie, żeby to dostosować.

---

## GitHub Actions

Workflow [`.github/workflows/update-catalog.yml`](.github/workflows/update-catalog.yml) uruchamia się **codziennie o 03:00 UTC** i commituje zmiany bezpośrednio do gałęzi głównej.

Można go uruchomić ręcznie z poziomu zakładki **Actions** i opcjonalnie podać `limit` (maksymalna liczba modeli do pobrania w tym uruchomieniu).

Wymagane: sekret repozytorium `HF_TOKEN` z tokenem Hugging Face.

---

## Jakie modele są katalogowane

Skrypt filtruje modele po:

- **library:** `transformers.js` — modele przygotowane do lokalnego wnioskowania w przeglądarce / Node.js
- **pipeline_tag:** `sentence-similarity` i `feature-extraction` — dwa tagi obejmujące modele embeddingowe na HF Hub

Modele są sortowane malejąco wg liczby pobrań.

---

## Przykład użycia danych

### Znalezienie modeli ONNX z wymiarem 384

```python
import yaml

with open("catalog.yaml") as f:
    catalog = yaml.safe_load(f)

results = [
    m for m in catalog
    if m["hidden_size"] == 384 and m["has_onnx"]
]
for m in results[:5]:
    print(m["id"], m["downloads"])
```

### Odczyt pełnych metadanych jednego modelu

```python
import yaml
from pathlib import Path

path = Path("data/sentence-transformers/all-MiniLM-L6-v2/model.yaml")
model = yaml.safe_load(path.read_text())

cfg = model.get("configJson") or {}
print(f"Wymiar:     {cfg.get('hidden_size')}")
print(f"Max tokeny: {cfg.get('max_position_embeddings')}")
print(f"Parametry:  {(model.get('safetensors') or {}).get('total', 0):,}")
print(f"Rozmiar:    {model['approxSizeMb']} MB")
print(f"Pliki ONNX: {model['onnxFiles']}")
```

### Kolumna pgvector

```sql
-- hiddenSize modelu = N w vector(N)
CREATE TABLE documents (
  id        bigserial PRIMARY KEY,
  content   text,
  embedding vector(384)   -- all-MiniLM-L6-v2
);
```
