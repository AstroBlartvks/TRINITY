# imoex_parser

Парсер истории индекса Московской Биржи (IMOEX) на Python.
Позволяет загружать данные за конкретный период через официальный ISS API МосБиржи и сохранять результат в JSON.

## Установка и зависимости

Для работы пакета требуются следующие зависимости:

```bash
pip install requests tqdm
```

## Быстрый старт

```python
import IMOEX_parser

parser = IMOEX_parser.IMOEXParser()

parser.start_parse_data("2010-01-01", "2026-09-18")
print(len(parser.get_data()), "записей")

parser.save_data_to_json("data.json")
```

## Пример с pandas

```python
import json
import pandas as pd

# Загружаем JSON
with open("data.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# Колонки MOEX
columns = [
    "BOARDID", "SECID", "TRADEDATE", "SHORTNAME", "NAME",
    "CLOSE", "OPEN", "HIGH", "LOW", "VALUE",
    "COL10", "COL11", "DECIMALS", "COL13", "CURRENCYID",
    "COL15", "COL16", "COL17", "ADMITTEDQUOTE", "COL19"
]

# DataFrame
df = pd.DataFrame(raw, columns=columns)
...
```
