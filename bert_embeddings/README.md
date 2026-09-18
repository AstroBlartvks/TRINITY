# rubert_model

Обёртка над `DeepPavlov/rubert-base-cased` для превращения новостных заголовков в эмбеддинги по дням.  
RuBERT замораживается и используется как фиксированный энкодер - обучается только надстройка над ним.

## Установка и зависимости

```bash
pip install torch transformers tqdm
```

## Быстрый старт

```python
from RUBERT_model import RuBERT

ruBERT = RuBERT()

texts = ["Выборы в России пройдут с 18 по 20 числа."]
print(ruBERT.get_cls_embedding(texts).shape)  # torch.Size([768])
```

## Работа с новостями

```python
import json
import pandas as pd
from RUBERT_model import RuBERT

news_path = "./../news_cache/ria_news.json"
with open(news_path, 'r', encoding='utf-8') as f:
    news_cache = json.load(f)

all_cal_dates = pd.date_range(
    '2010-01-01', '2026-09-18', freq='B'
).strftime('%Y-%m-%d').tolist()

ruBERT = RuBERT()
daily_embs = ruBERT.build_daily_embeddings(news_cache, all_cal_dates)
```

Первый запуск считает эмбеддинги и сохраняет их в `./../embd_cache/daily_embeddings.pt`.  
Повторный - загружает из кэша мгновенно.

## Выравнивание по торговым датам

```python
trade_dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')

emb_tensor = ruBERT.align_embeddings(daily_embs, trade_dates)
print(emb_tensor.shape)  # (N_торговых_дней, 768)
```

Пропущенные дни заполняются последним известным эмбеддингом (forward-fill), чтобы избежать утечки будущего.
