# ria_parser

Парсер заголовков новостей РИА Новости (`ria.ru`) на Python.  
Позволяет получать заголовки за конкретную дату или диапазон дат с кэшированием результатов в JSON.

## Установка и зависимости

Для работы пакета требуются следующие зависимости:

```bash
pip install requests beautifulsoup4 tqdm
```

## Быстрый старт

```python
from ria_parser.RIA_parser import RiaParser

parser = RiaParser()

titles = parser.parse_ria_day('2022-01-15')
print(len(titles), 'заголовков')

dates = ['2022-01-15', '2022-01-16', '2022-01-17']
cache = parser.parse_ria_range(dates, news_cache_dir="news_cache", delay=0.1)
print('Всего дней в кэше:', len(cache))
```

