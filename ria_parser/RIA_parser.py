import requests
import os
import json
import time

from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup

class RiaParser:
    def __init__(self):
        ...

    def parse_ria_day(self, date_str):
        """
        Парсит заголовки РИА новостей за один день.
        date_str: '2022-01-15'
        Возвращает список строк (заголовки).
        """
        date = datetime.strptime(date_str, '%Y-%m-%d')
        url  = f'https://ria.ru/{date.strftime("%Y%m%d")}/'

        try:
            resp = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; research bot)'
            })
            if resp.status_code != 200:
                return []

            soup  = BeautifulSoup(resp.text, 'html.parser')
            texts = []

            for tag in soup.find_all(['h2', 'h3'],
                                     class_=lambda c: c and 'title' in c.lower()):
                t = tag.get_text(strip=True)
                if len(t) > 20:
                    texts.append(t)

            if not texts:
                for tag in soup.find_all('a',
                                         class_=lambda c: c and 'title' in str(c).lower()):
                    t = tag.get_text(strip=True)
                    if len(t) > 20:
                        texts.append(t)

            seen, unique = set(), []
            for t in texts:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)

            return unique[:128]

        except Exception as e:
            return []


    def parse_ria_range(self, dates, news_cache_dir="news_cache", delay=0.1):
        """
        Парсит РИА за список дат с кэшированием.
        При повторном запуске грузит только отсутствующие даты.
        """
        cache_file = os.path.join(news_cache_dir, 'ria_news.json')

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                news_cache = json.load(f)
            print(f'  Кэш загружен: {len(news_cache)} дней уже есть')
        else:
            news_cache = {}

        missing = [d for d in dates if d not in news_cache]
        print(f'  Нужно спарсить: {len(missing)} дней')

        for i, date_str in enumerate(tqdm(missing, desc='Парсинг РИА')):
            news_cache[date_str] = self.parse_ria_day(date_str)
            time.sleep(delay)

            if (i + 1) % 100 == 0:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(news_cache, f, ensure_ascii=False, indent=2)

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(news_cache, f, ensure_ascii=False, indent=2)

        filled = sum(1 for v in news_cache.values() if v)
        print(f'  Итого: {len(news_cache)} дней, с новостями: {filled}')
        return news_cache


if __name__ == "__main__":
    riaParser = RiaParser()
    test_news = riaParser.parse_ria_day('2022-02-24')

    print(f'Тест 2022-02-24: {len(test_news)} заголовков')
    for t in test_news[:10]:
        print(f' - {t}')