import os

import torch
from tqdm.notebook import tqdm

from transformers import AutoTokenizer, AutoModel


class RuBERT:
    def __init__(self, bert_dim=768, device="cuda", embd_cache="./../embd_cache"):
        self.embd_cache = embd_cache
        self.bert_dim = bert_dim
        self.device = device
        self.tokenizer = None
        self.bert = None
        self.__load_rubert()


    def __load_rubert(self):
        """
        Загружает DeepPavlov/rubert-base-cased, Все веса заморожены
        """
        model_name = 'DeepPavlov/rubert-base-cased'
        print(f'  Загружаем {model_name}...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert = self.bert.to(self.device).eval()
        n_params = sum(p.numel() for p in self.bert.parameters())
        print(f'  RuBERT загружен: {n_params/1e6:.1f}M параметров (заморожено)')


    def get_cls_embedding(self, texts, max_len=128, batch_size=16):
        """
        Получает дневной эмбеддинг: среднее CLS по всем заголовкам.
        CLS токен = первый токен последнего слоя BERT.
        Возвращает tensor (768,).
        """
        if not texts:
            return torch.zeros(self.bert_dim)

        all_cls = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = self.tokenizer(batch, padding=True, truncation=True,
                              max_length=max_len, return_tensors='pt')
            enc   = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.bert(**enc)
            all_cls.append(out.last_hidden_state[:, 0, :].cpu())  # CLS

        return torch.cat(all_cls, dim=0).mean(dim=0)  # (768,)


    def build_daily_embeddings(self, news_cache, dates):
        """
        Строит dict {date_str: tensor(768,)} для всех дат. Кэш в ./embd_cache/daily_embeddings.pt.
        """
        cache_file = os.path.join(self.embd_cache, 'daily_embeddings.pt')

        if os.path.exists(cache_file):
            print('  Загружаем эмбеддинги из кэша...')
            embs = torch.load(cache_file, map_location='cpu')
            print(f'  Загружено: {len(embs)} дней')
            return embs

        print(f'  Считаем эмбеддинги для {len(dates)} дат...')
        daily_embs = {}
        for date_str in tqdm(dates, desc='RuBERT embeddings'):
            texts = news_cache.get(date_str, [])
            daily_embs[date_str] = self.get_cls_embedding(texts)

        torch.save(daily_embs, cache_file)
        print(f'  Сохранено → {cache_file}')
        return daily_embs


    def align_embeddings(self, daily_embs, trade_dates):
        """
        Выравнивает эмбеддинги по торговым датам.
        Forward-fill: если на дату нет новостей - берём последний известный день.
        Возвращает tensor (N, 768).
        """
        sorted_news_dates = sorted(daily_embs.keys())
        last_emb = torch.zeros(self.bert_dim)
        result   = []

        for td in trade_dates:
            td_str = str(td)[:10]
            if td_str in daily_embs:
                last_emb = daily_embs[td_str]
            else:
                for nd in reversed(sorted_news_dates):
                    if nd <= td_str:
                        last_emb = daily_embs[nd]
                        break
            result.append(last_emb)

        return torch.stack(result, dim=0)  # (N, 768)


if __name__ == "__main__":
    ruBERT = RuBERT()
    tensor = ruBERT.get_cls_embedding("Выборы в России пройдут с 18 по 20 числа.")
    print(tensor)