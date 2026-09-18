from datetime import datetime

import requests
import json
import time

from tqdm.notebook import tqdm


class IMOEXParser:
    def __init__(self):
        self.get_url = lambda fr, to: f"https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json?from={fr}&till={to}"
        self.all_data = []

    def parse_data(self, fr: str, to: str):
        """fr, to - даты формата YYYY-mm-dd"""
        url: str = self.get_url(fr, to)
        json_data = requests.get(url)
        data = json.loads(json_data.text)
        return data['history']['data'][1:], data['history']['data'][-1][2]

    def start_parse_data(self, fr: str, to: str, delay: int = 2.0):
        start_date = datetime.strptime(fr, "%Y-%m-%d")
        end_date = datetime.strptime(to, "%Y-%m-%d")
        total_days = (end_date - start_date).days

        with tqdm(total=total_days, desc="Парсинг IMOEX", unit="дн") as pbar:
            current = fr
            while (datetime.strptime(to, "%Y-%m-%d") - datetime.strptime(current, "%Y-%m-%d")).days > 1:
                try:
                    data, new_fr = self.parse_data(current, to)
                    self.all_data += data
                    try:
                        passed = (datetime.strptime(new_fr, "%Y-%m-%d") - datetime.strptime(current, "%Y-%m-%d")).days
                        pbar.update(max(passed, 0))
                    except ValueError:
                        pass

                    current = new_fr
                    pbar.set_postfix({"записей": len(self.all_data)})
                except Exception as exp:
                    time.sleep(delay)
                    tqdm.write(f"{exp} | {current} -> {to}")

    def get_data(self):
        return self.all_data

    def save_data_to_json(self, json_name: str = "data.json"):
        json_data_all = json.dumps(self.all_data, indent=4, ensure_ascii=False,)
        with open(json_name, "w", encoding="utf-8") as File:
            File.write(json_data_all)


if __name__ == "__main__":
    imoexParser = IMOEXParser()
    imoexParser.start_parse_data("2026-07-20", "2026-07-24")
    print(imoexParser.get_data())