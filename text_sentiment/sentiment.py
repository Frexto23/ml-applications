import config # задает переменные окружения

from statistics import mean
from transformers import pipeline, logging

class StockAnalyzer:

    LABEL_NAMES = {"LABEL_0": "SELL", "LABEL_1": "NEUTRAL", "LABEL_2": "BUY"}
    WEIGHT = {"SELL": -1, "NEUTRAL": 0, "BUY": 1}

    def __init__(self):
        logging.enable_progress_bar() # включить показ прогресс бара скачивания модели в локальный кэш
        self._classifier = pipeline("text-classification", model="hse-teddy-bear/xlm-roberta-russian-stock-sentiment")

    def get_comment_score(self, pipe_result: list[dict]):
        impact = 0
        for item in pipe_result:
            score = item['score']
            label = self.LABEL_NAMES[item['label']]
            impact += self.WEIGHT[label] * score

        return impact

    def analyze(self, data):
        verdicts = {}
        for ticker, comments in data.items():
            scores = []
            for comment in comments:
                result = self._classifier(comment, top_k=None)
                scores.append(self.get_comment_score(result))
            verdicts[ticker] = mean(scores)

        return verdicts

if __name__ == "__main__":
    data = {
        "SBER": [
            "Покупаю Сбер, думаю пойдём выше.",
            "Сбер пробил сопротивление, беру ещё.",
            "Вышла отчётность, дивиденды не объявлены.",
        ],
        "GAZP": [
            "Продал Газпром, слишком рискованно.",
            "Газпром падает, лучше выйти.",
        ],
        "CTF": ["Беру, тут однозначно вверх."],
    }

    anal = StockAnalyzer()

    print(anal.analyze(data))
