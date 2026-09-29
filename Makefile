# Команды проекта. Требуется uv (https://docs.astral.sh/uv/); без него - см. README, раздел "Запуск".
DATE ?= 2015-08-01
RESTAURANT ?= 75

.PHONY: all install data train predict test requirements api docker-build docker-up docker-down docker-predict

all: install data train test predict

install:            ## зависимости из uv.lock (все группы: data, notebooks, dev) и пакет customers_prediction
	uv sync --all-groups

data:               ## скачать датасет Rossmann и собрать data/processed
	uv run --group data python make_dataset.py

train:              ## обучить модель, посчитать метрики, сохранить модель и записать прогон
	uv run python train.py

predict:            ## прогноз на 7 дней: make predict DATE=2015-07-11 RESTAURANT=65
	uv run python predict.py --date $(DATE) --restaurant $(RESTAURANT)

test:
	uv run pytest

requirements:       ## обновить requirements.txt из uv.lock
	uv export --no-hashes --format requirements-txt --frozen --all-groups -o requirements.txt

api:                ## API локально: http://localhost:8000/docs
	uv run uvicorn customers_prediction.api:app --port 8000

docker-build:       ## собрать образ
	docker compose build

docker-up:          ## запустить API в Docker: http://localhost:8000/docs
	docker compose up -d

docker-down:
	docker compose down

docker-predict:     ## прогноз CLI в Docker: make docker-predict DATE=2015-07-11 RESTAURANT=65
	docker compose run --rm api python predict.py --date $(DATE) --restaurant $(RESTAURANT)
