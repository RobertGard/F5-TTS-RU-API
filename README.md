# F5-TTS-RU-API

## Описание


**F5-TTS-RU-API** — это REST API для синтеза русской речи на базе модели [F5-TTS](https://github.com/SWivid/F5-TTS) с автоматической расстановкой ударений с помощью [ruaccent](https://github.com/Den4ikAI/ruaccent). Используется русская модель [Misha24-10/F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN) с HuggingFace. Поддерживается генерация аудио с референсом и без, а также формат вывода wav/mp3.

### О проекте F5-TTS
F5-TTS — это современная, быстрая и качественная TTS (Text-to-Speech) модель с открытым исходным кодом, поддерживающая inference на CPU и GPU, а также работу с референсным аудио для имитации интонации.

### Особенности API
- Используется русская модель (Misha24-10/F5-TTS_RUSSIAN)
- Автоматическая расстановка ударений через ruaccent
- Эндпоинт: `POST /v1/audio/speech`
- Поддержка параметров:
  - `input` — текст для озвучивания (обязательный)
  - `out_format` — формат вывода (`wav` или `mp3`, по умолчанию `wav`)
  - `ref_audio` — путь или URL к эталонному аудиофайлу (опционально)
  - `ref_text` — текст или URL к текстовому файлу для эталонного аудио (опционально)

## Установка


## Быстрый старт (Docker)

Рекомендуется запускать через Docker Compose — это полностью автоматизировано и не требует ручной установки зависимостей:

```bash
docker compose up --build
```

## Пример запроса к API

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
		  "input": "Привет, мир!",
		  "out_format": "mp3"
		}' --output output.mp3
```

## Используемые технологии

- [F5-TTS](https://github.com/SWivid/F5-TTS) — генерация речи
- [F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN) — генерация речи
- [ruaccent](https://github.com/Den4ikAI/ruaccent) — автоматическая расстановка ударений
- [FastAPI](https://fastapi.tiangolo.com/) — REST API
- [PyTorch](https://pytorch.org/) — backend для inference


## Примечания
- В этой сборке по умолчанию используется GPU (DEVICE=cuda). Для использования CPU можно явно указать DEVICE=cpu.
- При работе с GPU убедитесь, что PyTorch и torchaudio установлены с поддержкой вашей версии CUDA (см. инструкцию [тут](https://github.com/SWivid/F5-TTS)).
- Поддерживаются как локальные файлы, так и URL для референсного аудио и текста.
- Все входные данные проходят базовую фильтрацию для безопасности.

---
**F5-TTS-RU-API** — быстрый и удобный способ синтеза русской речи с поддержкой референса и ударений!