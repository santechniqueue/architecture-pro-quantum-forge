# Задание 3. Создание векторного индекса базы знаний

## 1. Модель эмбеддингов
- Название: `intfloat/multilingual-e5-base`
- Источник: Hugging Face (Sentence-Transformers / Transformers)
- Размер эмбеддингов: 768
- Особенности: используется нормализация эмбеддингов и FAISS `IndexFlatIP` (cosine similarity ≈ inner product при нормализации).  
  Для E5 применяется префикс `passage:` для документов и `query:` для запросов.

## 2. База знаний
- Директория KB: `Task02/knowledge_base/renamed`
- Формат документов: `.md` (предварительно очищаются от Markdown-разметки и переводятся в “plain text”)

## 3. Чанкинг
- Сплиттер: `RecursiveCharacterTextSplitter` (LangChain)
- Параметры: `chunk_size=1400` символов, `chunk_overlap=200` символов
- Метаданные чанков: `source_path`, `title`, `chunk_in_doc`, `start_char`, `end_char`, `chunk_id`

## 4. Индекс
- Векторная БД: FAISS
- Тип индекса: `faiss.IndexFlatIP`
- Файлы результата:
  - `Task03/index/faiss.index` - индекс
  - `Task03/index/chunks.jsonl` - тексты чанков + метаданные
  - `Task03/index/index_meta.json` - мета-информация о сборке

## 5. Как собрать индекс

```bash
python3 Task03/scripts/build_index.py \
  --kb_dir Task02/knowledge_base/renamed \
  --out_dir Task03/index \
  --model intfloat/multilingual-e5-base \
  --chunk_size 1400 \
  --chunk_overlap 200 \
  --batch 64
```

## 6. Статистика сборки (по факту запуска)
- Документов: 32
- Чанков в индексе: 703
- Время генерации (embedding + build): 37.826s
- Batch size: 64

Выходные файлы:

- `Task03/index/faiss.index` - индекс
- `Task03/index/chunks.jsonl` - чанки + метаданные (по строке на чанк)
- `Task03/index/index_meta.json` - сводные метрики (модель, кол-во файлов/чанков, время сборки и т.д.)

## 7. Пример запроса к индексу

```bash
python3 Task03/scripts/query_index.py \
  --index_dir Task03/index \
  --model intfloat/multilingual-e5-base \
  --q "Что такое Бро'нвекмор Макморхел?" \
  --k 5
  
python3 Task03/scripts/query_index.py \
  --index_dir Task03/index \
  --model intfloat/multilingual-e5-base \
  --q "Кто такая Норкирзок Зулзирдранкхар?" \
  --k 5
  
python3 Task03/scripts/query_index.py \
  --index_dir Task03/index \
  --model intfloat/multilingual-e5-base \
  --q "Кто такая Кранзиркран Сагкир?" \
  --k 5
```

Скрипт выведет `k` наиболее релевантных чанков с:

- score
- `source_path`, `title`, `chunk_in_doc`
- текст чанка
