# Burnout Risk Prediction API

REST-сервіс на FastAPI для прогнозування рівня ризику вигорання студентів (`Burnout_Risk_Level`: Low / Medium / High). Модель навчається на даних з ЛР2–ЛР3 і зберігає прогнози в PostgreSQL (Supabase).

## Структура проєкту

```
bunt_out_risk_pred/
├── app/
│   ├── main.py              # запуск FastAPI, підключення роутерів
│   ├── endpoints/
│   │   ├── training.py      # POST /train-model
│   │   └── inference.py     # POST /predict
│   ├── services/
│   │   ├── model.py         # тренування, збереження, завантаження моделі
│   │   └── utils.py         # підготовка ознак (add_features)
│   ├── database/
│   │   ├── database.py      # підключення до БД
│   │   ├── models.py        # опис таблиць
│   │   └── crud.py          # читання і запис у таблиці
│   └── schemas.py           # Pydantic-схеми запиту і відповіді
├── src/                     # ноутбуки ЛР1–ЛР3
├── models/                  # збережені моделі, метадані, new_input
├── docs/
│   └── project_description.md
├── .env.example
├── requirements.txt
└── README.md
```

## Встановлення

Потрібен Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Створіть файл `.env` у корені проєкту за зразком `.env.example`:

```
DB_URL=postgresql://USER:PASSWORD@HOST:5432/postgres
```

`.env` не додається в репозиторій.

## Запуск сервера

З кореня проєкту:

```bash
uvicorn app.main:app --reload
```

Після запуску:
- Swagger UI: http://127.0.0.1:8000/docs
- Перевірка стану: http://127.0.0.1:8000/health

## Тренування моделі: POST /train-model

Запит без тіла:

```bash
curl -X POST http://127.0.0.1:8000/train-model
```

Що відбувається:
1. Зчитуються всі записи з `stress_data_processed`.
2. Додаються похідні ознаки: `GPA_Diff`, `Anxiety_Squared`, `Total_Study_Load`, `Exhaustion_Index`, `AI_Inefficiency`.
3. Дані діляться на train (90%) і new_input (10%) зі стратифікацією за класом.
4. Тренується SVR (RBF) з MinMaxScaler і PCA. Гіперпараметри підібрано в ЛР3 через Optuna.
5. Модель зберігається в `models/SVR_unbalanced_tuned_<run_id>.joblib`, метадані в `..._meta.json`, рядки new_input у `new_input_<run_id>.csv`.
6. Прогнози на train-даних записуються в `predictions` з `source = "train"`, метрики в `model_metrics`.

Відповідь містить `run_id`, кількість рядків train і new_input та метрики на train.

Тривалість: кілька хвилин.

## Передбачення: POST /predict

Є два способи передати дані. Передається рівно один з них.

### Варіант 1: за Student_ID

Ознаки беруться з `stress_data_processed`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"student_id": 128015}'
```

### Варіант 2: з JSON-ознаками

Ознаки передаються у форматі `stress_data_processed` без `Student_ID`, `Burnout_Risk_Level` і `timestamp`. Похідні ознаки сервер рахує сам, їх передавати не потрібно.

```json
{
  "features": {
    "Year_of_Study": 4.0,
    "Pre_Semester_GPA": 0.5969599263012,
    "Weekly_GenAI_Hours": 1.0,
    "Prompt_Engineering_Skill": 2.0,
    "Tool_Diversity": 0.25,
    "...": "інші базові ознаки"
  }
}
```

Скорочений приклад. У реальному запиті мають бути всі 24 базові ознаки. Найнадійніше згенерувати їх з `new_input`:

```bash
python -c "import pandas as pd, json; df=pd.read_csv('models/new_input_a1d9f436.csv'); r=df.drop(columns=['Student_ID','Burnout_Risk_Level','timestamp'],errors='ignore').iloc[0]; json.dump({'features': {k: float(v) for k,v in r.items()}}, open('features.json','w'))"

curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d @features.json
```

### Відповідь

```json
{
  "request_id": "062bdedfe2314b64b91996568dda0586",
  "run_id": "a1d9f436",
  "model_name": "SVR_unbalanced_tuned",
  "source": "inference",
  "student_id": 128015,
  "y_pred_raw": 2.0,
  "y_pred_class": 2,
  "label": "High"
}
```

- `y_pred_raw`: прогноз SVR після обрізання до діапазону [0, 2].
- `y_pred_class`: `round(y_pred_raw)`, тобто клас 0 (Low), 1 (Medium) або 2 (High).

### Помилки

| Код | Причина |
|---|---|
| 409 | Модель не знайдено. Спершу викличте `/train-model`. |
| 404 | `student_id` немає в `stress_data_processed`. |
| 422 | Передано обидва способи або жодного, чи бракує ознак. |

## Структура бази даних

| Таблиця | Призначення |
|---|---|
| `stress_data` | Вихідні дані (ЛР2), містить `timestamp`. |
| `stress_data_processed` | Оброблені ознаки (ЛР3), містить `timestamp` і `Student_ID`. |
| `predictions` | Усі прогнози. `source` = `train`, `inference` (або `test` у старих записах ЛР3, які перенесено в `predictions_lab3_test`). |
| `inference_inputs` | Вхідні дані кожного запиту `/predict`: `request_id`, `created_at`, `Student_ID`, `raw_payload` (JSON). |
| `model_metrics` | Метрики запусків: `run_id`, `stage`, `split`, `mae`, `rmse`, `r2`, `accuracy`, `bal_acc`, `qwk`, `params`. |

Зв'язок: `predictions.request_id` → `inference_inputs.request_id`. Для рядків `source = 'train'` поле `request_id` порожнє.

Основні колонки `predictions`: `run_id`, `created_at`, `model_name`, `source`, `student_id`, `y_true`, `y_pred_raw`, `y_pred_class`, `request_id`.

## Приклади збережених передбачень

Останні передбачення з `/predict`:

```sql
SELECT p.request_id, p.student_id, p.y_pred_raw, p.y_pred_class, p.created_at, i.raw_payload
FROM predictions p
JOIN inference_inputs i ON i.request_id = p.request_id
WHERE p.source = 'inference'
ORDER BY p.created_at DESC
LIMIT 5;
```

Приклад рядка:

| request_id | student_id | y_pred_raw | y_pred_class | source |
|---|---|---|---|---|
| 062bdedfe2314b64b91996568dda0586 | 128015 | 2.0 | 2 (High) | inference |

Розподіл прогнозів за класами для кожного джерела:

```sql
SELECT source, y_pred_class, COUNT(*)
FROM predictions
WHERE run_id = 'a1d9f436'
GROUP BY source, y_pred_class
ORDER BY source, y_pred_class;
```

## Модель і метрики

Метрики на train, запуск `a1d9f436`:

| Метрика | Значення |
|---|---|
| MAE | 0.512 |
| RMSE | 0.644 |
| R² | 0.272 |
| Accuracy | 0.531 |
| Balanced accuracy | 0.492 |
| QWK | 0.411 |

Найкращий CV MAE з ЛР3: 0.5249 (5-fold, на train).