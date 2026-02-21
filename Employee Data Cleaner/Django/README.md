# 🧹 Employee Data Cleaner — Django

An interactive, step-by-step data cleaning pipeline for Indian employee datasets,
built with Django + pure vanilla JS. No database required.

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dev server
python manage.py runserver
```

Open **http://127.0.0.1:8000** in your browser.

---

## Project Structure

```
employee_cleaner/
├── manage.py
├── requirements.txt
├── README.md
│
├── employee_cleaner/          # Django project package
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── cleaner/                   # Main app
    ├── views.py               # All step logic + AJAX endpoints
    ├── urls.py                # URL routing
    └── templates/
        └── cleaner/
            └── index.html     # Full SPA-style UI
```

---

## API Endpoints

| Method | URL                    | Description                        |
|--------|------------------------|------------------------------------|
| POST   | `/upload/`             | Upload CSV or Excel file           |
| POST   | `/step/preview/`       | Return first 10 rows + types       |
| POST   | `/step/missing/`       | Missing value audit                |
| POST   | `/step/convert/`       | Force numeric dtypes               |
| POST   | `/step/fill/`          | Fill missing values (configurable) |
| POST   | `/step/duplicates/`    | Remove duplicate rows              |
| POST   | `/step/negative/`      | Fix negative salaries              |
| POST   | `/step/outliers/`      | Remove outliers (Z-score or IQR)   |
| POST   | `/step/profile/`       | Generate data profile report       |
| POST   | `/reset/`              | Restore original dataset           |
| GET    | `/download/?format=csv`   | Download cleaned CSV            |
| GET    | `/download/?format=excel` | Download cleaned Excel          |

---

## Improvements Over the Streamlit Version

| Feature                       | Streamlit | Django |
|-------------------------------|-----------|--------|
| No page reload between steps  | ❌         | ✅ AJAX |
| Configurable fill strategies  | ❌         | ✅      |
| IQR outlier method            | ❌         | ✅      |
| Configurable outlier threshold| ❌         | ✅      |
| Salary histogram (Chart.js)   | ❌         | ✅      |
| Data profile report           | ❌         | ✅      |
| Categorical value breakdowns  | ❌         | ✅      |
| Excel export                  | ❌         | ✅      |
| Undo / Reset pipeline         | ❌         | ✅      |
| Tabbed results per step       | ❌         | ✅      |
| Accepts CSV + Excel upload    | ❌         | ✅      |
| No external CSS framework     | —         | ✅      |

---

## Production Notes

- Change `SECRET_KEY` in `settings.py` to a random value (use `django.core.management.utils.get_random_secret_key()`).
- Set `DEBUG = False` and configure `ALLOWED_HOSTS`.
- For persistent sessions across restarts, switch `SESSION_ENGINE` to `django.contrib.sessions.backends.db` (add `django.contrib.sessions` to `INSTALLED_APPS` and run `python manage.py migrate`).
- Use `gunicorn employee_cleaner.wsgi` with an nginx reverse proxy for production.
