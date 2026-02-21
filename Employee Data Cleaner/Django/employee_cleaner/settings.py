"""
Django settings for employee_cleaner project.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-change-me-in-production-use-env-var"

DEBUG = True

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "cleaner",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "employee_cleaner.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "employee_cleaner.wsgi.application"

# Sessions — stored in the filesystem for zero DB dependency
SESSION_ENGINE = "django.contrib.sessions.backends.file"
SESSION_FILE_PATH = BASE_DIR / "tmp" / "sessions"
SESSION_COOKIE_AGE = 60 * 60 * 4  # 4 hours

# File uploads — up to 50 MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 52_428_800
FILE_UPLOAD_MAX_MEMORY_SIZE = 52_428_800

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Create session directory automatically
os.makedirs(SESSION_FILE_PATH, exist_ok=True)
