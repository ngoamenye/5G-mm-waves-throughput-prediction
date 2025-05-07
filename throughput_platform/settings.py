from pathlib import Path
import os  # Ajouté pour gérer les chemins

# 📁 Chemin de base du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# ⚠️ Clé secrète Django (à garder privée en prod)
SECRET_KEY = "django-insecure-#+wm4taor7)j#i^gx-!xz3im-(ezw6je9yre-4&ot%9ee--p(x"

# 👨‍💻 Mode debug activé pour développement
DEBUG = True

ALLOWED_HOSTS = []

# ✅ Apps installées
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "simulator",  # Ton app principale
]

# ⚙️ Middleware
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "throughput_platform.urls"

# 🎨 Templates configuration (avec DIRS mis à jour)
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "simulator", "templates")],
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

WSGI_APPLICATION = "throughput_platform.wsgi.application"

# 💾 Base de données SQLite
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# 🔐 Validation des mots de passe
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# 🌍 Internationalisation
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# 🗂️ Fichiers statiques (JS, CSS, images)

STATIC_URL = 'static/'
STATICFILES_DIRS = [
    BASE_DIR / 'simulator' / 'static',
]

# 🆔 Clé primaire auto par défaut
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = '/login/'     # redirection automatique après @login_required
LOGIN_REDIRECT_URL = '/dashboard/'  # après connexion, aller ici
