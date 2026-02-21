from django.urls import path
from . import views

urlpatterns = [
    path("",                views.index,          name="index"),
    path("upload/",         views.upload,         name="upload"),
    path("step/preview/",   views.step_preview,   name="step_preview"),
    path("step/missing/",   views.step_missing,   name="step_missing"),
    path("step/convert/",   views.step_convert,   name="step_convert"),
    path("step/fill/",      views.step_fill,       name="step_fill"),
    path("step/duplicates/",views.step_duplicates, name="step_duplicates"),
    path("step/negative/",  views.step_negative,   name="step_negative"),
    path("step/outliers/",  views.step_outliers,   name="step_outliers"),
    path("step/profile/",   views.step_profile,    name="step_profile"),
    path("reset/",          views.reset,           name="reset"),
    path("download/",       views.download,        name="download"),
]
