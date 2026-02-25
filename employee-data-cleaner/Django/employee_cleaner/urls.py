from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Include all paths from the cleaner app directly at the root
    path('', include('cleaner.urls')),
]
