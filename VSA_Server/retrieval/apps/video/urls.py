"""Initial URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.conf.urls import include
from django.views.static import serve
from django.urls import path

from Initial import settings
from video import views


urlpatterns = [
    path('', views.index),
    path('cameraPath/', views.cameraPath),
    path('query/', views.query),
    path('capture/', views.capture),
    path('xianshi/', views.xianshi),
    path('save/', views.save),
    path('show/', views.show),


]

urlpatterns += static('/static/', document_root=settings.STATICFILES_DIRS)
