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
from django.urls import include, path
from django.views.static import serve

from Initial import settings
# from database import views
from .import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('video/', include('video.urls')),
    path('dataset_singleshot/', include('dataset_singleshot.urls')),

    path('login/', views.login),
    path('logout/', views.logout),
    path('register/', views.register),
    path('base/', views.base),
    path('index/', views.index),
    # url(r'^show', views.show),
    # url(r'^load/', views.load),
    # url(r'^test/', views.test),
    # url(r'^query$', views.query),
    # url(r'^PCM/', views.PCM),
    # url(r'^save$', views.save),
    # url(r'^showFeedback$', views.showFeedback),
    # url(r'^camera$', views.camera),
    # url(r'^camera_image$', views.camera_image),
    # url(r'^capture$', views.capture),
    # url(r'^xianshi$', views.xianshi),
    # # url(r'^captcha', include('captcha.urls'))
    url(r'^15/(?P<path>.*)$', serve, {'document_root': '/home/cliang/mmap/VSA_Server/retrieval/static/info/videos/15'}),
    url(r'^18/(?P<path>.*)$', serve, {'document_root': '/home/cliang/mmap/VSA_Server/retrieval/static/info/videos/18'}),
    url(r'^23/(?P<path>.*)$', serve, {'document_root': '/home/cliang/mmap/VSA_Server/retrieval/static/info/videos/23'}),
    url(r'^25/(?P<path>.*)$', serve, {'document_root': '/home/cliang/mmap/VSA_Server/retrieval/static/info/videos/25'}),
    url(r'^img/(?P<path>.*)$', serve,{'document_root': '/home/cliang/mmap/VSA_Server/retrieval/static/info/imgs_resize'}),
    url(r'^probe/(?P<path>.*)$',serve,{'document_root': '/home/cliang/mmap/VSA_Server/retrieval/static/Pic/cropProbe'}),

]

urlpatterns += static('/static/', document_root=settings.STATICFILES_DIRS)
