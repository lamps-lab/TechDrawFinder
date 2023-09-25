from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
	#path('', views.home, name = 'home'),
	#path('vector/', views.media, name= 'media'),
	#path('search', views.search, name=search),
	path('searched', views.searched, name='searched'),
]
