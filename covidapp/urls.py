from django.contrib import admin
from django.conf.urls import url 
from django.urls import path , include 
from  . import views 
from .views import InteractiveGraphAPIView

urlpatterns = [
    path('helloworld/' , views.index , name = 'index'),
    path('helloworld2/' , InteractiveGraphAPIView.as_view()) , 
]
