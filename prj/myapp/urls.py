from django.urls import path
from . import views

app_name = 'myapp'

urlpatterns = [
    path('home/', views.home, name='home'),
    path('result/', views.form_valid, name = 'form_valid'),
    path('input/', views.input, name = 'Input_View'),
    path('howtouse/', views.HowtoUse_View, name = 'HowtoUse_View'),
    path('user/', views.User_View, name = 'User_View'),
    path('database/', views.database, name='database'),
    path('nowloading/',views.nowloading, name='nowloading'),
    path('database/delete', views.delete, name='delete'),
]