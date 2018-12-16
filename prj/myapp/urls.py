from django.urls import path
from . import views

app_name = 'myapp'

urlpatterns = [
    path('home/', views.Index, name='home'),
    path('result/', views.form_valid, name = 'form_valid'),
    path('input/', views.Input_View, name = 'Input_View'),
    path('howtouse/', views.HowtoUse_View, name = 'HowtoUse_View'),
    path('user/', views.User_View, name = 'User_View')
]