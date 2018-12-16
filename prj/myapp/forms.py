from django import forms
from .models import Alcohol

class PhotoForm(forms.Form):
    image = forms.ImageField()

class AlcoholData(forms.ModelForm):
    class Meta:
        model = Alcohol
        fields = ['name', 'degree', 'value', 'number'] #user_id消去済み
