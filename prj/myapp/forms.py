from django import forms

class PhotoForm(forms.Form):
    image = forms.ImageField()