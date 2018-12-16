from django.shortcuts import render, redirect
from .models import Photo
from .forms import PhotoForm
from django.http import HttpResponse
from .models import Alcohol #models.pyのimport
from .forms import AlcoholData #forms.pyのAlcoholData()のimport
from time import sleep
from .name import NAME

#機械学習用
import sys
import cv2
import tensorflow as tf
import os
import random
import numpy as np
from . import main

# 識別ラベルと各ラベル番号に対応する名前
DRINK_NAMES = {
  0: u"ビール",
  1: u"赤ワイン",
  2: u"白ワイン"
}

def home(req):
    f = open('/Users/yuhi/django/prj/myapp/name.py', 'w')
    f.write("NAME = \"\" ")
    f.close()
    if req.method == 'GET':
        return render(req, 'myapp/home.html', {
            'form': PhotoForm(),
            'photos': Photo.objects.all(), #いらないかも
        })

    elif req.method == 'POST':
        form = PhotoForm(req.POST, req.FILES)
        if not form.is_valid():
            raise ValueError('invalid form')

        photo = Photo()
        photo.image = form.cleaned_data['image']
        photo.save()
        return redirect('/myapp/result')

# def Input_View(req):
#     return render(req, 'myapp/home.html')

def HowtoUse_View(req):
    return render(req, 'myapp/howtouse.html')

def User_View(req):
    return render(req,'myapp/user.html')

def nowloading(req):
    return render(req, 'myapp/nowloading.html')


def form_valid(request):

    image = []
    img = cv2.imread("//Users/yuhi/django/prj/media/images/image.jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (28, 28))
    image.append(img.flatten().astype(np.float32)/255.0)
    image = np.asarray(image)
    logits = main.inference(image, 1.0)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    ckpt_path = '/Users/yuhi/django/prj/myapp/model.ckpt'
    if ckpt_path:
        saver.restore(sess, ckpt_path)
    softmax = logits.eval()
    result = softmax[0]
    rates = [round(n * 100.0, 1) for n in result]
    humans = []
    for i, rate in enumerate(rates):
        name = DRINK_NAMES[i]
        humans.append({
            'label': i,
            'name': name,
            'rate': rate
        })

    rank = sorted(humans, key=lambda x: x['rate'], reverse=True)
    rank2 = rank[0]
    rank3 = rank2['name'] #順番に並べたやつから名前を取り出す
    # 推論した結果を、テンプレートへ渡して表示
    context = {
        'result':rank3,
    }

    f = open('/Users/yuhi/django/prj/myapp/number.py', 'w')
    f.write("NAME = '"+ rank3 +"'")
    f.close()

    return redirect( '/myapp/nowloading')
    # return render(request,'myapp/inputform.html', context)

def input(request):
    if(request.method == 'POST'):
        obj = Alcohol()
        alcohol = AlcoholData(request.POST, instance=obj)
        alcohol.save()
        return redirect('/myapp/database')

    params = {
        'title': 'ホームページ',
        'form': AlcoholData(),
        'result':NAME,
    }
    return render(request, 'myapp/inputform.html', params)

def database(request):
    params = {
        'title': 'データベース情報',
        'message': 'データベースの全データです。',
        'form': AlcoholData(),
        'data': [],
    }

    #POST送信のチェック
    if(request.method == 'POST'):
        id = request.POST['id']
        #params['data'] = Alcohol.objects.all().filter(user_id=id) #user_idとforms.pyの変数idの一致するものを表示
        params['form'] = AlcoholData(request.POST)
    else:
        params['data'] = Alcohol.objects.all()

    return render(request, 'myapp/database.html', params)




