from django.shortcuts import render, redirect
from .models import Photo
from .forms import PhotoForm
from django.http import HttpResponse
from .models import Alcohol #models.pyのimport
from .forms import AlcoholData #forms.pyのAlcoholData()のimport
from .forms import WeightForm #追記ver2
from .models import InfoUser #追記ver3!!!!!!!
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
    params = {
        'title': '体重などのユーザデータ情報',
        'form_weight': WeightForm(),
        'user_info': [],
    }
    if(req.method == 'POST'):
        obj = InfoUser()
        user_info = WeightForm(req.POST, instance=obj)
        # if not user_info.is_valid():
        #     raise ValueError('invalid form')
        user_info.save()
        return redirect( '/myapp/home')
    return render(req,'myapp/user.html', params)

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

    f = open('/Users/yuhi/django/prj/myapp/name.py', 'w')
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


def delete(request):
    Alcohol.objects.all().delete()
    return redirect('/myapp/database')#ここはパスを指定

def database(request):
    params = {
        'title': 'データベース情報',
        'message': 'データベースの全データです。',
        'condition': '',
        'data': [],
        'detail1': '',
        'detail2': '',
        'detail3': '',
        'detail4': '',
    }

    params['data'] = Alcohol.objects.all()
    alcohol_data = Alcohol.objects.all()
    user_info = InfoUser.objects.all().last()
    intake_alcohol = 0.0

    for item in alcohol_data:
        intake_alcohol += item.degree*float(item.value)

    intake_alcohol *= 0.8 #アルコール比重


    result = intake_alcohol /100.0 *0.15 / float(user_info.weight)
    result = round(result,2)

    if result < 0.02:
        params['condition'] = 'シラフ'
        params['message'] = '<font color="green">このアプリを通じて楽しいお酒の飲み方を覚えましょう!</font>'
    elif result < 0.05 :
        params['condition'] = '爽快期'
        params['message'] = '<font color="skyblue">お酒はとても楽しいですね。ほどほどに...</font>'
        params['detail1'] = 'さわやかな気分になる'
        params['detail2'] = '皮膚が赤くなる'
        params['detail3'] = '陽気になる'
        params['detail4'] = '判断力が少しにぶる'
    elif result < 0.1 :
        params['condition'] = 'ほろ酔い期'
        params['message'] = '<font color="blue">お酒との付き合い方が上手い方はここらへんで飲むのをやめます。</font>'
        params['detail1'] = '手の動きが活発になる'
        params['detail2'] = '抑制がとれる（理性が失われる）'
        params['detail3'] = '体温が上がる'
        params['detail4'] = '脈が速くなる'
    elif result < 0.15 :
        params['condition'] = '酩酊初期'
        params['message'] = '<font color="purple">これ以上飲むと取り返しのつかないことになり兼ねます。注意しましょう。</font>'
        params['detail1'] = '気が大きくなる'
        params['detail2'] = '大声でがなりたてる'
        params['detail3'] = 'おこりっぽくなる'
        params['detail4'] = '立てばふらつく'
    elif result < 0.3 :
        params['condition'] ='酩酊期'
        params['message'] = '<font color="magenta">人に迷惑をかけます。今すぐ帰る準備をしましょう。</font>'
        params['detail1'] = '何度も同じことをしゃべる'
        params['detail2'] = '千鳥足になる'
        params['detail3'] = '呼吸が速くなる'
        params['detail4'] = '吐き気、おう吐がおこる'
    elif result < 0.4 :
        params['condition'] = '泥酔期'
        params['message'] = '<font color="pink">記憶できない状態です。お酒をやめましょう。</font>'
        params['detail1'] = 'まともに立てない'
        params['detail2'] = '意識がはっきりしない'
        params['detail3'] = '言語がめちゃめちゃになる'
    else :
        params['condition'] = '昏睡期'
        params['message'] = '<font color="red">命が危険です！お酒をやめてください!!!</font>'
        params['detail1'] = 'ゆり動かしても起きない'
        params['detail2'] = '大小便はたれ流しになる'
        params['detail3'] = '呼吸はゆっくりと深い'
        params['detail4'] = '死亡'

    return render(request, 'myapp/database.html', params)
