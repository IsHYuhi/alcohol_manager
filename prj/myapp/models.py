from django.db import models
import hashlib
import os.path
from .name import NAME
# Create your models here.

# class Photo(models.Model):
#     title = models.CharField(null=True,max_length = 50)
#     image = models.ImageField(upload_to=get_image_path())
#     def get_filename(self):
#         return os.path.basename(self.file.name)
#########

def get_image_path(self, filename):
    """カスタマイズした画像パスを取得する.

    :param self: インスタンス (models.Model)
    :param filename: 元ファイル名
    :return: カスタマイズしたファイル名を含む画像パス
    """
    prefix = 'images/'
    name = 'image'
    extension = os.path.splitext(filename)[-1]
    return prefix + name + extension

def delete_previous_file(function):
    """不要となる古いファイルを削除する為のデコレータ実装.

    :param function: メイン関数
    :return: wrapper
    """
    def wrapper(*args, **kwargs):
        """Wrapper 関数.

        :param args: 任意の引数
        :param kwargs: 任意のキーワード引数
        :return: メイン関数実行結果
        """
        self = args[0]

        # 保存前のファイル名を取得
        result = Photo.objects.filter(pk=self.pk)
        previous = result[0] if len(result) else None
        path = '/Users/yuhi/django/prj/media/images' + '/' + 'image.jpg'
        path2 = '/Users/yuhi/django/prj/media/images' + '/' + 'image.png'
        if os.path.exists(path):
            os.remove('/Users/yuhi/django/prj/media/images' + '/' + 'image.jpg')
        if os.path.exists(path2):
            os.remove('/Users/yuhi/django/prj/media/images' + '/' + 'image.png')
        super(Photo, self).save()

        # 関数実行
        result = function(*args, **kwargs)


        return result
    return wrapper

class Photo(models.Model):
    @delete_previous_file
    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        super(Photo, self).save()

    @delete_previous_file
    def delete(self, using=None, keep_parents=False):
        super(Photo, self).delete()

    image = models.ImageField('画像', upload_to=get_image_path)


class Alcohol(models.Model):
    DEGREE_CHOICES = (
        ('', '度数を選択してください。'),
        (3, '3%'),
        (4, '4%'),
        (5, '5%'),
        (6, '6%'),
        (7, '7%'),
        (8, '8%'),
        (10, '10%'),
        (12, '12%'),
        (14, '14%'),
        (15, '15%'),
        (17, '17%'),
        (20, '20%'),
        (25, '25%'),
        (30, '30%'),
        (35, '35%'),
        (40, '40%'),
        (45, '45%'),
    )
    VALUE_CHOICES = (
        ('','量(サイズ)を選択してください。'),
        (180, '180ml(1合分)'),
        (200, '200ml(コップ1杯分)'),
        (350, '350ml(通常の1缶分)'),
        (500, '500ml(縦長の1缶分)'),
        (750, '750ml(4合瓶)'),
        (1800, '1800ml(1小瓶)'),
    )
    name = models.CharField(max_length=20, default = NAME)
    degree = models.IntegerField(default=0,choices=DEGREE_CHOICES) #百分率(%)でアルコール濃度の入力
    value = models.IntegerField(default=0,choices=VALUE_CHOICES) #飲んだ量(ml)
    #number = models.IntegerField(default=0) #飲んだ本数(本)
    #user_id = models.IntegerField() #user ID
class InfoUser(models.Model):
    # nickname = models.CharField(max_length=20)
    age = models.IntegerField()
    weight = models.IntegerField()


