from django.db import models
import hashlib
import os.path
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
