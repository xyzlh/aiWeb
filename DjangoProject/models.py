from django.db import models


class User(models.Model):
    GENDER_CHOICES = [
        ('M', '男'),
        ('F', '女'),
    ]
    username = models.CharField(max_length=150, primary_key=True)
    password = models.CharField(max_length=128)
    status   = models.IntegerField(default=0, null=True)  # 0 普通用户, 1 管理员
    age      = models.IntegerField(null=True, blank=True)
    gender   = models.CharField(max_length=1, choices=GENDER_CHOICES, null=True, blank=True)
    class Meta:
        db_table = 'user'
        app_label = 'DjangoProject'   # 指定归属 App 名称
class Record(models.Model):
    id = models.AutoField(primary_key=True)  # 显式设为主键（可省略，Django 默认会自动添加）
    username = models.CharField(max_length=150)
    tumor_name = models.CharField(max_length=255)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = 'record'
        app_label = 'DjangoProject'   # 指定归属 App 名称

