# Generated by Django 3.1.6 on 2021-07-06 18:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('adv_img_detect', '0002_auto_20210706_1220'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Result',
        ),
        migrations.AlterField(
            model_name='imageupload',
            name='pic',
            field=models.ImageField(blank=True, default='default.png', null=True, upload_to='images/'),
        ),
    ]
