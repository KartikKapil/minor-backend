# Generated by Django 3.2.5 on 2021-11-27 11:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_auto_20211102_1745'),
    ]

    operations = [
        migrations.AddField(
            model_name='notes',
            name='Title',
            field=models.CharField(default='Happy', max_length=100),
            preserve_default=False,
        ),
    ]
