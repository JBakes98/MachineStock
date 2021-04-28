from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('stockprediction', '0001_exchange'),
    ]

    operations = [
        migrations.CreateModel(
            name='Stock',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('ticker', models.CharField(max_length=4, unique=True)),
                ('exchange', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='stockprediction.exchange')),
            ],
            options={
                'verbose_name_plural': 'Stocks',
            },
        ),
    ]