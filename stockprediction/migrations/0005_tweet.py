from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('stockprediction', '0004_dividend_amount_column'),
    ]

    operations = [
        migrations.CreateModel(
            name='Tweet',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.CharField(max_length=280)),
                ('created_at', models.DateTimeField()),
                ('user_id', models.BigIntegerField()),
                ('user_screen_name', models.CharField(max_length=50)),
                ('verified', models.BooleanField(default=False)),
                ('followers_count', models.BigIntegerField()),
                ('friends_count', models.BigIntegerField()),
                ('favourites_count', models.BigIntegerField()),
                ('retweet_count', models.BigIntegerField()),
                ('stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='stock_prediction.stock')),
            ],
            options={
                'verbose_name_plural': 'Tweets',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='tweet',
            index=models.Index(fields=['stock', 'text'], name='stock_predi_stock_i_36014c_idx'),
        ),
        migrations.AddConstraint(
            model_name='tweet',
            constraint=models.UniqueConstraint(fields=('text', 'created_at'), name='tweet_data'),
        ),
    ]
