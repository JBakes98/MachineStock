
from decimal import Decimal
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('stockprediction', '0002_stock'),
    ]

    operations = [
        migrations.CreateModel(
            name='StockData',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('high', models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15)),
                ('low', models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15)),
                ('open', models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15)),
                ('close', models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15)),
                ('adj_close', models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15)),
                ('volume', models.BigIntegerField(blank=True, null=True)),
                ('change', models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15)),
                ('change_perc', models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15)),
                ('ma7', models.DecimalField(decimal_places=4, max_digits=15)),
                ('ma21', models.DecimalField(decimal_places=4, max_digits=15)),
                ('ema26', models.DecimalField(decimal_places=4, max_digits=15)),
                ('ema12', models.DecimalField(decimal_places=4, max_digits=15)),
                ('MACD', models.DecimalField(decimal_places=4, max_digits=15)),
                ('sd20', models.DecimalField(decimal_places=4, max_digits=15)),
                ('upper_band', models.DecimalField(decimal_places=4, max_digits=15)),
                ('lower_band', models.DecimalField(decimal_places=4, max_digits=15)),
                ('ema', models.DecimalField(decimal_places=4, max_digits=15)),
                ('momentum', models.DecimalField(decimal_places=4, max_digits=15)),
                ('log_momentum', models.DecimalField(decimal_places=4, max_digits=15)),
                ('stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='stock_prices', to='stockprediction.stock')),
            ],
            options={
                'verbose_name_plural': 'Stock Price Data',
                'ordering': ['stock', '-date'],
            },
        ),
        migrations.AddIndex(
            model_name='stockdata',
            index=models.Index(fields=['date', 'stock'], name='stock_predi_date_8755f6_idx'),
        ),
        migrations.AddConstraint(
            model_name='stockdata',
            constraint=models.UniqueConstraint(fields=('stock', 'date'), name='stocks_day_data'),
        ),
    ]
