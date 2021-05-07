from decimal import Decimal
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stock_prediction', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='stockdata',
            name='dividend_amount',
            field=models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15),
        ),
    ]
