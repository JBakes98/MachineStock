from decimal import Decimal
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockprediction', '0003_stock_data'),
    ]

    operations = [
        migrations.AddField(
            model_name='stockdata',
            name='dividend_amount',
            field=models.DecimalField(decimal_places=4, default=Decimal('0'), max_digits=15),
        ),
    ]
