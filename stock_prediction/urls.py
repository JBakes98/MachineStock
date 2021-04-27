from django.urls import path

from stock_prediction.views import StockList, StockDetail

urlpatterns = [
    path('', StockList.as_view(), name='stock-list'),
    path('<str:ticker>', StockDetail.as_view(), name='stock-detail')
]
