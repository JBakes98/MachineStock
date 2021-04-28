from django.urls import path

from stockprediction import views

urlpatterns = [
    path('', views.IndexView.as_view(), name='home'),
    path('stocks/', views.StockList.as_view(), name='stock-list'),
    path('stocks/<str:ticker>', views.StockDetail.as_view(), name='stock-detail'),
    path('add-stocks-endpoint/', views.AddStocksEndpoint.as_view(), name='add-stocks-background-endpoint'),
    path('collect-stock-data-endpoint/', views.CollectStockDataEndpoint.as_view(), name='collect-stock-data-endpoint'),
]
