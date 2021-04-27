from django.urls import path

from stock_prediction import views

urlpatterns = [
    path('', views.IndexView.as_view(), name='home'),
    path('stocks/', views.StockList.as_view(), name='stock-list'),
    path('stocks/<str:ticker>', views.StockDetail.as_view(), name='stock-detail'),
    path('add-stocks-endpoint/', views.add_stocks_background_endpoint, name='add-stocks-background-endpoint'),
    path('collect-stock-data-endpoint/', views.collect_stock_data_endpoint, name='collect-stock-data-endpoint'),
]
