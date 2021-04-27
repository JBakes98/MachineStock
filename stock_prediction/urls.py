from django.urls import path

from stock_prediction import views

urlpatterns = [
    path('', views.IndexView.as_view(), name='home'),
    path('stocks/', views.StockList.as_view(), name='stock-list'),
    path('stocks/<str:ticker>', views.StockDetail.as_view(), name='stock-detail'),
    path('add-stocks-background-task/', views.add_stocks_background_task, name='add-stocks-background-task'),
]
