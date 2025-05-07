from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predict/', views.predict_page, name='predict_page'),
    path('drift/', views.drift_page, name='drift_page'),
    path('map/', views.map_page, name='map_page'),
    path('', views.home_view, name='home'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('', views.home_view, name='home'),
   path('api/get_antennas/', views.get_antennas, name='get_antennas'),

    path('api/map_prediction/', views.map_prediction_api, name='map_prediction_api'),

    # API endpoints
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/detect_drift/', views.detect_drift, name='detect_drift'),
]
