"""
URL configuration for DjangoProject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path

from DjangoProject import views

urlpatterns = [
    path('monthly_stats_api/', views.monthly_stats_api,name='monthly_stats_api'),
    path('stats_api/', views.stats_api, name='stats_api'),

    path('statistics/', views.statistics, name='statistics'),
    path('get_username/', views.get_username_view, name='get_username'),
    path('history/', views.get_history_view, name='history'),
    path('auth/', views.auth_view, name='auth'),
    path('', views.login, name='login'),
    path('index/',views.index, name = 'index'),
    path('type/', views.infer_type, name = 'infer_type'),
path('location/', views.infer_location, name = 'infer_location'),
path('ai/', views.infer_ai, name = 'infer_ai'),

]