from django.urls import path
from . import views

app_name = 'demo'

urlpatterns = [
    path('',views.DisasterListView.as_view(),name='list'),
    path('<int:pk>/',views.DisasterDetailView.as_view(),name='detail'),
    path('create/',views.DisasterCreateView.as_view(),name='create'),
    path('update/<int:pk>/',views.DisasterUpdateView.as_view(),name='update'),
    #path('delete/<int:pk>/',views.SchoolDeleteView.as_view(),name='delete')
]
