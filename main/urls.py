from django.urls import path
from . import views

urlpatterns = [
    path('', views.apiOverview,name = "apiOverview"),
    path('notes-list/', views.NotesList,name = "notes-list"),
    path('notes-detail/<str:pk>', views.NotesDetail,name = "notes-detail"),
]
