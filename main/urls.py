from django.urls import path
from . import views

urlpatterns = [
    path('register/',views.user_register,name = "user_register"),
    path('Add-notes/',views.AddNote,name = "Add_notes"),
    path('notes-list/', views.NotesList,name = "notes-list"),
    path('notes-detail/<str:pk>', views.NotesDetail,name = "notes-detail"),
]
