from django.urls import path
from . import views

urlpatterns = [
    path('register/',views.user_register,name = "user_register"),
    path('Add-notes/',views.AddNote,name = "Add_notes"),
    path('Get-notes/',views.GetNotes,name = "Get_notes"),
    path('Get-Metrics/',views.GetMetrics,name = "Get_metrics"),
    path('notes-list/', views.NotesList,name = "notes-list"),
    path('notes-detail/<str:pk>', views.NotesDetail,name = "notes-detail"),
]
