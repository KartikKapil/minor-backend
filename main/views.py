from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework import permissions, status
from rest_framework.decorators import (
    api_view, permission_classes
)
from rest_framework.response import Response
from .serializers import NoteSerializer, UserSerializer
from .models import Notes


@api_view(['GET'])
def apiOverview(request):
	api_urls = {
		'List':'/notes-list/',
		'Detail View':'/notes-detail/<str:pk>/',
		'Create':'/notes-create/',
		'Update':'/notes-update/<str:pk>/',
		'Delete':'/notes-delete/<str:pk>/',
		}

	return Response(api_urls)

@api_view(['POST'])
@permission_classes((permissions.AllowAny, ))
def user_register(request):
	serializer = UserSerializer(data=request.data)
	if serializer.is_valid():
		serializer.save()
		return Response(serializer.data, status=status.HTTP_201_CREATED)
	return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes((permissions.AllowAny, ))
def NotesList(request):
	notes = Notes.objects.all()
	serializer = NoteSerializer(notes, many=True)
	return Response(serializer.data)

@api_view(['GET'])
@permission_classes((permissions.AllowAny, ))
def NotesDetail(request, pk):
	notes = Notes.objects.get(id=pk)
	serializer = NoteSerializer(notes, many=False)
	return Response(serializer.data)
