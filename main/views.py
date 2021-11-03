from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework import permissions, status
from rest_framework.decorators import (
    api_view, permission_classes
)
from rest_framework.response import Response
from .serializers import NoteSerializer, UserSerializer
from .models import Notes
# from .utility import incoming_message
from django.contrib.auth.models import User

@api_view(['POST'])
@permission_classes((permissions.AllowAny, ))
def user_register(request):
	"""For user registration """

	serializer = UserSerializer(data=request.data)
	if serializer.is_valid():
		serializer.save()
		return Response(serializer.data, status=status.HTTP_201_CREATED)
	return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def AddNote(request):
	"""For adding notes"""
	username = request.data.get('username')
	user = User.objects.get(username=username)
	Entry = request.data.get('entry')
	# Emotion = incoming_message(Entry)
	Emotion = 'Happy'
	Note = Notes(user=user, Entry=Entry, Emotion=Emotion)
	Note.save()
	return Response(status=200)
	

@api_view(['GET'])
def NotesList(request):
	notes = Notes.objects.all()
	serializer = NoteSerializer(notes, many=True)
	return Response(serializer.data)

@api_view(['GET'])
def NotesDetail(request, pk):
	notes = Notes.objects.get(id=pk)
	serializer = NoteSerializer(notes, many=False)
	return Response(serializer.data)
