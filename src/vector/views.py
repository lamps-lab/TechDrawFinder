from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
#from django.views.generic import ListView
#from .models import ListingImage
import faiss
import torch
import pandas as pd
import numpy as np
import glob
import os
import io
import codecs
from io import BytesIO
from torchvision import transforms
from  PIL import Image, ImageFile
import matplotlib.pyplot as plt
from urllib.request import urlopen
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# Create your views here.


## Loading CLIP model
def get_model_info(model_ID, device):
	model = CLIPModel.from_pretrained(model_ID).to(device)
	processor = CLIPProcessor.from_pretrained(model_ID)
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
	return model, processor, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"
model, processor, tokenizer = get_model_info(model_ID, device)

## Image Embedding
def get_single_image_embedding(my_image):
	image = processor(text = None, images = my_image, return_tensors="pt")["pixel_values"].to(device)
	embed = model.get_image_features(image)
	embed_np = embed.cpu().detach().numpy()
	return embed_np

## Text Embedding
def get_single_text_embedding(text):
	inputs = tokenizer(text, return_tensors = "pt").to(device)
	text_embeddings = model.get_text_features(**inputs)
	embedding_as_np = text_embeddings.cpu().detach().numpy()
	return embedding_as_np


#def home(request):
#	return render(request, 'events/home.html', {})

## To show the search icon  and camera button. We also  modified vector_search/urls.py, 
## vector/urls.py, and vector/models.py.
'''

class ImageListView(ListView):
	template_name = 'pciture.html'
	model = ListingImage
	def media(self):
		picture_list = ListingImage.objects.all()
		context = {'image_list'}
	return render(request, 'picture_list.html', {'picture_list': picture_list})
'''

#file_path = "./../vector_search/static/*.png"
#file_path = "/Users/muntabir/search-engine/vector_search/static/*.png"

def read_img_path(file):
	data = glob.glob(file)
	df = pd.DataFrame(data, columns = ['img_only'])
	return df

## This is a search function which will take user query and returns similar result

index = faiss.read_index('./../vector_search/vector/vector_db/large_v1.index')
fn = pd.read_csv('./../vector_search/patent_67000_segmented_metadata.csv')
caps = fn['object_infer']
viewpoint = fn['aspect']
def searched(request):
	
	if request.method == "POST" and request.FILES:
		data = []
		fs = FileSystemStorage() ##user query will be saved
		files = request.FILES.get('file')
		filename = fs.save(files.name, files)
		
		path = '/Users/muntabir/search-engine/vector_search/media/'
		
		## Displaying the user query
		url_img = fs.url(filename)


		query_image = Image.open((os.path.join(path)+ filename), 'r')
		query_image = query_image.resize((200, 200))
		
		query_embedding = get_single_image_embedding(query_image)
		
		query = np.array(query_embedding)
		D, I = index.search(query, k=9)
		
		images = []
		for x in I:
			for pic in x:
				result = open(fn['figure_path'][pic], 'r')
				img_captions = (caps[pic] + ", viewpoint: "+ viewpoint[pic])				
				images.append((result.name.split('/')[-1], img_captions))			

		return render(request, 'searched.html', {'query':url_img, 'img':images})

	else:
		txt = request.GET.get("query")
		
		#txt_idx = faiss.read_index('/Users/muntabir/search-engine/vector_search/vector/vector_db/text.index')
		
		if txt is not None:
			text_query = get_single_text_embedding(txt)
		
			txtQuery = np.array(text_query)
		
			D, I = index.search(txtQuery, k = 9)
		
			texts = []
			#fn = pd.read_csv('./../vector_search/patent_67000_segmented_metadata.csv')
			for x in I:
				for txt in x:
					res = open(fn['figure_path'][txt], 'r')
					txt_captions = (caps[txt] + ", viewpoint: "+ viewpoint[txt])
					texts.append((res.name.split('/')[-1], txt_captions))

			return render(request, 'searched.html', {'img':texts})

	return render(request, 'searched.html')
