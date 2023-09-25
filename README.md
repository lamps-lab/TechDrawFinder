# ImageSearchEngine
The goal is to build a vector search demo website for the DeepPatent3 dataset. The data includes at least 50,000 technical drawings collected from USPTO and their captions.

# Top-level Architecture
![architecture](https://github.com/lamps-lab/ImageSearchEngine/assets/32687449/4beea40f-b1c7-48ac-88da-c73f16ab0a24)

# Note
We have to create media/ and static/ in the Main Project Directory folder:
	a) the media/ folder is necessary to save the user query
	b) the static/ folder is necessary where all the patent images will be stored and display the images when user do a query.
	c) Also, the inside the app foler (i.e., vector/), the vector database (i.e., large.index) should reside.