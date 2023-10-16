# ImageSearchEngine
The goal is to build a vector search demo website for the DeepPatent3 dataset. The data includes at least 50,000 technical drawings collected from USPTO and their captions.

# Top-level Architecture
![architecture](https://github.com/lamps-lab/ImageSearchEngine/assets/32687449/4beea40f-b1c7-48ac-88da-c73f16ab0a24)

# exampledata
We have provided few test data for testing our search engine.

## src/
Under this directory, you will find our Django application with front-end and back-end configruation.

Please note that, the media/ directory is where the user provided query will be saved and the static/ is the directory where the USPTO images are stored (currently, we provided only few test data).

## Demo Link
Our TechDrawFinder search engine is available at:https://vector-search.cs.odu.edu/