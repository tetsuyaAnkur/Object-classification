Object Classification



Preprocess

For creating the training and testing dataset I had to extract useful information from the xml
file. For this I used Element-Tree, which is a very famous xml parser. Since I had to do object
classification so here I used Element-Tree to create a dictionary containing keys as the name
of the images and values as a list containing the corresponding classes of objects pesent in the
image.

I took a vector of 20 dimensions corresponding to 20 classes. If object of a particular class is
present in the image, that image would have the value 1, otherwise 0 in the place.

The dataset was divided into train(90% of the dataset) , validation(5%) and testing(5%) sets along
with the labels and was sent to train the model.



CNN Model

I have used Keras for building the CNN Model. I used a sequential model which has the embedding
input layer that takes in matrices of 500x500x3. This layer is followed by 4 Convolution layer,
2 Hidden Dense layer and an Output Dense layer. As the objects are to be classified into one of
the 20 classes, the output layer has 20 units.



Results

I was able to obtain an accuracy of 93% for classification. I have used 2700 random images for
training and 300 for testing.
