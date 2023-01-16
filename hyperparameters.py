
""" 
    Holds the hyperparameters for the model. Can be tweaked and changed to alter and influence the model.
    MNIST Dataset size is 70,000. By default the data is split into 70-20-10 ratio for training, testing and 
    validation respectively. The layer values are the number of neurons in each layer. The input layer is 784 for 28x28 pixels 
    for each image. The output layer is 10 for 10 digits. The hidden layers can be changed to any value. 
"""


train_ratio = 0.7
test_ratio = 0.2
validatin_ratio = 0.1

hidden_layer_1 = 512
hidden_layer_2 = 256
hidden_layer_3= 128
OUTPUT_LAYER = 10

epochs = 5