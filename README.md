# Handwritten Digit Prediction 

A Handwritten Number Prediction program made with Keras, Python and other Python modules. Trained with the popular MNIST dataset. 

# Demo

### Setup
1. Check if the modules in `requirements.txt` are installed or not.
2. If not, run `pip install -r requirements.txt` in the terminal.
3. Download and open the folder as the workspace.

### Running

4. Run `python train.py` in the terminal to create and train the model.
5. Run `python test.py` in the terminal to test the images in the digits folder.
6. Once an image shows up on a new window, close it for the next image to show up.
7. Images can be added to the digits folder to predict the digit by running `python test.py` again. 
 <br />NOTE: The image should be in **28x28 pixel size** with **white background** and the **digit in black color**. Each image should only contain **a single digit between 0 to 9.**
 Images bigger than 28x28 pixels size will be scaled down, but will lose quality. The bigger the image size, the more it will be distored and the model will be less likely to predict it correctly.
 8. Hyperparameters in `hyperparameters.py` can be tweaked.

