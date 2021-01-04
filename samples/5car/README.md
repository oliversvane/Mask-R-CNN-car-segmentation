# Car Segmentation Example

This is an example showing the use of Mask RCNN in a real application.
We train the model to detect and show the diffrent car parts in color while changing the rest of the image to
grayscale.

This example was developed as a project for Deloitte



## Run Jupyter notebooks
Open the `inspect_5car_data.ipynb` or `inspect_5car_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the model

The code in `5car.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.

You can train using the inspect_5car_model.ipynb notebook, or the train_5car.py file.


## Evaluate the model

You can train using the inspect_5car_model.ipynb notebook


## Dataset
The data has the following format:
./Mask_RCNN/datasets/5car'
    - /train       Containing original image in png format
    - /val         Containing original image in png format
    - /mask        Containing greyscale image with classes from 1-8 for the individual class. png format
    
The extension does not matter, just be consistant

You can use the transform_images.py to transform your data correctly


## Weights 
Weights can be downloaded here:
https://drive.google.com/file/d/1R7RA9lvurWwyXYgCZl0QMt4seHUeRSnL/view?usp=sharing

add this .h5 file to logs/5car* and run the model