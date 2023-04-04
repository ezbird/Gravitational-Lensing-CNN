'''
This is a basic code to augment images into 
a reasonable training set for a CNN, by 
rotation, flipping, zooming, etc.

Augmentor catalog is needed:
pip install Augmentor

Ezra Huscher
April 2023
'''
import Augmentor

# My goal is to augment the 60 images of the training set:
# 1. flipping (vertical, horizontal, and both) x4
# 2. rotation (15◦) x24
# 3. scaling (double and half) x3

data_dir = '/home/dobby/LensingProject/training_set/'

# Open Augmentor and choose alterations to perform.
# Images are saved automatically in the data_dir folder.
p = Augmentor.Pipeline(data_dir)
p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25) # max rotation here is 25 deg left or right
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)

# Other options to consider:
#p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
#p.zoom_random(probability=0.5, percentage_area=0.8)
#p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)

# Okay! How many new augmented images should be made?
p.sample(600)
