===========
DATA
===========
Training set: train.pkl
Validation set: val.pkl

--------------------------------
Data Descriptions:
--------------------------------
Training set:

train.pkl is a list of 20000 data points, where each data point is represented as a dictionary. Each data point (dictionary) consists of three keys: X, y, and file_name. X is our training data, where each item is a 3x96x96 image. y is our target labels, which in this care are saliency maps. Each saliency map is a 48x48 image. file_name is the filename of the images. 
--------------------------------
Validation set:

val.pkl is a list of 500 data points, where each data point is also represented as a dictionary. Here, each data point (dictionary) consists of five keys: X, y, y_original, X_original, and file_name. Again, X consists of 3x96x96 images and y consists of 48x48 saliency maps. y_original is the original saliency maps before downscaling to 48x48; these are 480x640. X_original is the original training data before downscaling; these are 3x480x640. You should use the downsized X and y data for all of your training. y_original and X_original should be used for evaluation and visualisation. 
--------------------------------
The DataLoader:

dataset.py defines a Pytorch Dataset. dataset.py will unpickle the dataset provided (either train.pkl or val.pkl) and store it in self.dataset. dataset.py also has a __getitem()__ function, which will return datapoints according to the provided index. You can use this dataset by passing it into a torch.utils.data.DataLoader, as shown in the labs. 

===========
CODE
===========

dataset.py: dataloader for the Salicon dataset. It takes as input the path to the training and validation files of the Salicon datasets (train.pkl and val.pkl)

evaluation.py: code for evaluating the trained model. It takes as input two arguments:
    1) --preds: path to saliency maps predictions of the validation set. It should be a pickle file containing a list of the models' outputs for each image in the validation set, i.e. a             flattened version of the predicted saliency map of size 2304     
    2) --gts: path to the provided file with the groundtruth of the validation set (val.pkl)

visualisation.py: code for visualising images, ground truth saliency maps and the model's predictions, similar to Figure 5 in the paper. It takes three arguments:
    1) --preds: path to saliency maps predictions of the validation set. It should be a pickle file containing a list of the models' outputs for each image in the validation set, i.e. a                flattened version of the predicted saliency map of size 2304     
    2) --gts: path to the provided file with the groundtruth of the validation set (val.pkl)
    3) --outdir: path to the output directory where you want to save the output image

