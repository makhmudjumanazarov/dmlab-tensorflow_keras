# Dmlab  Dataset Classification
-Transfer Learning using VGG19 CNN Model 

-Building the CNN architecture from scratch
### Description
The <a href= "https://www.tensorflow.org/datasets/catalog/dmlab">Dmlab dataset</a> contains frames observed by the agent acting in the DeepMind Lab environment, which are annotated by the distance between the agent and various objects present in the environment. The goal is to is to evaluate the ability of a visual model to reason about distances from the visual input in 3D environments. The Dmlab dataset consists of 360x480 color images in 6 classes. The classes are {close, far, very far} x {positive reward, negative reward} respectively.

-I have used the VGG19 CNN Model, which is pre-trained on the ImageNet dataset for classification. Data augementation hasn't been used for making the model generalize better. The model achieved an accuracy 53% on validation set. This is not a good situation, but I don't understand why.

![Images of dmlab](/images/vgg19_evaluate.jpg)

-Using Tensorflow, I built the architecture from scratch like exports' method. I didn't use data augementation in this either for making the model generalize better. The model achieved an accuracy 51% in 5 epochs on validation set. There is enough information for each class but the result is not good. This situation is incomprehensible to me. If you have a better solution please share with me!!!

![Images of dmlab](/images/scratch_5_epochs.jpg)

### Dataset
Contents of the dataset:
- Number of categories: 6
- Number of train images: 65550
- Number of test images: 22375
- Number of validation images: 22628

Sample images of 10 different categories from the dataset:

![Images of dmlab](/images/dmlab_images.jpg)


### Getting Started
The `dmlab.ipynb` notebook can be directly run on Jupyter Notebook or others. Use GPU for faster training and evaluation.

### Steps
<br />
<b>Step 1.</b> Clone <a href= "https://github.com/makhmudjumanazarov/dmlab-tensorflow_keras.git">this repository </a>
via Terminal, cmd or PowerShell
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv dmlab
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source dmlab/bin/activate # Linux
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install -r requirements.txt # With Tensorflow GPU
pip install ipykernel
python -m ipykernel install --user --name=dmlab
</pre>
<br/>
<b>Step 5.</b> 
<pre>
The dmlab.ipynb notebook can be directly run on Jupyter Notebook
</pre> 
<br/>


## Dmlab - Streamlit - Demo 

Dmlab via Streamlit 

### Model 
Download the models through this <a href= "https://drive.google.com/file/d/1eujUxgPtbBHQKi9qBqpxKR3UyPelsvTv/view?usp=share_link">link</a>

<pre>
  streamlit run stream.py
</pre> 
