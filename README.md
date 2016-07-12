# concarne 

#### a lightweight framework for learning with side information (aka privileged information), based on Theano and Lasagne 

concarne implements various patterns for learning with side information presented in this [paper](http://arxiv.org/abs/1511.06429).

### Quickstart

- Check out concarne from the repository<br/>
```git clone https://github.com/tu-rbo/concarne.git```
- Install concarne:<br/>
```python setup.py install```
- Run the simple example <br/>
```python example/simple_multiview.py```

For more information on how to use concarne, checkout out the documentation
or the code of the simple example example/simple_multiview.py

The experiments in the paper are implemented in example/synthetic.py and
example/handwritten.py


### What is concarne?

concarne implements a variety of different patterns that enable to apply
*side information*. As it depends on Theano and lasagne, you can use 
neural network structures that you have developed yourself and easily
combine them with the side information learning task.

### What is learning with side information?

Supervised, semi-supervised, and unsupervised learning estimate a function 
given input/output samples. Generalization to unseen samples requires making 
prior assumptions about this function. However, many priors assumptions cannot be defined 
by only taking the function, its input, and its output into account. 

We use *side information* to define such priors. Side information are
data that are neither from the input space nor from the output space of the function,
but include useful information for learning it. Importantly, these 
data *are not required during test time*, but only during training time.

Learning with side information subsumes a variety of related approaches, such as 
- multi-task learning 
- multi-view learning (or co-learning)
- Learning using Privileged Information
- Slow Feature Analysis
- and others

### Examples for learning with side information?

To apply learning with side information, you need to have an additional source of data (neither input nor output of your
classifier/regressor) available during training - the *side information*.
However, this additional data is *not required during test time*.

1. Imagine you want to classify images, and during test time you only have RGB
data available, but during training you also have 3D depth information 
available. Learning with side information, in particular the multi-view pattern allows you
to incorporate the depth data during training time to shape your classifier,
without making the depth data an input of your classifier.<br/>
Paper: [Chen et al., 2014: Recognizing RGB Images by Learning from RGB-D Data](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Chen_Recognizing_RGB_Images_2014_CVPR_paper.pdf)

2. Again consider image classification, but now imagine that in addition to the
labels for each training sample you also know the pose of the object in the image. 
You can use this pose information as an auxiliary prediction task using
the *multi-task* pattern.<br/>
Paper: [Zhao & Itti, 2016: Improved Deep Learning of Object Category using Pose Information](https://www.researchgate.net/publication/283734369_Improved_Deep_Learning_of_Object_Category_using_Pose_Information)

3. Another way to use pose information is to use relative poses between
*pairs of images*. This can be done using the *pairwise transformation pattern*.
<br/>
Paper: [Jayaraman & Grauman, 2015: Learning image representations equivariant to ego-motion](http://arxiv.org/pdf/1505.02206.pdf)

4. If you know want your side information is much better for predicting the
target than the input data, you can apply the *direct pattern* to do a regression 
of the input on the side information, and then use the resulting representation to
predict the targets.

All examples are examples for supervised learning, but learning with side information
is equally applicable to reinforcement learning:
[Jonschkowski & Brock, 2015: Learning state representations with robotic priors](http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Jonschkowski-15-AURO.pdf)

### Requirements

- [Theano](http://deeplearning.net/software/theano/)
- [Lasagne](https://github.com/Lasagne/Lasagne)

Follow the installation requirements for these frameworks.
In general, it is recommended to use the latest versions by installing it from github:

	```pip install --upgrade https://github.com/Theano/Theano/archive/master.zip```
	```pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip```

For using the nolearn compatibility extension, you will need to install the newest version of [nolearn](https://github.com/dnouri/nolearn) which requires pydotplus.

concarne was tested with Python 2.7.11+ and Python 3.5.2.

### Running tests

In the concarne repository root, type
   ```nosetests tests```

### Building the API documentation

In the concarne repository root, type

    cd docs
    make html
    
You can now read the documentation by opening docs/_build/html/index.html

### Citing concarne

If you use concarne in your scientific work, please consider citing the following paper:

    @article{
      author    = {Rico Jonschkowski and Sebastian H{\"{o}}fer and Oliver Brock},
      title     = {Patterns for Learning with Side Information},
      volume    = {arXiv:1511.06429 [cs.LG]},
      year      = {2016},
      url       = {http://arxiv.org/abs/1511.06429},
	}

We would also be glad if you sent us a copy of your paper!

### Additional data from the paper

The example scripts provided with concarne use some of the data from the paper "Patterns for Learning with Side Information".
The full datasets and a description of how these datasets are stored, formatted and how they have been generated can be found at [TU Berlin](https://owncloud.tu-berlin.de/index.php/s/865193384483af385172f5871aa5cd36).

### Disclaimer

Parts of concarne contain code from the [nolearn](https://github.com/dnouri/nolearn) library, Copyright (c) 2012-2015 Daniel Nouri.

No animals were harmed during the development of this framework.
