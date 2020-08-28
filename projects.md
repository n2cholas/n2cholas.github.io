---
layout: page
title: Projects
---

This page is out of date! Will be updated with my recent projects soon.


<hr>


<img src="/assets/images/projects/pixelshot2.png" class="center" width="50%"/>
### PixelShot 300: One-pixel Camera
_Java, Arduino, C++_

One-pixel camera that can capture a 300-by-300 photo in under 5 minutes. This
was achieved using a lens-phototransistor module with two stepper motors to
control direction. It was created by a team of four with nothing more than
basic electrical components along with an Arduino MEGA2560.

<p>The software controls the motors and sensors, undistorts the captured fisheye image, can demosaic a Bayer colour filter image, and balances colours and brightness.</p>

<p><a class="more-link" href="https://github.com/n2cholas/PixelShot300" target="_blank"><i class="fa fa-external-link"></i>View on Github</a></p>


<hr>


<img src="/assets/images/projects/distribution_of_types.png" class="center" width="50%"/>

### Competitive Pokemon Analysis
_Python, Pandas, Scipy, Scikit-learn, Seaborn, Jupyter_

I set out to predict a Pokemon’s competitive tier based on its in-game
characteristics. So, I scraped data from Smogon, Bulbapedia, and Veekun using
BeautifulSoup for Python. I then cleaned and analyzed various properties about
pokemon with respect to their tiers in a Jupyter Notebook. Finally, I built a
regression model using Scikit-learn.

<p><a class="more-link" href="https://github.com/n2cholas/pokemon-analysis" target="_blank"><i class="fa fa-external-link"></i>View on GitHub</a>&nbsp; &nbsp; &nbsp;<a class="more-link" href="https://nicholasvadivelu.com/2018/09/06/competitive-pokemon/" target="_blank"><i class="fa fa-external-link"></i>Read about Lessons I Learned</a> </p>


<hr>


<img src="/assets/images/projects/thrive_demo.gif" class="center" width="100%"/>
### Thrive Life Simulator
_Java_

Simulates a complex dinosaur ecosystem. Over a dozen species of carnivores,
herbivores, and omnivores interact in a forest-type environment. Phenomena such
as natural disasters, breeding, hunting, disease, aging, and simple genetic
variation are factored in. For this project, I wrote a 3D ray-casting engine
from scratch.

<p><a class="more-link" href="https://github.com/n2cholas/ThriveLifeSimulator" target="_blank"><i class="fa fa-external-link"></i>View on Github</a></p>


<hr>


<img src="/assets/images/projects/chrome.png" class="center"/>
### Chrome Tab Predictor
_Javascript, Synaptic_

Every time a tab is opened, this extension will predict what website the user
will want to visit. Given a user's browsing history, this chrome extension will
train an artificial neural network to make these predictions based on the day
and time.

<p><a class="more-link" href="https://github.com/n2cholas/chrome-tab-predictor" target="_blank"><i class="fa fa-external-link"></i>View on Github</a> &nbsp; &nbsp; &nbsp; <a class="more-link" href="https://chrome.google.com/webstore/detail/chrome-tab-predictor/dmifffdjjlkpobdffgpmeeiakgjmndfb" target="_blank"><i class="fa fa-external-link"></i>View on Chrome Webstore</a></p>


<hr>


<img src="/assets/images/projects/quora.png" width="50%" class="center"/>
### Kaggle - Quora Insincere Questions Classification
_Python, TensorFlow, Matplotlib, Jupyter/Colab_

This kernels-only competition involved classifying insincere questions. I
experimented with 1D CNNs and various types of RNNs, particularly LSTMs and
GRUs. For each of these, I tried adaptive optimization technique such as
RMSProp and Adam. Since there was a limited training time of 2 hours, I elected
to use a custom training loop along with an efficient text preprocessing
pipeline.

<p><a class="more-link" href="https://www.kaggle.com/n2cholas/preprocessing-lstm-in-tensorflow" target="_blank"><i class="fa fa-external-link"></i>View on Kaggle</a> </p>


<hr>


<img src="/assets/images/projects/sofasearch.gif" class="center"/>
### Sofa Search
_Python, TensorFlow, Keras, Flask, HTML/CSS, Javascript, Heroku_

Created at Hack the North 2017, this web app helped users find and purchase a
sofa. The app presents users with photos of sofas that they rate. Each rating
will train a convolutional neural network with their preferences to suggest
sofas that they may like.

<p><a class="more-link" href="https://devpost.com/software/sofa-search" target="_blank"><i class="fa fa-external-link"></i>View on DevPost</a> &nbsp; &nbsp; &nbsp; <a class="more-link" href="https://github.com/n2cholas/sofa-search" target="_blank"><i class="fa fa-external-link"></i>View on Github</a> </p>


<hr>


<img src="/assets/images/projects/rapchatz.jpg" class="center" width="50%"/>
### RapChatz
_Node.JS, Amazon Web Services_

App receives a keyword through Facebook messenger or the Amazon Echo and finds
a matching rap line based on rhyming and context. Created in a team of 5 using
Node.JS, Amazon Alexa, and Heroku, this application won top developer hack and
top use of Amazon Web Services out of 40 teams at StarterHacks 2017.

<p><a class="more-link" href="https://github.com/n2cholas/starterhacks" target="_blank"><i class="fa fa-external-link"></i>View on Github</a></p>


<hr>


<img src="/assets/images/projects/ted.gif" class="center" width="50%"/>
### Kaggle TED Talk Dataset Analysis
_Python, Pandas, TensorFlow, Keras, Matplotlib, Jupyter_

A quantitative measure of rating was extracted from qualitative descriptors in
the data set. Two neural networks were trained on various features, including
talk sentiment and themes, to predict this measure, achieving good accuracy.
Created with Lawrence Pang.

<p><a class="more-link" href="https://www.kaggle.com/lpang36/analysis-of-ted-talk-ratings/notebook" target="_blank"><i class="fa fa-external-link"></i>View on Kaggle</a> </p>


<hr>


<img src="/assets/images/projects/myolert.gif" class="center" width="100%"/>
### myolert
_Android (Java), MYO SDK, Google Location Services_

Android application that allows users to discretely call for help using the MYO
armband. A simple hand gesture will make your smartphone send out a distress
call or text to trusted contacts.

<p><a class="more-link" href="https://devpost.com/software/myolert" target="_blank"><i class="fa fa-external-link"></i>View on DevPost</a>&nbsp; &nbsp; &nbsp;<a class="more-link" href="https://github.com/n2cholas/myolert" target="_blank"><i class="fa fa-external-link"></i>View on GitHub</a>&nbsp; &nbsp; &nbsp;<a class="more-link" href="https://www.youtube.com/watch?v=pOat5nNqFxA" target="_blank"><i class="fa fa-external-link"></i>View Live Demo</a></p>


<hr>


<img src="/assets/images/projects/magnet.png" class="center"/>
### Magnetic Field Simulator
_MATLAB_

Simulates and visualizes magnetic field interactions between two or more
solenoids. Parameters include the physical and magnetic properties of the
solenoids, including the length, width, thickness of coils, number of windings,
and strength of the current.

<p><a class="more-link" href="https://github.com/n2cholas/TOPS-AP-Physics-C/tree/master/Lab4-AC_Circuits" target="_blank"><i class="fa fa-external-link"></i>View on Github</a></p>


<hr>


<img src="/assets/images/projects/collision.png" class="center" width="50%"/>
### Collision Simulator
Written in C#, this program simulates a 2D elastic collision between two
identical stress balls. The compression of the balls is factored into the
simulation. A spreadsheet is outputted with the force, acceleration, velocity,
and positions of both balls with respect to a dynamic time interval.

<p><a class="more-link" href="https://github.com/n2cholas/TOPS-AP-Physics-C/tree/master/Lab1A-CollisionSimulation" target="_blank"><i class="fa fa-external-link"></i>View on Github</a></p>


<hr>


<img src="/assets/images/projects/uomi.png" class="center"/>
### UOMi
_Python, PyMongo, Flask, Heroku, HTML/CSS, JavaScript_

Created at Hack the 6ix, this web app keeps track of money you owe and money
owed to you.

<p><a class="more-link" href="https://github.com/n2cholas/chrome-tab-predictor" target="_blank"><i class="fa fa-external-link"></i>View on Github</a> &nbsp; &nbsp; &nbsp; <a class="more-link" href="https://devpost.com/software/uomi" target="_blank"><i class="fa fa-external-link"></i>View on DevPost</a> &nbsp; &nbsp; &nbsp; <a class="more-link" href="hackthe6ix-project.herokuapp.com" target="_blank"><i class="fa fa-external-link"></i>Try it out!</a></p>
