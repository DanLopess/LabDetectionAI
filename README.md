# SmartLabDetection

AI System for detecting how many people are inside a lab that contains a set of sensors.

## Intro

During the COVID-19 pandemic, a restriction on the maximum amount of people that
could be simultaneously inside a room, was imposed. This capacity
depended on several factors, including the room dimension, ventilation, etc. The need
to automatically detect the number of persons inside a lab without affecting privacy, led
to the implementation of an experimental lab based on low-cost, non-intrusive sensors.

## Lab details and sensors

The lab consists of a 13m2 room where a **Zigbee** based wireless sensor network was
installed. The lab has three workstations (a chair, and a desk with a dock station and a
table lamp). There is a small window above workstation 3 and there is no
heating/ventilation/AC system active in the room.

The wireless network is a Zigbee-based star network with six slave nodes feeding data
to the master node. There is one CO2 sensor (MH-Z14A) in the center of the room, two
digital infrared motion sensors (PIR) in opposed walls, and, in each workstation, a node
containing a light sensor (BH1750) and a temperature sensor (LMT84LP) has been
installed.

PIR sensor data indicates if movement was detected during the last 30s. For the
remaining sensor nodes, the Arduino Uno microcontroller board sampled data from the
sensors and transmitted it periodically via a Zigbee module every 30s.
Sensor measurements were taken over a period of several days. Each student manually
annotated when entered and left the room during this period. Therefore, true
occupancy was annotated during the measurement period.
The resulting dataset has now been made available (Lab6Dataset.csv)

## Objective

During the worse times of the pandemic, Técnico imposed a limit of 2 persons inside the
above lab. However, the students that use the lab had frequent deadlines, and often
ignored the 2-person limit.

The objectives of this project are to develop a **NN-based classifier** that, using the dataset,
is able to:

- Detect when there are more than 2 persons inside the lab;
- Detect how many people are inside the lab.

The third objective is to detect if there are more than 3 people inside the lab using **Fuzzy Inference Systems** and compare the results and performance.

## Environment Setup

To be able to run this project you need to create an environment with all of the required packages.

For ease of use, we recommend creating a conda environment with our provided environment file. You simply need to execute the following command:

`conda env create -f environment.yml`

Then you can open the \*.ipynb file and run the notebook cells to perform the steps for prepping the data and hyperparameter tuning of both models.

The result is a pair of classifiers that are able to answer the two objectives mentioned earlier.

## Making new predictions

As a result of all of the process described in the **Jupyter Notebook** two predictors, or classifiers were trained using **neural networks** and hyperparameter tuning for obtaining a model that best generalizes the data, as well as a **Fuzzy Inference System** with a fixed fuzzy rule set.

To run both for a new dataset please execute the following command:

`python TestMe.py <Path-to-dataset> <path-to-results>`

The script will load the correspondent models, run the predictions print and plot the results.

## Results

The results for both the Neural Network and Fuzzy inference system can be found at the `results_report.pdf` file of each directory.

## Suggestions

All suggestions are appreciated so that we can improve and optimize this model.
