# Glasses Detection on Yolov5 using Jetson Nano 2gb

## Aim and Objective

### Aim

To create a Glasses detection system which will detect Human eyes and then check if Glasses is worn or not.

### Demo





https://github.com/prachipaunikar/Glasses-Detection-using-Jetson-Nano-2GB/assets/147481200/9ff22f19-db84-439a-9bf6-e303c33825ee

### link:- https://youtu.be/3JSmUSvM_1c




https://github.com/prachipaunikar/Glasses-Detection-using-Jetson-Nano-2GB/assets/147481200/7a67ad7c-62ba-4813-9092-71cd8a981d37

### link:- https://youtube.com/shorts/nJ6NLlKIIUw


### Objective

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.
• Using appropriate datasets for recognizing and interpreting data using machine learning.
• To show on the optical viewfinder of the camera module whether a person is wearing a Glasses or not.

## Abstract

• A person’s eyes is classified whether a Glasses is worn or not and is detected by the live feed from the system’s camera.
• We have completed this project on jetson nano which is a very small computational device.
• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.
• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.
• A Glasses is the most important and improves their ability to see clearly, which is crucial for daily activities such as reading, driving, or using digital devices.
• Glasses can protect the eyes from physical harm, such as debris, dust, or small particles that could potentially cause injury. Safety glasses, for example, are designed specifically for this purpose in environments where there is a risk of eye injury.

## Introduction

• This project is based on a Glasses detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.
• This project can also be used to gather information about who is wearing a Glasses and who is not.
• Glasses worn can further be classified into Sunglasses for protect the eyes from harmful UV rays and bright light. Computer Glasses designed to reduce eye strain and fatigue caused by prolonged exposure to digital screens. Safety Glasses are designed to protect the eyes from potential hazards in work or sports environments. Fashion Glasses are worn primarily for aesthetic purposes, enhancing personal style and complementing facial features based on the image annotation we give in roboflow. 
• Training in Roboflow has allowed us to crop images and change the contrast of certain images to match the time of day for better recognition by the model.
• Neural networks and machine learning have been used for these tasks and have obtained good results.
• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Glasses detection as well.

## Literature Review

• Wearing glasses helps to reduce the impact of several vision-related issues and challenges, thereby significantly improving daily life and overall well-being. Primarily, glasses correct refractive errors such as myopia (nearsightedness), hyperopia (farsightedness), and astigmatism. By addressing these common vision problems, glasses enable individuals to see clearly at various distances, whether reading up close, viewing screens, or engaging in activities requiring distance vision like driving.
• Driving Glasses with Anti-Reflective Coating glasses are designed to minimize reflections and glare, especially at night or in low-light conditions. They improve contrast and clarity, making it easier to see road signs and other vehicles. This type of Glasses protects your eyes from dust and high beam lights when driving your two-wheeler.
•It is observed that wearing glasses improves your vision, comfort, and overall quality of life significantly. Firstly, glasses correct refractive errors such as myopia, hyperopia, and astigmatism. By addressing these issues, glasses provide clear and sharp vision, allowing individuals to perform daily activities with greater ease and accuracy.
• Wearing glasses not only corrects vision impairments but also enhances various aspects of daily life and well-being. Beyond improving visual acuity, glasses contribute to comfort, safety, and overall quality of life in several important ways.
• The significance of glasses encompasses a multitude of benefits that collectively enhance everyday life and overall well-being. Primarily, glasses serve as vital tools for correcting vision impairments such as myopia (nearsightedness), hyperopia (farsightedness), and astigmatism. By providing precise vision correction, glasses enable individuals to perform daily tasks with clarity and accuracy, from reading books to navigating busy streets.
• Mandatory glasses laws, also known as vision standards for driving, are regulations implemented by governments to ensure road safety by requiring drivers with certain visual impairments to wear corrective lenses while operating a vehicle. These laws typically specify minimum visual acuity and field of vision requirements that drivers must meet to obtain or retain their driver's license.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.
• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.
• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.
• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Nano Jetson 2gb

![image](https://github.com/prachipaunikar/Glasses-Detection-using-Jetson-Nano-2GB/assets/147481200/7d2dd01c-329e-4a82-8c73-74429003afdd)


## Proposed System

1] Study basics of machine learning and image recognition.

2]Start with implementation

• Front-end development
• Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether a person is wearing a Glasses or not.

4] Use datasets to interpret the object and suggest whether the person on the camera’s viewfinder is wearing a Glasses or not.

## Methodology

The Glasses detection system is a program that focuses on implementing real time Glasses detection. It is a prototype of a new product that comprises of the main module: Glasses detection and then showing on viewfinder whether the person is wearing a Glasses or not. Glasses Detection Module

This Module is divided into two parts:

1] Eyes detection

• Ability to detect the location of a person’s eyes in any input image or frame. The output is the bounding box coordinates on the detected eyes of a person.
• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.
• This Datasets identifies person’s eyes in a Bitmap graphic object and returns the bounding box image with annotation of Glasses or no Glasses present in each image.

2] Glasses Detection

• Recognition of the eyes and whether Glasses is worn or not.
• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.
• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.
• YOLOv5 was used to train and test our model for whether the Glasses was worn or not. We trained it for 149 epochs and achieved an accuracy of approximately 92%. 

## Installation

Initial Configuration

sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
Create Swap

udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
/swapfile1	swap	swap	defaults	0 0
Cuda env in bashrc

vim ~/.bashrc

### add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
Update & Upgrade

sudo apt-get update
sudo apt-get upgrade
Install some required Packages

sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow
Install Torch

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
Install Torchvision

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
Clone Yolov5

git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
Download weights and Test Yolov5 Installation on USB webcam

sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0

## Glasses Dataset Training

We used Google Colab And Roboflow
train your model on colab and download the weights and past them into yolov5 folder link of project

colab file given in repo

## Running Glasses Detection Model

source '0' for webcam

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

## Advantages

• In fields like security and authentication systems, glasses detection helps minimize errors by accurately identifying individuals wearing glasses. This reduces false positives and improves the overall reliability of facial recognition technology.
• Glasses detection system shows whether the person in viewfinder of camera module is wearing a Glasses or not with good accuracy.
• Its ability to improve accuracy in facial recognition, enhance user experience in virtual try-ons, personalize retail recommendations, and contribute to better medical diagnostics.
• When completely automated no user input is required and therefore works with absolute efficiency and speed.
• It can work around the clock and therefore becomes more cost efficient.

## Application

• Detects a person’s eyes and then checks whether Glasses is worn or not in each image frame or viewfinder using a camera module.
• E-commerce platforms and retail stores use glasses detection to enable virtual try-on experiences. This technology allows customers to see how different styles of glasses look on their face without physically trying them on, enhancing the shopping experience.
• Can be used as a reference for other ai models based on Glasses Detection.

## Future Scope

• As we know technology is marching towards automation, so this project is one of the step towards automation.
• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.
• Glasses detection will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the situation in an efficient way.

## Conclusion

• In this project our model is trying to detect a person’s eyes and then showing it on viewfinder, live as to whether Glasses is worn or not as we have specified in Roboflow.
• The model tries to solve the problem of glasses detection to enhance virtual try-on experiences for eyewear used in E-commerce platforms and retail stores.
• The model is efficient and highly accurate and hence reduces the workforce required.

## Reference

1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.gettyimages.ae/search/2/image?phrase=glasses&sort=mostpopular&license=rf%2Crm

3] Google images

## Articles :-

1] https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses

2] https://www.kaggle.com/datasets/egorovlvan/glasses-dataset


