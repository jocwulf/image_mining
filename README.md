# image_mining
Python and R scripts for mining social media images

## Scraper
- scrape_images.py collects images and metadata from a social media site and puts data into mongodb

## Image Mining
- tf_image_recognition.py uses Resnet 101 image classification model provided by Google Open Images Dataset team to infer 5000 Labels and write them into mongodb
- tf_object_detection.py uses different object detection models from the tensorflow object detection model zoo and writes results into mongodb
- opencv.py uses opencv library for face and body detectoin
- visionapi.py connects to Google Vision API and writes results into mongodb

## Analytics
- pca_ols.R conducts principal component analysis and OLS regressions with object detection data
