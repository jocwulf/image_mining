'''
Created on 22.05.2018

@author: JWulf
'''

import numpy
import cv2
import os
import time
import pymongo
from pymongo import MongoClient
import requests
import json
from urllib.request import urlretrieve
import urllib.parse

##############specify parameters

#access token
api_key = ''

#group for which we want the image mining
group_url = ''

###########preparing models
frontalface_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml')
fullbody_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_fullbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_lowerbody.xml')
profileface_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_profileface.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')
upperbody_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_upperbody.xml')

#from here: https://github.com/Aravindlivewire/Opencv/blob/master/haarcascade/
aGest_cascade = cv2.CascadeClassifier('./haarcascades/aGest.xml')
fist_cascade = cv2.CascadeClassifier('./haarcascades/fist.xml')
palm_cascade = cv2.CascadeClassifier('./haarcascades/palm.xml')
closed_frontal_palm_cascade = cv2.CascadeClassifier('./haarcascades/closed_frontal_palm.xml')

#from here: https://github.com/Balaje/OpenCV/blob/master/haarcascades/
hand_cascade = cv2.CascadeClassifier('./haarcascades/hand.xml')

#hog = cv2.HOGDescriptor()
#hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

#####return group_id from link

def getgroupid(api_key,group_url):
    query1 = "https://api.flickr.com/services/rest/?&method=flickr.urls.lookupGroup&api_key="
    query2 = "&url="
    query3 = "&format=json&nojsoncallback=1"
    
    session = requests.Session()
    fullquery = query1+api_key+query2+group_url+query3
    response = session.get(fullquery)
    response_json=json.loads(response.text)
    return response_json['group']['id']

##############this is opencv classification
def opencvimage(imagepath):
    ############prepare image
    img = cv2.imread(imagepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ################generate the human classification information
    opencv=dict()
    
    frontalfaces = frontalface_cascade.detectMultiScale(gray, 1.3, 8)
    if len(frontalfaces)==0:
        opencv["frontalfaces"]=0
    else:
        opencv["frontalfaces"]=len(frontalfaces)
    
    profilefaces = profileface_cascade.detectMultiScale(gray, 1.3, 8)
    if len(profilefaces)==0:
        opencv["profilefaces"]=0
    else:
        opencv["profilefaces"]=len(profilefaces)
        
    eye = eye_cascade.detectMultiScale(gray, 1.1, 5)
    if len(eye)==0:
        opencv["eye"]=0
    else:
        opencv["eye"]=len(eye)
        
    smile = smile_cascade.detectMultiScale(gray, 1.7, 22)
    if len(smile)==0:
        opencv["smile"]=0
    else:
        opencv["smile"]=len(smile)

        
    fullbody = fullbody_cascade.detectMultiScale(gray, 1.1,5)
    if len(fullbody)==0:
        opencv["fullbody"]=0
    else:
        opencv["fullbody"]=len(fullbody)
    
    
    upperbody = upperbody_cascade.detectMultiScale(gray, 1.1, 5)
    if len(upperbody)==0:
        opencv["upperbody"]=0
    else:
        opencv["upperbody"]=len(upperbody)
        
    lowerbody = lowerbody_cascade.detectMultiScale(gray, 1.1, 5)
    if len(lowerbody)==0:
        opencv["lowerbody"]=0
    else:
        opencv["lowerbody"]=len(lowerbody)
        
        
    aGest = aGest_cascade.detectMultiScale(gray, 1.05, 5)
    if len(aGest)==0:
        opencv["aGest"]=0
    else:
        opencv["aGest"]=len(aGest)
        
    fist = fist_cascade.detectMultiScale(gray, 1.05, 5)
    if len(fist)==0:
        opencv["fist"]=0
    else:
        opencv["fist"]=len(fist)
        
    palm = palm_cascade.detectMultiScale(gray, 1.05, 5)
    if len(palm)==0:
        opencv["palm"]=0
    else:
        opencv["palm"]=len(palm)
        
    closed_frontal_palm = closed_frontal_palm_cascade.detectMultiScale(gray, 1.05, 5)
    if len(closed_frontal_palm)==0:
        opencv["closed_frontal_palm"]=0
    else:
        opencv["closed_frontal_palm"]=len(closed_frontal_palm)
        
    hand = hand_cascade.detectMultiScale(gray, 1.05, 5)
    if len(hand)==0:
        opencv["hand"]=0
    else:
        opencv["hand"]=len(hand)
        
    return opencv


###########################this is main function
#connect to db
client = MongoClient('localhost:27017')
db = client.flickr
            
starttime = time.time()
group_id = getgroupid(api_key,group_url)

finished=0
workcount=0
while finished==0:
    try:
        fotoidlist = list()
        
        #get all photoIDs
        for entry in db.flickr.find({"group_id":group_id,"saved_image_extension":{"$exists":1},"opencv":{"$exists":0}}):
            fotoid=entry['id']
            file= "/mnt/volume-fra1-01/flickr/picextraction/"+str(fotoid)+".jpg"
            try:
                entry["opencv"]= opencvimage(file)
            except Exception as e:
                try:
                    entry["opencv"]= opencvimage(file)
                except Exception as e:
                    entry["opencv_error"]=str(e)
            db.flickr.update_one({'_id': entry['_id']}, {"$set":entry}, upsert=False)
            workcount+=1
            if workcount%1000==0:
                print(str(workcount)+" images done with opencv processing time per image: "+str(workcount/(float(time.time()-starttime))))
        finished=1

    except Exception as e:
        print("ERROR, resetting mongodb connection: "+str(e))
        continue

print("finished at time: "+str(time.time()-starttime))
           



