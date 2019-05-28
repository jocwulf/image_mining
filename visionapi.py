import requests
from pymongo import MongoClient
import os
import json
import sys
import base64

######VARIABLES
GVISION_KEY = ''


####this is the function to retrieve google information for images in database
try:
    
    #Mongo connection
    client = MongoClient('localhost:27017')
    db = client.flickr
    
    ###first get all ids we want to analyze
    photoid_dicts = list()
    for entry in db.flickrCar.find({"gvision":{"$exists":0},'saved_image_extension':{'$exists':1}},{"id":1,"saved_image_extension":1}):    
        photoid_dicts.append(entry)
    totalusers = len(photoid_dicts)
    usercount = 0
    for photoid_dict in photoid_dicts:
        thephotoid = photoid_dict['id']
        
        if usercount%250==0:
            print(str(usercount)+" images processed of total: "+str(totalusers))
        if usercount==100000:
            print("BREAK after 100.000")
            break
        usercount = usercount+1
        imagename = "/mnt/volume_fra1_01/flickr/picextraction/"+photoid_dict['id']+"."+photoid_dict['saved_image_extension']
        
        request_list = []
    
        with open(imagename, 'rb') as image_file:
            content_json_obj = {
                'content': base64.b64encode(image_file.read()).decode('UTF-8')
            }
    
        feature_json_obj = []
        feature_json_obj.append({
            'type': 'OBJECT_LOCALIZATION',
            'maxResults': 1000,
        })
    
        feature_json_obj.append({
            'type': 'LABEL_DETECTION',
            'maxResults': 1000,
        })
    
        request_list.append({
            'features': feature_json_obj,
            'image': content_json_obj,
        })
    
        with open('vision.json', 'w') as output_file:
            json.dump({'requests': request_list}, output_file)
        
        data = open('vision.json', 'rb').read()
        GVISION_URL = 'https://vision.googleapis.com/v1/images:annotate?key='+ GVISION_KEY
        response = requests.post(url=GVISION_URL, data=data, headers={'Content-Type': 'application/json'})
        
        ###check if we have an error
        if 'error' in response.json():
            print("error in flickr id: "+photoid_dict['id']+"this is error message"+str(response))
        
        #print(str(response))
        #write the response to the database
        db.flickr.update_one({'_id':photoid_dict['_id']}, {"$set": {"gvision":response.json()}}, upsert=False)
except Exception as e:
    print("INTERRUPT WITH ERROR: "+str(e)+" at PHOTOID"+str(thephotoid))