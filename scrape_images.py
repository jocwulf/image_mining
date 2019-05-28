############moved picture folder to:
#"/mnt/volume-fra1-01/flickr/picextraction/"
#from:
#"/mnt/volume-fra1-01/flickr/picextraction/"

#############define imports
import requests
import json
import pymongo
from pymongo import MongoClient
from urllib.request import urlretrieve
import urllib.parse
import os
import base64


###############specify variables that go into main
api_key = ''
flickr_groups = ['']

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

########define function that iterates through all photos and saves them into mongodb

def scrapegroup(api_key, group_url):
    
    #get group_id
    group_id = getgroupid(api_key,group_url)
    
    #open database connection
    #this is for local purposes
    client = MongoClient('localhost:27017')
    db = client.flickr
    
    #check if we have the group already
    allgroups = db.flickr.distinct("group_id")
    if group_id in allgroups:
        print("This group is already scraped: "+str(group_url))
        return      
    
    # formulate query
    query1 = "https://api.flickr.com/services/rest/?&method=flickr.groups.pools.getPhotos&api_key="
    query2 = "&group_id="
    query3 = "&page="
    query4 = "&format=json&nojsoncallback=1&per_page=500&extras=count_comments,count_faves,date_taken,date_upload,geo,icon_server,last_update,license,machine_tags,media,o_dims,original_format,owner_name,path_alias,tags,url_c,url_m,url_n,url_o,url_q,url_s,url_sq,url_t,views"
    
    
    session = requests.Session()
    page=0
    pages=1

    while page<pages:
        #########workaround

        print("retrieving page: "+str(page+1)+" off total pages: "+str(pages))
        fullquery = query1+api_key+query2+group_id+query3+str(page+1)+query4
        try:
            response = session.get(fullquery)
            response_json=json.loads(response.text)
        except Exception as e:
            try:
                response = session.get(fullquery)
                response_json=json.loads(response.text)
            except Exception as e:
                print("ERROR: skipping page"+int(page)+" due to error: "+str(e))
                page = page+1
                continue
        stat=response_json['stat']
        if stat!="ok":
            print("error in query, we try again")
            response_json=json.loads(response.text)
            stat=response_json['stat']
            if stat!="ok":
                if (page+1)<pages and pages >1:
                    print("error is durable, we move to next page")
                    page = page+1
                    continue
                else:
                    print("error is durable, we abort scraping")
                    break
                
        page = int(response_json['photos']['page'])
        pages = int(response_json['photos']['pages'])
        for photo in response_json['photos']['photo']:
            photo['group_id']=group_id
            photo['group_url']=group_url
            db.flickr.insert(photo)
    
    return

############define function that iterates through all photos of the specific group and saves images into database

def downloadimages(group_id):
    
    print("start downloading images")
    #open database connection
    #this is for local purposes
    client = MongoClient('localhost:27017')
    db = client.flickr


    isdone=0
    while isdone==0:
        try:
            for entry in db.flickr.find({"group_id":group_id,"media":"photo","url_m":{"$exists":1},"saved_image_extension":{"$exists":0}}):#"group_id":group_id
                image_id = entry['id']
                image_url = entry['url_m']
                image_extension = image_url.split(".")[-1]
                imagename = "/mnt/volume-fra1-01/flickr/picextraction/"+image_id+"."+image_extension
                try:
                    urlretrieve(image_url, imagename)
                    entry['saved_image_extension']=image_extension
                except Exception as e:
                    try:
                        urlretrieve(image_url, imagename)
                        entry['saved_image_extension']=image_extension
                    except Exception as e:
                        print("ERROR: skipping "+imagename+" due to error: "+str(e))
                        continue
                
                db.flickr.update_one({'_id': entry['_id']}, {"$set":entry}, upsert=False)
            isdone=1
    
        except Exception as e:
                print("ERROR: restart mongo cursor after following error with mongodb :"+str(e))
                continue
    
        
    
    print("finished downloading images")
            


###########add context to each photo
def scrapecontext(api_key, group_id):
    print("start scraping context")

    #open database connection
    #this is for local purposes
    client = MongoClient('localhost:27017')
    db = client.flickr

    
    # prepare query
    query1 = "https://api.flickr.com/services/rest/?&method=flickr.photos.getContext&api_key="
    query2 = "&photo_id="
    query3 = "&format=json&nojsoncallback=1&extras=count_comments,count_faves,date_taken,date_upload,geo,icon_server,last_update,license,machine_tags,media,o_dims,original_format,owner_name,path_alias,tags,url_c,url_m,url_n,url_o,url_q,url_s,url_sq,url_t,views"
    
    #go through all group_id photos with no precontext defined
    isdone=0
    while isdone==0:
        try:
            for entry in db.flickr.find({"group_id":group_id, "precontext":{"$exists":0}},{"id":1}): #keep workload low by just exchanging required information and inserting new field afterwards

                entryid = entry["_id"]
                photoid = entry["id"]
                
                ###########now fetch the context
                session = requests.Session()
                fullquery = query1+api_key+query2+photoid+query3

                try:
                    response = session.get(fullquery)
                    response_json=json.loads(response.text)
                except Exception as e:
                    try:
                        response = session.get(fullquery)
                        response_json=json.loads(response.text)
                    except Exception as e:
                        print("ERROR retrieving context of photoid: "+str(photoid)+str(e))
                        continue
                stat=response_json['stat']
                if stat!="ok":
                    response_json=json.loads(response.text)
                    stat=response_json['stat']
                    if stat!="ok":
                        print("ERROR retrieving context of photoid: "+str(photoid))
                        continue
                db.flickr.update_one({'_id': entryid}, {"$set": {"precontext":response_json}}, upsert=False)
            isdone=1
    
        except Exception as e:
                print("ERROR: :"+str(e))
                continue

    print("finished scraping context")
    return


##########define main function that orchestrates process

if __name__ == '__main__':

    for group_url in flickr_groups:
        print("start processing "+str(group_url))
        group_id = getgroupid(api_key,group_url)
        print(str(group_url)+"+"+str(group_id))
        
        ##scrape image information
        scrapegroup(api_key, group_url)
    
        ##write photo data into database
        
        downloadimages(group_id)
        
        
        scrapecontext(api_key, group_id)
        print("finished processing "+str(group_id))
    print("all done")




