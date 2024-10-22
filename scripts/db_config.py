import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def connectToDB():
    try:
        print("Connecting to DB")
        mongo_uri = os.getenv("MONGO_URI")
        #Connect to client
        print("Connection string: ", mongo_uri)
        client = MongoClient(mongo_uri, tlsInsecure =True)
        db = client.get_database('pairit')
        collection = db.get_collection('classes')
        return collection
    except Exception as e:
        print("Error connecting to DB: ", e)
        return None
    

    