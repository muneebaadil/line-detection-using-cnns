import os

def CheckAndCreate(path): 
    if not os.path.exists(path): 
        os.makedirs(path)