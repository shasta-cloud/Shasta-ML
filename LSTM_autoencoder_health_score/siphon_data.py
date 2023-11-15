#!/usr/bin/python3

import sys
import os
import time
import sched
import http.client
import json
import csv
from csv import DictWriter
#from datetime import datetime
import pandas as pd
import numpy as np
import subprocess
#import tensorflow_serving
import grpc
import tensorflow as tf
#from tensorflow_serving.apis import predict_pb2
#from tensorflow_serving.apis import prediction_service_pb2_grpc
#from sklearn.model_selection import train_test_split # to split the data into two parts
#from sklearn.preprocessing import StandardScaler # for normalization
#from sklearn import preprocessing
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def getHeaders():
    if os.path.exists("PRIV-headers.json"):
        print("Authentication: Reading token from cached file")
        f = open("PRIV-headers.json")
        headers = json.load(f)
        f.close()
    elif os.path.exists("PRIV-creds.json"):
        print("Authentication: Logging in to retrieve token")
        f = open("PRIV-creds.json")
        creds = json.load(f)
        f.close()

        conn = http.client.HTTPSConnection("sec-dev.shastacloud.com", 16001)
        headers = {
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/api/v1/oauth2", json.dumps(creds), headers)
        res = conn.getresponse()
        data = json.loads(res.read())

        headers = {
                'Authorization': "Bearer " + data["access_token"]
        }
        with open("PRIV-headers.json", "w") as outfile:
            json.dump(headers, outfile)
        return headers
    else:
        print("Error: PRIV-creds.json and PRIV-headers.json not found")
        exit(1)

    return headers


def getInventory(dev_serialno):
    global inventory

    for i in inventory:
        if i['serialNumber'] == dev_serialno:
            return i
    
    return None

def getVenue(venue_uuid):
    global venues

    for v in venues:
        if v['id'] == venue_uuid:
            return v
    
    return None

def getEntity(entity_uuid):
    global entities

    for e in entities:
        if e['id'] == entity_uuid:
            return e
    
    return None

def loadStats(dev_serialno):
    global headers
    #global from_cache

    # Get Latest Single Device Stats

    conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
    payload = ''
    conn.request("GET", "/api/v1/device/" + dev_serialno + "/statistics?lastOnly=true", payload, headers)
    res = conn.getresponse()
    stats = json.loads(res.read());
    with open("last-device-stats.json", "w") as outfile:
        json.dump(stats, outfile, indent=2)
    return stats

def get_device_health(dev_serialno):
    global headers
    #global from_cache

    # Get Latest Single Device healthcheck

    conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
    payload = ''
    conn.request("GET", "/api/v1/device/"+ dev_serialno +"/healthchecks?lastOnly=true", payload, headers)
    res = conn.getresponse()
    healthcheck = json.loads(res.read());
    with open("last-device-health.json", "w") as outfile:
        json.dump(healthcheck, outfile, indent=2)
    return healthcheck

def check_response(dev_serialno):
    global headers
    
    conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
    if not conn: 
        return False
    payload = ''
    try:
        conn.request("GET", "/api/v1/device/" + dev_serialno + "/statistics?lastOnly=true", payload, headers)
        res = conn.getresponse()
        if res.status == 200:
            return True
        else:
            return False
    except:
        return False
    
def check_internet_connection():
   try:
      subprocess.check_output(["ping", "-c", "1", "8.8.8.8"])
      return True
   except subprocess.CalledProcessError:
      return False


def predict_from_serving_server(data):
    conn = http.client.HTTPConnection("localhost", 8605)
    #if not conn: 
    #   return False
    #print('no connection')
    payload = "{\r\n     \"instances\": "+str(data)+" \r\n }"
    #payload = {"instances": data}
    print(payload)
    #try:
    headers = {'Content-Type': 'text/plain'}
    conn.request("POST", "/v1/models/saved_model/versions/3:predict", payload, headers)
    res = conn.getresponse()
    prediction = json.loads(res.read())['predictions']

    conn.close()
    if res.status == 200:
        print('ok')
        return [float(i) for i in prediction[0][0]]
    else:
        print('not ok')
        return False
    #except:
    #    print('exception')
    #    return False


def evaluate_current_state(device):
    if not check_response(device):
        print('no responce')
        if check_internet_connection():
            print("Internet is connected. Renewing token.")
            getHeaders()
        else:
            print("Internet is not connected.")
        return False
    else:
        healthcheck = get_device_health(device)
        #predict in tensorflow serving server

        # Set up gRPC channel to TensorFlow Serving server
        channel = grpc.insecure_channel('localhost:8601')
        return True
    
dataset = pd.DataFrame()
dataset.tail()
headers = getHeaders()
fifo_counter = 0
print("Authentication: Using token '" + headers["Authorization"] + "'")

selected_devices = ['903cb3bb245d'] #, '903cb3bb2755'

def run_code(selected_devices, dataset, dataset_size=1000):
    global fifo_counter 
    for device in selected_devices:    
        if not check_response(device):
            print('no responce')
            if check_internet_connection():
                print("Internet is connected. Renewing token.")
                getHeaders()
            else:
                print("Internet is not connected. Trying again in 60 seconds.")
            return scheduler.enter(60, 1, run_code,(selected_devices,dataset)) 
        
        device_stats = loadStats(device)
        device_health = get_device_health(device)
        inter = device_stats['interfaces']
        data = {'mac': device,
                'temperature1': device_stats['unit']['temperature'][0],
                'temperature2': device_stats['unit']['temperature'][1],
                'uptime': device_stats['unit']['uptime'],
                'cpu1': device_stats['unit']['cpu_load'][0],
                'cpu2': device_stats['unit']['cpu_load'][1],
                'cpu3': device_stats['unit']['cpu_load'][2],
                'cpu4': device_stats['unit']['cpu_load'][3],
                'cpu5': device_stats['unit']['cpu_load'][4],
                'cpu_load_1m': device_stats['unit']['load'][0],
                'cpu_load_5m': device_stats['unit']['load'][1],
                'cpu_load_15m': device_stats['unit']['load'][2],
                'mem_buffered': device_stats['unit']['memory']['buffered'],
                'mem_cached': device_stats['unit']['memory']['cached'],
                'mem_free': device_stats['unit']['memory']['free'],
                'mem_total': device_stats['unit']['memory']['total'],
                'num_ifaces': len(inter), 
                'sanity': device_health['sanity'],
                'health_mem': device_health['values']['unit']['memory'],
                }

        dataset = dataset._append(data, ignore_index=True)
        dataset2 = dataset
        dataset2 = dataset2.drop(['mac'], axis=1)
        dataset2 = dataset2.drop(['uptime'], axis=1)
        dataset2 = dataset2.drop(['mem_buffered'], axis=1)
        dataset2 = dataset2.drop(['mem_cached'], axis=1)
        dataset2 = dataset2.drop(['mem_total'], axis=1)
        dataset2 = dataset2.drop(['sanity'], axis=1)
        dataset2 = dataset2.drop(['num_ifaces'], axis=1)
        dataset2['mem_free'] = dataset2['mem_free'].astype(float)/device_stats['unit']['memory']['total']
        df = pd.DataFrame(dataset2)
        print(df.shape[1], df.shape[0])
        df = np.reshape(df, (int(len(df)), 1, 12))
        prediction = predict_from_serving_server([df[-1].tolist()])
        print(prediction)
        mse = np.power((df[-1] - prediction), 2)
        mean_mse = np.mean(mse)
        print(mse)
        print(mean_mse) 
        

        # Create a new figure with a gauge subplot
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'indicator'}]])

        # Add a gauge trace to the subplot
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = (1-mean_mse/100000)*100,
            title = {'text': "MSE"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': 'darkred'},
                    {'range': [20, 40], 'color': 'red'},
                    {'range': [40, 60], 'color': 'orange'},
                    {'range': [60, 80], 'color': 'yellow'},
                    {'range': [90, 100], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))

        # Update the layout of the figure
        fig.update_layout(height=250, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0})

        # Display the figure
        fig.show()
          
        column_names = ['mac', 'temperature1', 'temperature2', 'uptime', 'cpu1', 'cpu2', 'cpu3', 'cpu4', 'cpu5', 'cpu_load_1m', 'cpu_load_5m', 'cpu_load_15m', 'mem_buffered', 'mem_cached', 'mem_free', 'mem_total', 'num_ifaces', 'sanity', 'health_mem']
        with open('temp_dataset_'+ str(device)+'.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=column_names)
            dictwriter_object.writerow(data)
            f_object.close()

        if len(dataset) < dataset_size:
            fifo_counter+=1
            pass
        else:
            dataset.drop(0, inplace=True)
            print(fifo_counter, dataset_size)
            if fifo_counter ==  dataset_size-1:
                dataset.to_csv('dataset.csv', index=False)
                fifo_counter = 0
            else:
                fifo_counter+=1
        print(dataset)
    # Schedule the next run
    scheduler.enter(60, 1, run_code,(selected_devices,dataset)) 



# Create a scheduler object
scheduler = sched.scheduler(time.time, time.sleep)

# Schedule the first run
scheduler.enter(0, 1, run_code,(selected_devices,dataset))

# Start the scheduler
scheduler.run()
'''
selected_devices = ['903cb3bb245d', '903cb3bb2755']
for device in selected_devices:
    device_stats = loadStats(device)
    print(device_stats)'''