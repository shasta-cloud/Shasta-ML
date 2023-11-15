#!/usr/bin/python3

import sys
import os
import time
import sched
import http.client
import json
import csv
from csv import DictWriter
from datetime import datetime
import pandas as pd
import subprocess

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


'''    
if (len(sys.argv) > 1 and sys.argv[1] == "-c"):
    from_cache = 1
    print("NOTE: Loading data from cache (last files)")
else:
    from_cache = 0

headers = getHeaders()
print("Authentication: Using token '" + headers["Authorization"] + "'")


# Get device connection statistics
if (from_cache):
    with open("last-device-conn-stats.json") as infile:
        cstats = json.load(infile)
else:
    print("-> Loading device connection statistics")
    conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
    payload = ''
    conn.request("GET", "/api/v1/devices?connectionStatistics=true", payload, headers)
    res = conn.getresponse()
    cstats = json.loads(res.read())
    with open("last-device-conn-stats.json", "w") as outfile:
        json.dump(cstats, outfile, indent=2)

# Get all devices with status
if (from_cache):
    with open("last-devices.json") as infile:
        devices = json.load(infile)
    dev_count = len(devices)
else:
    print("-> Loading device count")
    conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
    payload = ''
    conn.request("GET", "/api/v1/devices?countOnly=true", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read())
    dev_count = data['count']

    print("-> Loading all devices with status info, 50 at a time (" + str(dev_count) + " total)")

    left = dev_count
    offset = 0
    devices = []
    while(left > 0):
        if (left >= 50):
            end_cnt = offset + 50
        else:
            end_cnt = offset + left
        print("   -> Loading " + str((offset+1)) + " to " + str(end_cnt))
        conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
        payload = ''
        conn.request("GET", "/api/v1/devices?deviceWithStatus=true&limit=50&offset="+str(offset), payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read())
        page = data['devicesWithStatus']
        left -= len(page)
        offset += len(page)
        for d in page:
            devices.append(d)
        time.sleep(0.2)
    with open("last-devices.json", "w") as outfile:
        json.dump(devices, outfile, indent=2)

# Get Inventory
if (from_cache):
    with open("last-inv-devices.json") as infile:
        inventory = json.load(infile)
else:
    print("-> Loading all inventory with extended info")
    conn = http.client.HTTPSConnection("prov-dev.shastacloud.com", 16005)
    payload = ''
    conn.request("GET", "/api/v1/inventory?withExtendedInfo=true", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read());
    inventory = data['taglist']
    with open("last-inv-devices.json", "w") as outfile:
        json.dump(inventory, outfile, indent=2)

# Get Venues
print("-> Loading all venues")
if (from_cache):
    with open("last-inv-venues.json") as infile:
        venues = json.load(infile)
else:
    conn = http.client.HTTPSConnection("prov-dev.shastacloud.com", 16005)
    payload = ''
    conn.request("GET", "/api/v1/venue", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read());
    venues = data['venues']
    with open("last-inv-venues.json", "w") as outfile:
        json.dump(venues, outfile, indent=2)

# Get Entities
print("-> Loading all entities")
if (from_cache):
    with open("last-inv-entities.json") as infile:
        entities = json.load(infile)
else:
    conn = http.client.HTTPSConnection("prov-dev.shastacloud.com", 16005)
    payload = ''
    conn.request("GET", "/api/v1/entity", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read());
    entities = data['entities']
    with open("last-inv-entities.json", "w") as outfile:
        json.dump(entities, outfile, indent=2)
print(cstats)
print("\nThere are " + str(cstats['connectedDevices']) + " connected devices out of "
                    + str(dev_count) + " total (" + str(cstats['averageConnectionTime']) + " avg conn time)")
print("-> Processing online devices")
online_devices = [];
cnt_total = 0
cnt_online = 0

selected_devices = ['903cb3bb245d', '903cb3bb2755']
for x in devices:
    
    #print(str(x['serialNumber']))

    if x['connected'] and x['serialNumber'] in selected_devices:
        print(x)
    cnt_total = cnt_total + 1
    if x['connected']:
        cnt_online = cnt_online + 1
        nd = { "mac": x['serialNumber'] }
        inv = getInventory(x['serialNumber'])
        if (inv == None):
            nd['name'] = "unknown"
            if (len(x['manufacturer']) > 0):
                nd['model'] = x['compatible']
            else:
                nd['model'] = "unknown"
        else:
            nd['name'] = inv['name']
            nd['model'] = inv['deviceType']
        v = getVenue(x['venue']);
        if (v == None):
            nd['venue'] = "unknown"
            nd['org'] = "unknown"
        else:
            nd['venue'] = v['name']
            e = getEntity(v['entity']);
            if (e == None):
                nd['org'] = "unknown"
            else:
                nd['org'] = e['name']
        nd['num_assocs'] = x['associations_2G'] + x['associations_5G'] + x['associations_6G']
        idx = x['firmware'].find("Shasta")
        if (idx < 0):
            if (len(x['firmware']) == 0):
                nd['firmware'] = "unknown"
            else:
                nd['firmware'] = x['firmware']
        else:
            nd['firmware'] = x['firmware'][idx:]

        nd['conf_2g'] = "unknown"
        nd['conf_5g'] = "unknown"
        nd['conf_6g'] = "unknown"
        if 'configuration' in x and 'radios' in x['configuration']:
            for r in x['configuration']['radios']:
                if r['band'] == '2G':
                    nd['conf_2g'] = str(r['channel'])
                elif r['band'] == '5G':
                    nd['conf_5g'] = str(r['channel'])
                elif r['band'] == '6G':
                    nd['conf_6g'] = str(r['channel'])

        stats = loadStats(nd['mac'])
        if (from_cache == 0):
            time.sleep(0.2) # so we don't slam the API

        if 'unit' in stats:
            nd['uptime'] = stats['unit']['uptime']
            nd['up_days'] = round(nd['uptime'] / 86400, 2)
            nd['cpu_busy_pct'] = stats['unit']['cpu_load'][0]
            nd['cpu_load_1m'] = stats['unit']['load'][0]
            nd['cpu_load_5m'] = stats['unit']['load'][1]
            nd['cpu_load_15m'] = stats['unit']['load'][2]

            mem_free = stats['unit']['memory']['free']
            mem_total = stats['unit']['memory']['total']
            mem_used = mem_total - mem_free
            nd['mem_used_pct'] = round((mem_used * 100 / mem_total), 2)
            nd['mem_free_pct'] = round((mem_free * 100 / mem_total), 2)
        else:
            nd['uptime'] = -1
            nd['up_days'] = -1
            nd['cpu_busy_pct'] = -1
            nd['cpu_load_1m'] = -1
            nd['cpu_load_5m'] = -1
            nd['cpu_load_15m'] = -1
            nd['mem_used_pct'] = -1
            nd['mem_free_pct'] = -1
    
        nd['num_ssids'] = 0
        if 'interfaces' in stats:
            nd['num_ifaces'] = len(stats['interfaces'])
            for x in stats['interfaces']:
                if 'ssids' in x:
                    nd['num_ssids'] = nd['num_ssids'] + len(x['ssids'])
        else:
            nd['num_ifaces'] = 0

        for x in ['2g', '5g', '6g']:
            nd["chan_" + x] = 0
            nd["width_" + x] = 0

        if 'radios' in stats:
            for r in stats['radios']:
                if r['band'][0] == "2G":
                    x = '2g'
                elif r['band'][0] == "5G":
                    x = '5g'
                elif r['band'][0] == "6G":
                    x = '6g'
                else:
                    continue
                nd["chan_" + x] = r['channel']
                nd["width_" + x] = r['channel_width']
    
        
        print(nd['mac']
                + ", " + nd['name']
                + ", " + nd['org']
                + ", " + nd['venue']
                + ", " + nd['model']
                + ", " + nd['firmware']
                + ", " + str(nd['uptime'])
                + ", " + str(nd['up_days'])
                + ", " + str(nd['cpu_busy_pct'])
                + ", " + str(nd['cpu_load_1m'])
                + ", " + str(nd['cpu_load_5m'])
                + ", " + str(nd['cpu_load_15m'])
                + ", " + str(nd['mem_used_pct'])
                + ", " + str(nd['mem_free_pct'])
                + ", " + str(nd['num_ifaces'])
                + ", " + str(nd['num_ssids'])
                + ", " + str(nd['num_assocs'])
                + ", " + str(nd['chan_2g'])
                + ", " + str(nd['width_2g'])
                + ", " + str(nd['chan_5g'])
                + ", " + str(nd['width_5g'])
                + ", " + str(nd['chan_6g'])
                + ", " + str(nd['width_6g'])
            )
        
        online_devices.append(nd)

now = datetime.now()
prefix = now.strftime("%Y%m%d-%H%M%S")

print("\n[" + prefix + "] Processed " + str(cnt_online) + " online devices out of " + str(cnt_total) + " total")

# Write out JSON
fn = prefix + "-online-devices.json"
print("   -> Writing JSON of online devices to " + fn)
with open(fn, "w") as outfile:
    json.dump(online_devices, outfile, indent=2)

# Write out CSV
fn = prefix + "-online-devices.csv"
print("   -> Writing CSV of online devices to " + fn)
csv_fields = ['mac', 'name', 'org', 'venue', 'model', 'firmware', 'uptime', 'up_days',
              'cpu_busy_pct', 'cpu_load_1m', 'cpu_load_5m', 'cpu_load_15m',
              'mem_used_pct', 'mem_free_pct', 'num_ifaces', 'num_ssids', 'num_assocs',
              'chan_2g', 'width_2g', 'chan_5g', 'width_5g', 'chan_6g', 'width_6g',
              'conf_2g', 'conf_5g', 'conf_6g']
with open(fn, "w") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=csv_fields)
    writer.writeheader()
    for od in online_devices:
        writer.writerow(od)

'''
dataset = pd.DataFrame()
dataset.tail()
headers = getHeaders()
fifo_counter = 0
print("Authentication: Using token '" + headers["Authorization"] + "'")

selected_devices = ['903cb3bb2775'] 

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
                dataset.to_csv('dataset'+ str(device)+ '.csv', index=False)
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