#!/usr/bin/python3

import sys
import os
import time
import http.client
import json
import csv
from datetime import datetime

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
    global from_cache

    # Get Latest Single Device Stats
    print("   -> Loading latest stats for " + dev_serialno)
    if (from_cache):
        with open("device-stats.json") as infile:
            stats = json.load(infile)
    else:
        conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
        payload = ''
        conn.request("GET", "/api/v1/device/" + dev_serialno + "/statistics?lastOnly=true", payload, headers)
        res = conn.getresponse()
        stats = json.loads(res.read());
        with open("device-stats.json", "w") as outfile:
            json.dump(stats, outfile, indent=2)
    return stats
    
if (len(sys.argv) > 1 and sys.argv[1] == "-c"):
    from_cache = 1
    print("NOTE: Loading data from cache (last files)")
else:
    from_cache = 0

headers = getHeaders()
print("Authentication: Using token '" + headers["Authorization"] + "'")

def loadCapabilities(device):
    global headers
    global from_cache

    # Get Latest Single Device Stats
    print("   -> Loading latest capabilities for " + device)
    if (from_cache):
        with open("device-capabilities.json") as infile:
            cap = json.load(infile)
    else:
        conn = http.client.HTTPSConnection("gw-dev.shastacloud.com", 16002)
        payload = ''
        conn.request("GET", "/api/v1/device/" + device + "/capabilities", payload, headers)
        res = conn.getresponse()
        cap = json.loads(res.read());
        with open("device-capabilities.json", "w") as outfile:
            json.dump(cap, outfile, indent=2)
    return cap


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


def get_all_devices(from_cache):

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
    return devices, dev_count, cstats, inventory, venues, entities
devices, dev_count, cstats, inventory, venues, entities = get_all_devices(from_cache)
#print("\nThere are " + str(cstats['connectedDevices']) + " connected devices out of "
#                    + str(dev_count) + " total (" + str(cstats['averageConnectionTime']) + " avg conn time)")
print("-> Processing online devices")
online_devices = [];
cnt_total = 0
cnt_online = 0
dataframe = []
for device in devices:
    # Check that the device is connected and it is an AP
    if (device['connected'] == False):
        continue
    capabilities = loadCapabilities(device['serialNumber'])
    if (capabilities['capabilities']['platform'] != 'ap'):
        continue
    device_stats = loadStats(device['serialNumber'])
    #not looging if there are no radios #TODO: create an exeption in score
    if 'radios' not in device_stats:
        continue    
    device_health = get_device_health(device['serialNumber'])
    #inter = device_stats['interfaces']  
    # Initialize the device dataframe
    dataframe = { "mac": device['serialNumber'] }
    dataframe['model'] = capabilities['capabilities']['compatible']
    #venue = getVenue(device['venue'])
    #if (venue != None):
    #    dataframe['venue'] = venue['name']
    #    entity = getEntity(venue['entity'])
    #    dataframe['org'] = entity['name']
    #print(device_stats)
    dataframe['firmware'] = device['firmware']
    dataframe['total_num_assocs'] = device['associations_2G'] + device['associations_5G'] + device['associations_6G']
    dataframe['associations_2G'] = device['associations_2G']
    dataframe['associations_5G'] = device['associations_5G']
    dataframe['associations_6G'] = device['associations_6G']

    dataframe['rxBytes'] = device['rxBytes']
    dataframe['txBytes'] = device['txBytes']

    #unit data
    dataframe['temperature1'] = device_stats['unit']['temperature'][0]
    dataframe['temperature2'] = device_stats['unit']['temperature'][1]
    dataframe['uptime'] = device_stats['unit']['uptime']
    for cpu in device_stats['unit']['cpu_load']:
        dataframe['cpu'+str(device_stats['unit']['cpu_load'].index(cpu))] = cpu
    dataframe['cpu_load_1m'] = device_stats['unit']['load'][0]
    dataframe['cpu_load_5m'] = device_stats['unit']['load'][1]
    dataframe['cpu_load_15m'] = device_stats['unit']['load'][2]
    dataframe['mem_buffered'] = device_stats['unit']['memory']['buffered']
    dataframe['mem_cached'] = device_stats['unit']['memory']['cached']
    dataframe['mem_free'] = device_stats['unit']['memory']['free']
    dataframe['mem_total'] = device_stats['unit']['memory']['total']
    dataframe['uptime'] = device_stats['unit']['uptime']
    dataframe['boottime'] = device_stats['unit']['boottime']
    dataframe['localtime'] = device_stats['unit']['localtime']
    dataframe['timesinceboot'] = device_stats['unit']['localtime'] - device_stats['unit']['localtime']
    
    #radio data
    for radio in device_stats['radios']:
        if (radio['band'][0] == '2G'):
            dataframe['radio_2g'] = radio['band'][0]
            dataframe['chan_2g'] = radio['channel']
            dataframe['width_2g'] = radio['channel_width']
            for chan in radio['channels']:
                dataframe['channels_2g_'+str(radio['channels'].index(chan))] = chan
            for frq in radio['frequency']:
                dataframe['frequency_2g_'+str(radio['frequency'].index(frq))] = frq
            dataframe['phy_2g'] = radio['phy']
            dataframe['survey_2g'] = radio['survey']
            dataframe['temperature_2g'] = radio['temperature']
            dataframe['tx_power_2g'] = radio['tx_power']
        if (radio['band'][0] == '5G'):
            dataframe['radio_5g'] = radio['band'][0]
            dataframe['chan_5g'] = radio['channel']
            dataframe['width_5g'] = radio['channel_width']
            for chan in radio['channels']:
                dataframe['channels_5g_'+str(radio['channels'].index(chan))] = chan
            for frq in radio['frequency']:
                dataframe['frequency_5g_'+str(radio['frequency'].index(frq))] = frq
            dataframe['phy_5g'] = radio['phy']
            dataframe['survey_5g'] = radio['survey']
            dataframe['temperature_5g'] = radio['temperature']
            dataframe['tx_power_5g'] = radio['tx_power']
        if (radio['band'][0] == '6G'):
            dataframe['radio_6g'] = radio['band'][0]
            dataframe['chan_6g'] = radio['channel']
            dataframe['width_6g'] = radio['channel_width']
            for chan in radio['channels']:
                dataframe['channels_6g_'+str(radio['channels'].index(chan))] = chan
            for frq in radio['frequency']:
                dataframe['frequency_6g_'+str(radio['frequency'].index(frq))] = frq
            dataframe['frequency_6g'] = radio['frequency']
            dataframe['phy_6g'] = radio['phy']
            dataframe['survey_6g'] = radio['survey']
            dataframe['temperature_6g'] = radio['temperature']
            dataframe['tx_power_6g'] = radio['tx_power']
    #link-stats data
    for link in device_stats['link-state']:
        for network in device_stats['link-state'][link]:
            for carrier in device_stats['link-state'][link][network]:
                if carrier == 'delta_counters':
                    ind = str(list(device_stats['link-state'][link][network]).index(carrier))
                    print(carrier)
            
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_collisions'] = device_stats['link-state'][link][network][carrier]['collisions']
                    if 'multicast' in list(device_stats['link-state'][link][network][carrier]):
                        dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_multicast'] = device_stats['link-state'][link][network][carrier]['multicast']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_rx_bytes'] = device_stats['link-state'][link][network][carrier]['rx_bytes']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_rx_dropped'] = device_stats['link-state'][link][network][carrier]['rx_dropped']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_rx_errors'] = device_stats['link-state'][link][network][carrier]['rx_errors']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_rx_packets'] = device_stats['link-state'][link][network][carrier]['rx_packets']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_tx_bytes'] = device_stats['link-state'][link][network][carrier]['tx_bytes']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_tx_dropped'] = device_stats['link-state'][link][network][carrier]['tx_dropped']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_tx_errors'] = device_stats['link-state'][link][network][carrier]['tx_errors']
                    dataframe['link_'+str(link)+'_'+str(network)+'_'+ind+'_tx_packets'] =device_stats['link-state'][link][network][carrier]['tx_packets']
            
    #interfaces data
    dataframe['num_ifaces'] = len(device_stats['interfaces'])

    #device health data
    dataframe['sanity'] = device_health['sanity']
    dataframe['health_mem'] = device_health['values']['unit']['memory']    
    #dataframe['health_cpu'] = device['values']['unit']['cpu_load']



    '''
    if (inv == None):
            dataframe['name'] = "unknown"
            if (len(device['manufacturer']) > 0):
                dataframe['model'] = capabilities['compatible']
            else:
                dataframe['model'] = "unknown"
    else:
        dataframe['name'] = inv['name']
        dataframe['model'] = inv['deviceType']
    '''
    print(dataframe)
    #print(capabilities)
    #print(device)

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
'''
with open(fn, "w") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=csv_fields)
    writer.writeheader()
    for od in online_devices:
        writer.writerow(od)
'''
