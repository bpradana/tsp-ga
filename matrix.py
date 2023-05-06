import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import json


def preprocess_coordinate(coordinate_list):
  latitudes = coordinate_list['Koordinat'].map(lambda x: float(x.split(',')[0]))
  longitudes = coordinate_list['Koordinat'].map(lambda x: float(x.split(',')[1]))
  coordinate_list['latitude'] = latitudes
  coordinate_list['longitude'] = longitudes
  return coordinate_list

def clean_columns(coordinate_list):
  coordinate_list.drop(['No', 'Koordinat', 'Daerah', 'Keterangan (Opsional)'], axis=1, inplace=True)
  coordinate_list.rename(columns={'Customer': 'customer'}, inplace=True)
  return coordinate_list

def call_api(source, destination):
  s_lat, s_long = source
  d_lat, d_long = destination
  url = f'http://127.0.0.1:5000/route/v1/driving/{s_long},{s_lat};{d_long},{d_lat}?steps=false'
  response = requests.get(url)
  return response.json()

def parse_distance_duration(response):
  distance = response['routes'][0]['distance']
  duration = response['routes'][0]['duration']
  return distance, duration

def create_matrix(coordinate_list, type=None):
  matrix = {}
  for i in tqdm(range(len(coordinate_list))):
    source = (coordinate_list.iloc[i]['latitude'], coordinate_list.iloc[i]['longitude'])
    matrix[coordinate_list.iloc[i]['customer']] = {}
    for j in range(len(coordinate_list)):
      destination = (coordinate_list.iloc[j]['latitude'], coordinate_list.iloc[j]['longitude'])
      response = call_api(source, destination)
      distance, duration = parse_distance_duration(response)
      if type == 'distance':
        matrix[coordinate_list.iloc[i]['customer']][coordinate_list.iloc[j]['customer']] = distance
      elif type == 'duration':
        matrix[coordinate_list.iloc[i]['customer']][coordinate_list.iloc[j]['customer']] = duration
      else:
        matrix[coordinate_list.iloc[i]['customer']][coordinate_list.iloc[j]['customer']] = {'distance': distance, 'duration': duration}
  return matrix

if __name__ == '__main__':
  coordinate_list = pd.read_csv('coordinate_list.csv')
  coordinate_list = preprocess_coordinate(coordinate_list)
  coordinate_list = clean_columns(coordinate_list)

  matrix = create_matrix(coordinate_list)
  with open('matrix.json', 'w') as f:
    json.dump(matrix, f, indent=2)

  matrix = create_matrix(coordinate_list, type='distance')
  pd.DataFrame(matrix).to_csv('distance_matrix.csv', sep=';')

  matrix = create_matrix(coordinate_list, type='duration')
  pd.DataFrame(matrix).to_csv('duration_matrix.csv', sep=';')
