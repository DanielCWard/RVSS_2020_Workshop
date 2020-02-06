import numpy as np
import json

data_0_path = 'slam_0.json'

with open(data_0_path, 'r') as f:
    datastore_0 = json.load(f)

with open('slam_1.json', 'r') as f:
    datastore_1 = json.load(f)

with open('slam_2.json', 'r') as f:
    datastore_2 = json.load(f)

x_vec = 0
y_vec = 0
for i in np.arange(len(datastore_0['taglist'])):
    x_vec = x_vec + np.std( np.array( [datastore_0['map'][i][0], datastore_1['map'][i][0], datastore_2['map'][i][0]] ) )
    y_vec = y_vec + np.std( np.array( [datastore_0['map'][i][1], datastore_1['map'][i][1], datastore_2['map'][i][1]] ) )

x_vec_cov = x_vec/len(datastore_0['taglist'])
y_vec_cov = y_vec/len(datastore_0['taglist'])
print('x_vec_cov ', x_vec_cov)
print('y_vec_cov ', y_vec_cov)