import os
import ipdb 
st = ipdb.set_trace
path = "/home/mprabhud/dataset/preprocessed_shapenet_4"
instance_id_list = [li[:-4] for li in os.listdir(path) if li.endswith('.obj')]
# st()
dict_file = open("/home/mprabhud/shamit/shapenet_preprocess/dict_file.txt", "w")

dict_file.write('[')
for cnt, instanceid in enumerate(instance_id_list):
    if cnt == len(instance_id_list) - 1:
        dict_file.write("'{}'".format(instanceid))
    else:
        dict_file.write("'{}',".format(instanceid))
dict_file.write(']')
dict_file.close()

json_file = open("/home/mprabhud/shamit/shapenet_preprocess/json_file.txt", "w")
for cnt, instanceid in enumerate(instance_id_list):
    json_file.write('"{}":"{}",'.format(instanceid, instanceid))
json_file.close()