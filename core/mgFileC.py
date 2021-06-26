import io
import sys
import json

### NOTE: This script is tested only in Linux! ###
folder = sys.argv[1]


# Get current workdir
# currentPath = os.getcwd() + '/'
pathInContainer = '/tmp/{}'.format(folder) + '/'
dbFileInDocker = '/opt/AliceVision_bundle/share/aliceVision/cameraSensors.db'

# Define the path to the .mg file needed to be convert
mgFile = './project.mg'

with io.open(mgFile, 'r', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)



# text = 'D:/users/ggr/Projekte/Siemens_Surfacecracks/Photogrammetrie/Turbine/UV/P1260764.JPG'
# splitedText=text.split('/')
# print(splitedText)
# fileName = splitedText[-1]
# print(fileName)
# newPath = currentPath + fileName
# print(newPath)

data['graph']['CameraInit_1']['inputs']['sensorDatabase'] = dbFileInDocker

for viewPoint in data['graph']['CameraInit_1']['inputs']['viewpoints']:
    fileName = viewPoint['path'].split('/')[-1]
    newFilePath = currentPath + fileName
    viewPoint['path'] = newFilePath

for intrinsic in data['graph']['CameraInit_1']['inputs']['intrinsics']:
    fileName = intrinsic['serialNumber'].split('/')[-1]
    newFilePath = currentPath + fileName
    intrinsic['serialNumber'] = newFilePath

# write into filet
with io.open(mgFile, 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(data, f)

# Check again
with io.open(mgFile, 'r', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)

print(data['graph']['CameraInit_1']['inputs']['sensorDatabase'])
print(data['graph']['CameraInit_1']['inputs']['viewpoints'][0]['path'])
print(data['graph']['CameraInit_1']['inputs']['intrinsics'][0]['serialNumber'])
