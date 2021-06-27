"""
This script is supposed to run inside alicevision/meshroom2021.1 container
python ver. = 2.7
"""

import sys
import json


pipelineMgFile = sys.argv[1]
folder = sys.argv[2]

generatedMgFileName = '/tmp/{}/generatedMgTemplate.mg'.format(folder)

with open(pipelineMgFile) as pipelineFile:
    pipeline = json.load(pipelineFile)

with open(generatedMgFileName) as generatedFile:
    imagePath = json.load(generatedFile)

pipeline['graph']['CameraInit_1']['inputs']['viewpoints'] = imagePath['graph']['CameraInit_1']['inputs']['viewpoints']
pipeline['graph']['CameraInit_1']['inputs']['intrinsics'] = imagePath['graph']['CameraInit_1']['inputs']['intrinsics']
pipeline['graph']['CameraInit_1']['inputs']['sensorDatabase'] = imagePath['graph']['CameraInit_1']['inputs']['sensorDatabase']

pipeline['graph']['FeatureExtraction_1']['inputs']['maxThreads'] = 20

with open(pipelineMgFile, 'w') as f:
    json.dump(pipeline, f)

