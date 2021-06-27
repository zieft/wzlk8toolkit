"""
This script is supposed to run inside alicevision/meshroom2021.1 container
python ver. = 2.7
"""

import sys
import json


pipelineMgFile = sys.argv[1]
generatedMgFileName = 'generatedMgTemplate.mg'

with open(pipelineMgFile) as pipelineFile:
    pipeline = json.load(pipelineFile)

with open(generatedMgFileName) as generatedFile:
    imagePath = json.load(generatedFile)

pipeline['graph']['CameraInit_1']['viewpoints'] = imagePath['graph']['CameraInit_1']['viewpoints']
pipeline['graph']['CameraInit_1']['intrinsics'] = imagePath['graph']['CameraInit_1']['intrinsics']
pipeline['graph']['CameraInit_1']['sensorDatabase'] = imagePath['graph']['CameraInit_1']['sensorDatabase']

pipeline['graph']['FeatureExtraction_1']['maxThreads'] = 20

with open(pipelineMgFile, 'w') as f:
    json.dump(pipeline, f)

