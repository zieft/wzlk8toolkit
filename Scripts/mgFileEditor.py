"""
This script is supposed to run inside alicevision/meshroom2021.1 container
python ver. = 2.7
"""

import sys
import json


path_provided_mg_file = sys.argv[1]
# dataset_name = sys.argv[2]
path_generated_mg_file = sys.argv[2]

with open(path_provided_mg_file) as pipelineFile:
    pipeline = json.load(pipelineFile)

with open(path_generated_mg_file) as generatedFile:
    imagePath = json.load(generatedFile)

nodeList = list(pipeline['graph'].keys())

for node in nodeList:
    if 'ImageMatching_' in node:
        pipeline['graph'][node]['inputs']['tree'] = imagePath['graph']['ImageMatching_1']['inputs']['tree']
    elif 'CameraInit_' in node:
        pipeline['graph'][node]['inputs']['viewpoints'] = imagePath['graph']['CameraInit_1']['inputs']['viewpoints']
        pipeline['graph'][node]['inputs']['intrinsics'] = imagePath['graph']['CameraInit_1']['inputs']['intrinsics']
        pipeline['graph'][node]['inputs']['sensorDatabase'] = imagePath['graph']['CameraInit_1']['inputs']['sensorDatabase']
    elif 'FeatureExtraction_' in node:
        pipeline['graph'][node]['inputs']['maxThreads'] = 20


# pipeline['graph']['CameraInit_1']['inputs']['viewpoints'] = imagePath['graph']['CameraInit_1']['inputs']['viewpoints']
# pipeline['graph']['CameraInit_1']['inputs']['intrinsics'] = imagePath['graph']['CameraInit_1']['inputs']['intrinsics']
# pipeline['graph']['CameraInit_1']['inputs']['sensorDatabase'] = imagePath['graph']['CameraInit_1']['inputs']['sensorDatabase']
#
# pipeline['graph']['FeatureExtraction_1']['inputs']['maxThreads'] = 20

with open(path_provided_mg_file, 'w') as f:
    json.dump(pipeline, f)

