#include <iostream>
#include <cstring>
#include <cmath> // for exp()
#include <algorithm> // for sort()
#include "nvdsinfer_custom_impl.h"
#include <stdexcept>

using namespace std;

// Define the number of classes
#define NUM_CLASSES 7

// Sigmoid function
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C" bool NvDsInferParseCustomBbox(
    vector<NvDsInferLayerInfo> const& outputLayers,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayers.empty()) {
        cerr << "Error: No output layers found!" << endl;
        return false;
    }

    float* output = (float*)outputLayers[0].buffer;
    int num_queries = outputLayers[0].inferDims.d[0]; // Number of queries (300)
    int output_dim = outputLayers[0].inferDims.d[1];  // Values per query (11: 4 bbox + 7 classes)

    for (int i = 0; i < num_queries; ++i) {
        float* query = output + i * output_dim;

        float cx = query[0] * networkInfo.width;
        float cy = query[1] * networkInfo.height;
        float w = query[2] * networkInfo.width;
        float h = query[3] * networkInfo.height;

        // Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        float left = cx - (w / 2);
        float top = cy - (h / 2);
        float right = cx + (w / 2);
        float bottom = cy + (h / 2);

        float* class_scores = query + 4; // Skip pointer 4 for bbox -> 7 classes
     
        float max_score = 0.0f;
        int class_id = -1;
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float score = sigmoid(class_scores[j]);
            if (score > max_score) {
                max_score = score;
                class_id = j;
            }
        }

        // if (max_score < detectionParams.perClassThreshold[class_id]) {
        //     continue;
        // }
        if (max_score < 0.5f) {
            continue;
        }

        NvDsInferParseObjectInfo obj;
        obj.left = left;
        obj.top = top;
        obj.width = right - left;
        obj.height = bottom - top;
        obj.classId  = class_id;
        obj.detectionConfidence = max_score;

        objectList.push_back(obj);
    }

    return true;
}