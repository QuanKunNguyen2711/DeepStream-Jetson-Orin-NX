#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <Eigen/Dense> // Include Eigen library for tensor operations
#include <stdexcept>
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include <omp.h>

using namespace Eigen;
using namespace std;

const int REG_MAX = 16; // reg_max value
const int NUM_CLASSES = 7; // Number of classes (71 - 4 * REG_MAX)
const vector<int> STRIDES = {8, 16, 32}; // Strides for each output layer

// Function to perform 3D convolution with a 1x1x1 kernel
vector<MatrixXf> conv3d(const vector<vector<MatrixXf>>& anchor_x, const VectorXf& kernel) {
    int reg_max = anchor_x.size();
    int num_predictions = anchor_x[0].size();
    int height = anchor_x[0][0].rows();
    int width = anchor_x[0][0].cols();

    vector<MatrixXf> output(num_predictions, MatrixXf::Zero(height, width));

    #pragma omp parallel for collapse(3)
    for (int p = 0; p < num_predictions; ++p) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float sum = 0.0f;
                for (int r = 0; r < reg_max; ++r) {
                    sum += anchor_x[r][p](h, w) * kernel(r);
                }
                output[p](h, w) = sum;
            }
        }
    }

    return output;
}


class Anchor2Vec {
public:
    Anchor2Vec(int reg_max = 16) : reg_max(reg_max), num_predictions(4) {
        // Initialize reverse_reg
        reverse_reg = VectorXf::LinSpaced(reg_max, 0, reg_max - 1);
    }

    pair<vector<vector<MatrixXf>>, vector<MatrixXf>> forward(const vector<MatrixXf>& reg, int height, int width);

private:
    int reg_max;
    int num_predictions;
    VectorXf reverse_reg; // Equivalent to torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
};

pair<vector<vector<MatrixXf>>, vector<MatrixXf>> Anchor2Vec::forward(const vector<MatrixXf>& reg, int height, int width) {
    // Step 1: Reshape the input tensor
    // Input reg has shape (num_predictions * reg_max, height, width)
    // Reshape to (reg_max, num_predictions, height, width)
    vector<vector<MatrixXf>> anchor_x(reg_max, vector<MatrixXf>(num_predictions, MatrixXf::Zero(height, width)));

    // Perform the reshaping
    for (int r = 0; r < reg_max; ++r) {
        for (int p = 0; p < num_predictions; ++p) {
            anchor_x[r][p] = reg[p * reg_max + r];
        }
    }


    // Step 2: Apply softmax along the reg_max dimension
    // (16, 4, height, width)
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < num_predictions; ++p) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                VectorXf vec(reg_max);
                for (int r = 0; r < reg_max; ++r) {
                    vec(r) = anchor_x[r][p](h, w);
                }
                vec = vec.array().exp();
                float sum = vec.sum();
                vec /= sum;
                for (int r = 0; r < reg_max; ++r) {
                    anchor_x[r][p](h, w) = vec(r);
                }
            }
        }
    }

    VectorXf reverse_reg = this->reverse_reg;

    vector<MatrixXf> vector_x = conv3d(anchor_x, reverse_reg);

    return {anchor_x, vector_x};
}

vector<float> arange(float start, float stop, float step) {
    vector<float> result;
    for (float value = start; value < stop; value += step) {
        result.push_back(value);
    }
    return result;
}

pair<vector<float>, vector<float>> meshgrid(const vector<float>& x, const vector<float>& y) {
    vector<float> xv, yv;
    for (float yi : y) {
        for (float xi : x) {
            xv.push_back(xi);
            yv.push_back(yi);
        }
    }
    return {xv, yv};
}

pair<vector<vector<float>>, vector<float>> generate_anchors(int image_height, int image_width, const vector<int>& strides) {
    vector<vector<float>> anchors;
    vector<float> scalers;

    for (int stride : strides) {
        int anchor_num = (image_width / stride) * (image_height / stride);
        float offset = stride / 2.0f;

        // Generate h and w ranges
        auto h = arange(0, image_height, stride);
        auto w = arange(0, image_width, stride);

        // Add offset
        for (float& val : h) val += offset;
        for (float& val : w) val += offset;

        auto [anchor_w, anchor_h] = meshgrid(w, h);

        for (size_t i = 0; i < anchor_w.size(); i++) {
            anchors.push_back({anchor_w[i], anchor_h[i]});
        }

        for (int i = 0; i < anchor_num; i++) {
            scalers.push_back(static_cast<float>(stride));
        }
    }

    return {anchors, scalers};
}

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

MatrixXf concatMatrices(const vector<MatrixXf>& matrices) {
    if (matrices.empty()) {
        return MatrixXf();
    }

    int total_rows = 0;
    int cols = matrices[0].cols();
    for (const auto& mat : matrices) {
        total_rows += mat.rows();
    }

    MatrixXf result(total_rows, cols);

    int current_row = 0;
    for (const auto& mat : matrices) {
        result.block(current_row, 0, mat.rows(), cols) = mat;
        current_row += mat.rows();
    }

    return result;
}

vector<Detection> yolo_head_decode(
    const vector<vector<float>>& pred, 
    int image_height, 
    int image_width, 
    NvDsInferParseDetectionParams const& detectionParams, 
    Anchor2Vec anc2vec, 
    int reg_max = REG_MAX
) {
    int h = image_height;
    int w = image_width;

    auto [offset, scaler] = generate_anchors(h, w, STRIDES); // (8400,2), (8400,1)
    
    VectorXf scaler_eigen = Map<VectorXf>(scaler.data(), scaler.size());

    MatrixXf offset_eigen(offset.size(), offset[0].size());
    for (size_t i = 0; i < offset.size(); i++) {
        for (size_t j = 0; j < offset[i].size(); j++) {
            offset_eigen(i, j) = offset[i][j];
        }
    }

    vector<MatrixXf> pred_bbox_reg, pred_class_logits;
    for (const auto& layer_output : pred) {
        int split_idx = 4 * reg_max; // Index where classification logits start
        int channels = 4 * reg_max + NUM_CLASSES; // Total channels 71
        int split_size = layer_output.size() / channels; // Spatial size (height * width)
        int height = sqrt(split_size); // 80
        int width = height; // 80

        // Reshape regression outputs (64,80,80)
        vector<MatrixXf> reg(4 * reg_max, MatrixXf(height, width));
        for (int c = 0; c < 4 * reg_max; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int flat_index = c * (width * height) + h * width + w; // Index into layer_output
                    reg[c](h, w) = layer_output[flat_index];
                }
            }
        }

        // Reshape classification logits (7,80,80)
        vector<MatrixXf> class_logits(NUM_CLASSES, MatrixXf(height, width));
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int flat_index = (split_idx + c) * (width * height) + h * width + w; // Index into layer_output
                    class_logits[c](h, w) = layer_output[flat_index];
                }
            }
        }

        // Apply Anchor2Vec transformation
        auto [_, bbox_reg] = anc2vec.forward(reg, height, width);

        int num_predictions = 4;
        vector<MatrixXf> bbox_reg_permuted(height, MatrixXf::Zero(width, num_predictions));
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int p = 0; p < num_predictions; ++p) {
                    bbox_reg_permuted[h](w, p) = bbox_reg[p](h, w);
                }
            }
        }

        // Reshape (height, width, num_predictions) -> (height * width, num_predictions)
        MatrixXf bbox_reg_reshaped(height * width, num_predictions);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                bbox_reg_reshaped.row(h * width + w) = bbox_reg_permuted[h].row(w);
            }
        }

        vector<MatrixXf> class_logits_permuted(height, MatrixXf::Zero(width, NUM_CLASSES));
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int p = 0; p < NUM_CLASSES; ++p) {
                    class_logits_permuted[h](w, p) = class_logits[p](h, w);
                }
            }
        }

        // Reshape (height, width, num_classes) -> (height * width, num_classes)
        MatrixXf class_logits_reshaped(height * width, NUM_CLASSES);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                class_logits_reshaped.row(h * width + w) = class_logits_permuted[h].row(w);
            }
        }

        pred_bbox_reg.push_back(bbox_reg_reshaped);
        pred_class_logits.push_back(class_logits_reshaped);

    }

    MatrixXf bbox_reg_concat = concatMatrices(pred_bbox_reg);
    MatrixXf class_logits_concat = concatMatrices(pred_class_logits);

    class_logits_concat = class_logits_concat.unaryExpr([](float x) {
        return 1.0f / (1.0f + exp(-x)); // Sigmoid function
    });

    // Step 1: Compute pred_xyxy = pred_bbox_reg * scaler.view(1, -1, 1)
    MatrixXf pred_xyxy = bbox_reg_concat.array().colwise() * scaler_eigen.array();
    MatrixXf lt = pred_xyxy.block(0, 0, pred_xyxy.rows(), 2);
    MatrixXf rb = pred_xyxy.block(0, 2, pred_xyxy.rows(), 2);

    MatrixXf final_bbox = MatrixXf::Zero(pred_xyxy.rows(), 4);
    // final_bbox << offset - lt, offset + rb;
    final_bbox.col(0) = offset_eigen.col(0) - lt.col(0); // x1
    final_bbox.col(1) = offset_eigen.col(1) - lt.col(1); // y1
    final_bbox.col(2) = offset_eigen.col(0) + rb.col(0); // x2
    final_bbox.col(3) = offset_eigen.col(1) + rb.col(1); // y2

    vector<Detection> detections;

    for (int i = 0; i < final_bbox.rows(); i++) {
        Index class_id;
        float max_conf = class_logits_concat.row(i).maxCoeff(&class_id);
        if (max_conf >= 0.5f) {
            Detection detection;
            detection.x1 = final_bbox(i, 0); // x1
            detection.y1 = final_bbox(i, 1); // y1
            detection.x2 = final_bbox(i, 2); // x2
            detection.y2 = final_bbox(i, 3); // y2
            detection.confidence = max_conf; // Confidence score
            detection.class_id = static_cast<int>(class_id); // Predicted class ID

            detections.push_back(detection);
        }
    }  

    return detections;
}


extern "C" bool NvDsInferParseCustomYolo(
    vector<NvDsInferLayerInfo> const& outputLayers,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    vector<NvDsInferParseObjectInfo>& objectList) {
    
    vector<vector<float>> pred;
    for (size_t i = 0; i < outputLayers.size(); ++i) {
        // Cast layer.buffer to float*
        const auto& layer = outputLayers[i];
        float* buffer = static_cast<float*>(layer.buffer);
        vector<float> layerData(buffer, buffer + layer.inferDims.numElements);
        pred.push_back(layerData);
    }

    int image_width = networkInfo.width;
    int image_height = networkInfo.height;

    Anchor2Vec anc2vec(REG_MAX); 
    auto detections = yolo_head_decode(pred, image_height, image_width, detectionParams, anc2vec, REG_MAX);

    for (const auto& detection : detections) {
        NvDsInferParseObjectInfo obj;
        obj.left = detection.x1;
        obj.top = detection.y1;
        obj.width = detection.x2 - detection.x1;
        obj.height = detection.y2 - detection.y1;
        obj.detectionConfidence = detection.confidence;
        obj.classId = detection.class_id;
        objectList.push_back(obj);
    }   

    return true;
}