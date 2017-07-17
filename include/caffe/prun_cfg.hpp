// prun_cfg.hpp: include flages for pruning
// NOTE:
//  define follow signals in the follow file:
//        tools/upgrade_net_proto_binary.cpp
//        tools/upgrade_solver_proto_text.cpp
//        tools/caffe.cpp
//        tools/extract_features.cpp
//        tools/upgrade_net_proto_text.cpp
//        examples/mnist/convert_mnist_data.cpp
//        tools/convert_imageset.cpp
//        tools/compute_image_mean.cpp
//        examples/siamese/convert_mnist_siamese_data.cpp
//        examples/cifar10/convert_cifar_data.cpp
//        examples/cpp_classification/classification.cpp
//    eg. DEFINE_name(name, default value, "explain");

#ifndef PRUN_CFG_HPP_
#define PRUN_CFG_HPP_

#include <gflags/gflags.h>
#include <iostream>

using namespace std;



DECLARE_bool(prun_conv);
DECLARE_double(conv_ratio_0);
DECLARE_double(conv_ratio_1);
DECLARE_double(conv_ratio_2);
DECLARE_bool(prun_fc);
DECLARE_double(fc_ratio_0);
DECLARE_double(fc_ratio_1);
DECLARE_double(fc_ratio_2);
DECLARE_bool(prun_retrain);
DECLARE_int32(prun_fc_num);

#endif  // PRUN_CFG_HPP_
