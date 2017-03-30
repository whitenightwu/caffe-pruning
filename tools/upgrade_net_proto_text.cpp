// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_text v0_net_proto_file_in net_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/caffe.hpp"

#include "caffe/prun_cfg.hpp"

#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

// for pruning by zhluo
DEFINE_bool(prun_conv, false, "Optional; pruning CONV layers");
DEFINE_bool(prun_fc, false, "Optional; pruning FC layers");
DEFINE_bool(prun_retrain, false, "Optional; retrain net after pruning");
DEFINE_bool(sparse_csc, false, "Optional; blob use CSC sparse storage");
DEFINE_int32(prun_fc_num, 0, "Optional; the number of FC layers");
DEFINE_double(conv_ratio_0, 0, "Optional; conv layer prun ratio");
DEFINE_double(conv_ratio_1, 0, "Optional; conv layer prun ratio");
DEFINE_double(conv_ratio_2, 0, "Optional; conv layer prun ratio");
DEFINE_double(fc_ratio_0, 0, "Optional; fc layer prun ratio");
DEFINE_double(fc_ratio_1, 0, "Optional; fc layer prun ratio");
DEFINE_double(fc_ratio_2, 0, "Optional; fc layer prun ratio");

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;  // Print output to stderr (while still logging)
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: "
        << "upgrade_net_proto_text v0_net_proto_file_in net_proto_file_out";
    return 1;
  }

  NetParameter net_param;
  string input_filename(argv[1]);
  if (!ReadProtoFromTextFile(input_filename, &net_param)) {
    LOG(ERROR) << "Failed to parse input text file as NetParameter: "
               << input_filename;
    return 2;
  }
  bool need_upgrade = NetNeedsUpgrade(net_param);
  bool success = true;
  if (need_upgrade) {
    success = UpgradeNetAsNeeded(input_filename, &net_param);
    if (!success) {
      LOG(ERROR) << "Encountered error(s) while upgrading prototxt; "
                 << "see details above.";
    }
  } else {
    LOG(ERROR) << "File already in latest proto format: " << input_filename;
  }

  // Save new format prototxt.
  WriteProtoToTextFile(net_param, argv[2]);

  LOG(INFO) << "Wrote upgraded NetParameter text proto to " << argv[2];
  return !success;
}
