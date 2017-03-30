#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/lenet_test_solver.prototxt $@
