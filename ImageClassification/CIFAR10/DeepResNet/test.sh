#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

# test script
TEST_SCRIPT="resnet_eval.py"

# download the weights file
WEIGHTS_URL=`grep "\[S3_WEIGHTS_FILE\]:" readme.md  | sed "s/\[S3_WEIGHTS_FILE\]://" | sed "s/ //"`
WEIGHTS_FILE=${WEIGHTS_URL##*/}
echo "Downloading weights file from ${WEIGHTS_URL}"
curl -o $WEIGHTS_FILE $WEIGHTS_URL 2> /dev/null

python -u $TEST_SCRIPT  -i ${EXECUTOR_NUMBER} -vvv --model_file $WEIGHTS_FILE --manifest val:/data/CIFAR/val-index.csv --manifest_root /data/CIFAR -b gpu -z 32 --no_progress_bar 2>&1 | tee output.dat
rc=$?
if [ $rc -ne 0 ];then
    exit $rc
fi

# get the top-1 misclass
top1=`cat output.dat | sed -n "s/.*Accuracy: \(.*\) \% (Top-1).*/\1/p"`

top1pass=0
top1pass=`echo $top1'>'85 | bc -l`

rc=0
if [ $top1pass -ne 1 ];then
    echo "Top1 Accuracy too low "$top1
    rc=1
fi

exit $rc
