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

# download the weights file
WEIGHTS_URL=`grep "\[S3_WEIGHTS_FILE\]:" readme.md  | sed "s/\[S3_WEIGHTS_FILE\]://" | sed "s/ //"`
WEIGHTS_FILE=${WEIGHTS_URL##*/}
echo "Downloading weights file from ${WEIGHTS_URL}"
curl -o $WEIGHTS_FILE $WEIGHTS_URL 2> /dev/null

python -u image_caption.py --test_only -i ${EXECUTOR_NUMBER} -vvv --model_file $WEIGHTS_FILE --no_progress_bar > output.dat
rc=$?
if [ $rc -ne 0 ];then
    exit $rc
fi

# get the top-1 misclass
bleu=`tail -n 1 output.dat | sed "s/BLEU = //" | sed "s/\/.*//"`
bleupass=`echo $bleu'>'12 | bc -l`

rc=0
if [ $bleupass -ne 1 ];then
    echo "BLEU too low "$bleu
    rc=1
fi
exit $rc
