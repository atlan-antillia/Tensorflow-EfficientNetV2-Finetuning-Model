# Copyright 2022 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2022/09/16 Copyright (C) antillia.com

# pip install split-folders
# split_master.py

import os
import sys
import shutil
import splitfolders
import traceback


if __name__ == "__main__":
  # We don't need val dataset, because the train dataset generated by this 
  # script will be split into train and valid in CustomDataset class in a training process.
  #  CutomData
  #so val ratio = 0.0
  #       train, val, test
  ratio = (0.8,  0.0, 0.2)
  input_folder  = "./Osteosarcoma-master"
  output_folder = "./Osteosarcoma_Images"

  if len(sys.argv) == 3:
    input_folder  = sys.argv[1]
    output_folder = sys.argv[2]

  input("Hit any key to start")

  try: 
    if not os.path.exists(input_folder):
      raise Exception("Not found input_folder " + input_folder)
    if os.path.exists(output_folder):
      shutil.rmtree(output_folder)

    splitfolders.ratio(input_folder, output=output_folder, seed=37, ratio=ratio)

    val_folder = os.path.join(output_folder, "val")
    if os.path.exists(val_folder):
      shutil.rmtree(val_folder)
      print("Removed val_folder {}".format(val_folder))

  except:
    traceback.print_exc()
