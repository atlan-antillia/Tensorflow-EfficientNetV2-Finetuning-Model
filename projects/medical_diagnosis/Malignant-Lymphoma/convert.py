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

# 2022/09/13 Copyright (C) antillia.com

#Malignant_Lymphoma

import os, sys
import shutil

from PIL import Image
import glob
import traceback

def convert(base_image_dir, output_image_dir, CROPPED_SIZE=640):

  classes = ["CLL", "FL", "MCL"]
  for cls in classes:
    class_dir = os.path.join(base_image_dir, cls)
    files = glob.glob(class_dir  + "/*.tif")
    output_dir = os.path.join(output_image_dir, cls)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
      print("--- Removed {}".format(output_dir))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for file in files:
        print("file : {} ".format( file))

        basename = os.path.basename(file)
        name     = basename.split(".")[0]
        
        outfile = name +  ".jpg"
        im = Image.open(file)
        W, H = im.size
        image = im.convert("RGB")
      
        print("Original Image size W:{}  H]{}".format(W, H))
        x = int( (W - CROPPED_SIZE)/2 )
        y = int( (H - CROPPED_SIZE)/2 )
        
        w = CROPPED_SIZE
        h = CROPPED_SIZE
        print( "x:{}  y:{}  w:{}  h:{}".format(x, y, w, h))
        cropped_image = image.crop((x, y, x+w, y+h))
        
        output_file = os.path.join(output_dir, outfile)
        cropped_image.save(output_file, "JPEG", quality=90)
        print("--- Saved {}".format(output_file))
        


if __name__ == "__main__":
  base_image_dir =  "./Malignant_Lymphoma"
  output_image_dir = "./Malignant_Lymphoma-jpg-master_840x840"
  try:

    convert(base_image_dir, output_image_dir, CROPPED_SIZE=840)

  except:
    traceback.print_exc()

