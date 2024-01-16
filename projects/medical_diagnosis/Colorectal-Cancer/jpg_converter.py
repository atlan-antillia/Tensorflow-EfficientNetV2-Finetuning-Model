# jpg_converter.py
# 2022/11/21 to-arai
#

import os
import sys
import glob
from PIL import Image

import shutil
import traceback


def jpg_converter(input_dir, output_dir):
  subdirs = os.listdir(input_dir)
  
  for subdir in subdirs:
    fullpath = os.path.join(input_dir, subdir)
    tif_files = glob.glob(fullpath + "/*.tif")
    print("--- len tif_files {}".format(len(tif_files)))
    output_subdir = os.path.join(output_dir, subdir)
    if os.path.exists(output_subdir):
      shutil.rmtree(output_subdir)

    if not os.path.exists(output_subdir):
      os.makedirs(output_subdir)

    input("HIT any key") 
    for tif_file in tif_files:
      image = Image.open(tif_file)
      out = image.convert("RGB")
      basename = os.path.basename(tif_file)
      nameonly = basename.split(".")[0]
      print("--- tif file {}".format(tif_file))

      jpg_filename = nameonly + ".jpg"
      output_file = os.path.join(output_subdir, jpg_filename)
      print("--- saved {}".format(output_file))

      #input("HIT")

      out.save(output_file, "JPEG", quality=90)

if __name__ == "__main__":
  try:
    input_dir  = "./CRC-VAL-HE-7K"
    output_dir = "./CRC-VAL-HE-7K-jpg-master"

    jpg_converter(input_dir, output_dir)

  except:
    traceback.print_exc()
