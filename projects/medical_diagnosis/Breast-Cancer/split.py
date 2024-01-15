import splitfolders

input_folder = "./master" #Enter Input Folder
output = "./train/" #Enter Output Folder

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(0.8,0.2))