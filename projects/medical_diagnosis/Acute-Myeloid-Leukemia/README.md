<h2>EfficientNetV2-Acute-Myeloid-Leukemia</h2>
 This is an experimental Acute-Myeloid-Leukemia Classification project based on <b>efficientnetv2</b> 
 in <a href="https://github.com/google/automl">Brain AutoML</a><br>

Please see also our first expreiment <a href="https://github.com/sarah-antillia/EfficientNet-Acute-Myeloid-Leukemia">
EfficientNet-Acute-Myeloid-Leukemia </a>

<h3>1. Dataset Citation</h3>
The AML image dataset used here has been taken from the following web site;<br>
<pre>
CANCER IMAGING ARCHIVE

https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7

Package - AML-Cytomorphology
From:
Natasha Honomichl
To:
help@cancerimagingarchive.net 
CC (on download):
Natasha Honomichl
Date Sent:
17 Feb 2021 01:11 PM
</pre>


<h3>2. Download dataset</h3>

Please download Resampled_AMLImages dataset from google drive:<br>
<a href="https://drive.google.com/file/d/15W31ddTljTo4rZ8HCohvyg6XFggdEoH2/view?usp=sharing">Resampled_AML_Images.zip</a>
<br>
It contains the following test and train datasets.<br>
<pre>
Resampled_AML_Images
  ├─test
  │  ├─BAS
  │  ├─EBO
  │  ├─EOS
  │  ├─KSC
  │  ├─LYA
  │  ├─LYT
  │  ├─MMZ
  │  ├─MOB
  │  ├─MON
  │  ├─MYB
  │  ├─MYO
  │  ├─NGB
  │  ├─NGS
  │  ├─PMB
  │  └─PMO
  └─train
      ├─BAS
      ├─EBO
      ├─EOS
      ├─KSC
      ├─LYA
      ├─LYT
      ├─MMZ
      ├─MOB
      ├─MON
      ├─MYB
      ├─MYO
      ├─NGB
      ├─NGS
      ├─PMB
      └─PMO
</pre>

<br>


1 Sample images of Resampled_AML_Images/train/BAS:<br>
<img src="./asset/sample_train_images_BAS.png" width="840" height="auto">
<br> 

2 Sample images of Resampled_AML_Images/train/EBO:<br>
<img src="./asset/sample_train_images_EBO.png" width="840" height="auto">
<br> 

3 Sample images of Resampled_AML_Images/train/EOS:<br>
<img src="./asset/sample_train_images_EOS.png" width="840" height="auto">
<br> 

4 Sample images of Resampled_AML_Images/train/KSC:<br>
<img src="./asset/sample_train_images_KSC.png" width="840" height="auto">
<br> 

5 Sample images of Resampled_AML_Images/train/LYA:<br>
<img src="./asset/sample_train_images_LYA.png" width="840" height="auto">
<br> 

6 Sample images of Resampled_AML_Images/train/LYT:<br>
<img src="./asset/sample_train_images_LYT.png" width="840" height="auto">
<br>

7 Sample images of Resampled_AML_Images/train/MMZ:<br>
<img src="./asset/sample_train_images_MMZ.png" width="840" height="auto">
<br> 

8 Sample images of Resampled_AML_Images/train/MOB:<br>
<img src="./asset/sample_train_images_MOB.png" width="840" height="auto">
<br> 

9 Sample images of Resampled_AML_Images/train/MON:<br>
<img src="./asset/sample_train_images_MON.png" width="840" height="auto">
<br> 

10 Sample images of Resampled_AML_Images/train/MYB:<br>
<img src="./asset/sample_train_images_MYB.png" width="840" height="auto">
<br> 

11 Sample images of Resampled_AML_Images/train/MYO:<br>
<img src="./asset/sample_train_images_MYO.png" width="840" height="auto">
<br> 

12 Sample images of Resampled_AML_Images/train/NGB:<br>
<img src="./asset/sample_train_images_NGB.png" width="840" height="auto">
<br> 

13 Sample images of Resampled_AML_Images/train/NGS:<br>
<img src="./asset/sample_train_images_NGS.png" width="840" height="auto">
<br> 

14 Sample images of Resampled_AML_Images/train/PMB:<br>
<img src="./asset/sample_train_images_PMB.png" width="840" height="auto">
<br> 

15 Sample images of Resampled_AML_Images/train/PMO:<br>
<img src="./asset/sample_train_images_PMO.png" width="840" height="auto">
<br> 

<br> 

<h2>
3. Train
</h2>
<h3>
3.1 Training script
</h3>
Please run the following bat file to train our AML efficientnetv2 model by using
<b>Resampled_AML_images/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
rem 2024/01/12
python ../../../efficientnetv2/EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../../efficientnetv2/efficientnetv2-m/model ^
  --optimizer=adam ^
  --image_size=360 ^
  --eval_image_size=360 ^
  --data_dir=./Resampled_AML_images/train ^
  --data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --num_epochs=100 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config
; 2024/01/12
[training]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 90
;rotation_range     = 10
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.2
height_shift_range = 0.2
shear_range        = 0.01
zoom_range         = [0.8, 1.0]
data_format        = "channels_last"
brightness_range   = [0.8, 1.0]
fill_mode          =  "nearest"
</pre>

<h3>
3.2 Training result
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./eval/train_accuracies.csv">train_accuracies</a>
and <a href="./eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/train_at_epoch_16_0104.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./eval/train_losses.png" width="640" height="auto"><br>

<br>
<h3>
4. Inference
</h3>
<h3>
4.1 Inference script
</h3>
Please run the following bat file to infer the Acute-Myeloid-Leukemia images in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
rem 2024/01/01
python ../../../efficientnetv2/EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=360 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
BAS
EBO
EOS
KSC
LYA
LYT
MMZ
MOB
MON
MYB
MYO
NGB
NGS
PMB
PMO
</pre>
<br>
<h3>
4.2 Sample test images
</h3>

Sample test images generated by <a href="./create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./Resampled_AML_Images/test">Lymphoma/test</a>.
<br>
<img src="./asset/test.png" width="840" height="auto"><br>


<br>
<h3>
4.3 Inference result
</h3>
This inference command will generate <a href="./inference/inference.csv">inference result file</a>.
<br>
Inference console output:<br>
<img src="./asset/inference_at_epoch_16_0104.png" width="740" height="auto"><br>
<br>

Inference result (<a href="./inference/inference.csv">inference.csv</a>):<br>
<img src="./asset/inference_csv_at_epoch_16_0104.png" width="640" height="auto"><br>
<br>
<h2>
5. Evaluation
</h2>
<h3>
5.1 Evaluation script
</h3>
Please run the following bat file to evaluate <a href="./Resampled_AML_Images/test">
Resampled_AML_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
rem 2024/01/01
python ../../../efficientnetv2/EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Resampled_AML_Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --eval_image_size=360 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
5.2 Evaluation result
</h3>

This evaluation command will generate <a href="./evaluation/classification_report.csv">a classification report</a>
 and <a href="./evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/evaluate_at_epoch_16_0104.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/classfication_report.png" width="640" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./evaluation/confusion_matrix.png" width="740" height="auto"><br>

<br>
<h3>
References
</h3>
<b>1. AML-Cytomorphology</b><br>
<pre>
The AML image dataset used here has been taken from the following web site;
CANCER IMAGING ARCHIVE
https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7
</pre>

<b>2. Acute Myeloid Leukemia classification using a federated Convolutional Neural Network</b><br>
scaleoutsystems<br>
<pre>
https://github.com/scaleoutsystems/AML-tutorial
</pre>

<b>3. Deep learning detects acute myeloid leukemia and predicts NPM1 mutation status from bone marrow smears</b><br>
Jan-Niklas Eckardt, Jan Moritz Middeke, Sebastian Riechert, Tim Schmittmann, Anas Shekh Sulaiman, <br>
Michael Kramer, Katja Sockel, Frank Kroschinsky, Ulrich Schuler, Johannes Schetelig, Christoph Röllig,<br> 
Christian Thiede, Karsten Wendt & Martin Bornhäuser <br>

<pre>
https://www.nature.com/articles/s41375-021-01408-w
</pre>


<b>4. AMLnet, A deep-learning pipeline for the differential diagnosis of acute myeloid leukemia from bone marrow smears</b><br>
Zebin Yu, Jianhu Li, Xiang Wen, Yingli Han, Penglei Jiang, Meng Zhu, Minmin Wang, Xiangli Gao, <br>
Dan Shen, Ting Zhang, Shuqi Zhao, Yijing Zhu, Jixiang Tong, Shuchong Yuan, HongHu Zhu, He Huang & Pengxu Qian <br>
<pre>
https://jhoonline.biomedcentral.com/articles/10.1186/s13045-023-01419-3
</pre>
