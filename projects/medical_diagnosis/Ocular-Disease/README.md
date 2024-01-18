<h2>EfficientNetV2-Ocular-Disease</h2>

 This is an experimental EfficientNetV2 Ocular Disease Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>
Please see also our first experiment <a href="https://github.com/atlan-antillia/EfficientNet-Ocular-Disease">
EfficientNet-Ocular-Disease
</a>
<br>
<h3>1. Dataset Citation</h3>
The image dataset used here has been taken from the following website:<br>
<br>
<b>Ocular Disease Intelligent Recognition ODIR-5K</b><br>
https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72
<br>
<br>
<b>Ocular Disease Recognition</b><br>
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
<br>
<br>
About this Data<br>
See also: https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72<br>
<pre>
Ocular Disease Intelligent Recognition (ODIR) is a structured ophthalmic database of 5,000 patients with age, color fundus photographs from left and right eyes and doctors' diagnostic keywords from doctors.

This dataset is meant to represent ‘‘real-life’’ set of patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China. In these institutions, fundus images are captured by various cameras in the market, such as Canon, Zeiss and Kowa, resulting into varied image resolutions.
Annotations were labeled by trained human readers with quality control management. They classify patient into eight labels including:

Normal (N),
Diabetes (D),
Glaucoma (G),
Cataract (C),
Age related Macular Degeneration (A),
Hypertension (H),
Pathological Myopia (M),
Other diseases/abnormalities (O)

License
License was not specified on source
</pre>
Splash Image
Image from <a href="https://pixabay.com/pt/users/matryx-15948447/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=5061291">Omni Matryx </a>
by Pixabay<br>


<h3>2. Download dataset</h3>
If you would like to train EfficientNetV2 Ocular-Disease Model by yourself,
please download the resampled dataset from the google drive 
<a href="https://drive.google.com/file/d/15PqVDySqLfMqAkwXXcVux4GlGfpg01gi/view?usp=sharing">Resampled_Ocular_Disease_Images.zip</a>
<br>
It contains the following test and train datasets.<br>
<pre>
Resampled_Ocular_Disease_Images
├─test
│  ├─A
│  ├─C
│  ├─D
│  ├─G
│  ├─H
│  ├─M
│  ├─N
│  └─O
└─train
    ├─A
    ├─C
    ├─D
    ├─G
    ├─H
    ├─M
    ├─N
    └─O
</pre>
The number of images in this Resampled_Ocular_Disease_Images is the following:<br>
<img src="./_Resampled_Ocular_Disease_Images_.png" width="740" height="auto"><br>
<br>
<br>
Resampled_Ocular_Disease_Images/train/A (Age related Macular Degeneration) :<br>
<img src="./asset/Ocular_Disease_train_A.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/C (Cataract):<br>
<img src="./asset/Ocular_Disease_train_C.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/D (Diabetes):<br>
<img src="./asset/Ocular_Disease_train_D.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/G (Glaucoma):<br>
<img src="./asset/Ocular_Disease_train_G.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/H (Hypertension):<br>
<img src="./asset/Ocular_Disease_train_H.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/M (Pathological Myopia):<br>
<img src="./asset/Ocular_Disease_train_M.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/N (Normal):<br>
<img src="./asset/Ocular_Disease_train_N.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/O (Other diseases/abnormalities):<br>
<img src="./asset/Ocular_Disease_train_O.png" width="840" height="auto">
<br>
<br>


<h3>
3. Train
</h3>
<h3>
3.1 Training script
</h3>
Please run the following bat file to train our Ocular Disease Classification efficientnetv2 model by using
<b>Resampled_Ocular_Disease_Images/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
rem 2024/01/18
python ../../../efficientnetv2/EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../../efficientnetv2/efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=512 ^
  --eval_image_size=512 ^
  --data_dir=./Resampled_Ocular_Disease_Images/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=False ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --num_epochs=50 ^
  --batch_size=2 ^
  --patience=10 ^
  --debug=True
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config
; 2024/01/18
[training]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 180
horizontal_flip    = True
vertical_flip      = True 
width_shift_range  = 0.2
height_shift_range = 0.2
shear_range        = 0.01
zoom_range         = [0.1, 2.0]
data_format        = "channels_last"
</pre>

<h3>
3.2 Training result
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./eval/train_accuracies.csv">train_accuracies</a>
and <a href="./eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Ocular_Disease_train_at_epoch_33_0118.png" width="740" height="auto"><br>
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
Please run the following bat file to infer the Ocular-Disease in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
rem 2024/01/18
python ../../../efficientnetv2/EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=512 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
A
C
D
G
H
M
N
O
</pre>
<br>
<h3>
4.2 Sample test images
</h3>

Sample test images generated by <a href="./create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./Resampled_Ocular_Disease_Images/test">Resampled_Ocular_Disease_Imagess/test</a>.
<br>
<img src="./asset/Ocular_Disease_test.png" width="840" height="auto"><br>

<h3>
4.3 Inference result
</h3>
This inference command will generate <a href="./inference/inference.csv">inference result file</a>.
<br>
<br>
Inference console output:<br>
<img src="./asset/Ocular_Disease_infer_at_epoch_33_0118.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Ocular_Disease_inference_csv_at_epoch_33_0118.png" width="740" height="auto"><br>
<br>
<h3>
5. Evaluation
</h3>
<h3>
<5.1 Evaluation script
</h3>
Please run the following bat file to evaluate <a href="./Resampled_Ocular_Disease_Images/test">
Resampled_Ocular_Disease_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>

</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./evaluation/classification_report.csv">a classification report</a>
 and <a href="./evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Ocular_Disease_evaluate_at_epoch_33_0118.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Ocular_Disease_classificaiton_report_at_epoch_33_0118.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./evaluation/confusion_matrix.png" width="740" height="auto"><br>


<br>
<h3>
References
</h3>
<b>1. Ocular Disease Intelligent Recognition ODIR-5K</b><br>
<pre>
https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72
</pre>

<b>2. Ocular Disease Recognition</b><br>
<pre>
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
</pre>
