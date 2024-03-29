<h2>EfficientNetV2 Brain Tumor Classification (Updated: 2022/09/03)</h2>
<a href="#1">1 EfficientNetV2 Brain Tumor Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Install Python packages</a><br>
<a href="#2">2 Python classes for MonkeyPox Detection</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Brain Tumor Classification</a>
</h2>

 This is a simple Brain Tumor Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>
 The Brain Tumor dataset used here has been taken from the following web site:<br>
 <a href="https://github.com/sartajbhuvaji/brain-tumor-classification-dataset">brain-tumor-classificaiton-dataset</a>
 <br>
 We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<li>
2022/08/01: Modified <a href="./CustomDataset.py">CustomDataset</a> class to be able the ImageDataGeneration parameters 
from a data_generator.config file. 
</li>
<li>
2022/08/01: Modified <a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a> class to save the commandline training parameters
as a file to a model_dir. 
</li>
<li>
2022/08/03: Updated <a href="./projects/Brain-Tumor-Classification/data_generator.config">data_generator.config</a> to improve inference accuracy.
</li>
<li>
2022/08/04: Added <a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a> class to evaluate Testing dataset.
</li>
<li>
2022/09/03: Updated <a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a> class.
</li>
<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Brain-Tumor.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Brain-Tumor-Classification
        ├─Brain-Tumor-Images
        │  ├─Testing
        │  │  ├─glioma_tumor
        │  │  ├─meningioma_tumor
        │  │  ├─no_tumor
        │  │  └─pituitary_tumor
        │  └─Training
        │      ├─glioma_tumor
        │      ├─meningioma_tumor
        │      ├─no_tumor
        │      └─pituitary_tumor
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        └─test
</pre>
The images in test, Testing and Training folders have been taken from
 <a href="https://github.com/sartajbhuvaji/brain-tumor-classification-dataset">brain-tumor-classificaiton-dataset</a>.
<br><br>
The number of images in Brain-Tumor-Images dataset:<br>
<img src="./projects/Brain-Tumor-Classification/_Brain-Tumor-Images_.png" width="740 height="auto">
<br>
Sample images in Brain-Tumor-Classification/Brain-Tumor-Images/Training/glioma_tumor:<br>
<img src="./asset/Brain-Tumor-Classification_Training_glioma_tumor.png" width="820" height="auto">
<br>
<br>

Sample images in Brain-Tumor-Classification/Brain-Tumor-Images/Training/meningioma_tumor:<br>
<img src="./asset/Brain-Tumor-Classification_Training_meningioma_tumor.png" width="820" height="auto">
<br>
<br>

Sample images in Brain-Tumor-Classification/Brain-Tumor-Images/Training/no_tumor:<br>
<img src="./asset/Brain-Tumor-Classification_Training_no_tumor.png" width="820" height="auto">
<br>
<br>

Sample images in Brain-Tumor-Classification/Brain-Tumor-Images/Training/pituitary_tumor:<br>
<img src="./asset/Brain-Tumor-Classification_Training_pituitary_tumor.png" width="820" height="auto">
<br>
<br>

<h3>
<a id="#1.2">1.2 Install Python packages</a>
</h3>
<br>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Brain Tumor Classification</a>
</h2>
We have defined the following python classes to implement our Brain Tumor Classification.<br>

<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./TestDataset.py">TestDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train Brain Tumor Classification Model by using
 <a href="https://github.com/sartajbhuvaji/brain-tumor-classification-dataset">brain-tumor-classificaiton-dataset</a>.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Brain-Tumor-Classification
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our brain-tumor efficientnetv2 model 
by using <a href="./projects/Brain-Tumor-Classification/Brain-Tumor-Images/Training">Brain-Tumor-Classification/Brain-Tumor-Images/Training dataset</a>.<br>
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --num_classes=4 ^
  --image_size=384 ^
  --eval_image_size=480 ^
  --data_dir=./Brain-Tumor-Images/Training ^
  --model_dir=./models ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0002 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --num_epochs=50 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  
</pre>
, and data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = True
featurewise_std_normalization=False
samplewise_std_normalization  =True
zca_whitening                =False
rotation_range     = 6
horizontal_flip    = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.1
zoom_range         = [0.2, 2.0]
data_format        = "channels_last"

[validation]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = True
featurewise_std_normalization=False
samplewise_std_normalization  =True
zca_whitening                =False
rotation_range     = 6
horizontal_flip    = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.1
zoom_range         = [0.2, 2.0]
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Brain-Tumor-Classification/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Brain-Tumor-Classification/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Brain-Tumor-Classification_train_console_output_at_epoch_14_0905.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Brain-Tumor-Classification/eval/train_accuracies.png" width="740" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Brain-Tumor-Classification/eval/train_losses.png" width="740" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the brain tumors in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=480 ^
  --num_classes=4 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
glioma_tumor
meningioma_tumor
no_tumor
pituitary_tumor
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Brain-Tumor-Classification/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Brain-Tumor-Classification/Brain-Tumor-Images/Testing">Brain-Tumor-Images/Testing</a> taken from
 <a href="https://github.com/sartajbhuvaji/brain-tumor-classification-dataset">brain-tumor-classificaiton-dataset</a>.<br>
 <br>
Brain-Tumor-Classification/test:<br>
<img src="./asset/Brain-Tumor-Classification_test.png" width="820" height="auto"><br>
<br>
<br>
Sample images in Brain-Tumoer-Classification/test:<br>

glioma_tumor<br>
<img src="./projects/Brain-Tumor-Classification/test/glioma_tumor___image(8)_101.jpg"  width="400" height="auto"><br><br>
meningioma_tumor<br>
<img src="./projects/Brain-Tumor-Classification/test/meningioma_tumor___image(18)_104.jpg"  width="400" height="auto"><br><br>
no_tumor<br>
<img src="./projects/Brain-Tumor-Classification/test/no_tumor___image(4)_105.jpg"  width="400" height="auto"><br><br>
pituitary_tumor<br>
<img src="./projects/Brain-Tumor-Classification/test/pituitary_tumor___image(8)_106.jpg"  width="400" height="auto"><br><br>

<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Brain-Tumor-Classification/inference/inference.csv">inference result file</a>.
<br>
Inference console output:<br>
<img src="./asset/Brain-Tumor-Classification_infer_console_output_at_epoch_14_0905.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Brain-Tumor-Classification_inference_at_epoch_14_0905.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Brain-Tumor-Classification/Brain-Tumor-Images/Testing">
Brain-Tumor-Images/Testing dataset</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Brain-Tumor-Images/Testing ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --eval_image_size=480 ^
  --num_classes=4 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --debug=False 
</pre>


<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Brain-Tumor-Classification/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Brain-Tumor-Classification/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Brain-Tumor-Classification_evaluate_console_output_at_epoch_14_0905.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Brain-Tumor-Classification_classification_report_at_epoch_14_0904.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Brain-Tumor-Classification/evaluation/confusion_matrix.png" width="740" height="auto"><br>

