rem 1_train.bat
rem 2024/01/14
rem optimizer=adam
python ../../../efficientnetv2/EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../../efficientnetv2/efficientnetv2-m/model ^
  --optimizer=adam ^
  --image_size=480 ^
  --eval_image_size=480 ^
  --data_dir=./Osteosarcoma_Images/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=False ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.00001 ^
  --clipvalue=0.2 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --num_epochs=100 ^
  --batch_size=2 ^
  --patience=10 ^
  --debug=True  


