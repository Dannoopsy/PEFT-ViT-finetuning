# PEFT-ViT-finetuning
Исследование методов Parameter-Efficient Fine-Tuning для дообучения ViT бэкбонов


для запуска

1. docker build --tag "new" -f ./Dockerfile .
2. docker run --runtime=nvidia -v YOUR_PATH/PEFT-ViT-finetuning/:/home/PEFT-ViT-finetuning -it peft /bin/bash
3. pip install -e .
4. pip install -e ".[train]"
5. python ./finetuning/main.py peft=PEFT_OPTION data=DATASET_NAME
   
   list of possible options for PEFT_OPTION: lora16, lora32, ft, freeze
   list of possible options for DATASET_NAME: cifar10, tobacco3482, rvl_cdip
