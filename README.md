# VRDL_HW01
Selected Topics in Visual Recognition using Deep Learning  
Introduction: Bird images classification
6,033 bird images belonging to 200 bird species, e.g., tree sparrow or mockingbird (training: 3,000, test: 3,033)
competition link : https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07


Requirements
---
To install requirements:
```
pandas                             1.2.0
numpy                              1.18.5
torch                              1.8.1+cu101
torchvision                        0.9.1+cu101
```
Folder structure
---
```
Root/
   
├──data
    │
    ├── classes.txt
    │    ├ 001.Black_footed_Albatross       # class name with the number
    │    ├ 002.Laysan_Albatross
    │    ├ ...
    ├── testing_img_order.txt               # the testing images id you load to generate the answer.txt
    │    ├ 4282.jpg
    │    ├ 1704.jpg
    │    ├ ...
    ├── training_images                     # training image data 
    │    ├ 0003.jpg                         
    │    ├ 0008.jpg         
    │    ├ ...
    ├── testing_images                      # testing image data      
    │    ├ 0001.jpg                         
    │    ├ 0002.jpg
    │    ├ ...
├──model_saved                             # use 5 fold method saving the every best valid accuracy in each fold       
├──src                                     # some source code inside 
├──Inference.py                            # inference your testing data, generate the answer
├──main_1_train.py                         # train your model
└──README.md
```


Pre-trained Models
---
You can download pretrained models in model_saved:
eg. ep=018-acc=0.7467.hdf5
This name show that the best validation accuracy 0.7467 in the epoch 18. 
My model trained and the model be named by **** Mini-batch size 20. optimizer is AdamW with learning rate = 1e-4, and the learning schedule = 1e-4 with 


Train
---

```
python main_1_train.py
```


Inference
---
```
python Inference.py
```


