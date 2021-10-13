## Context-aware and Time-aware Attention-based Model for Disease Risk Prediction with Interpretability
---
The rpo contains the PyTorch implementation of the paper for demo purposes. The original dataset is a private dataset, you can refer to below to prepare your own dataset.



## Getting Started
---
### 1. Prepare Your Dataset
Please construct your dataset as the following strutures:  
{   
 &emsp; 'train_seqs': [],  
 &emsp; 'train_labels': [],  
 &emsp; 'train_days': [],  
 &emsp; 'valid_seqs': [],  
 &emsp; 'valid_labels': []  
 &emsp; 'valid_days': []  
 &emsp; 'test_seqs': []  
 &emsp; 'test_labels': []  
 &emsp; 'test_days': []  
}  
'\*_seqs' is a list of patients’ records, each patient is a list of visits, and each visit is a list of medical codes. For example, ''train_seqs': [[[1, 2, 3], [2, 3]], &emsp;[[2, 3], [3], [4]]]' means that the training set contains two patients, the first patient has two visits, and there are two medical code (2 and 3) assigned to his/her second visit;    
 '\*_labels' is a list of ground truth labels, each element is 0 or 1, where 0 indicates the patient will not have a certain disease in the future, 1 denotes the patient will be diagnosed with the disease. For example, ''train_labels':[0, 1]' means the second patient will be diagnosed with the disease in the future;  
 '\*_days' is a list of patients' visit time. For example, ''train_days': [[0, 5], &emsp;[0, 3, 10]]' means that there are two patient in the training set, the second patient has three visit, and the corresponding visit time is 0, 3 and 10.
### 2. Training
Once the data is prepared, now you are ready to train the model by using the following command.  

``
python main.py --data_root your_data_path --lr 1e-4 --weight_decay 1e-3 --dim 128 --epochs 50 --save_model --save_dir your_save_path
``  
The trained model will be saved in ``your_save_path``.
The best hyper-parameters combination may be different for different dataset and need to be seached.
### 3. Interpreting
To get the explaination of the corresponding model for all samples in the test set, you can execute the following command.
``
python interpret.py --data_root your_data_path --model_path your_save_path/your_trained_model_name --save_path your_save_exp_path
``  
The explainations of all samples will be saved in your_save_exp_path.


## Final Words
---
That's all for now and hope this repo is useful to your research. For any questions, please create an issue and we will get back to you as soon as possible.



