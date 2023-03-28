# Face Parsing in synthetic/real datasets with popular models


<li>Segmentaion experiments using DepplabV3+, Unet, FPN, and MobilenetV2
<li>Trained/tested on Helen, LaPa, and Microsoft's fully synthetic face DS
<li>Transfer learning tested on iBugMask300
  
 ## Usage

Research results can be found in /reasearch
* "ibug.ipynb" -> Transfer learning handler
* "helen.ipynb" -> Training and testing of models on Helen
* "label_adapter_fcn.ipynb" -> Training of FCN (to in-the-wild domain) Label Adapter
* "label_adapter_unet.ipynb" -> Training of UNET (to in-the-wild domain) Label Adapter
* "label_test_script.py" -> FCN and helper functions
* "lapa.ipynb" -> Training and testing of models on LaPa
* "mix.ipynb" -> Training and testing of models of domain mixed dataset (50:50) Synthetic:Real
* "wood.ipynb" -> Training and testing of models of purely synthetic face data

Unfinished and outdated scripts can be found in /outdated_src </br>

If you wish to repeat these experiments, you must ensure that the file paths match the ones used in my notebooks. The model weights/saved models used in my experiemnts can be found here: https://drive.google.com/file/d/1PPOSVSgwNE61uy9Zk5UgiXfTZckxgPEK/view?usp=sharing (They are not included in this repo due to their large file size). If using the saved models, please ensure you are using the subsquent model for naked predictions, and then adapting those predictions with the corresponding label adapter (as described in my paper).


# Training Flow
  
![img](report/train_flow2.png)
  
# Testing Flow

![img](report/test_flow4.png)
  
# Results

### Final Results (iBugMask Test Sample)
![img](report/ibug_test_table_short.png)

### Full Results (iBugMask Test Sample)
![img](report/ibug_test_table_full.png)

### Helen Test Results (Models trained on Helen)
![img](report/helen_test_table.png)

### LaPa Test Results (Models trained on LaPa)
![img](report/lapa_test_table.png)


## License

[MIT](https://choosealicense.com/licenses/mit/)
