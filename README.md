<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_detectron2_deeplabv3plus/main/icons/detectron2.png" alt="Algorithm icon">
  <h1 align="center">train_detectron2_deeplabv3plus</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_detectron2_deeplabv3plus">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_detectron2_deeplabv3plus">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_detectron2_deeplabv3plus/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_detectron2_deeplabv3plus.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

 Train DeepLabV3+ model for semantic segmentation. Implementation from Detectron2 (Meta Research). 

 ![Deeplabv3+ illustration](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/samples/city_1_overlay.png?raw=true)



## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add data loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add train algorithm 
train = wf.add_task(name="train_detectron2_deeplabv3plus", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters


- **epochs** (int) - default '1000': Number of complete passes through the training dataset.
- **max_iter** (int) - default '1000': Maximum number of iterations. 
- **classes** (int) - default '2': Number of classes
- **input_width** (int) - default '800': Size width of the input image.
- **input_height** (int) - default '800': Size height of the input image.
- **batch_size** (int) - default '4': Number of samples processed before the model is updated.
- **learning_rate** (float) - default '0.02': Step size at which the model's parameters are updated during training.
- **eval_period** (int) - default '100: Interval between evaluations.  
- **dataset_split_ratio** (float) – default '90': Divide the dataset into train and evaluation sets ]0, 100[.
- **output_folder** (str, *optional*): path to where the model will be saved. 
- **config_file** (str, *optional*): path to the training config file .yaml. 
- **warmupFactor** (float) - default '0.001': 
- **warmupIters** (int) - default '200': 
- **polyLRFactor** (float) - default '0.9': 
- **polyLRConstantFactor** (float) - default '0.0': 
- **resnetDepth** (int) - default '50': 
- **batchNorm** (str) - default 'BN': 
- **early_stopping** (bool) - default 'False': 
- **patience** (int) - default '10': 
- **numGPU** (int) - default '1': 

**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add data loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add train algorithm 
train = wf.add_task(name="train_detectron2_deeplabv3plus", auto_connect=True)
train.set_parameters({
    "batch_size": "4",
    "epochs": "50",
    "learning_rate": "0.02",
    "dataset_split_ratio": "80",
    "max_iter": "1000",
    "classes": "2",
    "warmupFactor": "0.001",
    "warmupIters": "200",
    "polyLRFactor": "0.9",
    "batchNorm": "BN",
    "early_stopping": "False"
}) 

# Launch your training on your data
wf.run()
```
