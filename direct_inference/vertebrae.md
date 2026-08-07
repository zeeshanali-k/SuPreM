<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Apply to Vertebrae Segmentation</h3>
<p align="center">
    <a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
    <a href='https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> 
    <a href='document/promotion_slides.pdf'><img src='https://img.shields.io/badge/Slides-PDF-orange'></a> 
    <a href='document/dom_wse_poster.pdf'><img src='https://img.shields.io/badge/Poster-PDF-blue'></a> 
    <a href='https://www.cs.jhu.edu/news/ai-and-radiologists-unite-to-map-the-abdomen/'><img src='https://img.shields.io/badge/WSE-News-yellow'></a>
    <br/>
    <a href="https://github.com/MrGiovanni/SuPreM"><img src="https://img.shields.io/github/stars/MrGiovanni/SuPreM?style=social" /></a>
    <a href="https://twitter.com/bodymaps317"><img src="https://img.shields.io/twitter/follow/BodyMaps" alt="Follow on Twitter" /></a>
</p>

##### 0. Download CT scans

```bash
wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
tar -xzvf AbdomenAtlasDemo.tar.gz
```

The CT scans are organized in such a way:

```
AbdomenAtlasDemo
    ├── BDMAP_00000006
    │   └── ct.nii.gz
    ├── BDMAP_00000031
    │   └── ct.nii.gz
```

##### 1. Clone and setup the GitHub repository
```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/direct_inference/pretrained_checkpoints/
wget http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
cd ..
```

<details>
<summary style="margin-left: 25px;">[Option] if you get certificate issues when using wget</summary>
<div style="margin-left: 25px;">

```bash
wget --no-check-certificate http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
```

</div>
</details>


##### 2. Google Colab (Python 3.12)

Start a Colab notebook and select **Runtime > Change runtime type > T4 GPU** (or another NVIDIA GPU). Colab already provides a CUDA-enabled PyTorch and torchvision build; do not replace them with the legacy PyTorch 1.11 packages and do not run the repository's legacy `requirements.txt` in Colab.

Verify the runtime before installing the remaining dependencies:

```python
import sys
import torch

print(sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available(), "Select a GPU runtime before running inference."
```

The Python version printed by the current Colab runtime should be Python 3.12.

Install the Python 3.12-compatible dependencies while keeping Colab's PyTorch installation:

```python
%pip install -q "monai==1.5.2" nibabel h5py connected-components-3d fastremap
```

Restart the runtime after installation (**Runtime > Restart session**). After reconnecting, mount Google Drive and verify the environment:

```python
from google.colab import drive
drive.mount("/content/drive")

import sys
import torch
import monai

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("MONAI:", monai.__version__)
assert torch.cuda.is_available(), "A CUDA-enabled NVIDIA GPU is required."
```

Clone the repository and download the vertebrae checkpoint:

```bash
%cd /content
!git clone https://github.com/MrGiovanni/SuPreM
%cd /content/SuPreM/direct_inference/pretrained_checkpoints
!wget http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
%cd /content/SuPreM/direct_inference
```

As a final import check, run:

```python
import inference
print("SuPreM inference imports successfully")
```

Place the CT data in Google Drive with one directory per case and a `ct.nii.gz` inside each directory:

```text
/content/drive/MyDrive/AbdomenAtlasDemo/
├── BDMAP_00000006/
│   └── ct.nii.gz
└── BDMAP_00000031/
    └── ct.nii.gz
```

Run vertebrae inference with a single-process data loader, which is the most reliable setting in Colab:

```bash
%cd /content/SuPreM/direct_inference
!python -W ignore inference.py \
    --save_dir /content/drive/MyDrive/AbdomenAtlasDemoPredict \
    --checkpoint ./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth \
    --data_root_path /content/drive/MyDrive/AbdomenAtlasDemo \
    --customize \
    --num_workers 0 \
    --device cuda:0
```

The output directory will contain `combined_labels.nii.gz` and the individual vertebra masks under `segmentations/` for every processed case.

##### 3. Legacy Linux environment (Python 3.9 / MONAI 0.9)

Use this environment on Linux systems where the original Python 3.9 and CUDA 11.3 stack is still available:

```bash
conda create -n suprem python=3.9
source activate suprem
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### 4. Generate vertebrae masks by the AI (legacy environment)

```bash
datarootpath=/path/to/your/AbdomenAtlasDemo # NEED MODIFICATION!!!

pretrainpath=./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=./AbdomenAtlasDemoPredict

python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize
```

The vertebrae masks will be saved as
```
AbdomenAtlasDemoPredict
    ├── BDMAP_00000006
    │   ├── combined_labels.nii.gz
    │   └── segmentations
    │       ├── vertebrae_L5.nii.gz
    │       ├── vertebrae_L4.nii.gz
    │       ├── ...
    │       ├── vertebrae_L1.nii.gz
    │       ├── vertebrae_T12.nii.gz
    │       ├── vertebrae_T11.nii.gz
    │       ├── ...
    │       ├── vertebrae_T1.nii.gz
    │       ├── vertebrae_C7.nii.gz
    │       ├── vertebrae_C6.nii.gz
    │       ├── ...
    │       └── vertebrae_C1.nii.gz
    ├── BDMAP_00000031
    │   ├── combined_labels.nii.gz
    │   └── segmentations
    │       ├── vertebrae_L5.nii.gz
    │       ├── vertebrae_L4.nii.gz
    │       ├── ...
    │       ├── vertebrae_L1.nii.gz
    │       ├── vertebrae_T12.nii.gz
    │       ├── vertebrae_T11.nii.gz
    │       ├── ...
    │       ├── vertebrae_T1.nii.gz
    │       ├── vertebrae_C7.nii.gz
    │       ├── vertebrae_C6.nii.gz
    │       ├── ...
    │       └── vertebrae_C1.nii.gz
```

##### 5. [Important!] Postprocess vertebrae masks

Check the AI-predicted vertebrae masks (`combined_labels.nii.gz`) and the original CT scans (`ct.nii.gz`) using software such as [ITK-SNAP](https://www.itksnap.org/pmwiki/pmwiki.php). If you look closely at the AI-predicted masks, you will see many errors. Please design an automatic postprocessing to reduce these errors as many as you can. The postprocessing should be formatted in a separated python file `postprocessing_vertebrae.py`.

This is an illustration of vertebrae (and rib) label refinement.

![Refinement](https://github.com/MrGiovanni/SuPreM/blob/main/document/LetsSegmentVertebrae.png)
</div>

To identify the errors, you will need some knowledge about vertebrae in the human body as follow.

![Vertebral anatomy](https://i0.wp.com/aneskey.com/wp-content/uploads/2023/08/f01-01-9780323882262.jpg)
</div>

##### Related Work

[1] Liu, Junqi, Dongli He, Wenxuan Li, Ningyu Wang, Alan L. Yuille, and Zongwei Zhou. ["ShapeKit."](https://www.cs.jhu.edu/~zongwei/publication/liu2025shapekit.pdf) In International Workshop on Shape in Medical Imaging, pp. 44-58. Cham: Springer Nature Switzerland, 2025. 

[2] Meng, Di, Edmond Boyer, and Sergi Pujades. ["Vertebrae localization, segmentation and identification using a graph optimization and an anatomic consistency cycle."](https://gitlab.inria.fr/spine/vertebrae_segmentation) Computerized Medical Imaging and Graphics 107 (2023): 102235.

[3] Jaus, Alexander, Constantin Seibold, Kelsey Hermann, Negar Shahamiri, Alexandra Walter, Kristina Giske, Johannes Haubold, Jens Kleesiek, and Rainer Stiefelhagen. ["Towards unifying anatomy segmentation: Automated generation of a full-body ct dataset."](https://github.com/alexanderjaus/AtlasDataset) In 2024 IEEE International Conference on Image Processing (ICIP), pp. 41-47. IEEE, 2024.

[4] Seng, Magdalena, Jakob Wasserthal, Michael Bach, Ashraya Kumar Indrakanti, Markus Obmann, Christian Meier, Dorothee Harder, Elmar M. Merkle, Martin Segeroth, and Hanns-Christian Breit. ["AI-Based Opportunistic CT Risk Assessment Using TotalSegmentator in Osteoporotic Vertebral Fractures."](https://link.springer.com/content/pdf/10.1007/s10278-026-02049-0.pdf) Journal of Imaging Informatics in Medicine (2026): 1-12.
