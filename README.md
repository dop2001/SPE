## 🚀 Introduction
Segmentation Performance Evaluator(SPE) is used to estimat segmentation
models’ performance on unlabeled data. It is adaptable
to various evaluation metrics and model architectures.

## 🚢 How to use
1. Create virtual environment and install requirement package.
    ```shell
    # Optional
    conda create -n SPE python=3.8
    conda activate SPE
    ```
    ```shell
    git clone https://github.com/dop2001/SPE.git
    cd SPE
    pip install -r requirements.txt
    ```
2. Download the dataset ([Baidu Drive](https://pan.baidu.com/s/1ZOnyP3N45GTD1txuNmSFlQ?pwd=8888) | [Google Drive](https://drive.google.com/drive/folders/1MQA-SlSFbwVF9nr7qAV89ACCO-B4sfrF?usp=sharing)) and place it in the `datasets` directory.
3. You can modify some SPE configurations in the `configs` folder.
4. Build SPE framework for every dataset.
    ```shell
    python spe_main.py
    ```
5. The fitting curve results will be generated in the `result/curve` directory

## 🥳 Citation
If you find our work or any of our materials useful, please cite our paper:
```
@inproceedings{Zou2025SPE,
  title={Performance Estimation for Supervised Medical Image Segmentation Models on Unlabeled Data Using UniverSeg},
  author={Jingchen Zou, Jianqiang Li, Gabriel Jimenez, Qing Zhao, Daniel Racoceanu, Matias Cosarinsky, Enzo Ferrante, and Guanghui Fu},
  booktitle={MICCAI 2025},
  year={2025}
}
```