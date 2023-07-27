# License-Plate-Recoginition(LPR)
## 模型介绍
LPR是一个基于深度学习技术的车牌识别模型，主要识别目标是自然场景的车牌图像。
## 模型结构
模型采用LPRNet，模型结构主要包含三部分：一个轻量级CNN主干网络、基于预定位置的字符分类头部、基于贪婪算法的序列解码。此外模型使用CTC Loss和RMSprop优化器。参考论文[LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/pdf/1806.10447v1.pdf)。
## 数据集
推荐使用一个车牌数据集[CCPD](https://github.com/detectRecog/CCPD "CCPD官网GitHub")，该数据集由中科大收集，可用于车牌的检测与识别。

我们提供了一个脚本cut_ccpd.py用于剪裁出CCPD数据集中的车牌位置，以便用于LPR模型的训练，在cut_ccpd.py中修改img_path和save_path即可，分别是CCPD数据集中ccpd_base文件夹的路径和剪裁出的图像保存路径。LPR用于训练的数据文件名就是图像的标签。**数据集使用固定的大小94x24。** 使用方法：

    python cut_ccpd.py \
        --ccpdpath CCPD数据集下ccpd_base文件夹路径 \
        --savepath 保存切割图像的路径 
## 训练及推理
### 环境配置
在[光源](https://www.sourcefind.cn/#/service-details)可拉取训练以及推理的docker镜像，LPR推荐的镜像如下：
* 训练镜像：docker pull image.sourcefind.cn:5000/dcu/admin/base/vscode-pytorch:1.10.0-centos7.6-dtk-22.10-py37-patch4
* 推理镜像：docker pull image.sourcefind.cn:5000/dcu/admin/base/custom:migraphx2.5.0_centos7.6-dtk-22.10.1

在[光合开发者社区](https://cancon.hpccube.com:65024/4/main/)可下载Migraphx和ONNXruntime安装包，python依赖安装：

    pip install -r requirement.txt
### 训练与Fine-tunning
LPR模型的训练程序是train.py，初次训练模型使用以下命令：

    python train.py \
        --train_img_dirs 训练集文件夹路径 \
        --test_img_dirs 验证集文件夹路径

Fine-tunning使用以下命令：

    python train.py \
        --train_img_dirs 训练集文件夹路径 \
        --test_img_dirs 验证集文件夹路径 \
        --pretrained_model 预训练模型路径 \
        --resume_epoch Fine-tuning训练的起始epoch \  
        --max_epoch 训练的最大epoch

Fine-tuning时只训练从起始epoch到最大epoch。
### 预训练模型
在model文件夹下我们提供了一个预训练模型以及对应的onnx模型和mxr模型，以下是相关子目录的介绍：

    LPR
    ├── imgs #测试图像
    ├── model
    │   ├── lprnet.pth #基于pytorch框架训练出的LPR预训练模型 
    │   ├── LPRNet.onnx #由lprnet.pth转换的onnx模型
    └── └── LPRNet.mxr #用migraphx编译LPRNet.onnx得到的离线推理模型
### 测试
LPR模型用test.py对训练出的模型进行测试，使用方法如下：

    python test.py \
        --model 需要测试的pth模型路径 \
        --imgpath 测试集路径(文件夹或图像皆可)  \
        --batch_size 测试时的batch size大小 \
        --export_onnx True/False(该参数用于选择是否需要将pth模型转为onnx模型) \
        --dynamic True/False(该参数用于选择onnx模型是否使用动态的batch size)

### 推理
我们分别提供了基于ONNXruntime(ORT)和Migraphx的推理脚本，版本依赖：
* ONNXRuntime(DCU版本) >= 1.14.0
* Migraphx(DCU版本) >= 2.5.0
#### ORT
LPRNet_ORT_infer.py是基于ORT的的推理脚本，使用方法：

    python LPRNet_ORT_infer.py \
        --model onnx模型路径 \
        --imgpath 数据路径(文件夹或图像皆可)
#### Migraphx
LPRNet_migraphx_infer.py是基于Migraphx的推理脚本，使用需安装好Migraphx，支持onnx模型和mxr模型推理，mxr模型是migraphx将onnx模型保存成的离线推理引擎，初次使用onnx模型会保存对应的mxr模型。使用方法：

    python LPRNet_migraphx_infer.py \
        --model mxr/onnx模型路径 \
        --imgpath 数据路径(文件夹或图像皆可) \
        --savepath mxr模型的保存路径以及模型名称

## 性能和准确率数据
测试数据使用的是[LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch/tree/master/data/test)，使用的加速卡是DCU Z100。**mxr格式的模型是migraphx创建的onnx模型的离线引擎。**

| Engine | Model Path| Model Format | Accuracy(%) |
| :------: | :------: | :------: | :------: |
| ONNXRuntime | model/LPRNet.onnx | onnx | 92.7 |
| Migraphx | model/LPRNet.onnx | onnx | 92.7 |
| Migraphx | model/LPRNet.mxr | mxr | 92.7 |
## 源码仓库及问题反馈
* https://developer.hpccube.com/codes/modelzoo/lpr
## 参考
* https://github.com/sirius-ai/LPRNet_Pytorch
* https://github.com/qzpzd/license-plate-detect-recoginition
