## 项目概述
本仓库包含一系列实验代码，重点在于对不同模型、任务和数据集应用**一种或多种**不同的模型压缩技术。主要目标是评估每种压缩方法在不同情境下的有效性和效率。

### 参与者须知
参与者需要对预定义的模型、任务和数据集应用压缩算法，记录每项任务的操作时间和每种算法的学习过程、时间、偏好等，最终对这些方法产生自己的评价。

### 实验流程说明：
在这项实验中，您将被要求完成**三次问卷**。这些问卷分别在实验的不同阶段进行：
实验前：您将填写第一份问卷，了解您对特定模型压缩算法的初始知识和态度。
学习后：在您学习了相关的基础知识和模型知识之后，您将填写第二份问卷，以评估学习效果。
实验后：在实际操作实验完成后，您将填写第三份问卷，评价您对不同压缩算法的体验和看法。

### 实验指导：
为确保实验的有效性，请遵循以下步骤：

1. 先学习后实操：在开始实验操作之前，请确保您已经完成了关于模型压缩算法的学习部分。
2. 按顺序进行：请在实验的每个阶段严格按照指示进行。这有助于我们准确收集和分析数据。
3. 诚实回答：我们鼓励您根据自己的真实体验和看法回答问题。您的诚实反馈对于我们的研究至关重要。

### 压缩技术
研究的四种压缩算法包括：
1. 剪枝 Pruning https://zhuanlan.zhihu.com/p/609126518?utm_id=0
2. 量化 Quantization https://zhuanlan.zhihu.com/p/619914824
3. 知识蒸馏 Knowledge Distillation https://zhuanlan.zhihu.com/p/258390817?utm_id=0

### 方法论
- **任务概览**：下表为本实验需要实现压缩技术的模型、算法、数据集。实验需要实现计算机视觉和自然语言处理任务之一（将会指定给实验者）。

| 任务 (Task)             | 数据集 (Dataset)                | 模型 (Model)            |
|-------------------------|---------------------------------|-------------------------|
| Image Classification    | CIFAR10                         | ResNet                  |
| Text Classification     | GLUE                            | bert-large-uncased      |

- **压缩应用**：将单一或多种压缩方法种应用于被分配的任务，并记录其对性能的影响。
- **时间记录**：参与者应详细记录在每项任务和学习一种或多种压缩方法过程中投入的时间。
- **个人见解**：根据实验结果和个人体验，反思并记录偏好的压缩技术。

### 指南
- 以下为推荐代码，但参与者可以自由选择其他开源代码、工具或自己编写代码完成指定压缩任务。
- nlp模型和数据集可能需要借助hugging face镜像网站下载：https://hf-mirror.com/ 。若租用autodl服务器，可直接从公共盘查找部分模型和数据集，unzip到自己的container路径。
- 每个任务可以使用一种或多种压缩技术进行实验。 
- 实验者需要记录每种算法的学习过程、时间、偏好，建议建立一个本次实验的文档，包括详细的实验设置、执行过程和观察记录、学习时间等。

### 压缩算法推荐代码
- 剪枝
   - https://github.com/VainF/Torch-Pruning [CVPR 2023] Towards Any Structural Pruning; LLMs / Diffusion / Transformers / YOLOv8 / CNNs
   - https://github.com/princeton-nlp/CoFiPruning CoFiPruning: Structured Pruning Learns Compact and Accurate Models
   - https://github.com/huggingface/nn_pruning Prune a model while finetuning or training.

- 量化 Quantization
    - https://github.com/ctribes/cifar10-resnet18-pytorch-quantization Tests on cifar10 with Resnet18 network using quantization from a full precision checkpoint
    - https://github.com/pytorch/tutorials/blob/main/advanced_source/dynamic_quantization_tutorial.py Dynamic Quantization on an LSTM word language model
    
          tutorial: https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html#beta-dynamic-quantization-on-an-lstm-word-language-model

 
- 知识蒸馏 Know Distillation
    - https://github.com/haitongli/knowledge-distillation-pytorch A PyTorch implementation for exploring deep and shallow knowledge distillation (KD) experiments with flexibility
    - https://huggingface.co/distilbert-base-uncased BERT Knowledge Distillation: distilbert-base-uncased
