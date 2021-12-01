# HOI-Net
This is the Pytorch implementation of HOI-Net：Wang J, Li N, Luo Z, et al. High-Order-Interaction for weakly supervised Fine-Grained Visual Categorization[J]. Neurocomputing, 2021, 464: 27-36. which can download from (https://www.sciencedirect.com/science/article/abs/pii/S0925231221013060)
## Abstract
Fine-Grained Visual Categorization (FGVC) is a challenging task due to the large intra-subcategory and small inter-subcategory variances. Recent studies tackle this task through a weakly supervised manner without using the part annotation from the experts. Of those, methods based on bilinear pooling are one of the main categories for computing the interaction between deep features and have shown high effectiveness. However, these methods mainly focus on the correlation within one specific layer but largely ignore the high interactions between multiple layers. In this study, we argue that considering the high interaction between the features from multiple layers can help to learn more distinguishing fine-grained features. To this end, we propose a High-Order-Interaction (HOI) method for FGVC. In our HOI, an efficient cross-layer trilinear pooling is introduced to calculate the third-order interaction between three different layers. Third-order interactions of different combinations are then fused to form the final representation. HOI can produce more discriminative representations and be readily integrated with the two popular techniques, attention mechanism and triplet loss, to obtain superposed improvement. Extensive experiments conducted on four FGVC datasets show the great superiority of our method over bilinear-based methods and demonstrate that the proposed method achieves the state of the art.
## Figure1 
  ![image](https://user-images.githubusercontent.com/19604312/144156945-f9cb7c2a-453f-4c82-9019-8c5753e9aa5a.png)
   Figure 1: (a) The challenge of FGVC on the CUB-200-2011 dataset. The bird samples of the same subcategory may have large differences, while the bird samples of different  subcategories may have great similarities. (b) Effectiveness of the proposed High-Order-Interaction (HOI). Ordinary CNN network can not find discriminative regions without part annotations and thus fails to recognize samples from similar subcategories. HOI can activate important parts and thus accurately distinguishes samples from similar subcategories.
## Figure2
   ![image](https://user-images.githubusercontent.com/19604312/144157195-8683b401-da5c-4d11-a7f6-3129599f2b07.png)
Figure 2: The overall framework of our proposed High-Order-Interaction (HOI) method for fine-grained visual categorization. The model mainly consists of two levels: (1) Feature Attention Pyramid, and (2) High Order Interaction. In level 1, we extract features from multiple different layers and use the attention mechanism to improve discrimination. In level 2, several features obtained by using the cross-layer trilinear pooling are concatenated together to form the final feature representation of a given image. Finally, the cross-entropy and the triplet loss function are jointly used to optimize the model.
## Compatibility
* The code is tested using Pytorch  1.9.0+cu111 under Ubuntu 18.04 with Python 3.6.9. CPU: 64  Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz  GPU:3090
* ![image](https://user-images.githubusercontent.com/19604312/144155729-e5e17b90-7ba0-4700-94d3-709bfd62a94e.png)
## Preparing FGVC CUB-200-2011 Datasets
   You can download FGVC CUB-200-2011 Datasets dataset from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html or https://pan.baidu.com/s/1JQxa3DYDrM329skC73kbzQ
## Running testing
    Python test.py 
## Results
You can download HOI-Net(resnet50) and HOI-Net(resnet101) model from link：链接：https://pan.baidu.com/s/1Nvk3J4gpNzeCe5LbqXfCqQ  code：abcd and https://pan.baidu.com/s/1Itdf4-fCRtjkkF3p5_C_tA  code：abcd  and our models achieves the following performance on FGVC CUB-200-2011 Datasets. 
## performance on FGVC CUB-200-2011 Datasets

  ![image](https://user-images.githubusercontent.com/19604312/144165257-3c1ced33-b90a-4faa-9ec0-f26ae08a8c6b.png)

