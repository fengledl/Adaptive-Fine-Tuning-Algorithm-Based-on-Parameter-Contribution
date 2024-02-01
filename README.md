# Adaptive Fine-Tuning Algorithm Based on Parameter Contribution
![Pipeline](https://github.com/fengledl/Adaptive-Fine-Tuning-Algorithm-Based-on-Parameter-Contribution/assets/152671236/07edaa7d-b09a-49e2-b253-b66b084ad6f9)
# Abstract
Fine-tuning is an important transfer learning technique that has achieved significant success in various tasks lacking training data, and requires only a small number of training epochs to achieve satisfactory results. However, with the increasing complexity of the model scale and structure, designing appropriate fine-tuning schemes for specific target tasks becomes increasingly difficult. In this paper, a contribution measure criterion is used to quantify the importance of the pre-trained model parameters to the target task, providing a basis for selecting fine-tuning parameters. In addition, we find that the fine-tuning ratio vary depending on the specific target task. Therefore, we propose an adaptive fine-tuning ratio search strategy to search the appropriate fine-tuning ratio for the given target task. Based on the above strategy, we propose an adaptive fine-tuning algorithm based on parameter contribution to customize the fine-tuning scheme for the target task. The experimental results show that the proposed algorithm can effectively quantify the contribution of model parameters, and our algorithm can adaptively adjust the fine-tuning ratio for the target task. Furthermore, our algorithm achieves state-of-the-art performance on seven publicly available visual classification datasets widely used in transfer learning.
# Requirements
Python (3.8)

PyTorch (2.0.1)

# Dataset download link
The following information was supplied regarding data availability: The public datasets are available at:
   
1. MIT Indoor–https://web.mit.edu/torralba/www/indoor.html
   
2. Stanford Dogs–http://vision.stanford.edu/aditya86/ImageNetDogs/
   
3. Caltech 256-30–https://data.caltech.edu/records/nyy15-4j048

4. Caltech 256-60–https://data.caltech.edu/records/nyy15-4j048
   
5. Aircraft–https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
  
6. UCF-101–https://www.robots.ox.ac.uk/~vgg/decathlon/#download
    
7. Omniglot–https://www.robots.ox.ac.uk/~vgg/decathlon/#download
