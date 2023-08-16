# ResNet Quantization: Faster Inference with Post-Training Quantization
Implemented and executed post-training static quantization techniques on a ResNet architecture to optimize model performance, reduce memory footprint, and enhance inference speed.

#### For Resnet-18 Model took ref. from:
 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
 
## Post-Training Static Quantization for ResNet18

*This repository demonstrates the application of state-of-the-art static quantization techniques to the ResNet18 architecture using PyTorch. The goal is to reduce the model's memory footprint and increase inference speed while maintaining acceptable accuracy. Please note that static quantization is currently supported for CPUs, so GPUs (CUDA) are used solely for training, while the model is tested on a CPU.*



Employed state-of-the-art static quantization methods to convert full-precision weights and activations into lower bit-width representations, effectively reducing model size while maintaining accuracy.

Here, you will see significant decreases in model size while increasing speed. Note that quantization is currently only supported for CPUs, so we will be utilizing GPUs(CUDA) only for training purposes and CPU for testing purposes. Here, we are using the MNIST dataset. But furthermore, while using a complex dataset the accuracy might decrease upon quantization. By using a quantization configuration.

## Introduction:
In this project, we utilize post-training static quantization to optimize the ResNet18 architecture on the MNIST dataset. Quantization is a technique that converts the full-precision weights and activations of a model into lower bit-width representations. This leads to a significant reduction in model size and a boost in inference speed. However, it's important to note that quantization can potentially result in a decrease in accuracy, especially when applied to complex datasets.

## Quantization Configuration:
<kbd>
<div class="my-section" style= border: 1px solid #e1e4e8; "background-color: #f1f1f1; padding: 10px;">

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

</div>
</kbd>


- *In this project, we employ the **torch.quantization.get_default_qconfig('fbgemm')** configuration for quantizing our model. This step is a crucial part of the post-training static quantization process.*

#### Note:
Training uses resnet model as is with addition operation and floating point inputs / outputs.
But when model is quantized while testing addition operation is replaced with FloatFunction and the inputs / outputs are quantized/dequantized.

### Result:
- After quantization, you can observe the reduction in model size and increased inference speed during testing on a CPU. The accuracy may vary based on the quantization settings and the complexity of the dataset.

## Acknowledgments:
- The ResNet18 architecture and quantization techniques are inspired by the PyTorch documentation and relevant research papers.
- The MNIST dataset was used for this project.
