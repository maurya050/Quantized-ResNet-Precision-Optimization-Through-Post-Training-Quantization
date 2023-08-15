# ResNet Quantization: Faster Inference with Post-Training Quantization
Implemented and executed post-training static quantization techniques on a ResNet architecture to optimize model performance, reduce memory footprint, and enhance inference speed.

#### For Resnet-18 Model taken ref. from:
 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

<kbd>
<div class="my-section" style= border: 1px solid #e1e4e8; "background-color: #f1f1f1; padding: 10px;">

## Quantization Strategy(Post-training static quantization (Pytorch) - ResNet18)

</div>
</kbd>

Employed state-of-the-art static quantization methods to convert full-precision weights and activations into lower bit-width representations, effectively reducing model size while maintaining accuracy.

Here, you will see significant decreases in model size while increasing speed. Note that quantization is currently only supported for CPUs, so we will be utilizing GPUs(CUDA) only for training purposes and CPU for testing purposes. Here, we are using the MNIST dataset. But furthermore, while using a complex dataset the accuracy might decrease upon quantization. By using a quantization configuration.


<kbd>
<div class="my-section" style= border: 1px solid #e1e4e8; "background-color: #f1f1f1; padding: 10px;">

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

</div>
</kbd>

### Note
Training uses resnet model as is with addition operation and floating point inputs / outputs.
But when model is quantized while testing addition operation is replaced with FloatFunction and the inputs / outputs are quantized/dequantized.
