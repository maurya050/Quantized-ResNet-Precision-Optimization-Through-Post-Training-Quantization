# Quantized-ResNet-Precision-Optimization-Through-Post-Training-Quantization
Implemented and executed post-training static quantization techniques on a ResNet architecture to optimize model performance, reduce memory footprint, and enhance inference speed.
<kbd>
<div class="my-section" style= border: 1px solid #e1e4e8; "background-color: #f1f1f1; padding: 10px;">

## Quantization Strategy(Post-training static quantization (Pytorch) - ResNet18)

</div>
</kbd>

Employed state-of-the-art static quantization methods to convert full-precision weights and activations into lower bit-width representations, effectively reducing model size while maintaining accuracy.

Here significant decreases in model size while increasing speed. Note that quantization is currently only supported for CPUs, so we will be utilizing GPUs / CUDA only for training and CPU for testing. Furthermore, while using complex dataset the accuracy might decrease upon quantization. By using a quantization configuration
