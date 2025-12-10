## FlashAttention for unsupported Tesla v100

This repository want to implement the official implementation of FlashAttention and [FlashAttention-2](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/attention.md) under unsupported in TriDao repo [Nvidia Tesla V100](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/volta.md)

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** by Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

![FlashAttention](docs/fa2.jpeg)

### TD; TR

* **What is suboptimal but necessary due to Volta:**

  * Review compute speed on D>64 with PyTorch
  * Error correction: Code need error correction implementation of conversion.
  * Implement: alibi, dropout

```plaintext
(env) test@test:~/python ./benchmarks/benchmark_unsloth.py
🦥 Volta shim installed
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
🦥 flash_attn_func is now in llama.py
==((====))==  Unsloth 2025.5.7: Fast Llama patching. Transformers: 4.51.3.
   \\   /|    Tesla V100-SXM2-16GB. Num GPUs = 2. Max memory: 15.766 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 7.0. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
🦥 Model loaded Volta FA2 is active
Unsloth 2025.5.7 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 2,505 | Num Epochs = 1 | Total steps = 20
O^O/ \_/ \    Batch size per device = 4 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (4 x 4 x 1) = 16
 "-____-"     Trainable parameters = 39,976,960/7,000,000,000 (0.57% trained)
0%|                                     | 0/20 [00:00<?, ?it/s]Unsloth: Will smartly offload gradients to save VRAM!
{'loss': 2.6459, 'grad_norm': 0.3353293538093567, 'learning_rate': 0.0, 'epoch': 0.01}
{'loss': 2.6588, 'grad_norm': 0.3407493531703949, 'learning_rate': 0.0002, 'epoch': 0.01}
{'loss': 2.5317, 'grad_norm': 0.38473910093307495, 'learning_rate': 0.00018947368421052632, 'epoch': 0.02}
{'loss': 2.3638, 'grad_norm': 0.46123337745666504, 'learning_rate': 0.00017894736842105264, 'epoch': 0.03}
{'loss': 2.2422, 'grad_norm': 0.5855149626731873, 'learning_rate': 0.00016842105263157895, 'epoch': 0.03}
{'loss': 2.0572, 'grad_norm': 0.6951074004173279, 'learning_rate': 0.00015789473684210527, 'epoch': 0.04}
{'loss': 1.8681, 'grad_norm': 0.7144551873207092, 'learning_rate': 0.00014736842105263158, 'epoch': 0.04}
{'loss': 1.6858, 'grad_norm': 0.7136746048927307, 'learning_rate': 0.0001368421052631579, 'epoch': 0.05}
{'loss': 1.4876, 'grad_norm': 6.725831985473633, 'learning_rate': 0.0001263157894736842, 'epoch': 0.06}
{'loss': 1.4434, 'grad_norm': 1.3606822490692139, 'learning_rate': 0.00011578947368421053, 'epoch': 0.06}
{'loss': 1.2594, 'grad_norm': 1.128129243850708, 'learning_rate': 0.00010526315789473685, 'epoch': 0.07}
{'loss': 1.0319, 'grad_norm': 1.1167758703231812, 'learning_rate': 9.473684210526316e-05, 'epoch': 0.08}
{'loss': 0.9161, 'grad_norm': 1.2807807922363281, 'learning_rate': 8.421052631578948e-05, 'epoch': 0.08}
{'loss': 0.7684, 'grad_norm': 0.9921256899833679, 'learning_rate': 7.368421052631579e-05, 'epoch': 0.09}
{'loss': 0.6775, 'grad_norm': 0.9302423000335693, 'learning_rate': 6.31578947368421e-05, 'epoch': 0.1}
{'loss': 0.6001, 'grad_norm': 0.8588138222694397, 'learning_rate': 5.2631578947368424e-05, 'epoch': 0.1}
{'loss': 0.5337, 'grad_norm': 0.7218591570854187, 'learning_rate': 4.210526315789474e-05, 'epoch': 0.11}
{'loss': 0.5128, 'grad_norm': 0.5994800329208374, 'learning_rate': 3.157894736842105e-05, 'epoch': 0.11}
{'loss': 0.5007, 'grad_norm': 0.5552202463150024, 'learning_rate': 2.105263157894737e-05, 'epoch': 0.12}
{'loss': 0.5188, 'grad_norm': 0.5264633893966675, 'learning_rate': 1.0526315789473684e-05, 'epoch': 0.13}
{'train_time': 98.8052, 'train_samples': 3.239, 'train_steps': 0.202, 'train_loss': 1.415199938416481, 'epoch': 0.13}
100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [01:38<00:00,  4.94s/it]    
```