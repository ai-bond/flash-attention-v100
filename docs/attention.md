- [Русский язык](#введение) / [English Language](#introduction)

---

#### Введение

---

FlashAttention-2 (FA2) — это оптимизированная реализация механизма внимания, разработанная для эффективного использования памяти и вычислительных ресурсов на GPU. В отличие от традиционных подходов, FA2 не материализует полную матрицу внимания $$A \in \mathbb{R}^{N \times N} $$, что позволяет сократить потребление памяти с $$O(N^2) $$до $$O(N) $$, при этом сохраняя численную стабильность и точность вычислений.

#### Обозначения

---

- $$Q \in \mathbb{R}^{N \times d_k} $$ — матрица запросов (queries)
- $$K \in \mathbb{R}^{N \times d_k} $$ — матрица ключей (keys)
- $$V \in \mathbb{R}^{N \times d_v} $$ — матрица значений (values)
- $$P = Q K^\top / \sqrt{d_k} \in \mathbb{R}^{N \times N} $$ — логиты внимания
- $$A = \text{softmax}(P) \in \mathbb{R}^{N \times N} $$— матрица внимания
- $$O = A V \in \mathbb{R}^{N \times d_v} $$ — выход внимания


#### Forward-Проход

---

В forward-проходе FA2 используется **онлайн-softmax** и **блочная обработка** для избежания материализации $$A $$. Это достигается с помощью отслеживания локальных максимумов и нормализующих сумм.

#### Инициализация

---

Для каждой строки $$i $$:

- $$m_i^{(0)} = -\infty $$ — текущий максимум по строке
- $$l_i^{(0)} = 0 $$ — текущая сумма экспонент
- $$O_i^{(0)} = 0 $$ — накопленный выход

#### Обработка по блокам

---

Для каждого блока столбцов $$j $$ (по $$K_j, V_j $$):

1. **Вычисление логитов**:
   $$
   P_{ij} = Q_i K_j^\top / \sqrt{d_k}
   $$

2. **Обновление максимума**:
   $$
   m_i^{(j)} = \max\left(m_i^{(j-1)}, \max_{k \in B_j} P_{ik}\right)
   $$

3. **Обновление нормализующей суммы**:
   $$
   l_i^{(j)} = l_i^{(j-1)} \cdot \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) + \sum_{k \in B_j} \exp\left(P_{ik} - m_i^{(j)}\right)
   $$

4. **Обновление выхода**:
   $$
   O_i^{(j)} = O_i^{(j-1)} \cdot \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) + \sum_{k \in B_j} \exp\left(P_{ik} - m_i^{(j)}\right) V_k
   $$

5. **После всех блоков**:
$$
O_i = \frac{O_i^{(\text{final})}}{l_i^{(\text{final})}}
$$

#### Ключевые особенности

---

- **Fused kernel**: все операции (матричные умножения, softmax, умножение на $$V $$) выполняются в одном CUDA-ядре.
- **Recomputation**: матрица $$A $$не сохраняется для backward — она будет пересчитана из $$Q, K $$.
- **Tensor Cores**: используются для ускорения $$QK^\top $$и $$AV $$(через `mma.sync`).
- **Shared memory tiling**: данные загружаются блоками в shared memory для минимизации обращений к глобальной памяти.

#### Backward-Проход

---

В backward-проходе FA2 вычисляются градиенты по входным тензорам $$Q, K, V $$, при этом матрица внимания $$A $$не сохраняется на forward и пересчитывается по блокам на backward.

#### Обозначения

---

- $$dO = \partial L / \partial O \in \mathbb{R}^{N \times d_v} $$ — градиент по выходу (извне)
- $$dQ = \partial L / \partial Q $$
- $$dK = \partial L / \partial K $$
- $$dV = \partial L / \partial V $$

#### Формулы для backward-прохода

---

1. **Градиент по V**:
   $$
   dV = A^\top dO
   $$

2. **Промежуточный градиент по A**:
   $$
   dA = dO \cdot V^\top
   $$

3. **Градиент через softmax**:
   $$
   dP = A \odot \left(dA - \text{rowsum}(dA \odot A) \cdot \mathbf{1}\right)
   $$
   где $$\odot $$ — поэлементное умножение, $$\mathbf{1} $$— вектор из единиц.

   Эквивалентно:
   $$
   \text{sum\_dA\_A} = \sum_{j=1}^{N} (dA_{ij} \cdot A_{ij}) \quad \text{(по строкам)}
   $$
   $$
   dP_{ij} = A_{ij} \cdot \left(dA_{ij} - \text{sum\_dA\_A}_i\right)
   $$

4. **Градиенты по Q и K**:
   $$
   dQ = \frac{1}{\sqrt{d_k}} dP \cdot K
   $$
   $$
   dK = \frac{1}{\sqrt{d_k}} dP^\top \cdot Q
   $$

#### Особенности реализации
---
- $$A $$ не материализуется на forward → пересчитывается по блокам на backward из $$Q, K $$.
- **Численная стабильность**: при вычислении softmax вычитается $$\max(P_i) $$по строке.
- **Блочная обработка**: все операции (включая $$dV, dQ, dK $$) выполняются по тайлам (обычно $$128 \times 128 $$), чтобы уместиться в shared memory.
- **Fused kernel**: все вычисления делаются в одном CUDA-ядре без промежуточных записей в глобальную память.

#### Реализация Forward/Backward Ядер

---

#### Forward-Ядро

- Использует fused kernel для вычисления $$QK^\top $$, softmax, и $$AV $$.
- Применяет online softmax с отслеживанием $$m_i, l_i $$.
- Обрабатывает данные по блокам размером $$128 \times 128 $$.
- Использует shared memory для минимизации обращений к глобальной памяти.
- Поддерживает Tensor Core для ускорения операций.

#### Backward-Ядро

- Пересчитывает $$A $$по блокам из $$Q, K $$.
- Вычисляет $$dV, dA, dP, dQ, dK $$в одном fused проходе.
- Применяет численно стабильные формулы softmax-градиента.
- Использует shared memory и Tensor Core для оптимизации.

---
#### Основные техники оптимизации
| Техника | Описание |
| --- | --- |
| 3D Tiling | Разбиение тензоров на меньшие блоки в трёх измерениях для лучшего использования кэша и уменьшения обращений к глобальной памяти. |
| Causal Attention | Оптимизация для причинного внимания, где каждый токен может обращаться только к предыдущим токенам в последовательности. |
| ALiBi | Использование линейных смещений для уменьшения необходимости в маске attention. |
| Head Dimension | Оптимизация размерности головы внимания для лучшего использования аппаратных возможностей. |
| Double Buffering | Использование двойной буферизации для перекрытия вычислений и операций чтения/записи. |
| Fused Kernels | Объединение нескольких операций в одно ядро для уменьшения обращений к памяти. |
| Mixed Precision | Использование смешанной точности для ускорения вычислений и снижения потребления памяти. |
| Memory Coalescing | Оптимизация доступа к памяти для улучшения использования пропускной способности. |
| Parallelism | Использование параллелизма на различных уровнях для ускорения вычислений. |
| Quantization | Использование квантизации для уменьшения размера модели и ускорения вычислений. |
| Sparse Attention | Использование разреженных матриц внимания для уменьшения вычислений и потребления памяти. |
| Static vs Dynamic Attention | Оптимизация внимания в зависимости от его статичности или динамичности. |
| Tile Recycling | Переиспользование ранее вычисленных плиток для улучшения производительности. |
| Warps and Blocks Optimization | Оптимизация использования варпов и блоков для лучшего использования аппаратных ресурсов. |
| Memory Hierarchy Optimization | Оптимизация использования различных уровней памяти для уменьшения времени доступа к данным. |


- [Russian Language](#введение) / [English Language](#introduction)

---

#### Introduction
---

FlashAttention-2 (FA2) is an optimized implementation of the attention mechanism designed for efficient memory usage and computational resources on GPU. Unlike traditional approaches, FA2 does not materialize the full attention matrix $$A \in \mathbb{R}^{N \times N} $$, which allows reducing memory consumption from $$O(N^2) $$ to $$O(N) $$, while maintaining numerical stability and computational accuracy.

#### Notations
---

- $$Q \in \mathbb{R}^{N \times d_k} $$ — query matrix
- $$K \in \mathbb{R}^{N \times d_k} $$ — key matrix
- $$V \in \mathbb{R}^{N \times d_v} $$ — value matrix
- $$P = Q K^\top / \sqrt{d_k} \in \mathbb{R}^{N \times N} $$ — attention logits
- $$A = \text{softmax}(P) \in \mathbb{R}^{N \times N} $$ — attention matrix
- $$O = A V \in \mathbb{R}^{N \times d_v} $$ — attention output

#### Forward Pass
---

In the forward pass of FA2, an **online softmax** and **block processing** are used to avoid materializing $$A $$. This is achieved by tracking local maxima and normalization sums.

#### Initialization
---

For each row $$i $$:

- $$m_i^{(0)} = -\infty $$ — current row maximum
- $$l_i^{(0)} = 0 $$ — current sum of exponentials
- $$O_i^{(0)} = 0 $$ — accumulated output

#### Block Processing
---

For each column block $$j $$(over $$K_j, V_j $$):

1. **Logits computation**:
   $$
   P_{ij} = Q_i K_j^\top / \sqrt{d_k}
   $$

2. **Update maximum**:
   $$
   m_i^{(j)} = \max\left(m_i^{(j-1)}, \max_{k \in B_j} P_{ik}\right)
   $$

3. **Update normalization sum**:
   $$
   l_i^{(j)} = l_i^{(j-1)} \cdot \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) + \sum_{k \in B_j} \exp\left(P_{ik} - m_i^{(j)}\right)
   $$

4. **Update output**:
   $$
   O_i^{(j)} = O_i^{(j-1)} \cdot \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) + \sum_{k \in B_j} \exp\left(P_{ik} - m_i^{(j)}\right) V_k
   $$

After all blocks:
$$
O_i = \frac{O_i^{(\text{final})}}{l_i^{(\text{final})}}
$$

#### Key Features
---

- **Fused kernel**: all operations (matrix multiplications, softmax, multiplication by $$V $$) are performed in a single CUDA kernel.
- **Recomputation**: matrix $$A $$is not saved for backward — it will be recomputed from $$Q, K $$.
- **Tensor Cores**: used to accelerate $$QK^\top $$and $$AV $$(via `mma.sync`).
- **Shared memory tiling**: data is loaded in blocks to shared memory to minimize global memory accesses.

#### Backward Pass
---

In the backward pass of FA2, gradients with respect to input tensors $$Q, K, V $$are computed, while the attention matrix $$A $$is not saved during forward and recomputed in blocks during backward.

#### Notations
---

- $$dO = \partial L / \partial O \in \mathbb{R}^{N \times d_v} $$ — gradient w.r.t. output (from outside)
- $$dQ = \partial L / \partial Q $$
- $$dK = \partial L / \partial K $$
- $$dV = \partial L / \partial V $$

#### Backward Pass Formulas
---

1. **Gradient w.r.t. V**:
   $$
   dV = A^\top dO
   $$

2. **Intermediate gradient w.r.t. A**:
   $$
   dA = dO \cdot V^\top
   $$

3. **Gradient through softmax**:
   $$
   dP = A \odot \left(dA - \text{rowsum}(dA \odot A) \cdot \mathbf{1}\right)
   $$
   where $$\odot $$ is element-wise multiplication, $$\mathbf{1} $$ is a vector of ones.

   Equivalently:
   $$
   \text{sum\_dA\_A} = \sum_{j=1}^{N} (dA_{ij} \cdot A_{ij}) \quad \text{(row-wise)}
   $$
   $$
   dP_{ij} = A_{ij} \cdot \left(dA_{ij} - \text{sum\_dA\_A}_i\right)
   $$

4. **Gradients w.r.t. Q and K**:
   $$
   dQ = \frac{1}{\sqrt{d_k}} dP \cdot K
   $$
   $$
   dK = \frac{1}{\sqrt{d_k}} dP^\top \cdot Q
   $$

#### Implementation Features
---

- $$A $$ is not materialized during forward → recomputed in blocks during backward from $$Q, K $$.
- **Numerical stability**: $$\max(P_i) $$is subtracted row-wise during softmax computation.
- **Block processing**: all operations (including $$dV, dQ, dK $$) are performed in tiles (usually $$128 \times 128 $$) to fit in shared memory.
- **Fused kernel**: all computations are done in a single CUDA kernel without intermediate writes to global memory.

#### Forward/Backward Kernel Implementation
---

#### Forward Kernel

- Uses fused kernel to compute $$QK^\top $$, softmax, and $$AV $$.
- Applies online softmax with tracking of $$m_i, l_i $$.
- Processes data in blocks of size $$128 \times 128 $$.
- Uses shared memory to minimize global memory accesses.
- Supports Tensor Core acceleration for operations.

#### Backward Kernel

- Recomputes $$A $$ in blocks from $$Q, K $$.
- Computes $$dV, dA, dP, dQ, dK $$in a single fused pass.
- Applies numerically stable softmax gradient formulas.
- Uses shared memory and Tensor Cores for optimization.
---
#### Used optimization's
| Technique | Description |
| --- | --- |
| 3D Tiling | Splitting tensors into smaller blocks in three dimensions for better cache utilization and reduced global memory accesses. |
| Causal Attention | Optimization for causal attention where each token can only attend to previous tokens in the sequence. |
| ALiBi | Using linear biases to reduce the need for attention masking. |
| Head Dimension | Optimizing the attention head dimension for better hardware utilization. |
| Double Buffering | Using double buffering to overlap computations and memory operations. |
| Fused Kernels | Combining multiple operations into a single kernel to reduce memory accesses. |
| Mixed Precision | Using mixed precision (e.g., FP16 and FP32) to speed up computations and reduce memory usage. |
| Memory Coalescing | Optimizing memory access patterns to improve memory bandwidth utilization. |
| Parallelism | Leveraging parallelism at different levels to accelerate computations. |
| Quantization | Using quantization to reduce model size and speed up computations while maintaining acceptable accuracy. |
| Sparse Attention | Using sparse attention matrices to reduce computations and memory usage. |
| Static vs Dynamic Attention | Optimizing attention based on whether it is static or dynamic. |
| Tile Recycling | Reusing previously computed tiles to reduce computations and improve performance. |
| Warps and Blocks Optimization | Optimizing warp and block usage for better hardware resource utilization. |
| Memory Hierarchy Optimization | Optimizing the use of different memory levels to reduce data access latency. |

