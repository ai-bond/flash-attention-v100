- [Русский язык](#сведения-о-nvidia-volta-gv100-inline-ptx-assembly-tensor-core-и-mma) / [English Language](#nvidia-volta-gv100-inline-ptx-assembly-tensor-core-and-mma)


## Сведения о NVIDIA Volta (GV100) Inline PTX Assembly: Tensor Core и MMA
---

Ключевые аспекты программирования Tensor Core на архитектуре NVIDIA Volta (sm_70, чип GV100) с использованием низкоуровневых встроенных инструкций PTX. Tensor Core — это выделенные функциональные блоки, оптимизированные для выполнения операций умножения матриц с последующим сложением (Matrix Multiply-Accumulate, MMA): D = A × B + C. В архитектуре Volta ключевой инструкцией для выполнения таких операций является `mma.sync.aligned.m8n8k4`.

### Инструкция mma.sync.aligned.m8n8k4
---

Инструкция `mma.sync.aligned.m8n8k4` в PTX выполняет операцию MMA на уровне варпа (32 потока). Она напрямую задействует Tensor Core для выполнения вычислений, что позволяет достичь высокой теоретической пиковой производительности.

### Полная расшифровка инструкции
---

Рассмотрим инструкцию `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`:

| Часть инструкции | Описание |
|------------------|----------|
| **mma**          | Назначение: указывает, что это инструкция для выполнения матричного умножения с накоплением: D = A × B + C. |
| **sync**         | Модификатор синхронизации. Требует синхронизации всех 32 потоков варпа перед выполнением операции. Tensor Core работает на уровне варпа, и все 32 потока должны участвовать. |
| **aligned**      | Модификатор выравнивания данных. Указывает, что входные данные выровнены в памяти (обычно по 16-байтной границе). Обеспечивает максимальную пропускную способность и корректную работу Tensor Core. |
| **m8n8k4**       | Размеры матриц, участвующих в операции: m = 8 (строки в D и A), n = 8 (столбцы в D и B), k = 4 (внутренняя размерность). Формула: A<sub>8×4</sub> · B<sub>4×8</sub> + C<sub>8×8</sub> → D<sub>8×8</sub>. |
| **row**          | Макет (layout) матрицы A. Указывает, что матрица A хранится по строкам (row-major). |
| **col**          | Макет (layout) матрицы B. Указывает, что матрица B хранится по столбцам (column-major). |
| **f32**          | Тип данных результата D. 32-битное число с плавающей запятой (float). |
| **f16**          | Тип данных матрицы A. 16-битное число с плавающей запятой (half). |
| **f16**          | Тип данных матрицы B. 16-битное число с плавающей запятой (half). |
| **f32**          | Тип данных аккумулятора C. 32-битное число с плавающей запятой (float). |

Итоговая операция: D<sub>8×8</sub>(f32) = A<sub>8×4</sub>(f16) · B<sub>4×8</sub>(f16) + C<sub>8×8</sub>(f32)

### Распределение данных по потокам
---
Весь варп (32 потока) совместно вычисляет одну 8×8 матрицу D. На Volta один варп фактически вычисляет 4 независимые операции m8n8k4. Потоки разбиты на 4 группы по 8 потоков (quadpair):

| Группа | Потоки |
|--------|--------|
| Группа 0 | lane_id 0–3 и 16–19 |
| Группа 1 | lane_id 4–7 и 20–23 |
| Группа 2 | lane_id 8–11 и 24–27 |
| Группа 3 | lane_id 12–15 и 28–31 |

Каждая группа из 8 потоков вычисляет свою собственную матрицу D<sub>8×8</sub>. Данные матриц размазаны по регистрам этих 32 потоков в строго определённом порядке.

### Пример распределения данных
---

Для f16 операндов A, B и f32 для C/D:

- **Матрица A** (32 элемента f16 для 4 операций): Каждый поток в своей группе из 8 держит 1 элемент (всего 4 элемента на группу). Эти элементы часто упаковываются по два в 32-битный регистр (f16x2) и передаются в инструкцию как целочисленные (r) операнды.
- **Матрица B** — аналогично A.
- **Матрицы C/D** (64 элемента f32 для 4 операций): Каждый поток в своей группе из 8 держит 2 элемента (всего 8 элементов на группу). Эти элементы хранятся в 32-битных FP-регистрах (f).

### Использование в inline assembly
---
Инструкция mma может быть встроена в код CUDA с использованием inline PTX assembly. Синтаксис соответствует спецификации GCC inline assembly.

```cpp
asm volatile(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n\t"
    "{%0, %1, %2, %3, %4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11}, "
    "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
    : "r"(a_reg0), "r"(a_reg1),
      "r"(b_reg0), "r"(b_reg1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7)
);
```
### Inline ASM Volta: Объяснение спецификаторов операндов
---
   Спецификатор | Описание                                                                                     |
 |--------------|----------------------------------------------------------------------------------------------|
 | "=f"         | Выходной операнд (часть результата D) в 32-битном FP-регистре. = означает только запись (write-only output). |
 | "+f"         | Входно-выходной операнд (часть аккумулятора C, который становится частью D) в 32-битном FP-регистре. + означает чтение и запись (read-write). |
 | "f"          | Входной операнд (часть аккумулятора C) в 32-битном FP-регистре. Без префикса — только чтение (input-only). |
 | "r"          | Входной операнд (упакованные f16x2 элементы A или B) в 32-битном целочисленном регистре. Используется для целых чисел, упакованных half-значений (__half2 → хранится в одном 32-битном регистре как два f16). |

### Почему "r" для f16, но "f" для f32?
---
Для f16 (A, B): В PTX/SM70 операнды A и B для mma.m8n8k4 должны быть в формате .f16x2, упакованном в 32-битный регистр. В C++ нет нативного типа для .f16x2, поэтому его эмулируют через uint32_t или __half2. __half2 — это union, который интерпретирует 32 бита как два half. При передаче в asm компилятор кладёт это значение в целочисленный регистр → отсюда "r".

Для f32 (C, D): Аккумулятор C и результат D — в f32, и они действительно хранятся в FP-регистрах → "f" и "=f".

### Сопоставление регистров в inline PTX
---
В строке ассемблера:
 | Группа | Описание                                                                                     | Регистры                          |
 |--------|----------------------------------------------------------------------------------------------|-----------------------------------|
 | D      | 8×f32                                                                                       | {%0, %1, %2, %3, %4, %5, %6, %7} |
 | A      | 2×f16x2 = 4×f16                                                                             | {%8, %9}                          |
 | B      | 2×f16x2 = 4×f16                                                                             | {%10, %11}                        |
 | C      | 8×f32   | {%12, %13, %14, %15, %16, %17, %18, %19} |
### Атом MMA в CuTe
---
Атом MMA в CuTe — это комбинация двух структур: Operation и MMA_Traits.

#### Operation
- **Инструкция**: Инкапсулирует конкретную инструкцию PTX (HMMA).
- **Имя**: Кодирует ключевые свойства (архитектура SM70, размер операции 8x8x4, типы данных F32F16F16F32, размещение матриц NT).
- **Регистры**: Определяет, сколько значений каждый поток передает в инструкцию для матриц A, B, C и D.

#### MMA_Traits
- **Логические типы данных**: Определяет типы для вычислений (например, ValTypeA = half_t).
- **Форма операции**: Задает размерность MMA как Shape<_8,_8,_4> (MxNxK).
- **Отображение потоков**: Описывает, как логические ID потоков операции сопоставляются с физическими потоками в warp'е с помощью Layout (для Volta это quadpair из 8 потоков).
- **Макеты данных**: ALayout, BLayout, CLayout описывают, как значения распределены между потоками quadpair и их координатами в матрицах.

### Аппаратная основа Volta
---
- **Аппаратная единица**: Операция MMA выполняется группой из 8 потоков, называемой quadpair.
- **Размер операции**: Один quadpair выполняет матричное умножение размерностью 8x8x4.
- **Поддерживаемые форматы**: Tensor Cores Volta поддерживают смешанную точность: входные матрицы A и B в формате FP16, а аккумуляция и результат — в FP32.

### Пример
Пример использования атома MMA в CuTe для операции матричного умножения.
---

```cpp
// Пример кода для операции матричного умножения
using namespace cute;
using namespace cute::tensor;

auto mma_op = SM70_8x8x4_F32F16F16F32_NT{};
auto mma_traits = MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>{};

auto A = make_tensor<half_t>(make_shape(8, 8));
auto B = make_tensor<half_t>(make_shape(8, 4));
auto C = make_tensor<float>(make_shape(8, 8));

auto result = mma_op(A, B, C);
```

### Использование nvcuda::wmma — это высокоуровневый C++ API
---
Не смотря на то, что nvcuda::wmma — это высокоуровневый C++ API, представленный в CUDA 9+, и он предназначен для программного доступа к Tensor Cores . Первой архитектурой, поддерживающей Tensor Cores, была Volta (sm_70) единственная функция умножения это m16n16k16 - появилось в Ampere (sm_80). nvcuda::wmma в CUDA 12.4 не может сгенерировать mma.sync.m8n8k4 напрямую через wmma::fragment — он генерирует только m16n16k16. По этой причине использование nvcuda::wmma на архитектуре Volta не возможна! и связана с аппаратными ограничениями!

### Использование ldmatrix
---
Ошибка компиляции "Feature 'ldmatrix' requires .target sm_75 or higher" возникает потому, что инструкция ldmatrix, которая используется nvcuda::wmma для загрузки данных в фрагменты для операций с Tensor Cores, требует целевой архитектуры sm_75 или выше . Хотя архитектура Volta (sm_70) впервые представила Tensor Cores и поддерживает инструкцию mma.sync.m8n8k4, инструкция ldmatrix была введена позже, с архитектурой Turing (sm_75) . Поэтому код, использующий ldmatrix, не может быть скомпилирован для целевой архитектуры sm_70

- [Русский язык](#сведения-о-nvidia-volta-gv100-inline-ptx-assembly-tensor-core-и-mma) / [English Language](#nvidia-volta-gv100-inline-ptx-assembly-tensor-core-and-mma)

---

## NVIDIA Volta (GV100) Inline PTX Assembly: Tensor Core and MMA
---
Key aspects of programming Tensor Cores on NVIDIA Volta architecture (sm_70, GV100 chip) using low-level inline PTX instructions. Tensor Cores are dedicated functional units optimized for matrix multiply-accumulate (MMA) operations: D = A × B + C. In Volta architecture, the key instruction for these operations is `mma.sync.aligned.m8n8k4`.

### The mma.sync.aligned.m8n8k4 instruction
---
The `mma.sync.aligned.m8n8k4` instruction in PTX performs MMA operation at warp level (32 threads). It directly engages Tensor Core for computations, enabling high theoretical peak performance.

### Full instruction decoding
---
Let's examine the instruction `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`:
   Instruction part | Description |
 |------------------|-------------|
 | **mma**          | Purpose: Indicates this is an instruction for matrix multiply-accumulate operation: D = A × B + C. |
 | **sync**         | Synchronization modifier. Requires synchronization of all 32 warp threads before operation. Tensor Core operates at warp level and requires all 32 threads to participate. |
 | **aligned**      | Data alignment modifier. Indicates that input data is aligned in memory (typically 16-byte boundary). Ensures maximum bandwidth and correct Tensor Core operation. |
 | **m8n8k4**       | Matrix dimensions in the operation: m = 8 (rows in D and A), n = 8 (columns in D and B), k = 4 (inner dimension). Formula: A<sub>8×4</sub> · B<sub>4×8</sub> + C<sub>8×8</sub> → D<sub>8×8</sub>. |
 | **row**          | Layout of matrix A. Indicates that matrix A is stored row-major. |
 | **col**          | Layout of matrix B. Indicates that matrix B is stored column-major. |
 | **f32**          | Data type of result D. 32-bit floating point (float). |
 | **f16**          | Data type of matrix A. 16-bit floating point (half). |
 | **f16**          | Data type of matrix B. 16-bit floating point (half). |
 | **f32**          | Data type of accumulator C. 32-bit floating point (float). |

Final operation: D<sub>8×8</sub>(f32) = A<sub>8×4</sub>(f16) · B<sub>4×8</sub>(f16) + C<sub>8×8</sub>(f32)

### Data distribution across threads
---
The entire warp (32 threads) collectively computes one 8×8 matrix D. On Volta, one warp actually computes 4 independent m8n8k4 operations. Threads are divided into 4 groups of 8 threads (quadpair):
 | Group | Threads |
 |-------|---------|
 | Group 0 | lane_id 0-3 and 16-19 |
 | Group 1 | lane_id 4-7 and 20-23 |
 | Group 2 | lane_id 8-11 and 24-27 |
 | Group 3 | lane_id 12-15 and 28-31 |

Each group of 8 threads computes its own D<sub>8×8</sub> matrix. Matrix data is distributed across these 32 threads' registers in a strictly defined order.

### Example of data distribution
---
For f16 operands A, B and f32 for C/D:
- **Matrix A** (32 f16 elements for 4 operations): Each thread in its group of 8 holds 1 element (total 4 elements per group). These elements are often packed two into a 32-bit register (f16x2) and passed to the instruction as integer (r) operands.
- **Matrix B** - similar to A.
- **Matrices C/D** (64 f32 elements for 4 operations): Each thread in its group of 8 holds 2 elements (total 8 elements per group). These elements are stored in 32-bit FP registers (f).

### Usage in inline assembly
---
The mma instruction can be embedded in CUDA code using inline PTX assembly. Syntax follows GCC inline assembly specification.

```cpp
asm volatile(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n\t"
    "{%0, %1, %2, %3, %4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11}, "
    "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
    : "r"(a_reg0), "r"(a_reg1),
      "r"(b_reg0), "r"(b_reg1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7)
);
```
### Inline ASM Volta: Explanation of operand specifiers
---
   Specifier | Description |
 |-----------|-------------|
 | "=f"      | Output operand (part of result D) in 32-bit FP register. = means write-only output. |
 | "+f"      | Input-output operand (part of accumulator C that becomes part of D) in 32-bit FP register. + means read-write. |
 | "f"       | Input operand (part of accumulator C) in 32-bit FP register. Without prefix - input-only. |
 | "r"       | Input operand (packed f16x2 elements of A or B) in 32-bit integer register. Used for integers, packed half values (__half2 → stored in one 32-bit register as two f16). |

### Why "r" for f16 but "f" for f32?
---
For f16 (A, B): In PTX/SM70, A and B operands for mma.m8n8k4 must be in .f16x2 format packed in 32-bit register. In C++ there's no native type for .f16x2, so it's emulated via uint32_t or __half2. __half2 is a union that interprets 32 bits as two halves. When passed to asm, compiler puts this value in integer register → hence "r".
For f32 (C, D): Accumulator C and result D are in f32 and actually stored in FP registers → "f" and "=f".

### Register mapping in inline PTX
---
In the assembly line:
 | Group | Description | Registers |
 |-------|-------------|-----------|
 | D     | 8×f32       | {%0, %1, %2, %3, %4, %5, %6, %7} |
 | A     | 2×f16x2 = 4×f16 | {%8, %9} |
 | B     | 2×f16x2 = 4×f16 | {%10, %11} |
 | C     | 8×f32 | {%12, %13, %14, %15, %16, %17, %18, %19} |

### MMA Atom in CuTe
---
The MMA atom in CuTe is a combination of two structures: Operation and MMA_Traits.

#### Operation
- **Instruction**: Encapsulates specific PTX instruction (HMMA).
- **Name**: Encodes key properties (SM70 architecture, 8x8x4 operation size, F32F16F16F32 data types, NT matrix layout).
- **Registers**: Defines how many values each thread passes to the instruction for A, B, C, and D matrices.

#### MMA_Traits
- **Logical data types**: Defines types for computations (e.g., ValTypeA = half_t).
- **Operation shape**: Specifies MMA dimensionality as Shape<_8,_8,_4> (MxNxK).
- **Thread mapping**: Describes how operation's logical thread IDs map to physical threads in warp using Layout (for Volta it's quadpair of 8 threads).
- **Data layouts**: ALayout, BLayout, CLayout describe how values are distributed among quadpair threads and their coordinates in matrices.

### Volta Hardware Basis
---
- **Hardware unit**: MMA operation is performed by a group of 8 threads called quadpair.
- **Operation size**: One quadpair performs 8x8x4 matrix multiplication.
- **Supported formats**: Volta Tensor Cores support mixed precision: input matrices A and B in FP16 format, while accumulation and result are in FP32.

### Example
---
Example of using MMA atom in CuTe for matrix multiplication operation.

```cpp
// Example code for matrix multiplication operation
using namespace cute;
using namespace cute::tensor;

auto mma_op = SM70_8x8x4_F32F16F16F32_NT{};
auto mma_traits = MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>{};

auto A = make_tensor<half_t>(make_shape(8, 8));
auto B = make_tensor<half_t>(make_shape(8, 4));
auto C = make_tensor<float>(make_shape(8, 8));

auto result = mma_op(A, B, C);
```

### Using nvcuda::wmma — High-Level C++ API
---
Although `nvcuda::wmma` is a high-level C++ API introduced in CUDA 9+ and designed for programmatic access to Tensor Cores, the first architecture to support Tensor Cores was Volta (sm_70). The only multiply function available is `m16n16k16` - `m8n8k4` appeared in Ampere (sm_80). `nvcuda::wmma` in CUDA 12.4 cannot generate `mma.sync.m8n8k4` directly via `wmma::fragment` — it only generates `m16n16k16`. For this reason, using `nvcuda::wmma` on Volta architecture is not possible! This is related to hardware limitations.

### Using ldmatrix
---
The compilation error "Feature 'ldmatrix' requires .target sm_75 or higher" occurs because the `ldmatrix` instruction, which is used by `nvcuda::wmma` to load data into fragments for Tensor Core operations, requires target architecture sm_75 or higher. Although Volta architecture (sm_70) first introduced Tensor Cores and supports the `mma.sync.m8n8k4` instruction, the `ldmatrix` instruction was introduced later with Turing architecture (sm_75). Therefore, code using `ldmatrix` cannot be compiled for target architecture sm_70.
