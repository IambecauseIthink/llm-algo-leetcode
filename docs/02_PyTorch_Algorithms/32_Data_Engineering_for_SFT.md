# 32. Data Engineering for SFT | SFT 数据工程
**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `SFT`, `数据清洗` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

SFT 训练经常不是输在优化器或学习率，而是输在数据没有被稳定组织成训练样本。字段缺失、空回答、模板不统一、超长样本混进来，这些问题都会直接污染 loss 和评测。`09` 先回答最小训练循环怎么跑，`30` 再把上下文长度拉长，而 `32` 往更上游退一步，先回答“在开始训练之前，数据本身是否已经整理到足够可靠”。

这一节把 SFT 数据工程收敛成三个可执行动作：将原始记录清洗成稳定字段，审计空值、重复和长度异常，再把清洗结果落成训练样本。你会把这条“原始记录 → 数据审计 → 训练样本 → SFT 闭环”的链路接到 `09` 的 `input_ids / attention_mask / labels` 和 loss 对齐。

**关键词：** `dataset audit`, `prompt template`, `record cleaning`, `tokenizer`, `packing`

本节同时记录字符级长度，并把它作为后续 tokenizer 检查的输入；真正的 token 长度、chat template、截断、padding 和 labels mask 会在 `09`、`13` 和真实项目中继续展开。

---

## 前置阅读

**导语：** 进入本节前，先能说出训练接口需要哪些字段，再检查原始指令数据如何整理成稳定的 `prompt / response` 样本。
- [04. Python Config and Data Entry | Python 配置与数据组织](../00_Prerequisites/04_Python_Config_and_Data_Entry.md)
- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)

---

### Step 1: 先把原始记录清洗成稳定结构

- 统一保留 `instruction / input / response` 三个核心字段。
- 去掉多余空格和无意义空串，避免相同样本因为格式差异被误判成不同记录。
- 若字段缺失，要尽早补空值或剔除，而不是把异常留到训练阶段。
- 这里的清洗只处理结构和明显异常，不自动判断事实正确性、答案质量或训练数据泄漏。

### Step 2: 做最小数据审计

![SFT Data Engineering Flow](/02_PyTorch_Algorithms/32_sft_data_engineering_flow.svg)

- 统计总样本数、空回答数、重复样本数和超长样本数。
- 长度问题要同时看 prompt 和 response，而不是只看单字段。
- 如果异常比例已经很高，优先返工数据，而不是继续调训练超参。

### Step 3: 把清洗结果变成训练样本

- 把 instruction 和 input 拼成 prompt。
- 保留 response 作为监督目标。
- 给后续训练保留 metadata，例如 prompt 长度和原始索引，方便排查异常样本。
- 先输出 tokenizer-ready 文本；进入模型前还要用目标 tokenizer 重新统计 token 长度，并检查 chat template 是否匹配。

| 阶段 | 本节是否负责 | 需要记录 |
|:---|:---:|:---|
| 字段清洗与格式统一 | 是 | 原始索引、字段缺失、空回答 |
| tokenizer 与 chat template | 部分 | tokenizer 名称、模板版本、token 长度 |
| labels mask 与 loss 对齐 | 否 | 转到 09 / 13 |
| 数据是否值得进入训练 | 否 | 转到 33 / 64 |

### Step 4: 动手实战

1. 补全 `clean_sft_records`，统一字段并去掉首尾空白。
2. 补全 `audit_sft_dataset`，统计空回答、重复和超长样本。
3. 补全 `build_sft_training_examples`，把清洗后的记录转成训练样本。
4. 对生成的 `prompt / response` 调用目标 tokenizer，记录 token 长度分布；不要用字符数代替 token 数。

### 提示

- `TODO 1` 先把每条原始记录统一成 `instruction / input / response` 三个字段，并去掉首尾空白。
- `TODO 2` 先统计总量，再依次统计空回答、重复样本、超长样本和平均长度。
- `TODO 3` 先把 `instruction` 和可选 `input` 拼成 prompt，再补训练所需的 metadata。


```python
from typing import Dict, List

```


```python
def summarize_token_lengths(tokenizer, texts: List[str]) -> Dict[str, float]:
    """TODO 4: 用目标 tokenizer 统计文本 token 长度。"""
    # 提示：调用 tokenizer(texts, add_special_tokens=True)，从 input_ids 计算每条长度。
    # token_lengths = ???
    # return min / max / average；空列表要返回 0。
    raise NotImplementedError("请先完成 TODO 代码！")


def clean_sft_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    TODO 1: 统一字段并去掉首尾空白。
    """
    # 提示：先创建 cleaned，再逐条把 instruction / input / response 转成字符串并 strip。
    # cleaned = ???
    raise NotImplementedError


def audit_sft_dataset(records: List[Dict[str, str]], max_total_chars: int) -> Dict[str, float]:
    """
    TODO 2: 统计空回答、重复和超长样本。
    """
    # 提示：先建立 seen 和各项计数器，再逐条统计 total_chars、duplicate、empty_response 和 over_length。
    # total_samples = ???
    # duplicate_count = ???
    # avg_total_chars = ???
    raise NotImplementedError


def build_sft_training_examples(records: List[Dict[str, str]], template_prefix: str) -> List[Dict[str, object]]:
    """
    TODO 3: 把清洗后的记录转成训练样本。
    """
    # 提示：先创建 examples，再拼 prompt；如果有 input，再追加 Input 段。
    # examples = ???
    # prompt = ???
    raise NotImplementedError

```


```python
def test_sft_data_engineering_template():
    try:
        records = [
            {'instruction': ' 解释 LoRA ', 'input': '', 'response': ' 一种低秩适配方法。 '},
            {'instruction': '解释 LoRA', 'input': '', 'response': '一种低秩适配方法。'},
            {'instruction': '给出总结', 'input': '结合训练成本', 'response': ''},
        ]
        cleaned = clean_sft_records(records)
        assert cleaned[0]['instruction'] == '解释 LoRA'
        assert cleaned[0]['response'] == '一种低秩适配方法。'
        audit = audit_sft_dataset(cleaned, max_total_chars=30)
        assert audit['total_samples'] == 3
        assert audit['empty_response_count'] == 1
        assert audit['duplicate_count'] == 1
        class MockTokenizer:
            def __call__(self, texts, add_special_tokens=True):
                return {'input_ids': [list(range(len(text.split()))) for text in texts]}
        token_stats = summarize_token_lengths(MockTokenizer(), ['one two', 'one two three'])
        assert token_stats['count'] == 2 and token_stats['max_tokens'] == 3
        examples = build_sft_training_examples(cleaned[:1], template_prefix='### Instruction\n')
        assert examples[0]['has_input'] is False
        assert examples[0]['prompt'].startswith('### Instruction')
        print('测试通过：SFT 数据工程模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, IndexError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_sft_data_engineering_template()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
def summarize_token_lengths(tokenizer, texts: List[str]) -> Dict[str, float]:
    """TODO 4: 用目标 tokenizer 统计文本 token 长度。"""
    encoded = tokenizer(texts, add_special_tokens=True)
    token_lengths = [len(ids) for ids in encoded['input_ids']]
    if not token_lengths:
        return {'count': 0, 'min_tokens': 0, 'max_tokens': 0, 'avg_tokens': 0.0}
    return {
        'count': len(token_lengths),
        'min_tokens': min(token_lengths),
        'max_tokens': max(token_lengths),
        'avg_tokens': sum(token_lengths) / len(token_lengths),
    }


def clean_sft_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    TODO 1: 统一字段并去掉首尾空白。
    """
    # 提示：先创建 cleaned，再逐条把 instruction / input / response 转成字符串并 strip。
    # cleaned = ???
    cleaned: List[Dict[str, str]] = []
    for record in records:
        cleaned.append({
            'instruction': str(record.get('instruction', '')).strip(),
            'input': str(record.get('input', '')).strip(),
            'response': str(record.get('response', '')).strip(),
        })
    return cleaned


def audit_sft_dataset(records: List[Dict[str, str]], max_total_chars: int) -> Dict[str, float]:
    """
    TODO 2: 统计空回答、重复和超长样本。
    """
    # 提示：先建立 seen 和各项计数器，再逐条统计 total_chars、duplicate、empty_response 和 over_length。
    # total_samples = ???
    # duplicate_count = ???
    # avg_total_chars = ???
    seen = set()
    empty_response_count = 0
    duplicate_count = 0
    over_length_count = 0
    total_chars = 0
    for record in records:
        total_text = record.get('instruction', '') + record.get('input', '') + record.get('response', '')
        total_chars += len(total_text)
        key = (record.get('instruction', ''), record.get('input', ''), record.get('response', ''))
        if not record.get('response', '').strip():
            empty_response_count += 1
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
        if len(total_text) > max_total_chars:
            over_length_count += 1
    total_samples = len(records)
    return {
        'total_samples': total_samples,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_length_count': over_length_count,
        'avg_total_chars': total_chars / total_samples if total_samples else 0.0,
    }


def build_sft_training_examples(records: List[Dict[str, str]], template_prefix: str) -> List[Dict[str, object]]:
    """
    TODO 3: 把清洗后的记录转成训练样本。
    """
    # 提示：先创建 examples，再拼 prompt；如果有 input，再追加 Input 段。
    # examples = ???
    # prompt = ???
    examples = []
    for record in records:
        instruction = record.get('instruction', '')
        input_text = record.get('input', '')
        prompt = f"{template_prefix}{instruction}"
        if input_text:
            prompt += f"\n### Input\n{input_text}"
        examples.append({
            'prompt': prompt,
            'response': record.get('response', ''),
            'prompt_chars': len(prompt),
            'has_input': bool(input_text),
        })
    return examples

```

### 解析

**1. TODO 1：统一字段并去掉首尾空白**
- 先把原始记录统一成 `instruction / input / response` 三个核心字段，再对每个字段做字符串化和 `strip()`。
- 数据工程的第一步不是上 tokenizer，而是先把结构收紧，避免同一条样本因为格式噪声被当成不同记录。

**2. TODO 2：统计空回答、重复和超长样本**
- 先建立 `seen` 和各项计数器，再逐条统计 `empty_response_count`、`duplicate_count`、`over_length_count` 和平均长度。
- 审计的意义是先确认数据问题究竟出在空回答、重复样本，还是整体长度失控，而不是盲目进入训练。

**3. TODO 3：把清洗后的记录转成训练样本**
- 先把 `instruction` 和可选 `input` 拼成统一 `prompt`，再补 `response`、`prompt_chars` 和 `has_input` 等训练所需 metadata。
- 训练循环真正消费的是统一模板下的 `prompt + response`，不是原始记录本身。

**4. TODO 4：统计 token 长度**
- **实现方式**：把 tokenizer 作为参数注入，读取返回的 `input_ids` 长度并输出样本数、最小值、最大值和平均值。
- **关键点**：字符长度只能用于清洗阶段的粗筛；真正的截断、padding 和显存估算必须使用目标 tokenizer 的 token 长度。

**5. 这页的定位**
- 数据工程的起点是把结构收紧，而不是直接进入 tokenizer。
- 审计至少要覆盖空回答、重复和超长样本三类高频问题。
- 训练循环真正消费的是统一模板下的 `prompt + response`，不是原始记录本身。

## 相关阅读

完成字段清洗、数据审计和样本生成后，可以继续阅读 Datasets 工具和微调项目，验证样本如何进入真实训练闭环。

- [Hugging Face Datasets 官方文档](https://huggingface.co/docs/datasets)
- [Hugging Face Datasets 官方仓库](https://github.com/huggingface/datasets)
- [30. Long Context Fine-Tuning | 长上下文微调](./30_Long_Context_Fine_Tuning.md)
- [62. Instruction Fine-Tuning Project | 指令微调项目](./62_Instruction_Fine_Tuning_Project.md)
- [64. SFT Data Quality Project | SFT 数据质量项目](./64_SFT_Data_Quality_Project.md)
- [2.3](./2_3.md)
