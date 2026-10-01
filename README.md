# pdftrans — 保留排版的 PDF 翻译器

英文（及其他语言）PDF 翻译成中文，排版保持不变。

带图形界面的 PDF 翻译工具：调用大模型 API 翻译文字，把译文写回原来的位置，
图片、表格线、背景色、页面尺寸都保持不变。输出效果类似「沉浸式翻译」的 PDF 翻译：

| 输出方式 | 说明 | 文件名 |
| --- | --- | --- |
| 仅译文 | 原排版，文字替换为译文 | `论文.zh-CN.mono.pdf` |
| 双语对照 | 每页左边原文、右边译文 | `论文.zh-CN.dual.pdf` |
| 双语交替 | 原文页后紧跟对应的译文页 | `论文.zh-CN.alt.pdf` |

## 安装

需要 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

## 使用

### 图形界面

```bash
python -m pdftrans            # 打开窗口
python -m pdftrans 论文.pdf   # 打开窗口并载入文件
```

1. 把 PDF 拖进窗口（或点「选择 PDF…」）。
2. 选择服务商，填入 API Key，按需修改模型名，可以先点「测试连接」。
3. 选择目标语言和输出方式，点「开始翻译」。
4. 右侧预览区并排显示原文和译文，可以翻页、缩放。点「打开结果 PDF」查看文件。

设置会自动保存到 `~/.pdftrans/config.json`（API Key 也存在这里，文件权限为仅本人可读）。
API Key 也可以留空，改用环境变量：OpenAI 兼容接口读 `PDFTRANS_API_KEY` 或 `OPENAI_API_KEY`，
Claude 读 `ANTHROPIC_API_KEY`。

### 命令行

```bash
python -m pdftrans --cli 论文.pdf --preset DeepSeek --api-key sk-... --lang zh-CN --modes mono,dual
python -m pdftrans --cli a.pdf b.pdf --pages 1-5 -o 输出目录
```

没有给出的选项沿用图形界面里保存的设置。`python -m pdftrans --cli --help` 查看全部参数。

## 支持的大模型

所有兼容 OpenAI `chat/completions` 协议的服务都能用，界面里内置了预设：
OpenAI、DeepSeek、通义千问 (DashScope)、智谱 GLM、Moonshot (Kimi)、SiliconFlow、
Ollama（本地模型，无需 Key）。其他服务选「自定义 OpenAI 兼容接口」并填写 Base URL 与模型名即可。

也直接支持 Claude（Anthropic 官方 SDK），默认模型 `claude-opus-5-5`，可以在「推理强度」里
选择 low / medium / high（翻译任务用 low 通常就够，速度快、费用低）。使用官方接口时会开启
服务端兜底（`fallbacks: "default"`），模型拒答时自动换用备选模型重试；填写了自定义 Base URL
（第三方中转）时不发送该参数。

## 选项说明

- **页码范围**：如 `1-5,8`，留空表示全部页面。只输出所选页面。
- **并发请求数 / 每批文本量**：多个段落会打包成一个请求（JSON 格式）发送，减少请求次数。
  遇到限流 (HTTP 429) 时把并发调小。
- **缓存译文**：译文按「模型 + 目标语言 + 提示词 + 原文」缓存在 `~/.pdftrans/cache.sqlite3`，
  同一文件重新翻译（比如中途失败或取消后）不会重复计费。
- **附加要求**：术语表或风格要求，会加入系统提示词，例如 `attention → 注意力`。
- **字体**：默认译文用内置的 Droid Sans Fallback（中日韩）和 Times/Helvetica（西文），
  可以指定自己的 `.ttf/.otf` 字体文件（如思源宋体）。输出时只嵌入用到的字形，文件不会明显变大。

## 工作原理

1. 用 PyMuPDF 读出每一行文字的位置、字号、颜色和字体，再按字号、行距、缩进、列表符号等
   把行合并成段落（标题、正文、图注、表格单元格分别成段）。
2. 跳过不需要翻译的内容：页码、纯数字、公式（数学字体或符号占比高）、网址、已是目标语言的文字。
3. 把段落打包发给大模型翻译，提示词要求保留公式、引用、数字、网址不变。
4. 用 redaction 只删除被翻译的那些文字（图片和矢量图形保持不动，也不画白底，背景色得以保留），
   再在原位置排版译文：中日韩文按字断行、西文按词断行，避免标点出现在行首，保留粗体、颜色、
   居中、两端对齐和列表缩进。译文放不下时先利用段落下方的空白，仍不够再缩小字号。
5. 生成仅译文 / 左右对照 / 交替页三种 PDF。

## 已知限制

- 扫描版 PDF（整页是图片）没有可提取的文字，需要先做 OCR。
- 竖排和旋转的文字保持原样不翻译。
- 复杂公式与正文混排的段落会作为整体送去翻译，公式中的特殊符号可能变成普通字符。
- 加密的 PDF 需要先解除密码。

## 开发

```bash
python -m unittest discover -v
```

测试使用假的翻译后端，不会调用任何 API。代码结构：

| 文件 | 内容 |
| --- | --- |
| `pdftrans/layout.py` | 段落提取、过滤、断行排版、对照页生成 |
| `pdftrans/llm.py` | OpenAI 兼容接口与 Claude 后端、批量翻译、缓存 |
| `pdftrans/pipeline.py` | 整体流程：解析 → 翻译 → 排版 → 输出 |
| `pdftrans/gui.py` | PySide6 图形界面 |
| `pdftrans/cli.py` | 命令行 |
| `pdftrans/config.py` | 设置、服务商预设、页码解析 |
