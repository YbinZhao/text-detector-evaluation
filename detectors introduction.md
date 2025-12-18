## BiScope  
[https://openreview.net/forum?id=Hew2JSDycr&referrer=%5Bthe+profile+of+Siyuan+Cheng%5D%28%2Fprofile%3Fid%3D~Siyuan_Cheng1%29&utm_source=chatgpt.com](https://openreview.net/forum?id=Hew2JSDycr&referrer=%5Bthe+profile+of+Siyuan+Cheng%5D%28%2Fprofile%3Fid%3D~Siyuan_Cheng1%29&utm_source=chatgpt.com)  
数据集：arxiv，code，yelp，essay，creative   
  

| 数据集 | 内容/领域 | 类型 | 人类文本来源（论文描述） | 规模（Normal） | 规模（Paraphrased，仅机器） | 长度特征（Normal，平均/最小–最大/中位数） |
| -------- | -------------------------- | ------------ | ------------------------------------------ | ------------------------------------ | ------------------- | ---------------------------------------------------------------------- |
| Arxiv | arXiv 学术写作（论文段落/摘要类） | 自然语言（短文本组之一） | 复用已有 Arxiv 人类数据，并用 5 个商业 LLM 生成对应机器文本 | Human 350 / Machine 1750（All 2100） | 1400 | Human 786.7 / 132–1736 / 715.0；Machine 787.1 / 101–1701 / 810.5 |
| Code | 编程代码（论文明确写为 Python code） | 代码 | 复用已有代码数据，并用 5 个商业 LLM 生成机器版本；作者强调这是“代码域”检测 | Human 164 / Machine 819（All 983） | 656 | Human 631.5 / 132–1993 / 572.0；Machine 413.3 / 41–1908 / 352.0 |
| Yelp | Yelp 评论（餐饮/服务评价文本） | 自然语言（短文本组之一） | 复用已有 Yelp 人类评论数据 + 5 个商业 LLM 生成对应机器文本 | Human 2000 / Machine 9740（All 11740） | 8000 | Human 554.9 / 37–4959 / 407.0；Machine 461.1 / 10–2548 / 414.0 |
| Essay | 学生作文/论文式长文（student essays） | 自然语言（长文本组之一） | 复用已有 Essay 人类长文 + 5 个商业 LLM 生成机器长文 | Human 1000 / Machine 4897（All 5897） | 3999 | Human 4249.9 / 1276–41470 / 3301.5；Machine 3827.7 / 515–21094 / 3486.0 |
| Creative | 创意写作（creative writing） | 自然语言（长文本组之一） | 复用已有 Creative 人类创作文本 + 5 个商业 LLM 生成机器创作文本 | Human 1000 / Machine 4840（All 5840） | 4000 | Human 2899.0 / 499–9933 / 2462.5；Machine 2851.9 / 176–13716 / 2620.0 |
  
  
对于所有数据集，我们重复使用人类生成的数据，并使用五个最新的商业LLM制作人工智能生成的文本：GPT-3.5-Turbo [36]、GPT-4-Turbo [1]、Claude-3-Sonnet [2]、Claude-3-Opus [2]和Gemini-1.0-Pro [42]。 因此，对于每个数据集，我们都有一段人工生成的数据和五段人工智能生成的数据。  
  
## fast detect gpt，detectgpt  
[[2310.05130] Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature](https://arxiv.org/abs/2310.05130)  
  
[https://arxiv.org/abs/2301.11305](https://arxiv.org/abs/2301.11305)  
  
数据集：  

| XSum           | 新闻摘要     | 新闻文本，常用于生成与检测  |
| -------------- | -------- | -------------- |
| SQuAD          | 百科 / 问答  | Wikipedia 语境文本 |
| WritingPrompts | 故事写作     | 长文本创作（fiction） |
| WMT16-EN       | 机器翻译（英文） | 多语言鲁棒性评测       |
| WMT16-DE       | 机器翻译（德文） | 非英文检测能力        |
| PubMedQA       | 生物医学问答   | 技术/学术写作场景      |
  
  
## Binoculars  
  
[https://arxiv.org/abs/2401.12070](https://arxiv.org/abs/2401.12070)  
  

| 数据集 | 领域/文本类型 | 人类文本来源（论文描述） | 机器文本来源（论文描述） | 数据集特性/用途 |
| ------------------------------------------------------ | ------------------------------------------------ | --------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| Writing Prompts | 创意写作（故事/续写类） | 来自 Ghostbuster 工作引入的三套基准之一（与 News、Student Essay 同组） | ChatGPT 生成的机器样本 | balanced（人/机数量相等）；用于“ChatGPT 检测”主对比基准 |
| News | 新闻文本 | 同上，Ghostbuster 引入的基准之一 | ChatGPT | balanced；用于主实验对比与阈值/低 FPR 指标展示 |
| Student Essay | 学生作文/论文式写作 | 同上，Ghostbuster 引入的基准之一 | ChatGPT | balanced；用于主实验对比，强调低 FPR（0.01%）下的 TPR |
| CCNews | 新闻语料（Common Crawl News） | 论文从 CCNews 抽取人类文本 | 用人类文本前 50 tokens 作 prompt，让 LLaMA-2 和 Falcon-7B 续写生成（再去掉 prompt，只保留纯机器续写段） | 用于构造“非 ChatGPT”的机器文本，做跨模型检测与OOD 阈值设定 |
| CNN | 新闻摘要/新闻文章语料 | 论文从 CNN 抽取人类文本 | 同 CCNews：用 LLaMA-2 / Falcon-7B 续写生成 | 同 CCNews：用于 OOD 阈值与跨模型检测对比 |
| PubMed | 生物医学论文摘要/文本 | 论文从 PubMed 抽取人类文本 | 同 CCNews：用 LLaMA-2 / Falcon-7B 续写生成 | 同 CCNews：增加“专业领域”分布外测试与 OOD 阈值设定 |
| Orca（Open-Orca / Orca dataset） | 指令数据（instruction / system prompt → completion） | 论文使用 Orca 提供的大规模指令对与回答 | 论文说明其包含来自 GPT-3 / GPT-4 的机器生成 completions | 用于评测/压力测试更强 API 模型输出（如 GPT-4）相关场景 |
| M4 dataset（Multi-generator/Multi-domain/Multi-lingual） | 多域+多语言（含 Arxiv/Reddit/Wikihow/Wikipedia 等，含多种语言） | 来自 M4 检测基准汇总的多源文本 | 论文说明该数据集的机器文本由 ChatGPT 生成 | 用于验证 跨域/跨语言泛化（论文在“Reliability in the Wild”里重点用它） |
  
  
  
  
  
  
