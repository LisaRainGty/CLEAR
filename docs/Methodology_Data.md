# Methodology/Data

## Methodology

### 1\. Data Pipeline

#### Stage A — 评论属性抽取与品类内标准化

**目标：**把 28,556 条评论中的每一次属性提及，映射到**每个品类内部标准化**的 `attribute_id`，为下游 \(product, attribute\) 粒度的聚合提供干净、可对齐的 aspect 流

**范围与原则：** 按品类独立处理；品类内部两阶段标准化（先以商品参数锚定，再以评论自由生成扩展）

**总体流程：**

```Plain Text
A0  品类 CAS 构建           商品"产品参数"字段            →  CAS_{cat}.json
A1  CAS 约束下评论抽取      评论 + CAS                    →  raw_aspects.jsonl
A2  自由生成聚合 → CAS+     FREE:: 类 aspect              →  CAS+_{cat}.json
A3  评论标签重写            FREE:: → 标准 id              →  resolved_aspects.jsonl
```

##### A0\. 品类 CAS（Category Attribute Schema）构建

输入：每品类下所有商品的 `产品参数` 字典。步骤：

1. 收集所有 param key，做粗归一化（去空格、全半角、大小写、标点）；

2. 用 BGE\-large\-zh 计算 embedding，层次聚类（阈值保守，宁可少合不可错合）；

3. 每个 cluster 交 LLM 裁决：判断是否同一属性 → 输出 `canonical_name` \+ `aliases`；

4. 扫该 key 下所有 value，自动标注 `value_type`：全数字 \+ 单位 → `numeric:<unit>`；有限枚举集 → `enum:[…]`；布尔 → `boolean`；其他 → `text`。

产出 `CAS_{cat}.json`，每条结构如下：

```JSON
{
  "attribute_id":   "FOOD_PROTEIN_CONTENT",
  "canonical_name": "蛋白质含量",
  "aliases":        ["蛋白质", "蛋白含量", "优质蛋白"],
  "value_type":     "numeric:g/100mL",
  "source_keys":    ["蛋白质(g/100mL)", "每百毫升蛋白质"],
  "source":         "param",
  "category":       "food.beverages"
}
```

##### A1\. CAS 约束下的评论开放抽取

对每条评论，把其所属商品品类的 CAS（`canonical_name` \+ `aliases`）作为优先映射表传入 LLM，要求：能映射到 CAS 就用 CAS 的 `attribute_id`；不能映射才以 `FREE::<电商参数风格的名词短语>` 自由生成。LLM 同时输出 `polarity / evidence_span / type / explicit_fact_hit / mention_strength`。

Prompt 核心结构：

```Plain Text
[品类标准属性]  列出该品类 CAS 全部 canonical_name + aliases
[评论]          review_text

对评论中提及的每个 aspect 输出 JSON：
  attribute_id:
    - 能映射到 CAS → 用该 attribute_id
    - 不能         → "FREE::<电商参数风格的名词短语>"
                     (名词短语、客观、产品或服务导向，不带评价形容词)
  polarity:          pos | neg | neu
  evidence_span:     原文 ≤ 30 字
  type:              attribute | service | personal
  explicit_fact_hit: 是否含"说是/宣传的/写的/标的"等对宣传的印证或反驳信号
  mention_strength:  strong | weak
```

过滤规则：

|`type`|处理|
|---|---|
|`attribute`|保留，进主流程|
|`service`|保留，进主流程（服务类虚假宣传同为本研究范围）|
|`personal`|丢弃（消费情境与个人偏好，不可对齐）|

产出 `raw_aspects.jsonl`，每行：

```JSON
{
  "review_id": "...", "product_id": "...", "category": "...",
  "attribute_id": "FOOD_PROTEIN_CONTENT",
  "polarity": "neg", "evidence_span": "蛋白没说的那么高",
  "type": "attribute", "explicit_fact_hit": true, "mention_strength": "strong"
}
```

##### A2\. 自由生成属性聚合 → CAS\+

只处理 `attribute_id` 以 `FREE::` 开头的条目，**按品类独立分桶，不做跨品类合并**。步骤：

1. 字符串 Jaccard \> 0\.8 粗去重；

2. BGE embedding \+ Agglomerative 聚类（距离阈值 ≈ 0\.25）；

3. 每个 cluster 交 LLM，对照该品类现有 CAS 判定：

    - 是某 CAS 属性的同义变体**且** `value_type` 一致 → 并入（追加到 `aliases`）；

    - 否则 → 作为该品类 CAS 的新条目追加，标 `source = "review"`。

##### A3\. 评论标签重写

遍历 `raw_aspects.jsonl`：

- 非 `FREE::` 条目 → 原样保留；

- `FREE::` 条目 → 按 A2 输出的合并/新增表替换为 CAS\+ 的标准 `attribute_id`；

- 实在找不到的（预计 \< 1%）→ 写入 `unresolved_pool.jsonl`，人工抽查。

产出 `resolved_aspects.jsonl`：字段与 A1 输出一致，`attribute_id` 全部标准化。

##### 最终 CAS\+ Schema 字段

```JSON
{
  "attribute_id":   "<品类前缀>_<属性ID>",
  "canonical_name": "<品类内唯一标准名>",
  "aliases":        ["<原始 param key + 评论变体>"],
  "value_type":     "numeric:<unit> | enum:[...] | boolean | text | text:subjective",
  "source":         "param | review | both",
  "category":       "<品类>"
}
```

##### 结果示例（娟姗牛奶，`food.beverages`）

|评论原文|`type`|`attribute_id`|`polarity`|`explicit_fact_hit`|
|---|---|---|---|---|
|"每100ml蛋白4g确实挺高"|attribute|`FOOD_PROTEIN_CONTENT`|pos|true|
|"感觉蛋白没宣传的那么高"|attribute|`FOOD_PROTEIN_CONTENT`|neg|true|
|"口感说的丝滑 没觉得"|attribute|`FOOD_MILK_TASTE_SMOOTH`|neg|true|
|"产地内蒙古"|attribute|`FOOD_ORIGIN_PLACE`|neu|false|
|"娟姗奶就是贵"|attribute|`FOOD_PRICE_PERCEPTION`|neg|false|
|"发货很慢"|service|`FOOD_DELIVERY_SPEED`|neg|false|
|"包装破了"|service|`FOOD_PACKAGING_INTEGRITY`|neg|false|
|"我闺女爱喝"|personal|—（丢弃）|—|—|

#### Stage B — \(商品, 属性\) 聚合、主播话术 claim 抽取与评论对齐

**目标。** 在 Stage A 已完成属性标准化的基础上，以 `(product_id, attribute_id)` 为单位完成三件事：\(i\) 对每个商品 $p$ 取**评论侧已命中的属性集合** $A_{\text{cmt}}(p) \subseteq A_{\text{cat}}$ 作为 pair 的候选集与 claim 抽取的强约束 schema；\(ii\) 在 $A_{\text{cmt}}(p)$ 限定下从该商品全部 SRT 中抽出与每个属性相关的主播原话片段并保留时间戳；\(iii\) 对每条评论与对应主播话术做对齐判定，追加评论级 `y_supportability` 字段。

**核心方法选型。** 主播 claim 抽取采用 **LangExtract**（Google, 2025；Apache 2\.0）——schema\-guided extraction \+ 强制 source\-span grounding 的开源 Python 库，后端 **Gemini 2\.5 Flash**（全量启用 Batch API，标准价五折）。

**总体流程。**

```Plain Text
B0  评论侧候选属性集       Stage A 输出按商品聚合       →  A_cmt(p) for each product
B1  原子 claim 抽取        SRT + A_cmt(p)（商品级 schema） →  claim_list[product_id].jsonl
B2  pair 枚举              直接复用 A_cmt(p)             →  pair_skeleton.jsonl
B3  claim passage 拼接     按 pair groupby + 时序去重    →  pair（含 passage 与 segments）
B4  评论 × claim 对齐      Gemini Flash JSON mode        →  per-review y_supportability
B5  统计量收尾             3 个原子计数                  →  pair_records.jsonl
```

##### B1\. 原子 claim 抽取（每商品 1 次 LangExtract 调用）

**SRT 预处理。** 把同一 `product_id` 名下所有 clip 的 SRT 按时间升序拼接成单一长文本，clip 间插入可识别的分隔符 `\n===CLIP_BREAK|<srt_file>===\n`；同时为每条 cue 记录一张边界表，字段包括字符偏移区间 `char_start / char_end`、来源文件 `srt_file`、cue 序号 `cue_idx`、起止时间戳 `start_ts / end_ts`。该边界表是后续把 LangExtract 返回的 `char_interval` 字符区间反查为 SRT 时间戳的唯一依据。

**LangExtract 输出 schema。** 保留三件事：抽取类、原话子串、属性归类。

|字段|来源|含义|
|---|---|---|
|`extraction_class`|固定 `"attribute_claim"`|标识本任务的抽取类|
|`extraction_text`|**LangExtract 内置顶层字段**|主播 SRT 中的连续原话子串；库自动校验是否真实出现于源文本，未对齐者回 `char_interval = None` 由后处理丢弃|
|`attributes.attribute_id`|用户自定义|必须 ∈ 该商品评论侧候选集 $A_{\text{cmt}}(p)$，越界条目丢弃|

**Prompt 主体。**

```Plain Text
角色：电商直播事实抽取员。
任务：从下面主播口播文本中，抽取所有针对【给定候选属性集合内某个属性】的独立陈述。

硬约束：
1. extraction_text 必须是 SRT 原文中真实存在的连续字符串，禁止改写、概括或跨段拼接。
2. attributes.attribute_id 必须取自下方候选属性集合（即该商品评论侧已命中的属性子集）；
   不能归类到其中任何一项的，一律不要输出（包括主播提到但用户从未评论的属性）。
3. 同一属性的口语复读只抽表达最完整的一次；不同角度（如容量 vs 件数）拆为独立 extraction。
4. 忽略下单话术 / 主播八卦 / 与商品属性无关的内容。

本商品候选属性集合 A_cmt(p):
{该商品评论侧已命中的属性子集，每条形如 {attribute_id, canonical_name, aliases}}
```

**Few\-shot 示例。** 每品类准备 2–3 条人工示例；LangExtract 会自动校验示例 `extraction_text` 是否在示例文本中存在（不存在会触发 `Prompt alignment` 警告，需修正）。

**后处理（schema 硬校验 \+ 时间戳反查）。** 对 LangExtract 返回的每条 extraction 依次执行四步：

1. 若 `char_interval = None`（库自动判定无法在源文本对齐的幻觉抽取），丢弃；

2. 若 `attributes.attribute_id` 不在该商品评论侧候选集 $A_{\text{cmt}}(p)$ 内（即模型违反硬约束 2 自创了集合外标签），丢弃；

3. 用 `char_interval.start_pos / end_pos` 在 cue 边界表中反查得到 `srt_file / start_ts / end_ts`；

4. 写入 `claim_list[product_id].jsonl` 一条记录，字段如下：

```JSON
{
  "claim_id":     "<product_id>_<seq>",
  "attribute_id": "FOOD_PROTEIN_CONTENT",
  "claim_text":   "每100毫升蛋白4克，远高于普通牛奶",
  "srt_file":     ".../241206_151019.srt",
  "start_ts":     "00:12:45,300",
  "end_ts":       "00:12:49,800"
}
```

其中 `claim_text` 直接取自 LangExtract 的 `extraction_text` 字段，确保所有 claim 都是 SRT 原话连续子串。

##### B2\. \(商品, 属性\) pair 枚举

pair 总数 $\le \sum_p |A_{\text{cmt}}(p)|$，按 Stage A 实测 $\bar{|A_{\text{cmt}}(p)|} \approx 5\text{–}15$，预计 ≈ 4,000–9,000 pairs。

每个 pair 的 claim 侧根据 B1 实际抽到的命中数分两类：

- $A_{\text{cmt}}(p)$ 中**主播在 SRT 中确实有相关表述**的属性：`has_claim_srt = True`，主进入 §2 的标签判定；

- $A_{\text{cmt}}(p)$ 中**主播 SRT 未提及**的属性（用户评论了但主播没讲）：`has_claim_srt = False`，pair 仍保留，由 §2 的 $f_{\text{cov}}$ 自动降权——这一类 pair 仍可能因评论侧出现强烈不满\+`explicit_fact_hit` 而被打 $y=1$（"主播未提"也是一种隐性虚假宣传线索）

##### B3\. Claim passage 拼接

对每个保留的 pair，从 `claim_list[product_id]` 取出全部匹配 `attribute_id` 的 claim，按下列四步生成 passage：

1. 按 `(srt_file, start_ts)` 升序；

2. 相邻 claim 做归一化字符 Jaccard ≥ 0\.9 去重（防主播复读）；

3. 时序拼接 `claim_text`，相邻 claim 之间用分隔符 `\n---\n` 隔开；

4. 长度截断：若 passage token \> 600，沿时间轴均匀下采样保留 ≈ 500 tokens。所有 `segments` 元数据完整保留——passage 仅用于 B4 的 LLM prompt，溯源不受影响。

##### B4\. 评论 × claim 对齐判定

**职责唯一**：为每条评论新增字段 `y_supportability ∈ {0, 1}`。

**Prompt**

```Plain Text
角色：电商直播虚假宣传审查员。
属性: {attribute_canonical}（同义词：{aliases}）
主播口播（按时间拼接）:
  {passage 或 "【本商品无主播相关口播】"}

任务：对下面每条评论判断它是否针对主播口播中关于该属性的"具体表述"做出了直接回应？

判 y_supportability=1 的条件：
  评论指向 claim 中具体说法（同向肯定 / 反向否定均算）。例如：
    主播："每盒 125 毫升"  + 评论："125ml 这个量也太少了" → 1
    主播："口感丝滑"        + 评论："说丝滑没觉得"         → 1

判 y_supportability=0 的条件：
  评论仅泛泛谈论该属性、未指向具体说法（如"还行"、"还可以"）；
  或评论与主播口播无可比较点；
  当主播口播为空时，强制 y_supportability=0。

输入评论（编号 1..K）: {reviews}

对每条评论严格输出: {"cid": int, "y_supportability": 0|1}
```

##### B5\. Pair 级聚合记录（最小 schema）

```JSON
{
  "pair_id":             "p<product_id>__<attribute_id>",
  "product_id":          "...",
  "category":            "food.beverages",
  "attribute_id":        "FOOD_PROTEIN_CONTENT",
  "attribute_canonical": "蛋白质含量",

  "claim": {
    "has_claim_srt": true,
    "passage":       "每100毫升蛋白4克，远高于普通牛奶\n---\n这个营养价值是真的高",
    "segments": [
      {
        "claim_id":   "12345_3",
        "claim_text": "每100毫升蛋白4克，远高于普通牛奶",
        "srt_file":   ".../241206_151019.srt",
        "start_ts":   "00:12:45,300",
        "end_ts":     "00:12:49,800"
      }
    ]
  },

  "reviews": [
    {
      "comment_id":        "...",
      "text":              "感觉蛋白没宣传的那么高",
      "polarity":          "neg",         *// ← from Stage A，原样透传*
      "mention_strength":  "strong",      *// ← from Stage A*
      "explicit_fact_hit": true,          *// ← from Stage A*
      "evidence_span":     "蛋白没说的那么高",   *// ← from Stage A*
      "y_supportability":  1              *// ← B4 新增*
    }
  ],

  "stats": {
    "N_total":       12,    *//  = len(reviews)*
    "N_aligned":     5,     *//  = Σ 1[y_supportability=1]*
    "N_aligned_neg": 2      *//  = Σ 1[y_supportability=1 ∧ polarity=neg]*
  }
}
```

#### Stage C — 商品事实抽取（产品参数 \+ 图文详情）

**目标。** 以 $A_{\text{cmt}}(p)$ 中的**每一个 attribute\_id 为主体**，依次到三个商品事实源中**定向取证**并**全部保留**：\(i\) 结构化产品参数字典；\(ii\) 详情图 OCR 文本；\(iii\) 详情图/主图视觉。

**总体流程。**

```Plain Text
C1  详情图分流         单图 VLM 9 类分类 + 代表图采样           →  image_index[p]
C2  逐属性 × params    按 alias 反查 + LLM 兜底定向匹配         →  evidence_params[p][a]
C3  逐属性 × OCR       PaddleOCR + Gemini Flash 按 attr 取证    →  evidence_ocr[p][a]
C4  逐属性 × VLM       主图 + 代表图 multi-image 按 attr 取证   →  evidence_vlm[p][a]
C5  三源证据并列输出   三源 evidence list 原样保留              →  fact_records.jsonl
```

##### C1\. 详情图分流（VLM 单图 9 类分类）

**作用。** 给后续 C3 \(OCR\) 与 C4 \(VLM\) 决定该扫哪些图。这一步本身不做属性级取证，只标 category。

**分类体系：**

|category|描述|下游处理|
|---|---|---|
|`spec_table`|文字规格表 / 参数列表图|C3 OCR|
|`certificate`|认证检测图 / 资质证书|C3 OCR|
|`size_chart`|尺寸表 / 码数对照|C3 OCR|
|`material_closeup`|材质特写 / 纹理细节|C4 VLM|
|`product_photo`|产品实拍 / 多角度展示|C4 VLM|
|`scene_demo`|场景使用图 / 模特展示|C4 VLM|
|`packaging`|包装展示|C4 VLM|
|`comparison`|颜色 / 款式对比|C4 VLM|
|`other`|其他|C4 VLM|

只输出 `{"category": "<one_of_9>"}`，无视觉描述。

##### C2\. 逐属性 × 产品参数证据抽取

**输入。** $A_{\text{cmt}}(p)$ \+ 该商品 `产品参数` 字典（来自 `product_index.json[products][p]['产品参数']`，例如 `{"蛋白质含量":"4g/100mL", "品牌":"娟姗", "产地":"内蒙古", ...}`）。

**主体倒装：以 attribute 为索引**，对每个 attribute\_id 在 params 字典中收集所有语义对应的 \(key, value\) 对。

**两阶段定向匹配策略。**

1. **alias 反查（无 LLM）：** 对每个 attribute\_id $a$，把 CAS\+ 中 $a$ 的 `aliases ∪ canonical_name` 作为字符串模式集，与 params 字典所有 key 做归一化匹配（小写 \+ 去标点 \+ 简繁统一）。

2. **LLM 兜底（仅 alias 反查后仍空 list 的 attribute）：** 把这些"漏网" attribute 列表 \+ 该商品全部 params key\-value 一并送 Gemini 2\.5 Flash 一次调用，prompt：

```Plain Text
角色：电商产品参数到属性的语义匹配员。
任务：对下列每个 attribute_id，从 params 字典中挑出语义对应的所有条目（多对一，
全部列出）；params 中没有任何条目能对应的，明确返回空。
硬约束：
1. 只允许从下方 params 中取条目，禁止改写或自创取值。
2. attribute_id 必须取自下方候选属性列表。
3. 同一 attribute 下的多条 params 命中全部保留。

候选属性列表 A_cmt(p):
{该商品评论侧已命中的属性子集}

products 字典:
{key1: value1, key2: value2, ...}

每个 attribute 输出: {"attribute_id": "...", "matches": [{"param_key": "...", "raw_text": "..."}]}
```

**输出。** `evidence_params[product_id][attribute_id] = list of {param_key, raw_text}`；alias 反查 \+ LLM 兜底两轮均未命中的 attribute，list 为空。

##### C3\. 逐属性 × OCR 文本证据抽取

**输入。** $A_{\text{cmt}}(p)$ \+ 该商品所有归入 `spec_table / certificate / size_chart` 类的详情图。

**OCR 引擎。** PaddleOCR PP\-OCRv4（中文优化、本地、零 API 成本）。每张文字密集类图片产出一段 `ocr_raw_text`。

**LLM 定向匹配。** 把多张图的 `ocr_raw_text`（每段前缀 `=== <image_path> ===`）拼接成一段长文本，送 Gemini Flash 一次调用，prompt：

```Plain Text
角色：电商详情页结构化字段定向抽取员。
任务：对下方候选属性列表中的每一个 attribute_id，扫遍 OCR 文本，
找出所有能直接支撑该 attribute 的连续原文片段（多条全部列出，禁改写）；
找不到的明确返回空。
硬约束：
1. raw_text 必须是 OCR 原文中真实存在的连续字符串。
2. attribute_id 必须取自下方候选属性列表。
3. 每条 raw_text 必须报告它所在的 image_path（取拼接文本中相应的 === <path> === 前缀）。
4. 同一 attribute 在多张图、多句话中出现的，全部保留。
候选属性列表 A_cmt(p):
{该商品评论侧已命中的属性子集}
OCR 文本（多图拼接）:
=== <image_path_1> ===
<ocr_raw_text_1>
=== <image_path_2> ===
<ocr_raw_text_2>
...

每个 attribute 输出: {"attribute_id": "...", "matches": [{"raw_text": "...", "image_path": "..."}]}
```

**输出。** `evidence_ocr[product_id][attribute_id] = list of {raw_text, image_path}`；未命中的 attribute list 为空。

##### C4\. 逐属性 × VLM 视觉证据抽取

**输入。** $A_{\text{cmt}}(p)$ \+ 商品标题 \+ 主图 \+ C1 选出的代表图（≤ 8 张，含 category 标签）。

**调用方式。** 一次 multi\-image 调用（Qwen2\.5\-VL\-Max 或 Gemini 2\.5 Flash multi\-image），prompt 主体：

```Plain Text
角色：电商商品多模态视觉取证员。
任务：对下方候选属性列表中的每一个 attribute_id，逐一审视主图与详情图，
给出所有能在视觉上直接观察到的客观证据（多张图、多个细节全部列出）；
图中观察不到的明确返回空，禁出现"看起来 / 可能 / 大概"。
硬约束：
1. 只输出能被视觉证据直接支撑的项；不能在图中观察到的不要硬编。
2. attribute_id 必须取自下方候选属性列表。
3. 每条证据必须报告对应的 image_path。
4. raw_quote 是对图片视觉细节的简短客观描述（如"主图盒身正面标'4g 蛋白'"
   /"详情图 4 中模特年龄约 20–30 岁"），不复述商品标题或参数文本。
5. 同一 attribute 在多张图中出现的视觉证据全部保留。

候选属性列表 A_cmt(p):
{该商品评论侧已命中的属性子集}

输入图（已含 category 标签）: <主图> + <代表图 1> + ... + <代表图 N>

每个 attribute 输出: {"attribute_id": "...", "matches": [{"raw_quote": "...", "image_path": "..."}]}
```

**输出。** `evidence_vlm[product_id][attribute_id] = list of {raw_quote, image_path}`；未命中的 attribute list 为空。

##### C5\. 三源证据并列输出（无需 LLM）

对每个 \(p, a\) ∈ $A_{\text{cmt}}(p)$，把 C2 / C3 / C4 三源各自的 evidence list **原样并列**到一条 `fact_record`，不做任何裁决、归一化或合并。配套统计两个轻量字段：

- `evidence_count = {"params": n1, "ocr": n2, "vlm": n3}`：三源各自命中的证据条数；

- `coverage = sum(1[ni > 0]) ∈ {0, 1, 2, 3}`：命中的源数。

**Confidence 三档。** 仅依据 `coverage` 决定，不引入"哪源更可信"的先验：

|条件|confidence|
|---|---|
|`coverage = 3`（三源都有证据）|high|
|`coverage = 2`|medium|
|`coverage = 1`|low|
|`coverage = 0`|absent|

##### Pair 级输出 schema（最小必要字段）

```JSON
{
  "fact_id":         "f<product_id>__<attribute_id>",
  "product_id":      "...",
  "category":        "food.beverages",
  "attribute_id":    "FOOD_PROTEIN_CONTENT",

  "evidence_params": [
    {"param_key": "蛋白质含量", "raw_text": "4g/100mL"}
  ],
  "evidence_ocr": [
    {"raw_text": "蛋白≥4.0g/100mL", "image_path": "/mnt/gty/product_images/<pid>/detail_00jpeg"}
  ],
  "evidence_vlm": [
    {"raw_quote": "主图盒身正面标'4g 蛋白'", "image_path": "/mnt/gty/product_images/<pid>/main.webp"}
  ],

  "evidence_count":  {"params": 1, "ocr": 1, "vlm": 1},
  "coverage":        3,
  "confidence":      "high"
}
```

### 2\.标签生成与样本权重

本节描述如何把 Stage D 产出的"评论—话术"对齐信号聚合成 \(product, attribute\) pair 级的**硬二元标签** $y \in \{0, 1\}$（0 = truthful，1 = misleading）与**样本权重** $c \in [0, 1]$，作为下游分类器的训练监督

设计围绕两条在电商语境下已被反复证实的非对称性：

1. **负向偏差（negativity bias）。** 负向反馈是高成本、高诊断性的信号 \[[Baumeister et al\., 2001;](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-DkVNdrXNqoWYtCxpUhRceNZkntg) [Chen, Fay \& Wang, 2011](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-SBA4dnmU8ooktWxMCBWcJtevn4b)\]；相较于正评，负评更少被操纵。

2\. 正评的系统性操纵。 国内外电商平台的 fake\-positive review 比例被多项实证研究测算在 15–30% 区间 \[[Mayzlin, Dover \& Chevalier, 2014, ](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-B2ITdqetjoYGqUxoCcGcgn2AnsU)[*American Economic Review*](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-B2ITdqetjoYGqUxoCcGcgn2AnsU); [Luca \& Zervas, 2016, ](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-KaPedgRZNoDeLfxuWnRc5Y9knLf)[*Management Science*](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-KaPedgRZNoDeLfxuWnRc5Y9knLf); [He, Hollenbeck \& Proserpio, 2022, ](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-SjVidTsDzoFY4FxJlFMcRmM4nCe)[*Marketing Science*](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-SjVidTsDzoFY4FxJlFMcRmM4nCe)\]；正评不可直接视为等效证据。

#### 2\.1 硬标签规则

记 aligned 评论集合为 $\mathcal{A}_{p,a}$（Stage D 对 pair $(p, a)$ 做 per\-review 对齐判定得到的子集，`y_supportability_i = 1`）。

$y_{p,a} = \begin{cases} 1 & \text{若 } \exists\, i \in \mathcal{A}_{p,a}:\ \text{polarity}_i = \text{neg} \\ 0 & \text{otherwise} \end{cases}$

即**只要存在至少一条既与主播话术对齐、又表达负向情感的评论**，即判该 pair 为 misleading。标签的粗糙性通过 sample weight $c$ 进一步调节（见 §2\.3–2\.4）。范式来源：远监督下用启发式规则生成硬标签 \[[Mintz et al\., 2009, ACL](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-B2cYdqFltoS2gOxOKT4cSO8VnTg)\]，再由后续的噪声学习机制加以校正 \[[Han et al\., 2018, NeurIPS](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-XAR1djSOBokWi9xZNiHc8LT7ntc); [Northcutt et al\., 2021, JAIR](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-FK5adbz49o8gBexUgbdc5YxQnrb)\]。

#### 2\.2 评论级证据分

对每条 aligned 评论 $i$ 计算证据强度 $w_i$：

$w_i = (1 + \gamma \cdot \mathbb{1}[\text{explicit\_fact\_hit}_i]) \cdot s_i$

- $\gamma$ — 显式反驳/印证信号（Stage A 的 `explicit_fact_hit = 1`）的加成权重，默认 $\gamma = 2$，使含显式信号的评论有效权重变为普通评论的 3 倍；

- $s_i$ — 提及强度乘子，`mention_strength = strong` 时 $s_i = 1.2$，`weak` 时 $s_i = 0.7$。

显式反驳信号的诊断价值已在消费者抱怨行为文献中被证实；强度乘子反映 "bad is stronger than good" 的认知加权 \[[Baumeister et al\., 2001](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-Gtk7dJAAkoksnQxnkvRcth5cn7b)\]。

#### 2\.3 样本权重的三因子

定义 pair 级聚合量：



$S_{\text{neg}}(p,a) = \sum_{\substack{i \in \mathcal{A}_{p,a} \\ \text{polarity}_i=\text{neg}}} w_i, \qquad
S_{\text{pos}}(p,a) = \sum_{\substack{i \in \mathcal{A}_{p,a} \\ \text{polarity}_i=\text{pos}}} w_i$

$N_{\text{aligned}} = |\mathcal{A}_{p,a}|$，$N_{\text{total}}$ 为该 pair 的总评论数。基础权重由下列因子相乘：

**\(1\) 证据饱和（evidence saturation）。**

$f_{\text{sat}} = 1 - \exp(-N_{\text{aligned}} / k), \qquad k = 3$

聚合标签的可靠性随证据数呈边际递减，饱和曲线是 crowd\-aggregation / Bayesian label model 的标准形式 \[[Dawid \& Skene, 1979, ](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-VBZEdeCPtozQUCxPtq7cCVUinRf)[*Applied Statistics*](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-VBZEdeCPtozQUCxPtq7cCVUinRf); [Raykar et al\., 2010, JMLR](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-DrsRd5I6voxGjFxB1E6cgiXXn3n)\]。

**\(2\) 对齐覆盖率（alignment coverage）。**

$f_{\text{cov}} = \frac{N_{\text{aligned}}}{N_{\text{total}} + 1}$

该因子在两条独立文献线上同时有 grounding：

- 弱监督 / Data Programming 视角：每条 aligned 评论是一个 labeling function 的一次触发，$N_{\text{aligned}}/N_{\text{total}}$ 对应 Snorkel 标签模型中的 *labeling function coverage* \[[Ratner et al\., 2016, 2017](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-SCRUd1akzo1zkcxQYeOcNdBInKb)\]，低 coverage 源自动降权。

- 在线评论自选择视角：评论者对话题发言存在系统性选择偏差，低覆盖意味着 effective sample size 远小于 $N_{\text{total}}$，需做置信度校正 \[[Li \& Hitt, 2008, ](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-QORrdIoMjoCmyPxWqs1cj62Dnuh)[*Information Systems Research*](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-QORrdIoMjoCmyPxWqs1cj62Dnuh)\]。

- **证据检索视角**：证据密度与 claim verification 置信度同向 \[[Thorne et al\., 2018, NAACL](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-QcfadQtDwoQ4MVxyzFqcm09XnCg); [Soleimani et al\., 2020, ECIR](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-FaBed6Ro1oDHWXxTDe5cBgqCnbg)\]。

基础权重：

$c_{\text{base}}(p,a) = f_{\text{sat}} \cdot f_{\text{cov}}$

**\(3\) 非对称一致性（asymmetric consistency） — 仅用于 **$y = 1$** 分支。**

$f_{\text{asym}} = \frac{S_{\text{neg}}}{S_{\text{neg}} + \lambda \cdot S_{\text{pos}}}$

$\lambda < 1$ 把正评按可信度折扣，反映 fake\-positive 的系统性操纵 \[[Mayzlin et al\., 2014](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-ETsOd2NTqoeyWCxdRDtc8VBZnUf); [He et al\., 2022](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-PK39dJeproqgf3x9DDzcEo0anHb)\]。默认 $\lambda = 0.3$，近似对应文献中实证的 fake\-positive 比例上限；$\lambda = 1$ 退化为对称聚合，可作为消融对照。

**\(4\) 刷单嫌疑惩罚（fake\-review discount） — 仅用于 **$y = 0$** 分支。**

$f_{\text{fake}} = 1 - \rho \cdot \mathbb{1}[\text{suspected fake}]$

suspected fake判定规则（任一满足即标记）：

- \(a\) $N_{\text{total}} < 10$ 且全部 polarity = pos；

- \(b\) aligned 评论在 bigram 空间的 Jaccard 多样性低于阈值（措辞高度同质化）；

- \(c\) aligned 评论全部来自 $\leq 3$ 天时间窗（突发集中灌水）。

默认 $\rho = 0.4$。规则依据：opinion spam detection 的核心特征工程 \[[Jindal \& Liu, 2008, WSDM;](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-J51WdD86poiKZ1xueI0cajFnnSd) [Mukherjee et al\., 2013, ICWSM](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-MeyZdyXQaoDnxdxkiBCcXIBGnLb)\]。

#### 2\.4 最终权重公式

$c = \begin{cases}
  \min\!\bigl(1,\; c_{\text{base}} \cdot f_{\text{asym}} \cdot \phi_{\text{bonus}} \bigr) & \text{if } y = 1 \\[4pt]
  c_{\text{base}} \cdot f_{\text{fake}} & \text{if } y = 0
\end{cases}$

$c \leftarrow \max(c,\; 0.05) \quad \text{(下限防止样本被完全丢弃)}$

$\phi_{\text{bonus}} = 1.2$ 当 $\mathcal{A}_{p,a}$ 中的 neg 评论至少存在一条 `strong` 或 `explicit_fact_hit = 1`，否则 $\phi_{\text{bonus}} = 1.0$

#### 2\.5 训练时的权重用法

样本权重以 per\-sample multiplier 形式进入所有训练损失项：

$\mathcal{L}_{\text{total}} = \sum_{j} c_j \cdot \mathcal{L}_{\text{CE}}(y_j, \hat{y}_j) + \lambda_{\text{con}} \sum_{j} c_j \cdot \mathcal{L}_{\text{contrastive},\, j}$

该形式参考 robust deep learning 中的 *learning to reweight* \[[Ren et al\., 2018, ICML](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-V82FdX9D1oatNqxQbfecNEgdnFg)\]、*MentorNet* \[[Jiang et al\., 2018, ICML](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-XdEZdU9mponAFkxeblFcjDpknch)\]、*Co\-teaching* \[[Han et al\., 2018, NeurIPS](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-G3audaEr6oUq0IxVHm7cnkUznVf)\] 

#### 2\.6 默认参数和敏感性分析

|参数|含义|默认值|消融 sweep|
|---|---|---|---|
|$\gamma$|explicit 信号加成|2|\{1, 2, 3\}|
|$k$|证据饱和速率|3|\{2, 3, 5\}|
|$\lambda$|pos 证据不对称折扣|0\.3|\{0\.2, 0\.3, 0\.5, **1\.0**\}|
|$\rho$|刷单嫌疑惩罚|0\.4|\{0\.2, 0\.4, 0\.6\}|
|$\phi_{\text{bonus}}$|强 neg 证据加成|1\.2|\{1\.0, 1\.2, 1\.5\}|
|\(可选\) $\beta$|coverage 软化指数 $f_{\text{cov}}^{\beta}$|1\.0|\{0, 0\.5, 1\}|

$\lambda = 1.0$ 的 sweep 对应**对称聚合基线**；若不对称设置显著优于 $\lambda = 1.0$，即构成中国电商语境下不对称弱监督显著优于对称聚合的发现。$\beta$ sweep 可借鉴 BM25 频率软化思路 \[[Robertson \& Zaragoza, 2009](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-XSpZdyNZBogUHPxkimDcpif6nyf)\]。

**鲁棒性诊断。** 在一个人工清洗子集上注入 fake\-positive / outlier\-negative 噪声，验证 $c$ 能否正确降权噪声样本，对应 Confident Learning 的标签噪声检测思路 \[[Northcutt et al\., 2021, JAIR](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-NqSfdbIKjoR25Uxw3S5cPnxDn2c)\]。

### 模型输入编码

由于样本粒度已下沉到“商品 × 属性”，每条样本的监督目标即：**该商品在该属性上的主播话术，是否存在用户感知的虚假宣传风险**。本节负责把一条 record 转成 §4 模型输入张量的统一接口——双流 tokenization。

#### 3\.1 Dual\-Stream Tokenization 

每条 pair record 被切成两条独立 token 流，分别经 §4\.2 的共享 encoder 编码，再在 §4\.3 的 fusion 层做跨流交互：

|流|内容来源|角色|
|---|---|---|
|Claim Flow $X^c$|`attribute_name` \+ `claim.segments[]` 聚合后的属性级主播话术 bundle|该属性下主播对消费者形成预期的全部语言证据|
|Evidence Flow $X^e$|`attribute_name` \+ 三源商品事实（`evidence_params / evidence_ocr / evidence_vlm`）|该属性下商品事实能够支持的客观证据集合|

#### 3\.2 特殊 Token Schema

新增 6 类 task\-specific special tokens（追加进 BGE tokenizer 的 added\_tokens 列表，embedding 随机初始化、随训练更新）：

|Token|用途|出现位置|
|---|---|---|
|`[ATTR]`|属性锚点|Claim Flow 与 Evidence Flow 起始处，后接 `attribute_name`|
|`[CLM]`|属性级主播话术块锚点|Claim Flow 中每个 claim segment 起始处|
|`[CLM_NULL]`|主播未提及该属性时的占位锚点|Claim Flow 中（`has_claim_srt = False` 时）|
|`[EVD]`|Evidence Flow 的总锚点|Evidence Flow 的 `[ATTR] {name}` 之后|
|`[PARAM] / [OCR] / [VLM]`|三源证据的源标识|Evidence Flow 中每条证据 list 起始处|
|`[SEP_C] / [SEP_E]`|claim segment / evidence item 分隔符|Claim Flow 与 Evidence Flow 内部|

`[CLM]` 仅作为主播话术片段的边界标记，帮助 encoder 区分该属性下多次重复表述的内部结构，本身不承载任何额外监督信号。

#### 3\.3 Claim Flow 构造

```Plain Text
X^c = [CLS] [ATTR] {attribute_name}
      [CLM] {segment_1.text} [SEP_C]
      [CLM] {segment_2.text} [SEP_C]
      ...
      [CLM] {segment_K.text} [SEP]
```



- $K = $ `len(claim.segments)`；当 `has_claim_srt = False` 时退化为 `[CLS] [ATTR] {attribute_name} [CLM_NULL] [SEP]`，保持 batch shape 不变；

- 每个 `segment.text` 经 `tokenizer.encode(text, add_special_tokens=False)` 后接到对应 `[CLM]` 之后；

- 总长度上限 $L_c = 384$ tokens；超出时按 §1\.2 B3 的同策略沿时间轴均匀下采样 segments

#### 3\.4 Evidence Flow 构造

```Plain Text
X^e = [CLS] [ATTR] {attribute_name} [EVD]
      [PARAM] {evidence_params[0].raw_text} [SEP_E] {evidence_params[1].raw_text} ... [SEP_E]
      [OCR]   {evidence_ocr[0].raw_text}    [SEP_E] {evidence_ocr[1].raw_text}    ... [SEP_E]
      [VLM]   {evidence_vlm[0].raw_quote}   [SEP_E] {evidence_vlm[1].raw_quote}   ... [SEP_E]
```

- 三源 list 经 §1\.3 已做定向取证，按命中顺序拼接（不重排序，保留三源各自的命中顺序弱信号）；

- 当某源 `evidence_count[src] = 0` 时，对应整段（含源标识 token）省略；

- 总长度上限 $L_e = 384$ tokens，按 “PARAM 全保 → OCR 后截 → VLM 后截” 的优先级裁剪。

### 4\.Models — CLAIMARC Framework

CLAIMARC = Claim\-Aware Misleading\-Advertising detector with Retrieval\-augmented Contrastive learning

#### 4\.1 Architecture Overview

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTZlODY5NjEwMGY5NmI1MWMzM2YwODFhZDExNGMzOTJfN2QyMGM2MDUzZWQ0ZmU4NzkyOWEwZTA4ZThjZDkyNTNfSUQ6NzYzNTU1ODMzODkwNTg2OTI3NF8xNzgwNTQwNzczOjE3ODA2MjcxNzNfVjM)

#### 4\.2 Shared Encoder with LoRA

**骨干选择。** BGE\-large\-zh\-v1\.5（BAAI, 2024；24 层 Transformer，hidden $d = 1024$）。理由：\(1\) 中文电商域已有大量公开 retrieval benchmark 验证；\(2\) 与 §4\.5 的 retrieval pipeline 天然兼容；\(3\) 预训练目标本身是语义检索/排序，对 claim–evidence 一致性判别是合适的迁移起点。

虽然 instruction\-tuned LLM 具有更强的世界知识和生成能力，但本阶段的任务已经由 §1 转化为结构化的属性级 claim–evidence pair 判别；该任务更依赖双向、对称、检索友好的文本对表征，而非长文本生成。因此，CLAIMARC 将 BGE \+ TwoStreamFusion 作为主模型，并在 §5\.3\.2 中加入 Qwen2\.5\-7B \+ LoRA 作为同等监督下的大模型微调强基线，以实证检验轻量显式交互结构相对于 LLM 隐式交互的收益。

**双流共享。** Claim Flow 与 Evidence Flow 共用同一份 BGE encoder，共享理由：

1. 两路信号都是中文电商话术性文本，没有跨模态/跨语言 gap；

2. 共享参数让 §4\.3 cross\-attention 的 Q/K/V 投影面落在同一表征空间内，cross\-attn 的几何意义（“声称位置 vs 证据位置”）更清晰；

**LoRA 适配。** 仅在 BGE 每层的 `q_proj` 与 `v_proj` 上插 LoRA，rank = 16, $\alpha$ = 32, dropout = 0\.05。

#### 4\.3 TwoStreamFusion Module

**单层公式（**$l = 1..N$**）**

**Step 1: 流内 self\-attention。**

$\tilde{H}_c^{(l)} = \text{LN}\bigl(H_c^{(l-1)} + \text{SelfAttn}(H_c^{(l-1)})\bigr), \quad
\tilde{H}_e^{(l)} = \text{LN}\bigl(H_e^{(l-1)} + \text{SelfAttn}(H_e^{(l-1)})\bigr)$

**Step 2: 双向 cross\-attention（claim ↔ evidence）。**

$\hat{H}_c^{(l)} = \text{LN}\bigl(\tilde{H}_c^{(l)} + \text{CrossAttn}(Q{=}\tilde{H}_c^{(l)},\, K{=}V{=}\tilde{H}_e^{(l)})\bigr)$

$\hat{H}_e^{(l)} = \text{LN}\bigl(\tilde{H}_e^{(l)} + \text{CrossAttn}(Q{=}\tilde{H}_e^{(l)},\, K{=}V{=}\tilde{H}_c^{(l)})\bigr)$

**关键设计：Q/K/V 投影矩阵在两个方向（c→e 与 e→c）共享权重。** 共享权重强制 fusion 学到“对照不一致”这一对称几何，而非两个独立映射，是注入对称先验的轻量方式。

**Step 3: FFN。**

$H_c^{(l)} = \text{LN}\bigl(\hat{H}_c^{(l)} + \text{SwiGLU}(\hat{H}_c^{(l)})\bigr), \quad
H_e^{(l)} = \text{LN}\bigl(\hat{H}_e^{(l)} + \text{SwiGLU}(\hat{H}_e^{(l)})\bigr)$

层数 $N = 2$；每层 8 head（$d_{\text{head}} = 128$）。

#### 4\.4 Parallel Task Heads

##### Pair\-level Logistic Regression Classifier \(LRC\)

**目标。** 直接预测该 `(product_id, attribute_id)` pair 是否存在用户感知的虚假宣传风险。

**输入。** Claim 流 $[\text{CLS}]$ 与 Evidence 流 $[\text{CLS}]$ 的 fused 表征：

$\bar{\mathbf{h}}_c = H_c^{(N)}[\text{CLS}], \quad \bar{\mathbf{h}}_e = H_e^{(N)}[\text{CLS}]$

**ESIM\-style 4\-tuple 特征。**

$\mathbf{z}_{\text{pair}} = [\bar{\mathbf{h}}_c;\; \bar{\mathbf{h}}_e;\; \bar{\mathbf{h}}_c - \bar{\mathbf{h}}_e;\; \bar{\mathbf{h}}_c \odot \bar{\mathbf{h}}_e] \in \mathbb{R}^{4d}$

其中 $\bar{\mathbf{h}}_c - \bar{\mathbf{h}}_e$ 显式编码“属性话术与证据之间的差异方向”，$\bar{\mathbf{h}}_c \odot \bar{\mathbf{h}}_e$ 编码“话术与证据一致的方向”。

**输出。**

$\hat{y} = \sigma\bigl(\mathbf{w}^\top \text{LayerNorm}(\mathbf{z}_{\text{pair}}) + b\bigr)$

**损失（带 §2 sample weight）。**

$\mathcal{L}_{\text{CE}} = -\sum_i c_i \cdot \bigl[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\bigr]$

##### Retrieval Embedding Head

**目标。** 把 fused pair 表征投影到 256 维单位球面，作为 §4\.5 retrieval\-augmented contrastive learning 与 §4\.7 RKC inference 的 embedding 源。

$\mathbf{g} = \text{L2Norm}\bigl(\text{MLP}_{\text{ret}}([\bar{\mathbf{h}}_c;\, \bar{\mathbf{h}}_e])\bigr) \in \mathcal{S}^{255}$

$\text{MLP}_{\text{ret}}$：$2d \to 512 \to 256$，GELU \+ Dropout\(0\.1\)。L2 normalization 让 $\mathbf{g} \in \mathcal{S}^{255}$，cosine similarity = inner product，与 §4\.5 InfoNCE 度量一致。

#### 4\.5 Attribute\-Blocked Retrieval\-Augmented Contrastive Learning

同一 attribute 下，两件商品的商品事实接近、主播话术微差就可能反标签。如果 hard negative 从全 batch 随机采，模型容易学到“食品 vs 数码”这类粗粒度差异，而不是同属性事实接近但话术边界不同的风险差异。Attribute\-Blocked Retrieval 把 hard negative 限制在同 `attribute_id` 内采样，强制模型把判别面落在属性内部的细粒度语言边界上

##### 4\.5\.1 Memory Bank with FAISS

每 epoch 开始前一次 forward pass 把训练集所有样本的 $\mathbf{g}$ 写入 FAISS 索引 $\mathcal{M}$，索引按 `(attribute_id, y)` 二级 partition：

|配置|值|
|---|---|
|Index type|`IVF1024,PQ32`（4,000 样本下足够）|
|Metric|inner product（与 L2\-norm 后 cosine 等价）|
|Refresh|每 epoch 末完整重建|
|Partition key|`(attribute_id, y) → list[g_index]`|

##### 4\.5\.2 Pseudo\-gold Positives

对当前 batch 样本 $i$（attribute $a_i$, label $y_i$），从 $\mathcal{M}$ 的 $(a_i, y_i)$ partition 取最近 $K_p = 3$ 个邻居作为 pseudo\-gold positive 集合 $\mathcal{P}_i^+$。这些样本与 anchor 同 attribute、同 label，理论上语义最接近，承担“拉近正确同类”的几何角色。

##### 4\.5\.3 Attribute\-Blocked Hard Negatives

同 `attribute_id` 但反标签的训练样本（即 $\mathcal{M}$ 的 $(a_i, 1-y_i)$ partition）记作 $\mathcal{N}_i^{\text{attr}}$。在 $\mathcal{N}_i^{\text{attr}}$ 内按 cosine 距离排序，取 top\-$K_n = 5$ 作为 hard negatives。

当某 attribute 内反标签样本 $|\mathcal{N}_i^{\text{attr}}| < K_n$（小 attribute 长尾），fallback 到全 batch 内反标签样本最近邻补足。

##### 4\.5\.4 InfoNCE Loss

对 \(anchor $i$, positive $j \in \mathcal{P}_i^+$\) 对：

$\ell_{i,j} = -\log \frac{\exp(\mathbf{g}_i^\top \mathbf{g}_j / \tau)}{\exp(\mathbf{g}_i^\top \mathbf{g}_j / \tau) + \sum_{k \in \mathcal{N}_i^{\text{attr}}} \exp(\mathbf{g}_i^\top \mathbf{g}_k / \tau)}$

总对比损失（带 sample weight）：

$\mathcal{L}_{\text{CL}} = \sum_i \frac{c_i}{|\mathcal{P}_i^+|} \sum_{j \in \mathcal{P}_i^+} \ell_{i,j}$

温度 $\tau = 0.07$（SimCLR / RA\-HMD 共同默认值）。

#### 4\.6 Joint Loss and Two\-Stage Training

##### 4\.6\.1 Joint Loss

$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{CL}} \mathcal{L}_{\text{CL}} + \lambda_{\text{reg}} \|\Theta_{\text{LoRA}} \cup \Theta_{\text{fusion}} \cup \Theta_{\text{heads}}\|_2^2$

默认权重：$\lambda_{\text{CL}} = 0.3$，$\lambda_{\text{reg}} = 1\text{e-}4$。

##### 4\.6\.2 Two\-Stage Training Schedule

|Stage|Epochs|Active losses|$\lambda_{\text{CL}}$|设计意图|
|---|---|---|---|---|
|**Stage 1: Warm\-up**|3|$\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{reg}}$|0|让 encoder 与 fusion 先学到基本的属性级 claim–evidence 判别几何，避免初期 embedding $g$ 是噪声而 hard negative 退化为随机采样|
|**Stage 2: Contrastive**|6|$\mathcal{L}_{\text{CE}} + \lambda_{\text{CL}}\mathcal{L}_{\text{CL}} + \mathcal{L}_{\text{reg}}$|0\.3|启动 retrieval CL；每 epoch 末重建 FAISS 索引|

**两阶段必要性。** RA\-HMD \[[Mei et al\., 2025](https://rcns7iissy8j.feishu.cn/docx/MCjmdRTutoIhoCxSu2TcIpmcneb#share-L7dMd6HDuozhYGxvTjtczt5insg)\] 报告 SFT\-then\-CL 比联合训练稳定。warm\-up 阶段不开 CL 的核心原因：初期 $\mathbf{g}$ 表征还没收敛，按 cosine 排出来的 hard negative 是几何噪声而非真正的“同 attribute 反标签近邻”，对学习有害。

#### 4\.7 Inference Pipeline

##### Forward Classifier

测试 pair $i$ 走 §4\.2–4\.4 forward 一次，输出分类概率 $\hat{y}_i$ 与检索表征 $\mathbf{g}_i$

##### Retrieval\-augmented K\-NN Classifier \(RKC\)

在训练集 FAISS 索引上检索 $\mathbf{g}_i$ 的最近 $K_R = 10$ 个邻居。推理阶段不再做 attribute\-block，使 RKC 成为独立于当前 attribute 约束的检索式校验器；按 cosine 与样本权重 $c_j$ 加权投票：

$\hat{y}_i^{\text{RKC}} = \frac{\sum_{j \in \text{kNN}(i)} c_j \cdot \cos(\mathbf{g}_i, \mathbf{g}_j) \cdot y_j}{\sum_{j \in \text{kNN}(i)} c_j \cdot \cos(\mathbf{g}_i, \mathbf{g}_j)}$

RKC 不替代 forward classifier，而是检查 $\mathbf{g}$ 空间中的近邻标签是否支持同一结论。

##### Self\-Consistency Abstain

最终决策融合 forward classifier 与 RKC：

$\hat{y}_i^{\text{final}} = \begin{cases}
\text{abstain} & \text{if } |\hat{y}_i - \hat{y}_i^{\text{RKC}}| > \delta \\[3pt]
\mathbb{1}\bigl[\tfrac{1}{2}(\hat{y}_i + \hat{y}_i^{\text{RKC}}) \ge 0.5\bigr] & \text{otherwise}
\end{cases}$

$\delta = 0.3$。abstain 不作为主指标参与二分类比较；它用于选择性预测实验，表示模型自身与近邻空间判断明显冲突，需要人工复核。

## Data

### 1\.数据概览

|维度|数量 / 取值|
|---|---|
|切片（clips）总数|**732** 条|
|独立商品（products）总数|**628** 个（其中 570 个在 clips 中被引用）|
|直播间一级分类|**10** 个|
|直播间二级分类|**39** 个|
|涉及直播间（主播）|**115** 个|
|评论总条数|**28,556** 条|
|商品详情图总张数|**11,389** 张（平均 18\.1 张/商品）|

#### 1\.1 类目分布

##### 一级分类（10 类）

|一级分类|切片数|
|---|---|
|apparel\_and\_underwear（服装内衣）|212|
|general（综合）|124|
|baby\_kids\_and\_pets（母婴宠物）|86|
|shoes\_and\_bags（鞋包）|83|
|smart\_home（智能家居）|62|
|food\_and\_beverages（食品饮料）|45|
|digital\_and\_electronics（数码电子）|33|
|sports\_and\_outdoor（运动户外）|30|
|jewelry\_and\_collectibles（珠宝收藏）|29|
|beauty\_and\_personal\_care（美妆个护）|28|

##### 二级分类（39 类，前6）

|二级分类|切片数|
|---|---|
|mens\_clothing|78|
|womens\_clothing|72|
|fashion\_accessories|56|
|bags|46|
|childrens\_clothing|43|
|hardware\_tools|39|

#### 2\.数据索引

##### 2\.1 `clips` 字段（切片级，共 732 条）

每条记录包含以下字段：

|字段|说明|备注|
|---|---|---|
|`商品序号`|在索引中的序号||
|`商品名称`|商品标题（字符串）|可能含下划线/特殊字符|
|`product_id`|商品 ID，关联到 `products` 字典|570 个 unique|
|`直播间一级分类`|10 类||
|`直播间二级分类`|39 类||
|`直播间名称`|主播/直播间名|115 个 unique|
|`srt切片存储路径`|主播话术字幕文件||
|`商品评价存储路径`|`.xls` 评论文件||
|`评论文件数`|对应评论 xls 数量||
|`评论总数`|对应商品下总评论数|sum=28,556；mean=39\.0；max=1,501；min=0|

##### 2\.2 `products`字段（商品级，共 628 条）

每个 `product_id` 对应如下字段：

|字段|说明|
|---|---|
|`product_id`|抖音商品 ID|
|`商品名称` / `title`|商品名与展示标题|
|`price`|售价（float）|
|`sales_info`|形如 `"已售 235"` 的销量描述字符串|
|`shop_name`|店铺名|
|`product_url`|抖音商城商品详情页 URL|
|`产品参数`|结构化属性字典|
|`images.主图`|`{原始链接, 本地路径}`，本地存放于 `product_images/{product_id}/main.webp`|
|`images.详情图`|`[{原始链接, 本地路径}, ...]`，本地存放于 `product_images/{product_id}/detail_XXX.jpeg`|

### 2\.raw data

#### 2\.1 主播话术字幕

- **格式：** 标准 SRT（时间戳 \+ 纯文本），部分以 `.txt` 保存。样例：

    ```Plain Text
    1
    00:00:00,000 --> 00:00:29,840
    是的,都是没问题的,是吧,你们如果说是冬天想要出门的话……
    ```

#### 2\.2 商品详情

- **参数（****`产品参数`****）** 来自抖音商品详情页的结构化字段。最常见字段 TOP\-20：

    |    字段|    出现商品数|
    |---|---|
    |    品牌|    370|
    |    货号|    267|
    |    产地|    225|
    |    面料材质|    217|
    |    风格|    217|
    |    功能|    166|
    |    适用性别|    162|
    |    厚度|    149|
    |    生产企业名称|    143|
    |    适用季节|    134|
    |    材质|    131|
    |    款式|    125|
    |    适用人群|    113|
    |    保质期|    105|
    |    上市时间|    98|
    |    组合件数|    97|
    |    适用年龄|    93|
    |    产品名称|    92|
    |    型号|    73|
    |    衣长|    70|

- **主图：** `/mnt/gty/product_images/{product_id}/main.webp`（每商品 1 张）

- **详情图：** `/mnt/gty/product_images/{product_id}/detail_XXX.jpeg`（共 11,389 张）

#### 2\.3 商品评论

- **字段（共 6 列）：**

    |    列名|    类型|    说明|
    |---|---|---|
    |    评论内容|    str|    用户原始评论文本|
    |    评论时间|    str / datetime|    形如 `2024/12/20 13:24:51`|
    |    评论情感|    str|    **好评 / 中评 / 差评** 三分类标签|
    |    来自商品|    str|    冗余商品名|
    |    来源小店|    str|    店铺名|
    |    商品链接|    str|    飞瓜跳转链接|

- **观测窗口：** 2024\-10\-03 至 2025\-03\-31

- **覆盖：** 绝大多数 clips 对应 1 个评论文件；少数商品可能跨多个 xls。

### 3\.实验数据集

> §8\.1–8\.6 描述的是上游原始数据；本节描述经 §1 数据流水线（Stage A → B → C）\+ §2 标签生成与样本权重之后得到的、**直接喂给下游分类器**的训练数据集。
> 
> 

#### 3\.1 定义与规模

数据集中的**一条记录 = 一个 \(product, attribute\) pair**。每条记录把同一个商品的同一个属性下的三类信号绑成一个训练样本：\(i\) Stage B 产出的**主播话术 claim**（passage \+ segments \+ 时间戳）；\(ii\) Stage C 产出的**商品事实证据**（params / OCR / VLM 三源原样并列）；\(iii\) §2 产出的**硬二元标签 **$y$** 与样本权重 **$c$。

|项|说明|
|---|---|
|记录粒度|\(product\_id, attribute\_id\) pair|
|候选属性|$A_{\text{cmt}}(p)$（每个商品评论侧已命中的属性子集，来自 Stage A）|
|主播信号|claim\.passage（拼接全段）\+ claim\.segments\[\]（带 SRT 起止时间戳）|
|事实信号|evidence\_params \+ evidence\_ocr \+ evidence\_vlm（三源原样并列，不裁决）|
|监督信号|$y \in \{0, 1\}$（虚假宣传硬标签）\+ $c \in [0, 1]$（per\-sample weight）|
|评论审计字段|aligned\_neg / aligned\_pos 计数、$S_{\text{neg}} / S_{\text{pos}}$ 等 §2 中间量|
|估计规模|628 商品 × 平均 \~6\.5 attribute / 商品 ≈ **4,000 条 pair\-level 记录**（pilot 抽样估算，需以 Stage A 产出为准）|

#### 3\.2 单条记录的字段 schema

下面这条统一 dataset record 由 Stage B 的 `pair_records.jsonl` × Stage C 的 `fact_records.jsonl` × §2 的 `labels.jsonl` 在 \(product\_id, attribute\_id\) 键上 inner\-join 得到：

```JSON
{
 *// ===== 键与元信息 =====*
  "pair_id":         "<product_id>__<attribute_id>",
  "product_id":      "...",
  "category":        "food.beverages",
  "attribute_id":    "FOOD_PROTEIN_CONTENT",
  "attribute_name":  "蛋白质含量",          *// 来自 Stage A CAS+*

  *// ===== 主播话术（Stage B）=====*
  "claim": {
    "passage":  "...全段拼接的主播表述...",
    "segments": [
      {"clip_id": "...", "t_start": 123.4, "t_end": 138.7, "text": "..."},
      ...
    ]
  },

  *// ===== 商品事实证据（Stage C，三源原样并列）=====*
  "evidence_params": [{"param_key": "...", "raw_text": "..."}],
  "evidence_ocr":    [{"raw_text": "...",  "image_path": "..."}],
  "evidence_vlm":    [{"raw_quote": "...", "image_path": "..."}],
  "evidence_count":  {"params": 1, "ocr": 1, "vlm": 1},
  "coverage":        3,
  "confidence":      "high",

  *// ===== 监督信号（§2）=====*
  "y":      1,
  "c":      0.32,

  *// ===== 标签生成审计字段（§2 中间量，便于消融）=====*
  "label_audit": {
    "n_aligned":    12,
    "n_total":      20,
    "n_neg_aligned":2,
    "n_pos_aligned":10,
    "S_neg":        4.3,
    "S_pos":        15.7,
    "f_sat":        0.98,
    "f_cov":        0.57,
    "f_asym":       0.477,
    "phi_bonus":    1.2,
    "c_base":       0.56
  },

  *// ===== 划分 =====*
  "split": "train"   *// {"train", "val", "test"}*
}
```

#### 3\.3 Train / Val / Test 划分

为防止主播个人话术风格泄漏与同直播间样本相关性，采用**按主播分组的 grouped split**（GroupKFold\-style），而非随机拆分：

|划分|比例|分组键|
|---|---|---|
|Train|70%|room\_id|
|Val|10%|room\_id|
|Test|20%|room\_id|

同一直播间的所有 pair 严格落在同一 split。此设计使评估结果对"换一个新主播"具有泛化外推意义。Test 集的 $c$ 不参与训练，但保留以做 weighted\-evaluation（带置信度的混淆矩阵）与 abstain\-on\-low\-confidence 报告。

## References

\[Baumeister\_2001\_Bad\_is\_Stronger\_than\_Good\_RGP\.pdf\]

\[Chen\_Fay\_Wang\_2011\_Role\_of\_Marketing\_Social\_Media\_JIM\.pdf\]

\[Dawid\_Skene\_1979\_Observer\_Error\_Rates\_EM\_ApplStat\.pdf\]

\[Han\_2018\_Co\-teaching\_NeurIPS\.pdf\]

\[He\_Hollenbeck\_Proserpio\_2022\_Market\_for\_Fake\_Reviews\_MS\_WP\.pdf\]

\[Jiang\_2018\_MentorNet\_ICML\.pdf\]

\[Jindal\_Liu\_2008\_Opinion\_Spam\_WSDM\.pdf\]

\[Li\_Hitt\_2008\_Self\_Selection\_Online\_Product\_Reviews\_ISR\.pdf\]

\[Luca\_Zervas\_2016\_Fake\_It\_Till\_You\_Make\_It\_MS\_WP\.pdf\]

\[Mayzlin\_2014\_Promotional\_Reviews\_AER\_NBER\_WP\.pdf\]

\[Mintz\_2009\_Distant\_Supervision\_ACL\.pdf\]

\[Mukherjee\_2013\_Yelp\_Fake\_Review\_Filter\_ICWSM\.pdf\]

\[Northcutt\_2021\_Confident\_Learning\_JAIR\.pdf\]

\[Ratner\_2016\_Data\_Programming\_NeurIPS\.pdf\]

\[Ratner\_2017\_Snorkel\_VLDB\.pdf\]

\[Raykar\_2010\_Learning\_from\_Crowds\_JMLR\.pdf\]

\[Ren\_2018\_Learning\_to\_Reweight\_ICML\.pdf\]

\[Robertson\_Zaragoza\_2009\_BM25\_and\_Beyond\_FnTIR\.pdf\]

\[Soleimani\_2020\_BERT\_Evidence\_Retrieval\_ECIR\.pdf\]

\[Thorne\_2018\_FEVER\_NAACL\.pdf\]

\[Cite Before You Speak\.pdf\]

\[Li\_Xu\_etal\_2025\_IJoC\_SarcasmCause\_VoR\.pdf\]

\[Mei\_etal\_2025\_EMNLP\_RAHMD\.pdf\]







