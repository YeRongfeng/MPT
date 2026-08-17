# Academic English Style Guide for MPT

> 本文件总结用户提供的写作参考、Purdue OWL 和 University of Manchester Academic
> Phrasebank 的共同原则，用于后续 Introduction、Method、Abstract、Results 和
> Discussion 的句子级修改。它规定的是表达标准，不替代本文的技术事实和主张边界。

## 1. 核心目标

学术英文的目标不是把句子写得更短，也不是把普通词替换成看起来更高级的词。高质量
句子应同时满足：

1. **Precision：** 每个关键名词都有明确对象和范围；每个动词都说明实际发生的关系。
2. **Objectivity：** 让方法、数据和证据充当主语，避免情绪性评价和无依据的优越性描述。
3. **Formal rigor：** 使断言强度与证据或数学保证匹配；必要时对解释和外推进行对冲。
4. **Cohesion：** 句子从已建立的信息出发，把新的技术信息放在句尾或下一句的开头。
5. **Information density：** 删除不承担独立语义的词，但保留原因、条件、对比、范围和
   结果等逻辑关系。

“精炼”因此不等于把
`faces two difficulties in producing ...`
改写成更短的
`must generate ...`。前者包含问题定位和困难框架，后者可能只剩任务要求。

## 2. 参考资料真正支持的原则

### 2.1 Purdue OWL：精炼是保留最有效的词

Purdue 的 Conciseness 资源明确指出，concise writing 不一定拥有最少的词，而是保留
最有效、最具体的词。它建议逐词检查：一个词如果没有提供新的、独立的信息，就删除或
改写。主要策略包括：

- 用具体动词和名词替代多个模糊的小词，而不是机械追求拉丁源词；
- 删除解释显而易见事实的词、重复限定语和冗余类别词；
- 将能够合并的信息放入同一句，但不牺牲因果、条件和对比；
- 在语义允许时，把不必要的名词化或 `that/which` 从句压缩成短语；
- 不要过度使用 `there is/there are` 和空的 `it is` 开头；如果外置主语没有修辞作用，
  直接让真正的研究对象充当主语。

这并不意味着所有被动句或名词化都应删除。方法段中，研究对象有时比执行者更重要；
名词化也可以把一个已知过程压缩为稳定的技术概念。判断标准始终是：当前句子是否更
准确地表达了研究关系。

### 2.2 Academic Phrasebank：先确定句子的功能，再选择表达

Manchester Phrasebank 将研究写作按功能组织，而不是提供一张可以任意拼接的“高级
句式表”。对本文最有用的功能区分是：

- **Introduction：** 建立领域和问题的重要性，指出具体缺口，说明研究目标和贡献；
- **Method：** 说明采用了什么方法、如何实施，以及为什么采用该方法；方法应足够清楚，
  使有经验的读者能够重复研究；
- **Results：** 先定位数据或表格，再指出其中真正相关的结果；观察结果与进一步解释
  不应无意混在一起；
- **Discussion：** 通常沿着“重述结果 -> 与已有研究比较 -> 解释差异 -> 给出含义”
  的循环推进；解释和外推应比直接报告数据更谨慎；
- **Conclusion：** 回收主要发现，并给出与证据相称的意义、限制或后续方向。

这说明同一个动词不能脱离句子功能评判。例如 `show` 在描述图表时完全自然；在一个
需要突出证据强度的结论句中，则可能需要 `indicate`、`suggest` 或 `demonstrate`，
具体取决于证据和主张范围。

## 3. 信息密度的句法来源

### 3.1 已知信息到新信息

句首应承接前文已经建立的对象，句尾引出新的动作、限制或推论。不要让每个句子都重新
介绍主体：

```text
Progressive mapping provides only partial terrain observations within the input window.
This incomplete support leaves the planner dependent on an explicit observation mask.
```

第二句的 `This incomplete support` 承接前句，句尾的 `explicit observation mask` 引入
新机制。若把两句改成两个互不相干的模块介绍，信息仍然存在，但逻辑密度会下降。

### 3.2 用平行结构承载分类和对比

当问题有两个并列困难时，保留相同的句法骨架通常比拆成许多短句更紧凑：

```text
One is that progressive mapping may yield only partial terrain observation within the
fixed-size input window. The other is that endpoint violations persist, since the boundary
poses merely condition the network and are satisfied through learning rather than by
construction.
```

这里 `One is ... The other is ...` 同时完成分类、对齐和递进；`since` 把现象与机制连接
起来。不能为了所谓“简洁”把它改成 `The input is partial. Endpoints are difficult.`，
那会丢失问题的来源和技术含义。

### 3.3 合并信息，不是删除信息

高密度句子可以使用限制性从句、原因从句、分词结构、同位语和并列谓语，但每个附加
结构必须承担清楚的逻辑角色：

- `since` / `because`：原因；
- `while` / `whereas`：并列条件或对比；
- `although` / `despite`：让步；
- `where` / `in which`：定义空间、接口或机制；
- `thereby` / `thus`：由前一动作产生的结果；
- `rather than`：方法选择或责任主体的对照。

一个句子可以包含多个从句，但不应同时承担互不相关的五个技术动作。控制标准不是
“短”，而是读者能否一次识别主句、限定条件和新结论。

### 3.4 控制主语和谓语的距离

复杂限定应放在句首或句尾，避免主语与核心动词之间插入过长的修饰块。尤其检查：

- 分词短语的逻辑主语是否与主句主语一致；
- `which/that` 从句是否真的提供新信息；
- 主句谓语是否在读者读完一大串定语后才出现；
- 句尾是否被空泛的 `in order to`, `in terms of`, `with respect to` 占据。

## 4. 动词和词汇的使用边界

### 4.1 先选关系，再选动词

不要建立脱离语境的“普通词 -> 高级词”替换表。应先确定句子想表达的关系：

| 关系 | 可用动词 | 语义边界 |
| --- | --- | --- |
| 输入导致结果 | `yield`, `produce`, `give rise to` | `yield` 说明输入或过程产生什么，不等同于证明优越性 |
| 条件输入 | `condition`, `parameterize`, `specify` | `condition` 说明模型接收条件，不表示条件被严格满足 |
| 结构携带性质 | `encode`, `incorporate`, `embed`, `carry` | 必须确实存在结构或先验支持，不用来装饰普通输入 |
| 强制约束 | `enforce`, `impose`, `fix`, `satisfy` | 只用于硬约束、解析构造或明确的优化约束 |
| 训练适配 | `adapt`, `shape`, `refine`, `calibrate` | 说明训练如何改变已有映射，不自动暗示泛化改善 |
| 数据显示 | `indicate`, `reveal`, `show`, `demonstrate` | 由数据和证据强度决定；`demonstrate` 不是 `show` 的无条件升级 |
| 研究提出 | `propose`, `formulate`, `introduce`, `develop` | `formulate` 偏问题重构，`propose` 偏提出方法，`develop` 暗示形成了具体系统 |

`use` 不必强行替换成 `utilize`，后者常常更笨重；`show` 也不必一律替换成
`demonstrate`。正式度来自动词的准确论元和语义关系，而不是词典中看起来更生僻的词。

### 4.2 避免弱动词和空短语，但保留必要的逻辑

通常需要检查下列表达：

- `make ... difficult` -> 直接说明困难机制，如 `leaves ... to learning`；
- `is able to` -> `can` 或直接使用实义动词；
- `is used to` -> `computes`, `conditions`, `evaluates` 等具体动作；
- `makes use of` -> `uses`, `adopts`, `employs`，按语义选择；
- `has the ability to` -> 直接写能力对应的动词；
- `in order to` -> 若不需要强调目的，通常用 `to`；
- `a number of`, `various`, `several different` -> 只有在数量或差异确实重要时保留。

但不能为了删词而删除 `since`, `while`, `only during training` 等信息边界；它们不是
填充，而是技术主张的一部分。

### 4.3 名词化要服务于压缩，不要掩盖动作

名词化适合把已经确定的过程变成讨论对象，例如 `endpoint satisfaction`、
`terrain-feasibility objective`、`route-prior learning`。过度名词化会隐藏谁做了什么：

```text
Weak:    The computation of the objective was performed using complete terrain.
Better:  We compute the objective from complete terrain available only during training.
```

反过来，如果一个技术概念在后文需要反复引用，保留名词化通常更有效；不要为了“主动
语态”把所有稳定概念拆成冗长动词短语。

## 5. 客观性、对冲和断言强度

### 5.1 对冲不是所有句子都要变弱

需要按主张类型分层：

| 主张类型 | 表达方式 |
| --- | --- |
| 定义、接口、解析保证 | 直接陈述：`the representation exactly satisfies ...` |
| 当前方法的机制现象 | 在明确设定下直接陈述：`endpoint violations persist ...` |
| 冻结实验的直接比较 | 指标绑定基线和条件：`improves X over Y under Z` |
| 对结果的原因解释 | 使用 `may`, `could`, `appears to`, `is consistent with` |
| 跨数据集或未来泛化 | 使用范围限定和谨慎动词：`suggests`, `may extend`, `does not establish` |

不能因为“学术写作要谨慎”就把已经由构造保证的事实写成 `may satisfy`；也不能把没有
冻结证据的 OOD 泛化写成 `generalizes better`。对冲应当缩小推论范围，而不是模糊技术
定义。

### 5.2 观察、解释和含义要分开

结果句先说观测到什么，再说它可能意味着什么：

```text
The adapted model increases hard feasibility over imitation-only training.
This improvement may indicate that the training-time terrain objective reshapes the
deployment mapping toward physically feasible paths.
```

第一句是直接比较，第二句是解释，因此使用 `may indicate`。不要用一个 `demonstrate`
同时承担结果、原因和广泛意义。

### 5.3 主动和被动按信息焦点选择

- 摘要、Introduction、贡献：通常用 `we propose`, `we formulate`, `the planner generates`，
  让研究动作和方法承担主语；
- 方法流程、实验步骤：当对象比执行者重要时可用被动，如 `the trajectories were
  generated ...`；
- 结果和图表：让数据、表格或结果作主语，如 `Figure 3 reveals ...`、`the results
  indicate ...`。

主动语态不是绝对规则，被动语态也不是“不学术”。选择标准是读者需要关注谁或什么。

## 6. MPT 的技术语义边界

后续润色不得为了句式顺滑改动下列含义：

- `fixed-size input window` 是当前 diffusion planner 的输入接口，不是所有学习型规划
  器的共同属性；
- progressive mapping 使固定窗口内只有部分区域具有地形观测；这不是输入张量尺寸
  变化，也不自动等同于宽泛的 `partial map`；
- 扩散从轨迹高斯先验出发，该轨迹先验位于边界对齐的轨迹空间，并自带边界和相关物理
  性质；不能擅自把它改写成普通 `trajectory representation`；
- 当边界位姿仅作为条件输入时，endpoint violations persist 是当前条件生成机制下的
  问题现象；不要仅为了“谨慎”把它弱化成没有信息的 `difficult to enforce`；
- `available only during training` 必须保留，用来区分完整地形监督和部署可见输入；
- 解析保证、hard feasibility、route coverage、source diversity 和 OOD generalization
  是不同主张，不能用一个笼统的 `better` 或 `safer` 代替。

因此，当前摘要前几句的语言基准是：

```text
Learning-based global path planning on rough terrain faces two difficulties in producing
a traversable path under prescribed start and goal poses. One is that progressive mapping
may yield only partial terrain observation within the fixed-size input window. The other is
that endpoint violations persist, since the boundary poses merely condition the network and
are satisfied through learning rather than by construction. To address both, we propose a
Path MeanFlow planner that generates paths from masked terrain observations in a
boundary-aligned trajectory space, where the trajectory prior inherently encodes the
boundary poses.
```

这里不能为了套用 `must generate`、`Two factors make this difficult` 或 `representation
exactly fixes ...` 而改变原有的信息结构和技术责任主体。

## 7. 句子级修改流程

每次修改现有句子时，按以下顺序处理，而不是先做同义词替换：

1. 写出这句话唯一的核心命题，以及它的适用范围。
2. 标出主语、实义谓语、对象和逻辑从句，确认谁执行了什么动作。
3. 删除不提供独立信息的词，保留原因、条件、对比、时间和信息边界。
4. 用一个准确动词替代弱动词结构，但先确认该动词的论元和断言强度。
5. 将已知对象放在句首，将新机制、限制或结果放在句尾。
6. 检查长定语、悬垂修饰、代词指代和主谓距离。
7. 最后才检查正式度、对冲和词汇变化；不因追求高级词而改变技术命题。

## 8. 摘要审校清单

- 第一段是否直接建立任务张力，而不是泛泛介绍“学习方法很重要”？
- 每个困难是否同时说明现象和产生现象的机制？
- 方法句是否回答了“方法改变了哪一层信息”，而不是平铺模块名？
- 是否保留了轨迹先验、边界条件、mask 和训练/部署信息边界的准确责任主体？
- 是否有弱动词、重复名词、空主语或为了短而删掉的逻辑关系？
- 是否把结构保证、观测现象、实验比较和解释性推论分开？
- 每个 `improves`, `outperforms`, `demonstrates` 是否都有对应指标、基线和条件？
- 是否存在把当前模型接口推广成整个领域属性的表述？

## 9. Sources

本指南依据以下资料整理，网页资料于 2026-08-07 读取：

- 用户提供的参考：[pasted-text.txt](/home/yrf/.codex/attachments/ac09a7ba-618e-4e37-bc41-aeef54b112d9/pasted-text.txt)
- Purdue OWL, [Academic Writing Introduction](https://owl.purdue.edu/owl/general_writing/academic_writing/index.html)
- Purdue OWL, [Conciseness](https://owl.purdue.edu/owl/general_writing/academic_writing/conciseness/index.html)
- Purdue OWL, [Eliminating Words](https://owl.purdue.edu/owl/general_writing/academic_writing/conciseness/eliminating_words.html)
- Purdue OWL, [Changing Phrases](https://owl.purdue.edu/owl/general_writing/academic_writing/conciseness/changing_phrases.html)
- Purdue OWL, [Avoid Common Pitfalls](https://owl.purdue.edu/owl/general_writing/academic_writing/conciseness/avoid_common_pitfalls.html)
- University of Manchester, [Academic Phrasebank: Introducing Work](https://www.phrasebank.manchester.ac.uk/introducing-work/)
- University of Manchester, [Academic Phrasebank: Describing Methods](https://www.phrasebank.manchester.ac.uk/describing-methods/)
- University of Manchester, [Academic Phrasebank: Reporting Results](https://www.phrasebank.manchester.ac.uk/reporting-results/)
- University of Manchester, [Academic Phrasebank: Discussing Findings](https://www.phrasebank.manchester.ac.uk/discussing-findings/)
- University of Manchester, [Academic Phrasebank: Being Cautious](https://www.phrasebank.manchester.ac.uk/using-cautious-language/)
- University of Manchester, [Academic Phrasebank: Writing Conclusions](https://www.phrasebank.manchester.ac.uk/writing-conclusions/)
