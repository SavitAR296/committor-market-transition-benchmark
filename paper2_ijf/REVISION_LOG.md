# 修订记录：论文一（PRE）与论文二（IJF）

2026-10-19。依据《两篇论文审稿评估与修改指令_PRE_IJF》第 6、7 节逐条执行；第 8 节只做本地部分（GitHub 仓库与 Zenodo 按要求未动）。每条后面写明改动位置与核验方式。所有新数字都由脚本生成为宏，正文中没有手敲的结果数字（论文一残余的手敲数是模型参数，见 `P1_check_prose.py` 输出）。

## 0 交付物

| 文件 | 说明 |
|---|---|
| `paper1_PRE/main.pdf` | 12 页，REVTeX 4.2 |
| `paper1_PRE/supplement.pdf` | 6 页，表 S1–S12 连续 |
| `paper1_PRE/cover_letter.pdf` | 2 页 |
| `paper1_PRE/main_diff_tracked_changes.pdf` | latexdiff 对照上传版 |
| `paper2_IJF/main.pdf` | 37 页，elsarticle，双倍行距 |
| `paper2_IJF/cover_letter.pdf` | 2 页，新写 |
| `paper2_IJF/main_diff_tracked_changes.pdf` | latexdiff 对照 draft v1 |
| `paper1_pre_package_revised.zip`、`paper2_ijf_package_revised.zip` | 完整包（源码、脚本、CSV、SHA 清单） |

## 1 需要作者填写或确认（AUTHOR_INPUT_NEEDED）

1. **Zenodo 新版本 DOI 与 commit hash**
   - 宏 `\ZenodoDOI`、`\RepoCommit` 位于论文一 `main.tex` 顶部；论文二 `main.tex` 与两封 cover letter 里有同名宏。
   - PDF 里目前显示为加粗方括号占位，投稿前必须替换。
2. **互引**：两篇都还没有上 arXiv。
   - 现在的写法：论文一 `[xing2026b]` 为 “submitted to Int. J. Forecast. (2026)”，论文二 `xingpan2026` 为 “Submitted to Physical Review E”。
   - 这个写法只有两篇同周投出后才成立。上 arXiv 后替换为 arXiv 号，`refs.tex` 中已注释位置。
3. **论文二 CRediT 作者贡献**：在 “CRediT authorship contribution statement” 段，现为占位。
4. **论文一文献 `duplessis2026`（arXiv:2610.01677）**：联网未能检索到，标题无法核实，请补标题（B10）。
5. **请核对两个文献号**：
   - Hartich & Godec, Nat. Commun. **15**, 8678 (2024)：DOI 10.1038/s41467-024-52602-0 已核实；文章号 8678 是按配对 Reply 的 8679 推断的。
   - Xu, Patterson, Levin & Wang, PNAS **120**, e2218663120 (2023)：核对文献时检索到并新加，文章号凭记忆。
6. **论文二 Funding 声明**：写的是 “no specific grant”，请确认。
7. **一位作者已发表的单作者 CSF 论文**：
   - 若与任一篇主题相关，必须在两篇中引用，并在 cover letter 中说明（APS 与 Elsevier 都要求披露）。
   - 若它就是公开仓库 README 中的 “Committor benchmarks…”，则指令中“未投稿”的说法需要统一。
8. **建议审稿人**：按用户确认（无合作关系）写入了 cover letter，投稿前请再核对单位与邮箱。
9. **AI 声明**：已写明 Claude (Anthropic) 用于文字编辑、AI 编程助手用于代码。如还用了其他具体工具，请补名称（指令 8.2）。

## 2 对指令本身的更正（以数据为准，正文已按数据写）

1. **论文一 B6**
   - 指令的说法：反演后“|ω| 几乎不变而 𝒢 变化”。
   - 数据显示的是：在重拟的 PCA 平面中，|ω_fit| 从 0.019 变为 0.034（平方后是 3.1 倍），𝒢 反而从 47 降到 41，ė_p 因此从 0.035 变为 0.095；固定基下 ė_p 只到 0.038。
   - 结论：变化来自保留平面中的电流，不是 𝒢。
2. **论文一 A7**：`ratio_std` 是通量/漂移比 R（即表 S3 的 R 列），不是 ω_fit/ω。因此表 I 直接新增一列 ω_fit。
3. **论文一 P1-4 附注**：`T2_table1_closed_form.csv` 的 `omega_C1` 是未标准化值（−0.0124）；表 I 中的 −0.0135 是 `tab_p1.py` 标准化后的值。
4. **论文一 P1-10**：0.022–0.047 是把 ω_lev 插值匹配到数据值得到的，本身正确；0.044 是网格点。正文已写明插值方法。
5. **论文二 B4**：只有 episode 置换检验保持 size（4.2% / 5.3%）。quasi-binomial 检验偏大（6.6% / 11.9%），不是指令预期的 0.03–0.07。已据此改写第 5.1 与第 8 节：1957–1999 的 quasi-binomial 拒绝只算弱证据。
6. **论文二 B5**
   - GW 拒绝来自常数项（单独检验 p = 0.002–0.022）；滞后项单独检验 p = 0.058–0.093。原文“哪一个更好取决于上一个 episode”的解读不成立，已改写。
   - Giacomini–Rossi 波动检验在扩展窗下不拒绝。在滚动窗下对 AMC 拒绝 2 次，committor 在最近的窗口中更好；没有出现指令预期的“2008、2020 年翻转”。
7. **论文二 B2**
   - 20 天：s 经同样 seq-cal 后与 committor 无差别（BSS 0.086 vs 0.085，p = 0.94）。
   - 5 天：s 更差（0.086 vs 0.119，区间刚好包含 0，p = 0.18）。两者 DSC 相同，差异全部来自校准，正文已写明。
8. **论文二 A6**：块长改为 2h 后，以下定性结论全部逐项核对，均不变：第 4 节的 DSC 区间、MCS 结论（8/8 族保留 committor）、各市场的 DSC 区间、“动力学替代项在 20 天低于 OU”。GBT 区间的行为与指令预期一致。

## 3 论文一（PRE）逐条

**A 级**

- **A1 ✅ 补充材料表号**
  - 现为 S1–S12 连续：S1–S9 按指令内容编号，新增 S10（bootstrap 标准误）、S11（市场信息损失）、S12（数据文件对应表）。
  - longtable 已改为 tabular。S3、S9 改由 `tab_p1.py` 从 CSV 生成，与旧表逐数一致（已比对）。
  - 主文改用 `\Sref{}` 引用，编号由 `sync_labels.py` 从 aux 文件写入。`check_table_numbers.py` 已接入 make，编号不一致时构建失败。
- **A2 ✅ surrogate 断言**：IV.D 改为 200 个 surrogate 的中位数 0.0027、95% 分位 0.0070、最大值 0.0117，观测值为最大值的 3.0 倍。这些数是由 `E4_irreversibility_surrogates.csv` 生成的宏。表 S6 增加“null max”列。
- **A3 ✅ 记号**：显式定义 Δγ^± ≡ Δ(γ^±)，证明展开为奇偶两行。
- **A4 ✅ 内部版本史**
  - 删去 III.B 的 earlier estimate，改写为方法学陈述；表 S5 列名改为 “classifier-skill estimate”；IV.F 改为 “2000–2020 subsample”。
  - 补充材料改为 vega-datasets snapshot 与 “second copy of the Yahoo series”。
  - grep `frozen|earlier estimate|original` 结果为 0。
- **A5 ◐ 互引**：已无 “in preparation”，等 arXiv 号（见第 1 节）。
- **A6 ◐ 数据可用性**：已写明仓库 URL、目录 `paper1_pre/`、commit 与 DOI（后两者为占位）。
- **A7 ✅ II.D 数字自洽**：正文给出 ω_fit = −0.0127，2ω_fit²𝒢 = 4.8。表 I 新增 ω_fit 列，任一行都能复算 ė_p（`T2d` 脚本内断言）。
- **A8 ✅ 步长统一**：表 I 的 ℓ 改为 h = 10，与图 2(a) 一致；`T2d` 同时断言能复现旧的 h = 20 值。

**B 级**

- **B1 ✅ 文献定位**
  - 新增 12 篇：Daw 2000、Porta 2008、Lacasa 2012、Flanagan 2016、Zanin 2021、Roldán–Parrondo 2010 与 2012、Skinner 2021、Harunari 2022、van der Meer 2022、Teza 2020，以及 Xu 2023 PNAS。
  - 新颖性段明确：已有工作研究的是隐藏自由度或投影；本文研究的是因果滤波核的时间偏移。
  - 文献按首次引用顺序自动排序，共 54 条。
- **B2 ✅** 引言首段加入“fitted currents are not idle”与模型错设两句；讨论把“The size of a fitted entropy production says little by itself”段提到首位。
- **B3 ✅ 第 III 节在市场落地**
  - 新增 IV.G 与表 S11（脚本 E9）。
  - 无噪声的 (5,20) 窗口损失 ℓ = 0.10；单窗 5/20/60/250 天分别为 0.11/0.40/0.72/0.95。
  - 用校准 SV 模型模拟：log RV 窗口对损失 0.45，PCA(2) 态损失 0.49；同一估计量对无噪声窗口对的复原误差为 0.003。
  - 结论：损失主要来自输入噪声，而不是窗口延迟。
- **B4 ✅ 非线性特征**：IV.C 新增 drawdown 段；IV.D 解释了 O 两种估计不一致的来源（drawdown 在两个主成分上的载荷为 −0.50 与 −0.27）；表 II 加表注。
- **B5 ✅ 不确定度**
  - 表 II 的数据行全部带 moving-block bootstrap 标准误（20 日块、1000 次、正向/反演/中心化三个序列联合重抽）；表 S10 给出 60 日块（与 20 日块之比 0.5–1.4）。
  - 杠杆通道 P 为 13 个标准误，PCA(2) 的 P 为 11 个；两个加密货币的杠杆项都不到 1 个标准误。
- **B6 ✅** 已写，含第 2 节的更正。
- **B7 ✅** Prop. 3 改为 Lemma 1，加引 Mehler 1866；摘要改为 “in closed form”。
- **B8 ✅ 压缩重复**：III.D 并入 II.D；III.A 压成一段；摘要 140 词。
- **B9 ✅ 数据溯源**
  - README 删除 Doubao；`spx_volume_patches.csv` 增加 `source_url`、`retrieved_on` 两列。
  - 敏感性：用相邻两日均值替代该成交量，所有拟合量的最大变化为 1.4×10⁻⁵（附录 E、表 S1 注、README）。
- **B10 ◐ 文献格式**：Hartich 与 Bisker 已补文章号；Kooloth 已联网核实；Du Plessis 的标题未能核实。

**C 级**

- **C1 ◐** 只加了一句相图描述（逆时针环流，ω < 0），未画新面板。
- **C2 ✗** 未做。按推论 1，单通道两窗口观察者有 ω − O ≡ 0；演示必须设计双通道 (x, c) 与事件定义，不宜仓促。
- **C3 ✅** 图表注中的 CSV 文件名已移入表 S12。
- **C4 ✅** README_P1 中解释了仓库名里的 “committor”。

**其他**

- **Cover letter ✅**：三段式加 5 位建议审稿人加 AI 声明。
- **第 8 节本地部分 ✅**
  - `P1_numbers.py`：约 90 个宏，来源记录在 `P1_numbers_sources.csv`。
  - `sync_labels.py`、`check_table_numbers.py`、`P1_check_prose.py`。
  - Makefile 新增 quick、tables、manuscript 目标；SHA256SUMS_P1 已重生成并通过 `sha256sum -c`。

## 4 论文二（IJF）逐条

**A 级**

- **A1 ✅** episode 数改为全部样本的范围 9–34；摘要、引言、讨论三处都用宏。
- **A2 ✅ 悬空引用**
  - 引言的 companion 措辞已改；附录 A 改为“fixed before any evaluation”加实现核对句；“rule of the protocol”；讨论中的同类说法也已改；表 A1 列名改为 headline/dropped。
  - grep 只剩指令允许的那一句，以及 Zenodo 的 “archived on” 字样。
- **A3 ✅** 表 A1 的 headline 列分位数直接取自 `P2_F2_bootstrap_exp.csv`，与第 4 节逐位一致；F2 与 F7 的重复次数改为 2000。
- **A4 ✅** 第 3 节写明两种 MCB 的子样本差异，并加入直观数字：B 外日的平均 S 为 0.051，事件率为 0.014；自身目标 DSC 0.0103，操作事件 DSC 0.0010。
- **A5 ✅** 第 4 节按四点重写：样本内 PAV 上界；年内排序恒等；跨年由单一全局映射最优；可实现差距为 isotonic 的 0.094 对 0.085。摘要改为 “in-sample”。
- **A6 ✅ 块长**
  - 全部 bootstrap 改为 2h 块（`p2_common.boot_block`），涉及 F2、F4（ΔBSS 主报告与 MCS）、F6、F7、X1，已全部重跑。
  - 新增表 A3，列出 5–60 日块。GBT 在 h = 20 时块长 ≥ 40 日区间含 0；h = 5 时任何块长都不含 0。
  - 图 2、图 4 已重绘为 2h 块区间。
- **A7 ✅** 写明首个样本外年份的规则，及各市场年份 1957 / 2018 / 2021。
- **A8 ✅** 补上 20 天的 quasi-binomial p = 0.033。
- **A9 ✅** 表 A4 给出 8 个族的 MCS p 值。
- **A10 ✅** 区分 event episodes（第 2 节、表 3）与 saturation episodes（第 5 节、表 2）。
- **A11 ◐** 数据可用性写明 `paper2_ijf/` 目录，DOI 为占位；README 中的日期改为 2004-12-22。

**B 级**

- **B1 ✅ 引言重新定位**：第二段讲一般问题；committor 只作为 case study；三条一般性方法贡献；四条实践在引言末尾预告。
- **B2 ◐ 最简基准**
  - “stress index s seq-cal” 已进入表 1、表 4 与正文（结果见第 2 节第 7 条）。
  - 未做：“OU in-sample-cal”。需要训练期逐日的 S，现有 CSV 没有；重跑预测管线又需要环境中未安装的 deeptime。已写入局限。
- **B3 ✅** 表 A5 给出年内 AUC，并注明年内 seq-cal 与原始分数的 AUC 恒等。
- **B4 ✅ 模拟研究**：新脚本 `P2_S1_size_simulation.py`（seed 20261027，每格 2000 次），对应第 5.1 节与新表。日级卡方的 size 为 92.5% 与 99.7%；δ = 1 时置换检验的功效只有 5.7% 与 7.0%。
- **B5 ◐**：已做 GW 常数项/滞后项拆分，以及 Giacomini–Rossi 波动检验（临界值 3.07 由模拟得到）；未画时间路径图。
- **B6 ✅** 第 2 节写明全部超参数；表 A6 给出替代设定，BSS 最大变化为 0.042。
- **B7 ✅** 新增 12 篇并在文中实际使用：triptych、Drehmann–Juselius、Candelon 两篇、HLN 1997、Coroneo–Iacone、Giacomini–Rossi、Politis–White、Bussière–Fratzscher、Beutel、Fouliard、Bluwstein。未做：把图 1(b,c) 改画成 triptych。
- **B8 ✅** 第 7 节扩写为 2011、2020、2025 三段，数字全部是宏。
- **B9 ✅ 可读性**：改用 elsarticle 双倍行距（37 页）；各节加入“对使用者意味着什么”；减少 few episodes 的重复；摘要末句改为面向一般读者的建议。
- **B10 ✅** drift penalty 在第 5 节正文落地。

**C 级与其他**

- **C1、C2、C4 ✗** 未做。
- **C3 ✅** `src/episode_tests.py`。
- **Cover letter ✅**：新写，含三句贡献、companion 说明、声明，以及 5 位建议审稿人。
- **SHA256SUMS_P2 ✅** 已重生成并通过校验。

## 5 复现

**论文一**

```
cd paper1_pre/reproducibility_pre
make -f Makefile_P1 quick   # 约 1 分钟：表、宏、三个 PDF 与交叉引用检查
```

新增分析的运行时间：T2d 4 s，E4g 60 s，E9 85 s。

**论文二**

```
cd paper2_ijf/reproducibility_p2
make -f Makefile_P2 quick
```

评估阶段重跑见 README_P2：F2 95 s，F4 25 s，F7 70 s，X1 95 s，S1 75 s。注意：这些步骤要在前台运行；本次沙箱中的后台任务会被杀掉。
