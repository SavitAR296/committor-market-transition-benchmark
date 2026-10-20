# 待确认清单（AUTHOR_INPUT_NEEDED）

两篇论文的 PDF 已是完整版，只剩下面这些需要作者提供或确认的信息。请直接在每条的“答：”后面填写，整份发回即可；收到后我一次性替换进 LaTeX 源码、cover letter、README 并重新编译。
标有【必填】的项不填不能投稿；标有【确认】的项只需回答“是/否”或给出更正；标有【可选】的项不影响投稿。

## 一、投稿元数据

1. 【必填】**Zenodo 新版本 DOI**
   - 用在：论文一 `manuscript/main.tex` 的 `\ZenodoDOI`、`cover_letter.tex`；论文二 `manuscript_p2/main.tex`、`cover_letter.tex`。
   - PDF 中现在显示为加粗的“[DOI of the new Zenodo version]”。
   - 答：10.5281/zenodo.________

2. 【必填】**仓库 commit hash**（大纲第 8 节第 9 条建议打标签 `v2.0-prepr-submission`）
   - 用在：论文一 `main.tex` 的 `\RepoCommit`，PDF 中显示为“[commit hash]”。
   - 答：________

3. 【必填】**两篇互引的写法**（二选一）
   - 现状：论文一引论文二为 “submitted to Int. J. Forecast. (2026)”，论文二引论文一为 “Submitted to Physical Review E”。这种写法只有两篇在同一周投出才成立。
   - (a) 两篇同周投稿，保持现状；
   - (b) 先挂 arXiv，再把 arXiv 号替换进去。大纲第 9 节建议：论文一主分类 cond-mat.stat-mech（交叉 physics.data-an、q-fin.ST），论文二主分类 q-fin.ST（交叉 stat.AP、physics.data-an），先提交论文一。
   - 答：选 (a) / (b)；若选 (b)：论文一 arXiv:________，论文二 arXiv:________

4. 【必填】**论文二 CRediT 作者贡献**（IJF 要求；论文一 cover letter 可同样写一句）
   - 可用角色：Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization, Supervision, Project administration。
   - 答：Boyan Xing：________；Jiayi Pan：________

5. 【确认】**通讯邮箱**：现为 t330025089@mail.bnbu.edu.cn（学生账号，大纲第 5 节提醒过）。是否改为长期可用的邮箱？
   - 答：保持 / 改为 ________

6. 【可选】**两位作者的 ORCID**（Zenodo 元数据、CITATION.cff 与两家期刊投稿表单都会要）
   - 答：Boyan Xing ________；Jiayi Pan ________

## 二、声明

7. 【确认】**资助声明**：论文二现写 “This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.” 论文一 cover letter 未提资助。
   - 答：正确 / 应写为 ________

8. 【确认】**AI 工具声明**：两篇现写 “Claude (Anthropic) 用于编辑文字；AI 编程助手用于开发分析代码”。大纲第 8 节第 2 条要求写明具体工具。请列出写代码时实际用过的工具名称（如 Claude Code、ChatGPT、Copilot、豆包等）。
   - 答：________

9. 【确认】**利益冲突**：两篇都写“无竞争利益”。
   - 答：正确 / 更正 ________

## 三、文献核对

10. 【必填】**论文一文献 `duplessis2026`**，arXiv:2610.01677（Du Plessis, Volodina, Boulton 等）。大纲 B10 要求补标题，我联网未检索到该编号，无法核实。
    - 答：标题为 ________（或：该条目应改为 ________）

11. 【确认】**Hartich & Godec (2024), Nat. Commun. 15, 8678**。DOI 10.1038/s41467-024-52602-0 已核实为 Martínez 等 2019 文章的 Comment；文章号 8678 是按其 Reply（Bisker 等，8679）推断的，请在 nature.com 页面确认。
    - 答：8678 正确 / 应为 ________

12. 【确认】**Xu, Patterson, Levin & Wang (2023), PNAS 120, e2218663120**（“Non-equilibrium early-warning signals for critical transitions in ecological systems”）。核对文献时发现它用“正向与反向互相关之差”作预警信号，正是命题 1 分解的量，故新加到论文一引言与讨论。文章号凭记忆，请核对。
    - 答：正确 / 应为 ________

13. 【必填】**已发表的单作者 CSF 论文**（你提到其中一位作者发过一篇）
    - 期刊要求披露作者的相关已发表工作。请给出题目和 DOI，并回答：它与这两篇是否主题相关？它是否就是公开仓库 README 里写的 “Committor benchmarks for market regime transitions…”？
    - 若相关：两篇都需引用并在 cover letter 说明；若它就是仓库里那篇，“未投稿”的说法要统一。
    - 答：题目/DOI ________；是否相关 ________；是否同一篇 ________

## 四、建议审稿人（已按“无合作关系”写入两封 cover letter，请确认）

14. 【确认】论文一（PRE）：Édgar Roldán（ICTP）、Stephen W. Teitsworth（Duke）、Lucas Lacasa（IFISC, CSIC–UIB）、Vasilis Dakos（ISEM, CNRS/Montpellier）、Jean-Philippe Bouchaud（CFM）。
    - 答：保持 / 增删 ________；希望回避的人选 ________

15. 【确认】论文二（IJF）：Tilmann Gneiting（HITS/KIT）、Timo Dimitriadis（Heidelberg）、Barbara Rossi（UPF）、Mathias Drehmann（BIS）、Christophe Hurlin（Orléans）。
    - 答：保持 / 增删 ________；希望回避的人选 ________
    - 投稿表单若要邮箱，需你们自行查找官方页面填写。

## 五、可选的补充工作（不影响投稿，回答后我再做）

16. 【可选】论文二 B2 的 “OU in-sample-cal” 基准没做：需要训练期逐日的 S 值，现有 CSV 没有，重跑预测管线需要 deeptime。若你们能在本机运行 `P2_F1_forecasts.py` 并导出训练期 S，我可以补上这一行。
    - 答：补 / 不补

17. 【可选】论文一 C1（相图面板）、C2（不可逆 EWS 的 AUC 演示）和论文二 C1、C2、C4 未做；论文二 B7 中“把图 1(b,c) 改画成 triptych”也未做。是否需要补做其中哪几项？
    - 答：________

18. 【可选】论文二现用 elsarticle 双倍行距（37 页）。IJF 接受 “Your Paper Your Way”，也可改回单倍行距的 article 格式。
    - 答：保持 elsarticle / 改回 article

## 六、提交前由你们自己完成的事（我在沙箱里做不了）

- 按大纲第 8 节整理公开仓库与 Zenodo（双作者、`paper1_pre/` 与 `paper2_ijf/` 子目录、删去 “Submitted to CSF” 与旧数字、CITATION.cff、数据许可说明）。公开仓库现状与论文不一致，是审稿人第一分钟就会看到的问题。
- 两个包里的 `SHA256SUMS_*.txt` 已重生成并通过 `sha256sum -c`；上传前请勿再改动包内文件，或改动后重跑清单。
- PRE 投稿时把论文二 PDF 作为 “related manuscript” 上传；IJF 投稿时同样附论文一。
