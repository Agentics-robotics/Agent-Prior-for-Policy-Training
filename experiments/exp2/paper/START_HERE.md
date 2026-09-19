# Exp2 论文材料入口

本包已在全部四组方法的结果核验后生成。先读 [英文完整报告](REPORT.md)，再读
[方法](METHODS.md)、[历史与协议区别](HISTORY.md) 和 [论文写作 AI 指引](WRITING_GUIDE.md)。

- `tables/`：结果、配对比较、任务、训练数据、种子/初态、计算量和 API 用量。
- `policy_cards/`：92 个 API 策略的原文档和源码，逐文件保持原样。
- `PRIOR_CATALOG.md`：按方法和任务排列的 API prior 原文摘要及源码入口。
- `execution_examples/index.html`：40 段可离线播放的原始视频，以及 APPL 调用与 notebook。
- `POLICY_INDEX.csv`：全部 97 个模型的来源、参数量、更新数、checkpoint 路径及哈希。
- `figures/`：可直接排版的 PNG/PDF/SVG 图。
- `FORMALISM.md`：数据窗口、扩散损失及 API 调用接口的数学定义。
- `framework/`、`protocols/`：冻结框架、环境锁、规范和配置的原样副本，供离线查阅。
- `ARTIFACTS.json`：原始证据路径；大模型权重、原始轨迹和视频保留在仓库运行目录。
- `completion.json`：完成状态与本包文件哈希。

可将本目录的 ZIP 交给写作 AI；它能直接读取主要报告、表格和 API 策略原文。
需要检查原始视频或 checkpoint 时，再按索引访问同一仓库。旧 APPL 5.5 中断已按授权补测，当前 1,200 个结果完整；补测记录与视频见
`recovery_20260919/`。原中断保留，不计为任务失败。不要将系统比较写成已经完成的单因素消融。
