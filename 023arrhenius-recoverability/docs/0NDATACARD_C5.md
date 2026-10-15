# 0NDATACARD_C5.md — C5 稿件数据卡（SHA-256 登记）

生成时间：2026-10-15 15:24:38 +0800 · 生成脚本：`02code/120datacard_sha256.py`

> 依据项目约定 §4.6 / §202。本卡只登记 **C5 稿件（`Elsevier_zh.tex` / `Elsevier_en.tex`）实际依赖**的文件，
> 不含 C6/光谱方向的历史产物（`03data/processed/` 下 35 项与本稿无关）。

> **诚实边界**：SHA-256 仅证明**自本次登记起**文件字节未变，不追认登记之前的历史，
> 也不构成对提取步骤正确性的证明——提取正确性由脚本 55 的 `read_html` 与人工/Europe PMC 逐条核验承担。

---

## 一、源数据

### apple_firmness — 苹果硬度温度梯度（4 品种 × 5 贮温 × 5 时间点）
- **角色**：C5 真实动力学 / 体系 1
- **来源**：Foods 2023, doi:10.3390/foods12112113 (PMC10253207) 表 A1（开放获取）
- **获取/提取方式**：pandas.read_html 提取（脚本 55）+ 人工逐条核验；非本组采集
- **许可**：CC BY 4.0（Foods 为全开放获取）
- **在盘路径**：`03data/processed/pmc_real/PMC10253207_T16.csv`
- **大小**：1,770 bytes
- **SHA-256**：`d22131f037a120e71a321d358e141ea0b8b2b8b97c4299706f4b0c8fad66665f`

### food_ammonia_tbars — 长货架期食品 氨/TBARS 累积动力学（5 食品 × 4 贮温，10 序列）
- **角色**：C5 真实动力学 / 体系 2
- **来源**：Foods 2022 11(14):2004, doi:10.3390/foods11142004 (PMC9319022) 表 3
- **获取/提取方式**：pandas.read_html 提取（脚本 55）+ Europe PMC fullTextXML 独立重抓逐值核对（132 行 0 处不一致）
- **许可**：CC BY 4.0
- **在盘路径**：`03data/processed/pmc_real/PMC9319022_T3.csv`
- **大小**：5,239 bytes
- **SHA-256**：`ee6da3d22db941fe2936bba9817e3aa7873e408024ae9ab95d6bf48d536ccfbe`

### hass_avocado — Hass 牛油果采后成熟（478 果 × 3 组贮藏，14,722 条记录）
- **角色**：C5 真实动力学 / 体系 3（真统计估计误差）
- **来源**：Mendeley Data, doi:10.17632/3xd9n945v8.1
- **获取/提取方式**：原始 xlsx 直接读取（未做 L1 变换）；按 4.5 只读引用，未复制进项目
- **许可**：CC BY 4.0（Mendeley Data 默认）
- **在盘路径**：`${ARIS_RAW_ROOT}/004_hass_avocado_rgb_ripening/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset.xlsx`
- **大小**：633,351 bytes
- **SHA-256**：`f8abeaba6eedf67869907b3a71ee132cd89b42bb3dcce6b25ddb22e6269b51a7`

---

## 二、派生产物（稿件每个数字的来源）

| 产物 | 内容 | bytes | SHA-256 (前 16) |
| --- | --- | ---: | --- |
| `43trichotomy_formal.json` | 半合成三分律：Ψ 终值、塌缩斜率、Krug 匹配 | 1,870 | `771986e2487d00f3` |
| `54c5_realkinetics.json` | 苹果体系：品种 Ea/R²、设计扫描、超定子集 | 6,711 | `b758bd9a95da3758` |
| `54c5_realkinetics-designsweep.csv` | 苹果 26 温度子集逐条（补充表 S2 数据源） | 4,759 | `7a598b3e1aa8494f` |
| `91c5_realkinetics_killer.json` | Ψ vs 廉价启发式判别力、Ψ_het 跨体系 | 7,404 | `79f6df8ca0e94179` |
| `99_realkinetics2.json` | 食品体系：Ea 范围、同质/异质 Ψ 复现 | 14,118 | `b41c66625cde8d18` |
| `102_avocado_c5_pilot.json` | 牛油果体系：真统计估计误差、n 轴扫描 | 2,519 | `125ba608e1256276` |
| `119pazman_pronzato_bridge.json` | 与 Pázman–Pronzato 2006 算例的坐标对照 | 2,236 | `6d5f827620897f26` |
| `122criterion_bakeoff.json` | 准则对决（含对自身假设的证伪：塌缩=代数、D 对比未建立） | 7,884 | `360a30060f01d3a8` |
| `123montecarlo_criterion_study.json` | 1000 独立设计 Monte-Carlo：两轴速率、饱和、决策 regret | 28,895 | `3a1470f119505724` |
| `123montecarlo_criterion_study-designs.csv` | 逐设计表：准则值 + 真值 RMSE（正文 §sec:mc 数据源） | 493,976 | `03fce89d2c5fae06` |
| `124coverage_study.json` | 置信区间覆盖率研究：Ψ 作为标定诊断被证伪 + 区间口径警告 | 7,521 | `82f0cb50a1edfde2` |
| `124coverage_study-designs.csv` | 逐设计覆盖率表 | 264,689 | `4eb6d48a5b9e45c7` |
| `126trichotomy_plugin_psi.json` | 去 oracle 的 plug-in Ψ(θ̂) 三分律复现 + 预拟合收敛守卫 | 6,406 | `10eec76103ae3992` |
| `126trichotomy_plugin_psi-percell.csv` | plug-in Ψ 逐 (regime,n,seed) 明细 | 35,457 | `d0cf0e1edafa780e` |
| `127budget_matched_sensitivity.json` | 条件数劣势对预算配平容差的敏感性（收紧到严格同预算） | 23,058 | `650ffc11c60f9678` |
| `128montecarlo_runtime_receipt.json` | §sec:mc 墙钟时间凭据 + 123 的字节级可复现性核验 | 3,087 | `72b4f6c61aea9824` |
| `129kgrid_oracle_ablation.json` | k 搜索网格去 oracle 消融：162 键无一变动 >2% | 4,206 | `15ea31f6b8b5ea52` |
| `43trichotomy_formal-agg.csv` | 图 fig:tri 面板 a/b 的逐 regime 聚合数据源（42paper_figs_v8.fig2 直读） | 3,209 | `c1ec85c4bf0dbabd` |
| `43trichotomy_formal-percell.csv` | 图 fig:krug 的逐格经验/Krug 斜率数据源（42paper_figs_v8.fig3 直读） | 23,494 | `c1fbcc1f02d66896` |
| `105multiseed_realdata.json` | 真实数据四脚本 × 5 正典种子的跨种子聚合：正文各处「5 种子均值」与品种表 ±std 的来源 | 193,286 | `a72ec2811affc424` |
| `130aggregate_pilot_robustness.json` | 正文 ±30% 档「96% 保持负号 / 87% 保持 ≤−0.5」两个比例的【现行生成步骤与结果】（脚本 130） | 2,700 | `577273d3eaffb48e` |
| `104ablation_psi_aggregation.json` | Ψ_n 聚合判据与 N 因子消融（订正后 Ψ_n=λmin(I_total) 口径） | 10,086 | `3616bdb5ec2fe061` |
| `107design_multivariety.json` | 三单调品种 48 子集的（品种×ΔT）分层置换检验 | 10,494 | `d8f6d9a44b73e6ab` |
| `131aggregate_c5_partial_tally.json` | 正文「逐次抽样 5 次 Ψ_n 严格大于 cond、1 次数值持平」这一【跨 6 次抽样】汇总的现行生成步骤与结果（脚本 131）；此前该汇总只由人手从 6 个 91c5 产物数出，仓库内无聚合器 | 6,096 | `f9eb3dac3bf3a4c8` |
| `133ablation_innovation_map.json` | 创新点-消融映射与 LAQ 消融：tab:claims「仿射时斜率 −0.508 至 −1.288 / ρ=1/2 处 +0.077」与增项闭式相对误差 <5e−7 的直接来源（脚本 133） | 21,408 | `d6ec22da0f06c1cf` |
| `133ablation_innovation_map-laq.csv` | LAQ 消融逐 (种子, n, ρ) 明细表，即上一行各斜率的被聚合件 | 27,091 | `563dc6e750088626` |
| `124coverage_study-bins.csv` | 覆盖率按 Ψ 十分位分箱表：正文「最低/最高十分位覆盖」「$z$ 分位 0.78--0.81」「无自由度修正 0.66--0.71」「报出 SE 为真实 SD 的 1.32--1.70 倍」四组数的直接来源 | 2,100 | `d89130eac78188a2` |
| `122criterion_bakeoff-percriterion.csv` | 准则对决逐判据明细：tab:bakeoff 各行边际/偏 Spearman 与成对 Spearman 的被聚合件 | 9,702 | `df8080276e3885e0` |
| `99_realkinetics2-overdet.csv` | 食品体系超定子集逐条明细：正文该体系同质/异质 Ψ 复现数字的被聚合件 | 9,139 | `b0fcaad1b82f89db` |
| `107design_multivariety-sweep.csv` | 三单调品种 48 子集的逐子集扫描表：（品种×ΔT）分层置换检验的输入 | 77,087 | `a15db8f41b5dca14` |
| `104ablation_psi_aggregation_regtest.json` | Ψ_n 聚合消融的回归测试基准件（133 的 REGTEST_EXCLUSION 按文件名点名引用它做取证依据，故它本身必须可核；见 §9.0 裁决『保留、不删、不改名』） | 10,152 | `9bacfdd76f736546` |
| `137design_variance_scale_invariance.json` | 经验/总体设计方差相对误差的尺度不变性核验：§sec:converse 证明第 (2) 步「总体矩读法不额外要价」所依据者（Weyl 上界对本设计族是松的）；正文不引用其数字，但审计 §L.14.3 引用，故须可核 | 3,670 | `e37352c945fb7ef0` |
| `138eigendirection_angle_check.json` | 同方差代理与异方差定义量的最小特征方向夹角预检（义务 1f9bcf45 的预注册预测依据）：正文不引用其数字，但审计 §L.16 引用，故须可核 | 3,728 | `dadb67e9736d4b00` |
| `141rya579_seedscope_receipt.json` | 「5 种子聚合 vs 原生抽样」的产物指向分界：稿面 §统计检验与误差报告按文件名点名引用它（设计侧逐字节相同的 8 列/122 键 vs 随种子变的 5 列/24 键），故须可核 | 4,959 | `08dae6277e01895d` |
| `142rya579_method_identity_receipt.json` | 两阶段管线与正过程对应的三条恒等式核验（合同残差 0、delta 恒等式残差 1.97e-16）：稿面方法节按文件名点名引用其数字，故须可核 | 1,893 | `29008af9afb97f73` |
| `143rya579_ledger_receipts.json` | 取证台账收据：a9930fd7 的「刻意声明」定位与重复组两步判据的实测结果；正文不引用其数字，但审计 §L.20.6/§L.20.7 引用，故须可核 | 6,307 | `5edcd0407cb97c5f` |
| `144rya579_critical_kl_check.json` | LAN 临界档精确 KL 恒等式（eq:klexact）的数值核验：二次近似最大相对误差 6.04e-5；正文不引用其数字，但审计 §L.20.1 引用，故须可核 | 8,263 | `9f8297ec6fd1b7ad` |
| `145citation_field_diff.json` | 参考文献逐字段机械比对（稿面 \bibitem vs Crossref，DOI 直查兜底）：57 条 → AGREE 53 / CHECK 1（donoho2009，见审计 §L.21.8）/ 无注册记录 3（预印本与专著）；正文不引用其数字，但审计 §L.21.8/§L.22.3 引用，故须可核 | 28,083 | `6b43ea0da4f06711` |
| `147oracle_plugin_error_pairing.json` | oracle 层(43) 对 plug-in 层(126) 两列退化方向误差的【配对】差（RYA-583 新增）：140/140 全不同、最大绝对差 3.6418e-4、最大相对差 8.3567e-4、两列 Spearman 1.000；即稿面 §sec:relate-id 那三个数此前无产物字段承载的那一处（外审核对判 real_gap）。由 126 脚本产出但【独立成件】，理由见该文件 why_separate：126 的 json 是取证台账义务 c13bd67a3388 的 resolve 收据所指证据，收据每次门禁重验哈希，改动它会让收据当场自失效 | 1,786 | `98983af916df32ca` |
| `155relabel_release_fields.json` | 已登记产物里几处评审写法的字段改写台账（2026-10-15）：103 的 10 份逐种子产物（105_seeds 与 RYA-103 复跑档案各 5 份）、143 收据与 145 比对产物，逐文件记改写前登记的 SHA-256、改写前后的 SHA-256 与各规则命中数；只改目标字段，其余字节不动，不重跑。正文不引用其数字，但审计 §L.34 引用，故须可核 | 8,134 | `5ca11c2b42c8c8fd` |
| `requirements.lock.txt` | 逐包精确到版本的依赖锁（含解释器 3.13.5 与平台），即 §sec:repro 明写的 requirements.lock.txt。取证义务 24e3754f 指出「稿面声明了依赖锁但清单里没有它」——锁文件本身在盘（发布树根目录），缺的是它未进 SHA-256 登记，故自本轮起登记，使读者可核到与本文数字同一份锁 | 2,783 | `4db56e55d692775a` |
| `91c5_realkinetics_killer_seed19810915.json` | 苹果/食品体系逐种子原始产物：正文「5 种子均值 −0.70 与 −0.37」的直接来源 | 7,386 | `67e75bb0436a0c9a` |
| `91c5_realkinetics_killer_seed20041210.json` | 苹果/食品体系逐种子原始产物：正文「5 种子均值 −0.70 与 −0.37」的直接来源 | 7,373 | `ce0069561ddc1ff6` |
| `91c5_realkinetics_killer_seed20060515.json` | 苹果/食品体系逐种子原始产物：正文「5 种子均值 −0.70 与 −0.37」的直接来源 | 7,394 | `ad9199d5e701e30f` |
| `91c5_realkinetics_killer_seed2023.json` | 苹果/食品体系逐种子原始产物：正文「5 种子均值 −0.70 与 −0.37」的直接来源 | 7,396 | `c8b8889517d9ce51` |
| `91c5_realkinetics_killer_seed2024.json` | 苹果/食品体系逐种子原始产物：正文「5 种子均值 −0.70 与 −0.37」的直接来源 | 7,528 | `60b71038a643f43b` |
| `102_avocado_c5_pilot-seed19810915.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,524 | `c2b4229b4ecfe8e8` |
| `102_avocado_c5_pilot-seed20041210.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,522 | `71613de2e32238cd` |
| `102_avocado_c5_pilot-seed20060515.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,523 | `e681817f86efdad0` |
| `102_avocado_c5_pilot-seed2023.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,521 | `8a9be192cee30757` |
| `102_avocado_c5_pilot-seed2024.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,519 | `d0eeb03c9a39f169` |
| `103sensitivity_pilot-seed19810915.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,390 | `20e3c98f866f2b9a` |
| `103sensitivity_pilot-seed20041210.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,401 | `b5a76a3175ae0a14` |
| `103sensitivity_pilot-seed20060515.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,416 | `efd6cd7c4b4387bd` |
| `103sensitivity_pilot-seed2023.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,418 | `1f43be8c144488bd` |
| `103sensitivity_pilot-seed2024.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 2,420 | `3edcfa95303f6c44` |
| `54c5_realkinetics-seed19810915.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 6,712 | `bf4b310b6600eb1a` |
| `54c5_realkinetics-seed20041210.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 6,711 | `b7fcfb4cefd3d110` |
| `54c5_realkinetics-seed20060515.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 6,712 | `f60808cc1ff85782` |
| `54c5_realkinetics-seed2023.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 6,713 | `9121def945338256` |
| `54c5_realkinetics-seed2024.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 6,710 | `356ead6ce4bb2d30` |
| `99_realkinetics2-seed19810915.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 14,120 | `b4cdfd1d6d60f7c3` |
| `99_realkinetics2-seed20041210.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 14,100 | `2dee35dcc9dbba62` |
| `99_realkinetics2-seed20060515.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 14,118 | `a4dca9cb64639010` |
| `99_realkinetics2-seed2023.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 14,121 | `606439547a108a26` |
| `99_realkinetics2-seed2024.json` | 105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件 | 14,120 | `61ab3d1a051cb2ec` |
| `103sensitivity_pilot-grid-seed19810915.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 38,542 | `641d7071fe436c11` |
| `103sensitivity_pilot-grid-seed20041210.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 38,495 | `13efdfc9c442fc36` |
| `103sensitivity_pilot-grid-seed20060515.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 38,282 | `f63d85cce0d55acb` |
| `103sensitivity_pilot-grid-seed2023.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 37,481 | `122504317ddd7777` |
| `103sensitivity_pilot-grid-seed2024.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 38,217 | `19ee7fda1c61aafd` |
| `54c5_realkinetics-designsweep-seed19810915.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 4,759 | `a6b753acbe2a74fb` |
| `54c5_realkinetics-designsweep-seed20041210.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 4,757 | `01f31ed2988fd7eb` |
| `54c5_realkinetics-designsweep-seed20060515.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 4,765 | `2434ae3ddefd5fdc` |
| `54c5_realkinetics-designsweep-seed2023.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 4,767 | `8116c520a9de7356` |
| `54c5_realkinetics-designsweep-seed2024.csv` | 105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格） | 4,763 | `50fd327fdcc594a7` |
| `104ablation_psi_aggregation_seed19810915.json` | 104 聚合消融的逐种子原始产物（5 个预注册种子）：正文「增量恒为负但跨度近三倍（−0.0765 至 −0.2353）」这一跨种子区间的被聚合件 | 10,069 | `f00beb34254b724a` |
| `104ablation_psi_aggregation_seed20041210.json` | 104 聚合消融的逐种子原始产物（5 个预注册种子）：正文「增量恒为负但跨度近三倍（−0.0765 至 −0.2353）」这一跨种子区间的被聚合件 | 10,102 | `8d88bfe985c96492` |
| `104ablation_psi_aggregation_seed20060515.json` | 104 聚合消融的逐种子原始产物（5 个预注册种子）：正文「增量恒为负但跨度近三倍（−0.0765 至 −0.2353）」这一跨种子区间的被聚合件 | 10,102 | `c6a30ed2a9f256bf` |
| `104ablation_psi_aggregation_seed2023.json` | 104 聚合消融的逐种子原始产物（5 个预注册种子）：正文「增量恒为负但跨度近三倍（−0.0765 至 −0.2353）」这一跨种子区间的被聚合件 | 10,100 | `2320409ad8f3be85` |
| `104ablation_psi_aggregation_seed2024.json` | 104 聚合消融的逐种子原始产物（5 个预注册种子）：正文「增量恒为负但跨度近三倍（−0.0765 至 −0.2353）」这一跨种子区间的被聚合件 | 10,112 | `b508d4ad39e14cef` |
| `103sensitivity_pilot-grid-seed19810915.csv` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 36,753 | `7c03ad8a3fbb6028` |
| `103sensitivity_pilot-grid-seed20041210.csv` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 38,421 | `b2938efeb1c21621` |
| `103sensitivity_pilot-grid-seed20060515.csv` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 38,366 | `6a575bdb81035b4a` |
| `103sensitivity_pilot-grid-seed2023.csv` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 37,824 | `aada557fe110705e` |
| `103sensitivity_pilot-grid-seed2024.csv` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 36,450 | `81f3fcb124ab51cb` |
| `103sensitivity_pilot-seed19810915.json` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 2,379 | `5f4889e01cc8837a` |
| `103sensitivity_pilot-seed20041210.json` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 2,401 | `43b9d99edfb7c142` |
| `103sensitivity_pilot-seed20060515.json` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 2,356 | `ca7b4b40c401eaab` |
| `103sensitivity_pilot-seed2023.json` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 2,385 | `427fc85e0ab8a05a` |
| `103sensitivity_pilot-seed2024.json` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 2,400 | `1e3db52ed824e5ee` |
| `REDACTION_LEDGER.md` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 4,879 | `53e6ec3d1646a34f` |
| `aggregate_result.json` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 2,207 | `0b1f72da1cf01608` |
| `audit_output.json` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 7,966 | `181ed4347146c81a` |
| `audit_prompt.txt` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 2,981 | `489949169cdf76c8` |
| `rya103_verify_runner.sh` | RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据** | 1,088 | `17c0b68300489b63` |

完整 64 位摘要见 `04outputs/0NDATACARD_C5_manifest.tsv`。

---

## 三、与稿件的对应
- 稿件表 `tab:datacard`（数据卡表）列出的三个真实体系即本卡第一节三项。
- 稿件正文与补充材料的每个数字，其来源产物见本卡第二节，并在 `06doc/02sub/00SUBMISSION_AUDIT.md` §一（原 `claim_evidence_map.md`，2026-07-27 原样并入）中逐条绑定。

## 四、缺失项

无。

## 五、覆盖闸门（反向核对）

> `缺失项` 只回答「清单里的文件在不在盘」，回答不了「在盘的该进清单的文件有没有进清单」。本节对已声明覆盖的目录反向再扫一遍：

- `04outputs/105_seeds`
- `05logs/rya103_verify_20260810`

**在盘但未登记**：无。
