# PROJECT CHARTER — DRL HW3: DQN and its Variants

> **文件版本**：v1.0  
> **建立日期**：2026-05-13  
> **作者**：Tony Lo  
> **語言原則**：所有報告以繁體中文撰寫，英文專業名詞可保留

---

## 1. 作業目標（Project Objective）

本專案目標是基於 **GridWorld 環境**，從 Naive DQN 出發，依序實作並比較多種 DQN 變體，最終提交一份具備：

- ✅ **可重現性（Reproducibility）**：固定隨機種子、記錄完整超參數
- ✅ **可驗證性（Verifiability）**：實驗結果有 CSV 數據與圖表佐證
- ✅ **研究報告水準（Research-grade Report）**：以繁體中文撰寫，包含實驗分析
- ✅ **乾淨 GitHub Repo（Clean Repo）**：結構清晰、commit 歷史清楚

的完整作業。

---

## 2. 子任務定位（Scope of Each Sub-task）

### HW3-1：Naive DQN for Static Mode（30 分）

**目的**：驗證基礎 DQN 能在最簡單（完全靜態）的 GridWorld 環境中收斂。

| 項目 | 說明 |
|------|------|
| 環境模式 | Static（所有物件固定） |
| 必要實作 | Basic DQN + Experience Replay Buffer |
| 繳交項目 | ① 可執行的程式碼 ② 理解報告（understanding_report.md） |
| 評分依據 | 程式能跑 + 理解報告品質 |
| 對應報告 | `report/understanding_report.md`（**必繳，不可遺漏**） |

> ⚠️ **understanding_report.md 是 HW3-1 的必繳項目，遺漏將直接扣分。**

---

### HW3-2：Enhanced DQN Variants for Player Mode（40 分）

**目的**：實作並比較 Double DQN 與 Dueling DQN，分析其相較 Naive DQN 的改進原理。

| 項目 | 說明 |
|------|------|
| 環境模式 | Player（Player 隨機，其他固定） |
| 必要實作 | Double DQN + Dueling DQN |
| 核心分析 | 兩種方法如何改善基礎 DQN 的問題（過估、表示能力） |
| 對應設定 | `configs/hw3_2_player/` |
| 對應實驗 | `experiments/hw3_2_player/` |

---

### HW3-3：Enhance DQN for Random Mode WITH Training Tips（30 分）

**目的**：在最難（完全隨機）的環境中，透過訓練技巧（Training Tips）穩定學習。

| 項目 | 說明 |
|------|------|
| 環境模式 | Random（所有物件隨機） |
| 必要實作 | DQN 轉換至 **Keras 或 PyTorch Lightning** |
| 訓練技巧 | Gradient Clipping、LR Scheduling、Epsilon Decay 等 |
| Bonus | 整合更多 training techniques（Rainbow 元素） |
| 對應設定 | `configs/hw3_3_random/` |
| 對應實驗 | `experiments/hw3_3_random/` |

---

### Bonus：Rainbow DQN Integration

**目的**：整合 Rainbow DQN 核心元素，展示對進階 DRL 技術的掌握。

| Rainbow 元素 | 說明 |
|-------------|------|
| Double DQN | 解決 Q 值過估問題 |
| Dueling Network | 分離 Value 與 Advantage 函數 |
| Prioritized Experience Replay (PER) | 優先重要樣本 |
| Multi-step Learning | N-step Returns |
| Noisy Networks | 探索策略改進（可選） |
| Distributional RL | C51 / QR-DQN（可選） |

---

## 3. 不可遺漏項目清單（Must-Have Checklist）

- [ ] `report/understanding_report.md` — HW3-1 理解報告（**最高優先**）
- [ ] HW3-1 Static Mode DQN 可執行
- [ ] HW3-2 Double DQN + Dueling DQN 實作與比較圖
- [ ] HW3-3 Random Mode + Training Tips（PyTorch Lightning 或 Keras）
- [ ] 完整研究型實驗報告（繁體中文）
- [ ] GitHub Repo 公開可存取

---

## 4. 技術架構原則（Technical Architecture Principles）

1. **模組化設計**：`src/` 下各模組獨立，可任意組合
2. **設定檔驅動**：超參數以 `configs/*.yaml` 管理，不寫死在程式碼中
3. **可重現性優先**：所有訓練固定隨機種子（seed），記錄完整實驗設定
4. **報告自動化**：`scripts/generate_report_assets.py` 自動生成圖表
5. **不假造結果**：所有實驗數據必須來自真實執行

---

## 5. 使用框架（Technology Stack）

| 層級 | 工具 |
|------|------|
| 深度學習 | PyTorch（主）+ PyTorch Lightning（HW3-3） |
| 環境 | GridWorld（自定義，基於 starter code） |
| 實驗管理 | 手動 CSV + matplotlib 圖表 |
| 版本控制 | Git + GitHub |
| 報告 | Markdown（繁體中文） |
| SDD Workflow | OpenSpec + Antigravity |

---

## 6. 實驗敘事方向（Experiment Narrative Direction）

本作業的實驗敘事應以「從簡單到複雜、從單一到整合」為主軸：

1. **HW3-1**：驗證基礎 DQN 可學習靜態環境，建立 baseline
2. **HW3-2**：引入 Double DQN 改善過估問題，引入 Dueling 改善表示
3. **HW3-3**：面對最難環境，用訓練技巧穩定學習
4. **Bonus**：Rainbow 整合，展示對 state-of-the-art 的理解

---

## 7. OpenSpec Change History

| Change ID | 階段 | 說明 |
|-----------|------|------|
| CHG-001 | Phase 1 | Repo Bootstrap + OpenSpec 初始化 |
| CHG-002 | Phase 2 | HW3-1 Naive DQN 實作（待辦） |
| CHG-003 | Phase 3 | HW3-2 Double/Dueling DQN 實作（待辦） |
| CHG-004 | Phase 4 | HW3-3 Random Mode + Training Tips（待辦） |
| CHG-005 | Phase 5 | Rainbow Bonus + 報告整合（待辦） |
