# CHG-001: Repo Bootstrap + OpenSpec 初始化

**Change ID**: CHG-001  
**建立日期**: 2026-05-13  
**作者**: Tony Lo (via Antigravity)  
**狀態**: Applied ✅  
**對應分支**: `chore/repo-bootstrap`

---

## Change Summary

Phase 1 bootstrap：建立乾淨、可維護、可重現的專案骨架，並初始化 OpenSpec workflow。

---

## Motivation

本作業需要一個結構清晰的 Repo 以確保：
1. 可重現性（不同機器可重現實驗）
2. 可維護性（各模組分離，易於迭代）
3. 可驗證性（實驗結果有據可查）
4. 可繳交性（GitHub Repo 結構符合評分標準）

---

## Changes Applied

### 目錄結構
- ✅ 建立完整 `src/` 模組目錄
- ✅ 建立 `configs/`, `experiments/`, `results/` 分層目錄
- ✅ 建立 `report/`, `scripts/`, `notebooks/` 目錄
- ✅ 建立 `openspec/changes/`, `designs/`, `tasks/` 目錄

### Source of Truth 文件
- ✅ `README.md` — 首頁說明
- ✅ `PROJECT_CHARTER.md` — 作業目標與子任務定位
- ✅ `ASSIGNMENT_REQUIREMENTS.md` — 評分標準整理
- ✅ `EXPERIMENT_STORYLINE.md` — 實驗敘事規劃
- ✅ `REPRODUCIBILITY.md` — 可重現性說明

### Python 環境
- ✅ `requirements.txt`
- ✅ `environment.yml`
- ✅ `pyproject.toml`

### OpenSpec
- ✅ `openspec/openspec.yaml` — 專案設定
- ✅ `openspec/changes/CHG-001-repo-bootstrap.md` — 本 change
- ✅ `openspec/designs/DES-001-project-architecture.md` — 架構設計
- ✅ `openspec/tasks/TASK-001-phase1-bootstrap.md` — 任務追蹤

### Starter Code
- ✅ 保留原始 `第3章程式_ALL_IN_ONE.ipynb` 於 `notebooks/starter/`
- ✅ 備份至 `data/raw/`

---

## Impact

- **破壞性更改**: 無
- **相依關係**: 所有後續 Phase 依賴此骨架
- **回滾方式**: 刪除整個目錄（無業務邏輯）

---

## Verification

- [x] `git status` 顯示所有檔案已追蹤
- [x] 目錄結構符合規格
- [x] 文件均以繁體中文撰寫
- [x] 無訓練程式碼包含於本 change
