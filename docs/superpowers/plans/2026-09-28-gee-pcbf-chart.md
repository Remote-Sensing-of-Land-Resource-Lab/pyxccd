# GEE PCBF Time-Series Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a polished interactive SWIR2 CCDC + PCBF chart to the public Earth Engine example without changing detection logic.

**Architecture:** Keep `applyPCBF` unchanged and add visualization-only helpers below the example configuration. Convert observations, regularly sampled CCDC predictions, and classified break markers to a common feature schema, then render them with one multi-series `ui.Chart.feature.byFeature` chart.

**Tech Stack:** Google Earth Engine JavaScript API, GEE `ui.Chart`, Python `unittest`, Node.js syntax checker.

## Global Constraints

- Do not change PCBF decision logic or default parameter values.
- Keep the fixed five-band list and fixed category threshold unchanged.
- Use concise professional English for comments, map layers, and Console output.
- Do not create or start export tasks.

---

### Task 1: Lock the presentation contract with tests

**Files:**
- Modify: `/Users/liaoronghua/Documents/codex_project/demo/analysis/tests/test_pcbf_gee_public_api.py`

**Interfaces:**
- Consumes: the standalone GEE JavaScript source.
- Produces: regression checks for series names, chart styling, concise English labels, fixed PCBF settings, and absence of removed comments.

- [ ] **Step 1: Add failing source-level tests**
- [ ] **Step 2: Run the test module and confirm failures identify the missing chart**

### Task 2: Implement the interactive chart

**Files:**
- Modify: `/Users/liaoronghua/Documents/codex_project/demo/runs/20260927-pcbf-gee-chinese-example/pcbf_gee_landsat_example.js`
- Modify: `/Users/liaoronghua/Documents/codex_project/demo/pyxccd_recovery_feature/.worktrees/pcbf-public/tutorials/gee/pcbf_gee_landsat_example.js`

**Interfaces:**
- Consumes: CCDC array bands, PCBF record-aligned masks, Landsat SWIR2 observations, display point and dates.
- Produces: one interactive chart with `Observations`, `CCDC fitted trajectory`, `Retained break`, and `PCBF-removed candidate` series.

- [ ] **Step 1: Add helpers to evaluate the active SWIR2 CCDC segment on regular dates**
- [ ] **Step 2: Build observation, fit, and break-marker feature collections**
- [ ] **Step 3: Replace the old scatter chart and verbose Console output with the styled chart and compact break summary**
- [ ] **Step 4: Copy the verified script to the repository tutorial location**

### Task 3: Verify and document use

**Files:**
- Modify: `/Users/liaoronghua/Documents/codex_project/demo/pyxccd_recovery_feature/.worktrees/pcbf-public/tutorials/gee/README.md`

**Interfaces:**
- Consumes: the revised example.
- Produces: concise instructions for changing the point, dates, and three public PCBF parameters and interpreting the chart.

- [ ] **Step 1: Update the README chart description**
- [ ] **Step 2: Run the source tests**
- [ ] **Step 3: Run JavaScript syntax checking and compare both script hashes**
- [ ] **Step 4: Inspect the final diff to confirm algorithm code and defaults are unchanged**
