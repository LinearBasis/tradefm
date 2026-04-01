# Diploma Project: Market Representation Learning for Trading

## 1. Overview

This project explores the application of sequence-based models (inspired by TradeFM) to financial market data, with the goal of learning a **latent representation of market dynamics**.

The learned representation is then used for **decision-making in trading**, rather than direct price prediction.

---

## 2. Goal

The objective of this diploma is:

> To develop and evaluate a method for learning market representations from trade/order flow data and to assess their usefulness for trading decision-making.

---

## 3. Key Idea

Instead of predicting prices directly, the model:

- learns **market dynamics** from event sequences
- builds a **latent state of the market**
- uses this state for downstream decision-making

---

## 4. High-Level Pipeline

Raw market data (trades / order flow)  
↓  
Feature engineering (scale-invariant features)  
↓  
Sequence construction  
↓  
Generative / sequence model (TradeFM-like)  
↓  
Latent representation (market state)  
↓  
Decision-making module  
↓  
Backtesting / evaluation  

---

## 5. Project Structure
data/
raw/
processed/
features/

src/
data/
models/
decision/
backtest/
utils/

configs/
experiments/
notebooks/

docs/
NOTES.md
ARCHITECTURE.md
PLAN.md

---

## 6. Approach

The project is inspired by TradeFM, which models market microstructure as an autoregressive sequence of trade events.

Key ideas:

- modeling **event-level data** instead of full LOB
- using **scale-invariant features**
- learning **general representations across assets**

However, this work focuses on:

- adapting these ideas for trading
- exploring alternative model outputs
- integrating decision-making on top of learned representations

---

## 7. Decision-Making Layer

The model output is not used directly.

Instead, a separate module will:

- take latent representation as input
- produce trading decisions:
  - direction (buy/sell/hold)
  - confidence / signal strength
  - optional execution parameters

Different approaches will be explored:

- simple ML models (baseline)
- neural heads
- (optionally) reinforcement learning

---

## 8. Evaluation

Evaluation is a critical part of the project.

It includes:

### 1. Statistical evaluation
- distribution of outputs
- stability across time
- robustness

### 2. Backtesting
- simulated trading performance
- PnL
- Sharpe ratio
- drawdown

### 3. Ablation studies
- impact of representation
- impact of model design

---

## 9. Key Challenges

- Non-stationarity of markets
- Partial observability (no full LOB)
- Data leakage / look-ahead bias
- Evaluation of generative models
- Linking representation → decisions

---

## 10. Expected Contributions

- A pipeline for learning market representations from event data
- Adaptation of sequence models for trading
- Empirical evaluation of representation usefulness
- Comparison of decision-making approaches

---

## 11. Current Status

- [ ] Paper decomposition (TradeFM)
- [ ] Define system design
- [ ] Build data pipeline
- [ ] Implement baseline model