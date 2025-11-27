# Retail Analytics Copilot - Hybrid Agent

## Graph Design
- **7-Node LangGraph Architecture**: Router → Retriever → Planner → SQL Generator → Executor → Synthesizer → Repair Loop
- **Intelligent Routing**: Classifies questions as RAG-only, SQL-only, or Hybrid based on document references and data requirements  
- **Constraint-Based Planning**: Extracts date ranges, KPI formulas, categories, and entities from questions and document context
- **Self-Repair Mechanism**: Implements up to 2 repair cycles for SQL errors, empty results, or invalid outputs
---
## Graph Architecture
<img width="1074" height="2892" alt="Image" src="https://github.com/user-attachments/assets/1c9d744e-b042-4371-bbb1-f3d33a08601a" />

---

## DSPy Optimization
- **Optimized Module**: SQL Generator
- **Before Optimization**: valid_sql_rate = 0.25 (1/4 test cases passed)
- **After Optimization**: valid_sql_rate = 0.5 (2/4 test cases passed)  
- **Improvement**: +0.25 (100% improvement rate)
- **Evidence**: See `agent_trace.log` for detailed optimization report and execution traces
---
## Key Assumptions & Trade-offs
- **Cost Approximation**: Used CostOfGoods ≈ 0.7 × UnitPrice as specified in requirements
- **Date Range Handling**: Marketing calendar dates (1997) mapped to actual database date ranges when available
- **Empty Results**: Queries returning no data maintain proper format types with "No data" placeholders
- **Retrieval Simplicity**: Used TF-IDF retrieval with paragraph-level chunking for local operation
---
## Execution & Tracing
- **Trace Log**: `agent_trace.log` provides complete execution visibility with timestamps, node transitions, and metadata
- **Confidence Scoring**: Combines retrieval quality, SQL success, and repair count for sensible confidence estimates  
- **Citation System**: Links answers to specific document chunks and database tables used

*Note: The Northwind sample database contains data primarily from 2012-2023. Historical queries for 1997 return empty results as expected, demonstrating proper handling of data availability constraints while maintaining format compliance.*
