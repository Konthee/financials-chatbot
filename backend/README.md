# financial-qa backend

FastAPI + LangGraph service for the grounded Financial QA chatbot. See the repository root
[`README.md`](../README.md) for full setup and run instructions.

```bash
PYTHONPATH=src uvicorn financial_qa.app.main:app --reload   # run the API
PYTHONPATH=src python -m financial_qa.app.agent.graph        # render workflow assets
```
