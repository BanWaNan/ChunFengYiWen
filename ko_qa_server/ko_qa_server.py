import uvicorn
from fastapi import FastAPI, Request
from zhihaiQA_api import zhihaiQA_router
from zhihai_rag_api import zhihai_rag_router
from Check_system.check_system import check_system_router


app = FastAPI(debug=True)
app.include_router(zhihaiQA_router)
app.include_router(zhihai_rag_router)
app.include_router(check_system_router)

if __name__ == '__main__':
    uvicorn.run(app='ko_qa_server:app', host="0.0.0.0", port=8989)
