from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    usage: dict

class CodeIssue(BaseModel):
    severity: str
    description: str
    fix: str

class StructuredResponse(BaseModel):
    issues: list[CodeIssue]

class HistoryRequest(BaseModel):
    conversation_id: str
    message: str


class ToolChatRequest(BaseModel):
    """Function Calling 聊天请求。

    用户用自然语言描述需求，LLM 自主决定调用哪个工具。
    document_id 可选：如果用户明确指定文档则传入，否则由 LLM 从对话推断。
    """
    message: str
    user_id: str = "default_user"
    document_id: str | None = None


class ToolChatResponse(BaseModel):
    """Function Calling 聊天响应"""
    response: str
    tools_called: list[str]
