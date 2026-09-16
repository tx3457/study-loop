from pydantic import BaseModel, ConfigDict, Field, field_validator

class ChatRequest(BaseModel):
    message: str

class HistoryRequest(BaseModel):
    conversation_id: str
    message: str


class ToolChatRequest(BaseModel):
    """Function Calling 聊天请求。

    用户用自然语言描述需求，LLM 自主决定调用哪个工具。
    document_id 可选：如果用户明确指定文档则锁定该文档；省略时不授予
    文档工具范围。user_id/document_id 不接受 ASCII 控制字符，避免把
    身份边界拆成额外的系统提示行。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(default="default_user", min_length=1, max_length=128)
    document_id: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("user_id", "document_id", mode="before")
    @classmethod
    def reject_ascii_control_characters(cls, value):
        if isinstance(value, str) and any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise ValueError("identity fields must not contain ASCII control characters")
        return value


class ToolChatResponse(BaseModel):
    """Function Calling 聊天响应"""
    response: str
    tools_called: list[str]
