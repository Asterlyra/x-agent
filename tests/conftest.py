from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class FakeMessagesListChatModelWithTools(FakeMessagesListChatModel):
    """让 LangGraph 测试模型接受工具绑定。"""

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self
