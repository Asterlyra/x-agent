from langchain_core.messages import AIMessage, HumanMessage

from agent.config import AgentSettings
from agent.graph import GraphFactory
from agent.tools import calculate
from conftest import FakeMessagesListChatModelWithTools


def test_graph_dialogue_returns_ai_reply():
    model = FakeMessagesListChatModelWithTools(responses=[AIMessage(content="你好！")])
    graph = GraphFactory(model, AgentSettings(tools=())).build()
    result = graph.invoke({"messages": [HumanMessage(content="你好")]})
    assert result["messages"][-1].content == "你好！"


def test_graph_executes_tools_in_declared_order():
    model = FakeMessagesListChatModelWithTools(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculate", "args": {"expression": "2+3"}, "id": "call-1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="答案是 5"),
        ]
    )
    graph = GraphFactory(model, AgentSettings(tools=(calculate,))).build()
    result = graph.invoke({"messages": [HumanMessage(content="2+3 等于几？")]})
    assert any(message.type == "tool" and message.status == "success" for message in result["messages"])
    assert result["messages"][-1].content == "答案是 5"
