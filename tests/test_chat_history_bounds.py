"""进程内对话历史的有界契约。

/chat/history 把历史存在进程内。没有上限时，每个出现过的 conversation_id
都会永久占住一份历史，进程活多久就涨多久。这些用例钉住 LRU 淘汰、淘汰顺序
和压缩写回的行为。

跑：python -m pytest tests/test_chat_history_bounds.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import routers.chat as chat
from models.chat import HistoryRequest


class TestConversationHistoryBounds(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        chat.conversations.clear()

    def tearDown(self) -> None:
        chat.conversations.clear()

    async def test_tracked_conversations_stay_bounded(self) -> None:
        overflow = 25
        with patch.object(chat, "chat_history", AsyncMock(return_value="ok")):
            for index in range(chat.MAX_TRACKED_CONVERSATIONS + overflow):
                await chat.llm_service_history(
                    HistoryRequest(conversation_id=f"c{index}", message="hi")
                )

        self.assertEqual(len(chat.conversations), chat.MAX_TRACKED_CONVERSATIONS)
        self.assertNotIn("c0", chat.conversations)
        self.assertIn(
            f"c{chat.MAX_TRACKED_CONVERSATIONS + overflow - 1}", chat.conversations
        )

    async def test_reuse_moves_a_conversation_out_of_the_eviction_line(self) -> None:
        with patch.object(chat, "chat_history", AsyncMock(return_value="ok")):
            for index in range(chat.MAX_TRACKED_CONVERSATIONS):
                await chat.llm_service_history(
                    HistoryRequest(conversation_id=f"c{index}", message="hi")
                )
            # c0 是最久未用的；再访问一次应把它移到队尾。
            await chat.llm_service_history(
                HistoryRequest(conversation_id="c0", message="again")
            )
            # 于是下一个新会话挤掉的是 c1，而不是 c0。
            await chat.llm_service_history(
                HistoryRequest(conversation_id="fresh", message="hi")
            )

        self.assertIn("c0", chat.conversations)
        self.assertNotIn("c1", chat.conversations)
        self.assertIn("fresh", chat.conversations)

    async def test_history_accumulates_within_one_conversation(self) -> None:
        with patch.object(chat, "chat_history", AsyncMock(return_value="answer")):
            await chat.llm_service_history(
                HistoryRequest(conversation_id="c", message="first")
            )
            await chat.llm_service_history(
                HistoryRequest(conversation_id="c", message="second")
            )

        self.assertEqual(
            chat.conversations["c"],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    async def test_compressed_history_replaces_the_stored_one(self) -> None:
        compressed = [{"role": "system", "content": "[摘要]"}]
        with (
            patch.object(chat, "chat_history", AsyncMock(return_value="ok")),
            patch.object(chat, "COMPRESS_THRESHOLD", 2),
            patch.object(
                chat, "compress_chat_history", AsyncMock(return_value=compressed)
            ) as compress,
        ):
            await chat.llm_service_history(
                HistoryRequest(conversation_id="c", message="one")
            )
            compress.assert_not_awaited()
            await chat.llm_service_history(
                HistoryRequest(conversation_id="c", message="two")
            )

        compress.assert_awaited_once()
        self.assertEqual(chat.conversations["c"], compressed)

    async def test_compression_keeps_the_conversation_tracked(self) -> None:
        compressed = [{"role": "system", "content": "[摘要]"}]
        with (
            patch.object(chat, "chat_history", AsyncMock(return_value="ok")),
            patch.object(chat, "COMPRESS_THRESHOLD", 2),
            patch.object(
                chat, "compress_chat_history", AsyncMock(return_value=compressed)
            ),
        ):
            for _ in range(3):
                await chat.llm_service_history(
                    HistoryRequest(conversation_id="c", message="x")
                )

        self.assertEqual(len(chat.conversations), 1)
        self.assertIn("c", chat.conversations)


if __name__ == "__main__":
    unittest.main()
