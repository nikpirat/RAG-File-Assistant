"""Conversation memory management."""
import json
from typing import List, Dict, Optional
from datetime import datetime, timezone
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from src.config.settings import settings
from src.config.logging_config import get_logger
from src.models.database import ChatHistory

logger = get_logger(__name__)


class ConversationMemory:
    """Manage conversation memory with Redis (short-term) and PostgreSQL (long-term)."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.redis_client: Optional[aioredis.Redis] = None
        self.max_messages = settings.memory_max_messages

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("memory_initialized")
        except Exception as e:
            logger.warning("redis_connection_failed", error=str(e))
            self.redis_client = None

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    def _get_redis_key(self, user_id: str, chat_id: str) -> str:
        """Get Redis key for conversation."""
        return f"chat:{user_id}:{chat_id}"

    async def add_message(
        self,
        user_id: str,
        chat_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        retrieved_files: Optional[List[str]] = None,
    ) -> None:
        """Add message to memory."""
        logger.info("adding_message", user_id=user_id, role=role)

        # Store in PostgreSQL (long-term)
        chat_msg = ChatHistory(
            user_id=user_id,
            chat_id=chat_id,
            role=role,
            content=content,
            message_type=message_type,
            retrieved_files=retrieved_files,
        )
        self.session.add(chat_msg)
        await self.session.commit()

        # Store in Redis (short-term cache)
        if self.redis_client:
            try:
                key = self._get_redis_key(user_id, chat_id)
                message_data = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Add to list
                await self.redis_client.lpush(key, json.dumps(message_data))

                # Keep only recent messages
                await self.redis_client.ltrim(key, 0, self.max_messages - 1)

                # Set expiry (7 days)
                await self.redis_client.expire(key, 604800)
            except Exception as e:
                logger.warning("redis_add_failed", error=str(e))

    async def get_recent_messages(
        self,
        user_id: str,
        chat_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Get recent messages from memory."""
        limit = limit or self.max_messages

        # Try Redis first (faster)
        if self.redis_client:
            try:
                key = self._get_redis_key(user_id, chat_id)
                messages_json = await self.redis_client.lrange(key, 0, limit - 1)

                if messages_json:
                    messages = [json.loads(msg) for msg in reversed(messages_json)]
                    logger.info("messages_from_redis", count=len(messages))
                    return messages
            except Exception as e:
                logger.warning("redis_get_failed", error=str(e))

        # Fallback to PostgreSQL
        result = await self.session.execute(
            select(ChatHistory)
            .where(
                ChatHistory.user_id == user_id,
                ChatHistory.chat_id == chat_id,
            )
            .order_by(desc(ChatHistory.created_at))
            .limit(limit)
        )

        chat_msgs = result.scalars().all()
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
            }
            for msg in reversed(chat_msgs)
        ]

        logger.info("messages_from_db", count=len(messages))
        return messages

    async def clear_conversation(
        self,
        user_id: str,
        chat_id: str,
    ) -> None:
        """Clear conversation memory."""
        logger.info("clearing_conversation", user_id=user_id, chat_id=chat_id)

        # Clear Redis
        if self.redis_client:
            try:
                key = self._get_redis_key(user_id, chat_id)
                await self.redis_client.delete(key)
            except Exception as e:
                logger.warning("redis_clear_failed", error=str(e))

        logger.info("conversation_cleared")