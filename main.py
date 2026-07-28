"""AstrBot plugin that exposes a safe visual editor for LLM context history."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.api.web import error_response, json_response, request
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .editor_core import (
    EditorValidationError,
    apply_submitted_edits,
    editor_messages,
    history_revision,
    strip_thinking_parts,
    validate_history_container,
)

PLUGIN_NAME = "astrbot_plugin_context_editor"
MAX_PAGE_SIZE = 100
MAX_SNAPSHOTS = 30
MAX_SEARCH_LENGTH = 200
SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


@register(
    PLUGIN_NAME,
    "local",
    "安全查看、修改和删减 AstrBot 的 LLM 对话上下文",
    "1.1.0",
)
class ContextEditorPlugin(Star):
    """Plugin Pages backend for the visual context editor."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._conversation_locks: dict[str, asyncio.Lock] = {}

        context.register_web_api(
            f"/{PLUGIN_NAME}/conversations",
            self.list_conversations,
            ["GET"],
            "List conversations for the visual context editor",
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/conversation",
            self.get_conversation,
            ["GET"],
            "Load one conversation and its editable context",
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/conversation/save",
            self.save_conversation,
            ["POST"],
            "Safely save edited conversation context",
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/snapshots",
            self.list_snapshots,
            ["GET"],
            "List automatic context snapshots",
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/snapshots/restore",
            self.restore_snapshot,
            ["POST"],
            "Restore an automatic context snapshot",
        )

    @property
    def conversation_manager(self):
        return self.context.conversation_manager

    def _ensure_dashboard_user(self) -> None:
        if not getattr(request, "username", None):
            raise PermissionError("需要登录 AstrBot WebUI 后才能编辑上下文。")

    def _auto_snapshots_enabled(self) -> bool:
        return bool(self.config.get("enable_auto_snapshots", False))

    def _filter_thinking_enabled(self) -> bool:
        return bool(self.config.get("filter_thinking_before_llm", False))

    @filter.on_llm_request()
    async def filter_thinking_before_llm(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Remove think parts from the request copy without altering saved history."""

        del event
        if self._filter_thinking_enabled():
            req.contexts = strip_thinking_parts(req.contexts)

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        lock = self._conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_id] = lock
        return lock

    @staticmethod
    def _validate_identity(conversation_id: Any, umo: Any) -> tuple[str, str]:
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise EditorValidationError("缺少 conversation_id。")
        if not isinstance(umo, str) or not umo.strip():
            raise EditorValidationError("缺少 umo。")
        if len(conversation_id) > 160 or len(umo) > 1024:
            raise EditorValidationError("对话标识过长。")
        return conversation_id, umo

    @staticmethod
    def _history_from_conversation(conversation: Any) -> list[Any]:
        raw_history = getattr(conversation, "history", "") or "[]"
        try:
            history = json.loads(raw_history)
        except (TypeError, json.JSONDecodeError) as exc:
            raise EditorValidationError(
                "数据库中的 history 不是有效 JSON，无法安全编辑。"
            ) from exc
        return validate_history_container(history)

    async def _load_owned_conversation(
        self,
        conversation_id: str,
        umo: str,
    ) -> tuple[Any, list[Any]]:
        conversation = await self.conversation_manager.get_conversation(
            umo,
            conversation_id,
        )
        if conversation is None:
            raise LookupError("没有找到该对话。")
        if getattr(conversation, "user_id", None) != umo:
            raise PermissionError("对话标识与消息来源不匹配。")
        return conversation, self._history_from_conversation(conversation)

    async def _detail_payload(
        self,
        conversation_id: str,
        umo: str,
    ) -> dict[str, Any]:
        conversation, history = await self._load_owned_conversation(
            conversation_id,
            umo,
        )
        messages = editor_messages(history)
        return {
            "conversation": {
                "conversationId": conversation.cid,
                "umo": conversation.user_id,
                "platformId": conversation.platform_id,
                "title": conversation.title or "",
                "personaId": conversation.persona_id or "",
                "createdAt": conversation.created_at or 0,
                "updatedAt": conversation.updated_at or 0,
                "tokenUsage": getattr(conversation, "token_usage", 0) or 0,
            },
            "revision": history_revision(history),
            "messageCount": len(history),
            "invalidCount": sum(1 for item in messages if item["issue"]),
            "messages": messages,
            "autoSnapshotsEnabled": self._auto_snapshots_enabled(),
            "filterThinkingForLlm": self._filter_thinking_enabled(),
        }

    async def list_conversations(self):
        try:
            self._ensure_dashboard_user()
            page = request.query.get("page", 1, type=int) or 1
            page_size = request.query.get("page_size", 30, type=int) or 30
            search = (request.query.get("search", "") or "").strip()
            if page < 1:
                page = 1
            page_size = max(1, min(page_size, MAX_PAGE_SIZE))
            if len(search) > MAX_SEARCH_LENGTH:
                raise EditorValidationError("搜索内容过长。")

            conversations, total = (
                await self.conversation_manager.get_filtered_conversations(
                    page=page,
                    page_size=page_size,
                    search_query=search,
                )
            )
            items = [
                {
                    "conversationId": conv.cid,
                    "umo": conv.user_id,
                    "platformId": conv.platform_id,
                    "title": conv.title or "",
                    "personaId": conv.persona_id or "",
                    "createdAt": conv.created_at or 0,
                    "updatedAt": conv.updated_at or 0,
                    "tokenUsage": getattr(conv, "token_usage", 0) or 0,
                }
                for conv in conversations
            ]
            return json_response(
                {
                    "items": items,
                    "page": page,
                    "pageSize": page_size,
                    "total": total,
                }
            )
        except EditorValidationError as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("可视化上下文：读取对话列表失败")
            return error_response("读取对话列表失败，请查看 AstrBot 日志。", 500)

    async def get_conversation(self):
        try:
            self._ensure_dashboard_user()
            conversation_id, umo = self._validate_identity(
                request.query.get("conversation_id"),
                request.query.get("umo"),
            )
            return json_response(
                await self._detail_payload(conversation_id, umo)
            )
        except EditorValidationError as exc:
            return error_response(str(exc), status_code=400)
        except LookupError as exc:
            return error_response(str(exc), status_code=404)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except Exception:
            logger.exception("可视化上下文：读取对话失败")
            return error_response("读取对话失败，请查看 AstrBot 日志。", 500)

    async def save_conversation(self):
        try:
            self._ensure_dashboard_user()
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise EditorValidationError("请求正文必须是 JSON 对象。")
            conversation_id, umo = self._validate_identity(
                payload.get("conversationId"),
                payload.get("umo"),
            )
            base_revision = payload.get("baseRevision")
            if not isinstance(base_revision, str) or len(base_revision) != 64:
                raise EditorValidationError("缺少有效的 baseRevision。")

            async with self._lock_for(conversation_id):
                _, original_history = await self._load_owned_conversation(
                    conversation_id,
                    umo,
                )
                current_revision = history_revision(original_history)
                if current_revision != base_revision:
                    return error_response(
                        "保存失败：对话已产生新消息或被其他页面修改，请刷新后重试。",
                        status_code=409,
                    )

                edited_history = apply_submitted_edits(
                    original_history,
                    payload.get("messages"),
                )
                if edited_history == original_history:
                    return json_response(
                        {
                            **await self._detail_payload(conversation_id, umo),
                            "saved": False,
                            "snapshotId": None,
                        }
                    )

                snapshot_id = None
                if self._auto_snapshots_enabled():
                    snapshot_id = await self._save_snapshot(
                        conversation_id,
                        umo,
                        original_history,
                        reason="before-edit",
                    )
                # Recheck immediately before the write. This catches a message
                # appended while an optional disk snapshot was being created.
                _, latest_history = await self._load_owned_conversation(
                    conversation_id,
                    umo,
                )
                if history_revision(latest_history) != current_revision:
                    return error_response(
                        "保存已取消：创建快照期间对话发生了变化，请刷新后重试。",
                        status_code=409,
                    )

                await self.conversation_manager.update_conversation(
                    unified_msg_origin=umo,
                    conversation_id=conversation_id,
                    history=edited_history,
                )
                return json_response(
                    {
                        **await self._detail_payload(conversation_id, umo),
                        "saved": True,
                        "snapshotId": snapshot_id,
                    }
                )
        except EditorValidationError as exc:
            return error_response(str(exc), status_code=400)
        except LookupError as exc:
            return error_response(str(exc), status_code=404)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except Exception:
            logger.exception("可视化上下文：保存对话失败")
            return error_response("保存失败，请查看 AstrBot 日志。", 500)

    def _snapshot_dir(self, conversation_id: str) -> Path:
        import hashlib

        safe_dir = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        return (
            Path(get_astrbot_plugin_data_path())
            / PLUGIN_NAME
            / "snapshots"
            / safe_dir
        )

    async def _save_snapshot(
        self,
        conversation_id: str,
        umo: str,
        history: list[Any],
        reason: str,
    ) -> str:
        snapshot_id = (
            f"{int(time.time() * 1000)}_{secrets.token_hex(4)}"
        )
        record = {
            "schemaVersion": 1,
            "snapshotId": snapshot_id,
            "createdAt": int(time.time()),
            "reason": reason,
            "conversationId": conversation_id,
            "umo": umo,
            "revision": history_revision(history),
            "messageCount": len(history),
            "history": history,
        }
        await asyncio.to_thread(
            self._write_snapshot_sync,
            self._snapshot_dir(conversation_id),
            snapshot_id,
            record,
        )
        return snapshot_id

    @staticmethod
    def _write_snapshot_sync(
        directory: Path,
        snapshot_id: str,
        record: dict[str, Any],
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{snapshot_id}.json"
        temporary = directory / f".{snapshot_id}.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

        snapshots = sorted(
            directory.glob("*.json"),
            key=lambda path: path.name,
            reverse=True,
        )
        for old_path in snapshots[MAX_SNAPSHOTS:]:
            try:
                old_path.unlink()
            except OSError:
                logger.warning("无法清理旧上下文快照：%s", old_path.name)

    async def list_snapshots(self):
        try:
            self._ensure_dashboard_user()
            conversation_id, umo = self._validate_identity(
                request.query.get("conversation_id"),
                request.query.get("umo"),
            )
            await self._load_owned_conversation(conversation_id, umo)
            items = await asyncio.to_thread(
                self._read_snapshot_summaries_sync,
                self._snapshot_dir(conversation_id),
                conversation_id,
                umo,
            )
            return json_response({"items": items})
        except EditorValidationError as exc:
            return error_response(str(exc), status_code=400)
        except LookupError as exc:
            return error_response(str(exc), status_code=404)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except Exception:
            logger.exception("可视化上下文：读取快照失败")
            return error_response("读取快照失败，请查看 AstrBot 日志。", 500)

    @staticmethod
    def _read_snapshot_summaries_sync(
        directory: Path,
        conversation_id: str,
        umo: str,
    ) -> list[dict[str, Any]]:
        if not directory.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(
            directory.glob("*.json"),
            key=lambda item: item.name,
            reverse=True,
        )[:MAX_SNAPSHOTS]:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    record = json.load(stream)
                if (
                    record.get("conversationId") != conversation_id
                    or record.get("umo") != umo
                ):
                    continue
                items.append(
                    {
                        "snapshotId": record.get("snapshotId", path.stem),
                        "createdAt": record.get("createdAt", 0),
                        "reason": record.get("reason", "unknown"),
                        "revision": record.get("revision", ""),
                        "messageCount": record.get("messageCount", 0),
                    }
                )
            except (OSError, json.JSONDecodeError, AttributeError):
                logger.warning("忽略无法读取的上下文快照：%s", path.name)
        return items

    async def restore_snapshot(self):
        try:
            self._ensure_dashboard_user()
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise EditorValidationError("请求正文必须是 JSON 对象。")
            conversation_id, umo = self._validate_identity(
                payload.get("conversationId"),
                payload.get("umo"),
            )
            base_revision = payload.get("baseRevision")
            snapshot_id = payload.get("snapshotId")
            if not isinstance(base_revision, str) or len(base_revision) != 64:
                raise EditorValidationError("缺少有效的 baseRevision。")
            if not isinstance(snapshot_id, str) or not SNAPSHOT_ID_RE.fullmatch(
                snapshot_id
            ):
                raise EditorValidationError("snapshotId 无效。")

            async with self._lock_for(conversation_id):
                _, current_history = await self._load_owned_conversation(
                    conversation_id,
                    umo,
                )
                if history_revision(current_history) != base_revision:
                    return error_response(
                        "恢复失败：当前对话已经变化，请刷新后重试。",
                        status_code=409,
                    )

                snapshot = await asyncio.to_thread(
                    self._read_snapshot_sync,
                    self._snapshot_dir(conversation_id),
                    snapshot_id,
                    conversation_id,
                    umo,
                )
                snapshot_history = validate_history_container(
                    snapshot.get("history")
                )
                before_restore_id = None
                if self._auto_snapshots_enabled():
                    before_restore_id = await self._save_snapshot(
                        conversation_id,
                        umo,
                        current_history,
                        reason="before-restore",
                    )
                _, latest_history = await self._load_owned_conversation(
                    conversation_id,
                    umo,
                )
                if history_revision(latest_history) != base_revision:
                    return error_response(
                        "恢复已取消：创建快照期间对话发生了变化，请刷新后重试。",
                        status_code=409,
                    )

                await self.conversation_manager.update_conversation(
                    unified_msg_origin=umo,
                    conversation_id=conversation_id,
                    history=snapshot_history,
                )
                return json_response(
                    {
                        **await self._detail_payload(conversation_id, umo),
                        "restored": True,
                        "snapshotId": snapshot_id,
                        "beforeRestoreSnapshotId": before_restore_id,
                    }
                )
        except EditorValidationError as exc:
            return error_response(str(exc), status_code=400)
        except FileNotFoundError:
            return error_response("快照不存在或已被清理。", status_code=404)
        except LookupError as exc:
            return error_response(str(exc), status_code=404)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except Exception:
            logger.exception("可视化上下文：恢复快照失败")
            return error_response("恢复快照失败，请查看 AstrBot 日志。", 500)

    @staticmethod
    def _read_snapshot_sync(
        directory: Path,
        snapshot_id: str,
        conversation_id: str,
        umo: str,
    ) -> dict[str, Any]:
        target = directory / f"{snapshot_id}.json"
        if not target.is_file():
            raise FileNotFoundError(snapshot_id)
        with target.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        if (
            not isinstance(record, dict)
            or record.get("conversationId") != conversation_id
            or record.get("umo") != umo
        ):
            raise EditorValidationError("快照归属校验失败。")
        return record

    async def terminate(self):
        self._conversation_locks.clear()





