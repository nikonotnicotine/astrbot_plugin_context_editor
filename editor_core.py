"""Pure validation and optimistic-concurrency helpers for the context editor."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from astrbot.core.agent.message import Message

MAX_HISTORY_BYTES = 16 * 1024 * 1024
MAX_MESSAGES = 5000
MAX_ROLE_LENGTH = 64
NEW_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})
THINKING_PART_TYPES = frozenset({"think", "thinking"})



class EditorValidationError(ValueError):
    """Raised when a context editor payload is unsafe or inconsistent."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashes and size checks."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise EditorValidationError(f"数据不是有效的 JSON：{exc}") from exc


def message_hash(message: Any) -> str:
    """Return a stable hash for one source message."""

    return hashlib.sha256(canonical_json(message).encode("utf-8")).hexdigest()


def history_revision(history: list[Any]) -> str:
    """Return a stable optimistic-concurrency revision for a history list."""

    return hashlib.sha256(canonical_json(history).encode("utf-8")).hexdigest()


def validate_history_container(history: Any) -> list[Any]:
    """Validate the outer history container and its serialized size."""

    if not isinstance(history, list):
        raise EditorValidationError("对话历史必须是 JSON 数组。")
    if len(history) > MAX_MESSAGES:
        raise EditorValidationError(f"消息数量不能超过 {MAX_MESSAGES} 条。")
    encoded_size = len(canonical_json(history).encode("utf-8"))
    if encoded_size > MAX_HISTORY_BYTES:
        limit_mb = MAX_HISTORY_BYTES // (1024 * 1024)
        raise EditorValidationError(f"对话历史不能超过 {limit_mb} MiB。")
    return history


def _model_issue(message: dict[str, Any]) -> str | None:
    """Validate one message against the AstrBot runtime message model."""

    try:
        Message.model_validate(message)
    except KeyError as exc:
        return f"CONTENT 中包含未知的消息段类型：{exc.args[0]}。"
    except Exception as exc:
        detail = str(exc).splitlines()[0].strip()
        return f"消息结构不符合 AstrBot 格式：{detail[:180]}"
    return None


def is_thinking_part(part: Any) -> bool:
    """Return whether a content part represents a thinking trace."""

    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    if isinstance(part_type, str) and part_type.lower() in THINKING_PART_TYPES:
        return True
    return part_type is None and any(key in part for key in THINKING_PART_TYPES)


def strip_thinking_parts(history: Any) -> list[Any]:
    """Return a request-only copy with thinking data removed.

    The persisted conversation is never changed by this helper. Only the
    ProviderRequest copy used for an LLM call should be passed here.
    """

    if not isinstance(history, list):
        return history
    sanitized: list[Any] = []
    for message in history:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue

        copied = dict(message)
        changed = False
        for key in THINKING_PART_TYPES:
            if key in copied:
                copied.pop(key)
                changed = True

        content = message.get("content")
        if isinstance(content, list):
            filtered_parts = [
                part for part in content if not is_thinking_part(part)
            ]
            if len(filtered_parts) != len(content):
                copied["content"] = filtered_parts
                changed = True

        sanitized.append(copied if changed else message)
    return sanitized

def message_issue(message: Any) -> str | None:
    """Return a repair hint when a stored message cannot be safely retained."""

    if not isinstance(message, dict):
        return "此记录不是对象，只能删除。"
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        return "ROLE 缺失或无效，只能删除。"
    if len(role) > MAX_ROLE_LENGTH:
        return f"ROLE 长度超过 {MAX_ROLE_LENGTH}，只能删除。"
    return _model_issue(message)


def is_checkpoint_message(message: Any) -> bool:
    """Return whether a message is AstrBot's internal checkpoint marker."""

    return isinstance(message, dict) and message.get("role") == "_checkpoint"


def validate_new_message(message: Any) -> dict[str, Any]:
    """Validate a newly appended message and return it unchanged."""

    if not isinstance(message, dict):
        raise EditorValidationError("新消息必须是 JSON 对象。")
    role = message.get("role")
    if role not in NEW_MESSAGE_ROLES:
        raise EditorValidationError(
            "新消息的 ROLE 只能是 system、user 或 assistant。"
        )
    issue = message_issue(message)
    if issue:
        raise EditorValidationError(f"新消息无效：{issue}")
    return message


def validate_editable_history(history: Any) -> list[dict[str, Any]]:
    """Validate the shape AstrBot expects after an edit."""

    validate_history_container(history)
    for index, message in enumerate(history):
        issue = message_issue(message)
        if issue:
            raise EditorValidationError(f"第 {index + 1} 条记录无效：{issue}")
    return history


def editor_messages(history: list[Any]) -> list[dict[str, Any]]:
    """Wrap stored messages with immutable source identity for the browser."""

    validate_history_container(history)
    return [
        {
            "sourceIndex": index,
            "sourceHash": message_hash(message),
            "message": message,
            "issue": message_issue(message),
        }
        for index, message in enumerate(history)
    ]


def apply_submitted_edits(
    original_history: list[Any],
    submitted_items: Any,
) -> list[dict[str, Any]]:
    """Apply edit/delete/append rows without permitting role or order mutation.

    The browser submits retained rows. Missing source indices are deletions.
    New rows use sourceIndex null and are only accepted after all source rows,
    so the history order remains stable. Every retained source row carries a
    source hash, preventing stale or fabricated indices from being accepted.
    """

    validate_history_container(original_history)
    if not isinstance(submitted_items, list):
        raise EditorValidationError("messages 必须是数组。")

    result: list[Any] = []
    previous_index = -1
    seen_new_message = False

    for position, item in enumerate(submitted_items):
        if not isinstance(item, dict):
            raise EditorValidationError(f"第 {position + 1} 个编辑项无效。")

        source_index = item.get("sourceIndex")
        source_hash = item.get("sourceHash")
        edited_message = item.get("message")

        if source_index is None:
            seen_new_message = True
            result.append(validate_new_message(edited_message))
            continue

        if seen_new_message:
            raise EditorValidationError("新消息只能追加在已有历史之后。")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise EditorValidationError("sourceIndex 必须是整数或 null。")
        if source_index <= previous_index:
            raise EditorValidationError("消息顺序不能改变，也不能重复。")
        if source_index < 0 or source_index >= len(original_history):
            raise EditorValidationError("sourceIndex 超出当前对话范围。")

        original_message = original_history[source_index]
        if not isinstance(source_hash, str) or source_hash != message_hash(
            original_message
        ):
            raise EditorValidationError("消息来源校验失败，请刷新后重试。")

        original_issue = message_issue(original_message)
        if original_issue:
            if edited_message != original_message:
                raise EditorValidationError(
                    f"第 {source_index + 1} 条损坏记录不能修改，只能删除。"
                )
        else:
            if not isinstance(edited_message, dict):
                raise EditorValidationError("消息必须保持为 JSON 对象。")
            if is_checkpoint_message(original_message) and edited_message != original_message:
                raise EditorValidationError("内部 checkpoint 记录只能删除，不能修改。")
            if edited_message.get("role") != original_message.get("role"):
                raise EditorValidationError("ROLE 是只读字段，不能修改。")
            edited_issue = message_issue(edited_message)
            if edited_issue:
                raise EditorValidationError(
                    f"第 {source_index + 1} 条消息修改后无效：{edited_issue}"
                )

        result.append(edited_message)
        previous_index = source_index

    return validate_editable_history(result)
