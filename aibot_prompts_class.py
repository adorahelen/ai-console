from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable, Tuple
from uuid import UUID
from datetime import datetime
import yaml
import struct
import time
import asyncio

class SSLException(IOError):
    pass


class Record:
    MAX_MAC_SIZE = 48
    MAX_DATA_SIZE = 16384
    MAX_PADDING = 256
    MAX_IV_LENGTH = 16
    MAX_FRAGMENT_SIZE = 18432
    ENABLE_CBC_PROTECTION = True
    OVERFLOW_OF_INT08 = 256
    OVERFLOW_OF_INT16 = 65536
    OVERFLOW_OF_INT24 = 16777216

    @staticmethod
    def verify_length(buffer: memoryview, length: int):
        if len(buffer) < length:
            raise SSLException("Insufficient space in the buffer, may be caused by an unexpected end of handshake data.")

    @staticmethod
    def get_int8(buffer: memoryview) -> int:
        Record.verify_length(buffer, 1)
        value = buffer[0]
        del buffer[:1]
        return value

    @staticmethod
    def get_int16(buffer: memoryview) -> int:
        Record.verify_length(buffer, 2)
        value = (buffer[0] << 8) | buffer[1]
        del buffer[:2]
        return value

    @staticmethod
    def get_int24(buffer: memoryview) -> int:
        Record.verify_length(buffer, 3)
        value = (buffer[0] << 16) | (buffer[1] << 8) | buffer[2]
        del buffer[:3]
        return value

    @staticmethod
    def get_int32(buffer: memoryview) -> int:
        Record.verify_length(buffer, 4)
        value = (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3]
        del buffer[:4]
        return value

    @staticmethod
    def get_bytes8(buffer: memoryview) -> bytes:
        length = Record.get_int8(buffer)
        Record.verify_length(buffer, length)
        value = bytes(buffer[:length])
        del buffer[:length]
        return value

    @staticmethod
    def get_bytes16(buffer: memoryview) -> bytes:
        length = Record.get_int16(buffer)
        Record.verify_length(buffer, length)
        value = bytes(buffer[:length])
        del buffer[:length]
        return value

    @staticmethod
    def get_bytes24(buffer: memoryview) -> bytes:
        length = Record.get_int24(buffer)
        Record.verify_length(buffer, length)
        value = bytes(buffer[:length])
        del buffer[:length]
        return value

    @staticmethod
    def put_int8(buffer: bytearray, value: int):
        buffer.append(value & 0xFF)

    @staticmethod
    def put_int16(buffer: bytearray, value: int):
        buffer.extend([(value >> 8) & 0xFF, value & 0xFF])

    @staticmethod
    def put_int24(buffer: bytearray, value: int):
        buffer.extend([(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])

    @staticmethod
    def put_int32(buffer: bytearray, value: int):
        buffer.extend([(value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])

    @staticmethod
    def put_bytes8(buffer: bytearray, data: Optional[bytes]):
        if data:
            Record.put_int8(buffer, len(data))
            buffer.extend(data)
        else:
            Record.put_int8(buffer, 0)

    @staticmethod
    def put_bytes16(buffer: bytearray, data: Optional[bytes]):
        if data:
            Record.put_int16(buffer, len(data))
            buffer.extend(data)
        else:
            Record.put_int16(buffer, 0)

    @staticmethod
    def put_bytes24(buffer: bytearray, data: Optional[bytes]):
        if data:
            Record.put_int24(buffer, len(data))
            buffer.extend(data)
        else:
            Record.put_int24(buffer, 0)

@dataclass
class LlmPrompt:
    id: Optional[int] = None
    guid: Optional[UUID] = None
    app_id: Optional[int] = None
    app_code: Optional[str] = None
    app_builtin: Optional[bool] = None
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = False
    type: Optional[str] = None
    prompt: Optional[str] = None
    prompt_map: Optional[Dict[str, Any]] = field(default_factory=dict)
    token_count: int = 0
    rev: int = 0
    created: Optional[datetime] = None
    updated: Optional[datetime] = None

    def get_prompt_map(self) -> Dict[str, Any]:
        if not self.prompt_map and self.prompt:
            try:
                self.prompt_map = yaml.safe_load(self.prompt)
            except yaml.YAMLError:
                self.prompt_map = {}
        return self.prompt_map

    def get_prompt(self) -> str:
        if not self.prompt and self.prompt_map:
            return yaml.dump(self.prompt_map, sort_keys=False)
        return self.prompt or ""

    def set_prompt(self, prompt_str: str):
        try:
            parsed = yaml.safe_load(prompt_str)
            if isinstance(parsed, dict):
                self.prompt_map = {k: parsed[k] for k in ['prompt', 'question', 'cot', 'answer', 'spec'] if k in parsed}
        except yaml.YAMLError:
            self.prompt_map = {}

    @staticmethod
    def parse(record: Record) -> 'LlmPrompt':
        return LlmPrompt(
            id=record.i("id"),
            guid=record.guid(),
            app_id=record.i("app_id"),
            app_code=record.str("app_code"),
            app_builtin=record.bool("app_built_in"),
            name=record.name(),
            description=record.description(),
            enabled=record.bool("enabled"),
            type=record.str("type"),
            token_count=record.i("token_count"),
            prompt=record.str("prompt"),
            rev=record.i("rev"),
            created=record.created(),
            updated=record.updated()
        )

    @staticmethod
    def parse_map(data: Dict[str, Any]) -> 'LlmPrompt':
        def parse_date(value: str) -> Optional[datetime]:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

        return LlmPrompt(
            guid=UUID(data["guid"]),
            name=data.get("name"),
            description=data.get("description"),
            enabled=data.get("enabled", False),
            type=data.get("type"),
            prompt=data.get("prompt"),
            token_count=data.get("token_count", 0),
            rev=data.get("rev", 0),
            created=parse_date(data.get("created")) if data.get("created") else None,
            updated=parse_date(data.get("updated")) if data.get("updated") else None
        )

    @staticmethod
    def parse_from_yaml(yaml_str: str) -> 'LlmPrompt':
        data = yaml.safe_load(yaml_str)
        prompt_map = {k: data[k] for k in ['prompt', 'question', 'cot', 'answer', 'spec'] if k in data}
        return LlmPrompt(
            guid=UUID(data["guid"]) if "guid" in data else None,
            name=data.get("name"),
            description=data.get("description"),
            enabled=data.get("enabled", False),
            type=data.get("type"),
            prompt_map=prompt_map
        )

    def marshal(self) -> Dict[str, Any]:
        return {
            "guid": str(self.guid),
            "app_code": self.app_code,
            "app_built_in": self.app_builtin,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "type": self.type.lower() if self.type else None,
            "prompt": self.get_prompt(),
            "token_count": self.token_count,
            "rev": self.rev,
            "created": self.created,
            "updated": self.updated
        }

    def __str__(self):
        return self.name or ""
    

@dataclass
class OpenAiPrompt:
    id: Optional[int] = None
    guid: Optional[UUID] = None
    subscription_id: Optional[int] = None
    subscription_guid: Optional[UUID] = None
    subscription_name: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    type: Optional[str] = None
    prompt: Optional[str] = None
    token_count: Optional[int] = None
    vector: Optional[List[float]] = None
    rev: int = 1
    source: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    tiny: bool = False

    def get_embedding(self) -> Optional[str]:
        if self.type == "system":
            return None

        try:
            data = yaml.safe_load(self.prompt or "")
            if not isinstance(data, dict):
                return None

            parts = []
            if "question" in data:
                parts.append("Question:\n" + str(data["question"]) + "\n")
            if "cot" in data:
                parts.append("Chain of Thought:\n" + str(data["cot"]) + "\n")
            if "answer" in data:
                parts.append("Answer:\n" + str(data["answer"]))

            return "\n".join(parts).strip()
        except yaml.YAMLError:
            return None

    def marshal(self) -> Dict[str, Any]:
        if self.tiny:
            return {
                "guid": str(self.guid),
                "enabled": self.enabled,
                "rev": self.rev
            }
        else:
            return {
                "guid": str(self.guid),
                "subscription_name": self.subscription_name,
                "name": self.name,
                "description": self.description,
                "enabled": self.enabled,
                "type": self.type,
                "prompt": self.prompt,
                "token_count": self.token_count,
                "rev": self.rev,
                "created": self.created,
                "updated": self.updated
            }

    @staticmethod
    def parse(record: 'Record') -> 'OpenAiPrompt':
        prompt = OpenAiPrompt()
        prompt.id = record.i("id")
        prompt.guid = record.guid()
        prompt.subscription_id = record.i("subscription_id")
        prompt.subscription_guid = record.guid("subscription_guid")
        prompt.subscription_name = record.str("subscription_name")
        prompt.name = record.name()
        prompt.description = record.description()
        prompt.enabled = record.bool("enabled")
        prompt.type = record.str("type")
        prompt.prompt = record.str("prompt")
        prompt.token_count = record.i("token_count")

        vec_bytes = record.get("vector")
        if vec_bytes:
            prompt.vector = list(struct.unpack(f'{len(vec_bytes) // 8}d', vec_bytes))

        prompt.rev = record.i("rev")
        prompt.source = record.str("source")
        prompt.created = record.created()
        prompt.updated = record.updated()
        return prompt

    @staticmethod
    def convert(llm_prompt: 'LlmPrompt') -> 'OpenAiPrompt':
        return OpenAiPrompt(
            guid=llm_prompt.get_guid(),
            name=llm_prompt.get_name(),
            description=llm_prompt.get_description(),
            enabled=llm_prompt.is_enabled(),
            type=llm_prompt.get_type(),
            prompt=llm_prompt.get_prompt()
        )

    def __str__(self):
        return self.name or ""


@dataclass
class StreamBufferState:
    guid: str
    type: str = ""
    query: str = ""
    answer: str = ""
    len_js: int = 0
    done: bool = False
    error: Optional[str] = None
    updated_at: float = field(default_factory=lambda: time.time())

class StreamBuffer:

    _cache: Dict[str, Dict[str, dict]] = {}

    def __init__(
        self,
        guid: str,
        qtype: str,
        query: str,
        js_len_fn: Callable[[str], int],
        js_slice_fn: Callable[[str, int, int], Tuple[str, int, int]],
    ):
        self.state = StreamBufferState(guid=guid, type=qtype, query=query)
        self._lock = asyncio.Lock()
        self._js_len = js_len_fn
        self._js_slice = js_slice_fn
        self._gate_marker: Optional[str] = None
        self._gate_js_offset: int = 0
        self._gate_reached: bool = False

    def enable_gate(self, marker: str) -> None:
        self._gate_marker = marker
        self._gate_js_offset = 0
        self._gate_reached = False

    def disable_gate(self) -> None:
        self._gate_marker = None
        self._gate_js_offset = 0
        self._gate_reached = False

    def _scan_gate_if_needed(self) -> None:
        if self._gate_reached or not self._gate_marker:
            return
        pos = self.state.answer.find(self._gate_marker)
        if pos == -1:
            return
        end_pos = pos + len(self._gate_marker)
        js_off = self._js_len(self.state.answer[:end_pos])
        self._gate_js_offset = js_off
        self._gate_reached = True

    async def append(self, chunk: str) -> None:
        if not chunk:
            return
        async with self._lock:
            self.state.answer += chunk
            self.state.len_js += self._js_len(chunk)
            self._scan_gate_if_needed()
            self.state.updated_at = time.time()

    async def finish(self) -> None:
        async with self._lock:
            self.state.done = True
            self.state.updated_at = time.time()

    async def fail(self, err: str) -> None:
        async with self._lock:
            self.state.done = True
            self.state.error = err
            self.state.updated_at = time.time()

    async def get_chunk(self, offset_js: int, count_js: int):
        async with self._lock:
            if self._gate_marker and not self._gate_reached:
                return {"seq": offset_js, "value": "", "is_finished": False}

            min_offset = self._gate_js_offset if (self._gate_marker and self._gate_reached) else 0
            eff_offset = max(offset_js, min_offset)

            total_js = self.state.len_js
            chunk, next_offset_js, computed_total_js = self._js_slice(
                self.state.answer, eff_offset, count_js
            )
            if computed_total_js != total_js:
                total_js = computed_total_js
                self.state.len_js = computed_total_js

            is_finished = bool(self.state.done) and (next_offset_js >= total_js)
            return {"seq": next_offset_js, "value": chunk, "is_finished": is_finished}

    async def to_cache(self, model_key: str) -> None:
        async with self._lock:
            cache_data = StreamBuffer.get_cache(self.state.guid, model_key) or {
                "answer": "",
                "query": self.state.query,
                "type": self.state.type,
                "delete": False,
            }
            cache_data["answer"] = self.state.answer
            cache_data["query"] = cache_data.get("query") or self.state.query
            cache_data["type"] = cache_data.get("type") or self.state.type

            StreamBuffer.set_cache(self.state.guid, model_key, cache_data)

    def meta(self) -> dict:
        s = self.state
        return {
            "guid": s.guid,
            "type": s.type,
            "query": s.query,
            "len_js": s.len_js,
            "done": s.done,
            "error": s.error,
            "updated_at": s.updated_at,
        }

    @staticmethod
    def get_cache(guid: str, model_key: str) -> Optional[dict]:
        return StreamBuffer._cache.get(guid, {}).get(model_key)

    @staticmethod
    def set_cache(guid: str, model_key: str, cache_data: dict) -> None:
        if guid not in StreamBuffer._cache:
            StreamBuffer._cache[guid] = {}
        StreamBuffer._cache[guid][model_key] = cache_data

    @staticmethod
    def delete_cache(guid: str, model_key: Optional[str] = None) -> bool:
        if guid not in StreamBuffer._cache:
            return False

        if model_key is None:
            had = bool(StreamBuffer._cache[guid])
            StreamBuffer._cache.pop(guid, None)
            return had

        if model_key in StreamBuffer._cache[guid]:
            StreamBuffer._cache[guid].pop(model_key)
            if not StreamBuffer._cache[guid]:
                StreamBuffer._cache.pop(guid, None)
            return True

        return False

class StreamBufferRegistry:
    def __init__(
        self,
        js_len_fn: Callable[[str], int],
        js_slice_fn: Callable[[str, int, int], Tuple[str, int, int]],
    ):
        self._items: Dict[str, StreamBuffer] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._js_len = js_len_fn
        self._js_slice = js_slice_fn

    async def create(self, guid: str, qtype: str, query: str) -> StreamBuffer:
        async with self._global_lock:
            if guid in self._items:
                return self._items[guid]
            buf = StreamBuffer(guid, qtype, query, self._js_len, self._js_slice)
            self._items[guid] = buf
            self._locks[guid] = asyncio.Lock()
            return buf

    def get(self, guid: str) -> Optional[StreamBuffer]:
        return self._items.get(guid)

    async def remove(self, guid: str) -> None:
        async with self._global_lock:
            self._items.pop(guid, None)
            self._locks.pop(guid, None)

    async def cleanup(self, ttl_sec: int = 600) -> int:
        now = time.time()
        to_delete = []
        for guid, buf in self._items.items():
            s = buf.state
            if s.done and (now - s.updated_at >= 5):
                to_delete.append(guid)
            elif (now - s.updated_at) > ttl_sec:
                to_delete.append(guid)

        if not to_delete:
            return 0

        async with self._global_lock:
            for guid in to_delete:
                self._items.pop(guid, None)
                self._locks.pop(guid, None)
        return len(to_delete)