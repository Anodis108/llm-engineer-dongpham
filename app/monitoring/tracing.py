"""Monitoring hooks — Buổi 7, Section 4 (LangSmith).

Tối thiểu: 1 trace cho mỗi lần gọi pipeline.answer*(), gắn question/answer/
latency/lỗi. KHÔNG bọc qua LangChain callback — dùng langsmith SDK trực tiếp
(Client.create_run/update_run) để giữ triết lý "native SDK" của repo.

Mặc định tắt (MONITORING_ENABLED=false) nên khi chưa điền LANGSMITH_API_KEY
trong .env, toàn bộ hàm ở đây là no-op — không ai bắt buộc phải cài/kích hoạt
LangSmith để chạy phần còn lại của codebase.

Khác LangFuse (bản trước của file này): LangSmith không có object "span" tiện
dụng với `.start_observation()`/`.update()`/`.end()` sẵn có — `Client.create_run`
chỉ ghi 1 run (cần tự sinh `id`), đóng run phải gọi `update_run(run_id=...)`
riêng. `_Run` bên dưới là wrapper MỎNG giả lập lại đúng 3 method đó, để
app/agent/graph.py và app/agent_m2/graph.py (gọi qua `t["_span"]`) KHÔNG PHẢI
sửa dòng nào khi đổi nền tảng tracing.

── Module III, Bài 7: hai thay đổi so với bản trước ────────────────────────────

1. TAIL SAMPLING (xem app/observability/sampling.py cho phần quyết định).
   `_Run` KHÔNG còn gọi `create_run` ngay lúc tạo. Nó giữ toàn bộ run trong RAM,
   và chỉ gửi đi ở `end()` — lúc đó mới biết request có lỗi/chậm/bị guardrail
   chặn hay không. Đây là điều kiện bắt buộc để "giữ 100% ca lỗi": head sampling
   quyết định lúc bắt đầu thì chưa biết request nào sẽ lỗi.

   Cây run được flush theo thứ tự CHA TRƯỚC CON, từ `_flush_tree()` — LangSmith
   cần run cha tồn tại trước khi con trỏ `parent_run_id` vào nó.

   Đánh đổi phải biết: trace nằm trong RAM tới khi request kết thúc, nên process
   bị kill giữa chừng là mất trace đó. Với chatbot (request ngắn) đổi lại là
   được, với job chạy hàng giờ thì không.

2. CHE PII trước khi gửi (app/observability/redaction.py). Trace đã lên
   LangSmith thì không rút lại được, nên che phải xảy ra ở đây.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import settings
from app.observability.redaction import redact_deep
from app.observability.sampling import should_sample

logger = logging.getLogger(__name__)


def _get_client():
    """Lazy import + lazy client — tránh phụ thuộc cứng vào package `langsmith`
    khi MONITORING_ENABLED=false, và tránh tạo client ở import-time."""
    from langsmith import Client

    return Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)


def _maybe_redact(value: Any) -> Any:
    return redact_deep(value) if settings.trace_redact_pii else value


class _Run:
    """Một run trong cây trace — buffer tại chỗ, flush ở `end()`.

    Giả lập API "span" của LangFuse (start_observation/update/end) để call site
    không cần biết nền tảng dưới. LangSmith yêu cầu tự sinh `run_id` (uuid4) và
    tự gọi `create_run`/`update_run` riêng biệt.
    """

    def __init__(
        self,
        client,
        name: str,
        input: Any,
        metadata: dict | None = None,
        parent: _Run | None = None,
    ):
        self._client = client
        self.run_id = uuid.uuid4()
        self._name = name
        self._input = input
        self._metadata = dict(metadata or {})
        self._parent = parent
        self._children: list[_Run] = []

        self._start = time.perf_counter()
        self._latency = 0.0
        self._output: Any = None
        self._error: str | None = None
        self._ended = False

        # Quyết định sampling chỉ có ở run GỐC, và chỉ ở `end()` (tail sampling).
        # Con không tự quyết: nó được ghi hay không là do cha.
        self._sampled: bool | None = None
        self._decision_reason = ""

        if parent is not None:
            parent._children.append(self)

    # ── API giống LangFuse, call site không phải đổi ─────────────────────────

    def start_observation(
        self, name: str, input: Any = None, metadata: dict | None = None
    ) -> _Run:
        """Tạo nested child run — dùng bởi trace_step() để lồng span trong 1 graph
        (giống app/agent/nodes.py: decompose/retrieve/grade/... lồng dưới 1 trace)."""
        return _Run(self._client, name, input, metadata, parent=self)

    def update(
        self,
        output: Any = None,
        metadata: dict | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        # Merge, không ghi đè — trace_answer/trace_step gọi update() 2 LẦN khi
        # có lỗi (1 lần set error trong except, 1 lần set output trong finally).
        # Ghi đè sẽ mất _error vì finally luôn gọi update(output=...) sau đó mà
        # không truyền lại level/status_message.
        if output is not None:
            self._output = output
        if metadata is not None:
            self._metadata.update(metadata)
        if level == "ERROR":
            self._error = status_message

    def end(
        self,
        *,
        guardrail: bool = False,
        cache_hit: bool = False,
        tags: dict | None = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        self._latency = time.perf_counter() - self._start
        if tags:
            self._metadata.update(tags)

        if self._parent is None:
            self._resolve_and_flush(guardrail=guardrail, cache_hit=cache_hit)

    # ── Tail sampling + flush ────────────────────────────────────────────────

    def _resolve_and_flush(self, *, guardrail: bool, cache_hit: bool) -> None:
        decision = should_sample(
            error=self._error is not None,
            guardrail=guardrail,
            latency_s=self._latency,
            cache_hit=cache_hit,
            slow_latency_s=settings.trace_slow_latency_s,
            rate=settings.trace_sample_rate,
        )
        self._sampled = decision.sample
        self._decision_reason = decision.reason

        if not decision.sample:
            logger.debug(
                "Bỏ trace %r (%.0fms, lý do: %s)",
                self._name,
                self._latency * 1000,
                decision.reason,
            )
            return

        self._flush_tree()

    def _flush_tree(self) -> None:
        """Ghi run này rồi tới các con — CHA TRƯỚC CON.

        LangSmith dựng cây theo `parent_run_id`, nên run cha phải được gửi trước;
        gửi con trước sẽ tạo ra cây mồ côi.
        """
        self._flush_one()
        for child in self._children:
            child._flush_tree()

    def _flush_one(self) -> None:
        metadata = {"latency_s": self._latency, **self._metadata}
        if self._decision_reason:
            metadata["sample_reason"] = self._decision_reason

        kwargs: dict[str, Any] = {
            "id": self.run_id,
            "name": self._name,
            "inputs": {"input": _maybe_redact(self._input)},
            "run_type": "chain",
            "metadata": _maybe_redact(metadata),
            "project_name": settings.langsmith_project,
        }
        if self._parent is not None:
            kwargs["parent_run_id"] = self._parent.run_id

        try:
            self._client.create_run(**kwargs)
            self._client.update_run(
                run_id=self.run_id,
                outputs={"output": _maybe_redact(self._output)}
                if self._output is not None
                else None,
                # Che cả error: exception có thể chứa nguyên văn câu hỏi của
                # người dùng, mà câu hỏi thì có thể có SĐT/CCCD.
                error=_maybe_redact(self._error),
                extra={"metadata": metadata},
            )
        except Exception as exc:  # noqa: BLE001
            # Monitoring là thứ PHẢI hỏng im lặng: LangSmith sập không được làm
            # request của người dùng thất bại. Đây là ranh giới quan trọng nhất
            # của cả file — trace là quan sát, không phải đường tới hạn.
            logger.warning("Không ghi được trace %r lên LangSmith: %s", self._name, exc)


@contextmanager
def trace_answer(
    name: str,
    question: str,
    metadata: dict[str, Any] | None = None,
):
    """Bọc quanh 1 lần gọi pipeline (answer/answer_stream/answer_structured).

    Dùng như:
        with trace_answer("answer", question) as t:
            result = ...
            t["output"] = result

    `t["_span"]` là run cha đang mở — truyền xuống cho trace_step() để tạo
    nested run (xem trace_step()). Chỉ có mặt khi monitoring bật; các call site
    dùng t.get("_span") nên tự an toàn khi tắt.

    Hai khoá đặc biệt mà call site có thể đặt để ảnh hưởng việc SAMPLING:
        t["guardrail"] = True   → luôn giữ trace (tín hiệu an toàn)
        t["cache_hit"]  = True  → bỏ qua trace (kết quả tất định, tái hiện được)
    """
    if not settings.monitoring_enabled:
        yield {}
        return

    client = _get_client()
    run = _Run(client, name, question, metadata)
    box: dict[str, Any] = {"_span": run}
    try:
        yield box
    except Exception as exc:
        run.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        run.update(output=box.get("output"))
        run.end(
            guardrail=bool(box.get("guardrail")),
            cache_hit=bool(box.get("cache_hit")),
            tags=box.get("tags"),
        )


@contextmanager
def trace_step(
    parent_span: Any,
    name: str,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
):
    """Nested child run dưới `parent_span` (lấy từ trace_answer's t["_span"]).

    Dùng trong LangGraph node để thấy từng bước (decompose/retrieve/grade/...)
    lồng nhau trong LangSmith — thay vì 1 run phẳng cho toàn bộ graph.invoke().
    No-op nếu parent_span là None (monitoring tắt, hoặc node chạy ngoài trace).

    Dùng như:
        with trace_step(parent_span, "grade_documents", input=question) as t:
            ...
            t["output"] = graded
    """
    if parent_span is None:
        yield {}
        return

    run = parent_span.start_observation(name=name, input=input, metadata=metadata or {})
    # `_span` để call site lồng được span SÂU HƠN nữa dưới bước này (vd
    # retrieve() mở 3 span con query_rewrite/search/rerank bên trong span
    # "retrieve" của pipeline, chứ không phải ngang hàng với nó).
    box: dict[str, Any] = {"_span": run}
    try:
        yield box
    except Exception as exc:
        run.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        run.update(output=box.get("output"))
        run.end()


def trace_stream(
    name: str,
    question: str,
    tokens: Iterator[str],
    metadata: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Bọc quanh answer_stream(): gom token rồi ghi 1 run khi stream kết thúc.

    Không thể tạo run "giữa chừng" cho streaming, nên cũng không thể có span con
    cho từng token — 1 run cho cả lần stream.

    Khác trace_answer ở chỗ đây là GENERATOR, nên client ngắt kết nối giữa chừng
    sẽ ném GeneratorExit vào điểm `yield`. `finally` vẫn chạy → trace được ghi
    với phần token đã phát. Đó là chủ ý: stream bị bỏ dở là tín hiệu đáng xem
    (người dùng sốt ruột bỏ đi), không phải rác cần bỏ.
    """
    if not settings.monitoring_enabled:
        yield from tokens
        return

    client = _get_client()
    run = _Run(client, name, question, metadata)
    chunks: list[str] = []
    try:
        for token in tokens:
            chunks.append(token)
            yield token
    except Exception as exc:
        run.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        run.update(output="".join(chunks))
        run.end()
