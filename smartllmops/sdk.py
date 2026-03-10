import uuid
import time
import functools
import threading

# Thread-local storage to track spans within a single request/execution thread
_context = threading.local()


class SDKTracer:

    def __init__(self, telemetry, environment="prod", model="unknown", provider="unknown"):
        self.telemetry = telemetry
        self.environment = environment
        self.model = model
        self.provider = provider

    # ---------------------------------------------------------
    # TRACE INITIALIZATION
    # ---------------------------------------------------------

    def start_trace(self):
        """Initializes a new span list and stack for the current thread."""
        _context.spans = []
        _context.active_span_stack = []
        _context.trace_id = f"trace-{uuid.uuid4().hex[:8]}"

    # ---------------------------------------------------------
    # SAFE SERIALIZATION
    # ---------------------------------------------------------

    def _safe_serialize(self, obj, max_length=150):

        def _serialize(o, depth=0):
            try:
                if depth > 2:
                    return "..."

                if o is None:
                    return "None"

                if isinstance(o, (int, float, bool)):
                    return str(o)

                if hasattr(o, "page_content"):
                    return f"Document(len={len(o.page_content)})"

                if isinstance(o, (list, tuple)):
                    items = [_serialize(i, depth + 1) for i in o[:3]]
                    if len(o) > 3:
                        items.append("...")
                    container = "[]" if isinstance(o, list) else "()"
                    return f"{container[0]}{', '.join(items)}{container[1]}"

                if isinstance(o, dict):
                    keys = list(o.keys())
                    items = [f"{k}: {_serialize(o[k], depth + 1)}" for k in keys[:2]]
                    if len(keys) > 2:
                        items.append("...")
                    return f"{{{', '.join(items)}}}"

                return str(o).replace("\n", " ")

            except Exception:
                return "[Error]"

        s = _serialize(obj)

        if len(s) > max_length:
            return s[:max_length] + "... [TRUNCATED]"

        return s

    # ---------------------------------------------------------
    # SPAN DECORATOR
    # ---------------------------------------------------------

    def trace(
        self,
        name=None,
        span_type=None,
        metadata=None,
        usage=None,
        include_io=True,
        result_parser=None,
        parent_span_id=None
    ):
        """Decorator to capture function execution as a span."""

        def decorator(func):

            @functools.wraps(func)
            def wrapper(*args, **kwargs):

                span_name = name or func.__name__
                span_id = f"span-{uuid.uuid4().hex[:8]}"
                start_time = int(time.time() * 1000)

                # Initialize stack if missing (e.g. decorators on classes)
                if not hasattr(_context, "active_span_stack"):
                    _context.active_span_stack = []

                # Determine parent
                effective_parent = parent_span_id
                if effective_parent is None and _context.active_span_stack:
                    effective_parent = _context.active_span_stack[-1]

                # Push to stack
                _context.active_span_stack.append(span_id)

                try:
                    result = func(*args, **kwargs)
                    status = "success"
                    output = result

                except Exception as e:
                    status = "error"
                    output = str(e)
                    raise e

                finally:
                    # Pop from stack
                    if _context.active_span_stack:
                        _context.active_span_stack.pop()

                    end_time = int(time.time() * 1000)

                    if not hasattr(_context, "spans"):
                        _context.spans = []

                    trace_id = getattr(_context, "trace_id", None)

                    final_metadata = (metadata or {}).copy()
                    final_usage = (usage or {}).copy()

                    if result_parser and status == "success":
                        try:
                            parsed = result_parser(output, args, kwargs)

                            if isinstance(parsed, dict):
                                final_metadata.update(parsed.get("metadata", {}))
                                final_usage.update(parsed.get("usage", {}))

                        except Exception as e:
                            final_metadata["_parser_error"] = str(e)

                    if include_io and not final_metadata:
                        final_metadata = {
                            "input": self._safe_serialize(args),
                            "output": self._safe_serialize(output),
                        }

                    span = {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": effective_parent,
                        "type": span_type or "generic",
                        "name": span_name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "latency_ms": end_time - start_time,
                        "status": status,
                        "metadata": final_metadata,
                        "usage": final_usage,
                    }

                    _context.spans.append(span)

                return result

            return wrapper

        return decorator

    # ---------------------------------------------------------
    # TRACE EXPORT
    # ---------------------------------------------------------

    def export_trace(
        self,
        output,
        query=None,
        session_id=None,
        user_id=None,
        timestamp=None,
        rag_docs=None,
    ):
        """Exports collected spans as a trace."""

        if isinstance(output, dict) and "output" in output:
            result_data = output
            answer = output.get("output")
        else:
            result_data = {}
            answer = output

        trace_id = result_data.get("trace_id") or getattr(_context, "trace_id", f"trace-{uuid.uuid4().hex[:8]}")

        collected_spans = getattr(_context, "spans", [])

        provider_raw = result_data.get("provider_raw")

        # fallback to span usage if not provided
        if not provider_raw:
            for span in collected_spans:
                if span.get("type") == "llm":
                    provider_raw = span.get("usage")
                    break

        latency = result_data.get("latency_ms")

        if latency is None and collected_spans:
            latency = collected_spans[-1]["end_time"] - collected_spans[0]["start_time"]

        trace = {
            "id": trace_id,
            "trace_id": trace_id,
            "trace_name": result_data.get("trace_name", "ai-trace"),
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp or int(time.time() * 1000),
            "environment": result_data.get("environment", self.environment),
            "provider": result_data.get("provider", self.provider),
            "model": result_data.get("model", self.model),
            "input": {"query": query},
            "output": {"answer": answer},
            "latency_ms": latency,
            "rag_docs": self._safe_serialize(rag_docs, max_length=1000) if rag_docs else None,
            "spans": collected_spans,
            "provider_raw": provider_raw
            if isinstance(provider_raw, dict)
            else self._safe_serialize(provider_raw, max_length=500),
            "metadata": result_data.get("metadata", {}),
            "status": "success",
            "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            + f".{int(time.time()*1000)%1000:03d}Z",
        }

        if hasattr(_context, "spans"):
            _context.spans = []

        self.telemetry.log_trace(trace)