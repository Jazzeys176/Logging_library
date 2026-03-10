import uuid
import time
import functools
import threading

# Thread-local storage to track spans within a single request/execution thread
_context = threading.local()

class SDKTracer:

    def __init__(self, telemetry):
        self.telemetry = telemetry

    def start_trace(self):
        """Initializes a new span list for the current thread."""
        _context.spans = []

    def _safe_serialize(self, obj, max_length=150):
        """Aggressively converts objects to concise summaries for minimal log size."""
        def _serialize(o, depth=0):
            try:
                if depth > 2: return "..."
                if o is None: return "None"
                if isinstance(o, (int, float, bool)): return str(o)
                if hasattr(o, 'page_content'): return f"Document(len={len(o.page_content)})"
                
                if isinstance(o, (list, tuple)):
                    items = [_serialize(i, depth + 1) for i in o[:3]]
                    if len(o) > 3: items.append("...")
                    container = "[]" if isinstance(o, list) else "()"
                    return f"{container[0]}{', '.join(items)}{container[1]}"
                
                if isinstance(o, dict):
                    keys = list(o.keys())
                    items = [f"{k}: {_serialize(o[k], depth + 1)}" for k in keys[:2]]
                    if len(keys) > 2: items.append("...")
                    return f"{{{', '.join(items)}}}"
                
                return str(o).replace("\n", " ")
            except:
                return "[Error]"

        s = _serialize(obj)
        if len(s) > max_length:
            return s[:max_length] + "... [TRUNCATED]"
        return s

    def trace(self, name=None, span_type=None, metadata=None, usage=None, include_io=True, result_parser=None):
        """Decorator to automatically capture function execution as a span."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span_name = name or func.__name__
                start_time = int(time.time() * 1000)
                
                # Execute the function
                try:
                    result = func(*args, **kwargs)
                    status = "success"
                    output = result
                except Exception as e:
                    status = "error"
                    output = str(e)
                    raise e
                finally:
                    end_time = int(time.time() * 1000)
                    
                    # Lazy initialize spans if not present
                    if not hasattr(_context, 'spans'):
                        _context.spans = []

                    final_metadata = (metadata or {}).copy()
                    final_usage = (usage or {}).copy()
                    
                    if result_parser and status == "success":
                        try:
                            # Pass result, args, and kwargs to the parser for richer context
                            parsed = result_parser(output, args, kwargs)
                            if isinstance(parsed, dict):
                                final_metadata.update(parsed.get("metadata", {}))
                                final_usage.update(parsed.get("usage", {}))
                        except Exception as e:
                            # Fallback if parser fails - record the error in metadata for debugging
                            final_metadata["_parser_error"] = str(e)

                    # If no specific metadata and include_io is True, fallback to args/output
                    if include_io and not final_metadata:
                        final_metadata = {
                            "input": self._safe_serialize(args),
                            "output": self._safe_serialize(output)
                        }

                    # Concise prefix: e.g. "intent-classification" -> "intent"
                    clean_type = (span_type.split('-')[0] if '-' in str(span_type) else str(span_type)) if span_type else "generic"
                    prefix = f"span-{clean_type}-"
                    
                    span = {
                        "span_id": f"{prefix}{uuid.uuid4().hex[:8]}",
                        "type": span_type or "generic",
                        "name": span_name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "latency_ms": end_time - start_time,
                        "status": status,
                        "metadata": final_metadata,
                        "usage": final_usage
                    }
                    
                    _context.spans.append(span)
                
                return result
            return wrapper
        return decorator

    def export_trace(self, result, query=None, session_id=None, user_id=None, timestamp=None, include_spans=True, exclude_span_names=None):
        trace_id = result.get("trace_id") or f"trace-{uuid.uuid4().hex[:8]}"
        if exclude_span_names is None:
            exclude_span_names = [] # Include all spans by default, let them be "thin"
        
        # Collect and filter spans
        combined_spans = []
        if include_spans:
            collected = getattr(_context, 'spans', [])
            explicit = result.get("spans") or []
            for s in (collected + explicit):
                if s.get("name") not in exclude_span_names:
                    combined_spans.append(s)

        # Build compatibility structure
        trace = {
            "id": trace_id,
            "partitionKey": trace_id,
            "trace_id": trace_id,
            "trace_name": result.get("trace_name", "rag-qa"),
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp or int(time.time() * 1000),
            "environment": result.get("environment", "prod"),
            "intent": result.get("intent"),
            "routing_decision": result.get("routing_decision"),
            "provider": result.get("provider", "groq"),
            "model": result.get("model", "llama-3.1-8b-instant"),
            "input": {"query": query},
            "output": {"answer": result.get("output")},
            "latency_ms": result.get("latency_ms"),
            "documents_found": result.get("documents_found"),
            "retrieval_executed": result.get("retrieval_executed"),
            "retrieval_confidence": result.get("retrieval_confidence"),
            "rag_data": result.get("rag_data"),
            "spans": combined_spans,
            "provider_raw": result.get("provider_raw") if isinstance(result.get("provider_raw"), dict) else self._safe_serialize(result.get("provider_raw"), max_length=500),
            "status": "success",
            "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time()*1000)%1000:03d}Z"
        }

        # Clear context after export to prevent leak between requests
        if hasattr(_context, 'spans'):
            _context.spans = []

        self.telemetry.log_trace(trace)