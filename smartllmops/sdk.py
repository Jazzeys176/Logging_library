import uuid
import time
import functools
import threading
import statistics
import hashlib

# Thread-local storage to track spans within a single request/execution thread
_context = threading.local()

class SDKTracer:

    def __init__(self, telemetry, environment="prod", model="unknown", provider="unknown"):
        self.telemetry = telemetry
        self.environment = environment
        self.model = model
        self.provider = provider

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

    def export_trace(self, output, query=None, session_id=None, user_id=None, timestamp=None, rag_docs=None):
        """Autonomous trace assembly (LangFuse style)."""
        
        # 1. Handle legacy dict or new raw output
        if isinstance(output, dict) and "output" in output:
            # Legacy/Hybrid support
            result_data = output
            answer = output.get("output")
        else:
            result_data = {}
            answer = output

        trace_id = result_data.get("trace_id") or f"trace-{uuid.uuid4().hex[:8]}"
        
        # 2. Extract spans and analyze them for metrics
        collected_spans = getattr(_context, 'spans', [])
        
        intent = result_data.get("intent")
        routing_decision = result_data.get("routing_decision")
        provider_raw = result_data.get("provider_raw")
        
        # Scan spans for missing info
        retrieval_span = next((s for s in collected_spans if s.get("name") == "vector-search"), None)
        llm_span = next((s for s in collected_spans if s.get("type") == "llm"), None)
        intent_span = next((s for s in collected_spans if s.get("type") == "intent-classification"), None)

        if not intent and intent_span:
            intent = intent_span.get("metadata", {}).get("intent")
            
        if not provider_raw and llm_span:
            provider_raw = llm_span.get("usage")

        # 3. Calculate Retrieval Metrics (Self-Assembly)
        rag_data = result_data.get("rag_data")
        retrieval_confidence = result_data.get("retrieval_confidence", 0)
        documents_found = result_data.get("documents_found", 0)
        
        if not rag_data and rag_docs:
            scores = [float(score) for _, score in rag_docs]
            documents_found = len(rag_docs)
            
            # Confidence Math migrated from RAGEngine
            if scores:
                avg_score = statistics.mean(scores)
                coverage = min(len(scores) / 3, 1)
                std_score = statistics.pstdev(scores) if len(scores) > 1 else 0
                consistency = 1 / (1 + std_score)
                retrieval_confidence = round(0.6 * avg_score + 0.25 * coverage + 0.15 * consistency, 4)
            
            rag_data = {
                "documents_found": documents_found,
                "retrieval_scores": {
                    "avg_score": statistics.mean(scores) if scores else 0,
                    "min_score": min(scores) if scores else 0,
                    "max_score": max(scores) if scores else 0,
                    "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0,
                    "per_doc_scores": [round(s, 4) for s in scores]
                },
                "retrieved_documents": [
                    {
                        "doc_id": hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
                        "score": round(float(score), 4),
                        "content_preview": doc.page_content[:200]
                    }
                    for doc, score in rag_docs
                ]
            }

        # Determine trace name if not provided
        trace_name = result_data.get("trace_name")
        if not trace_name:
            if routing_decision == "out_of_scope": trace_name = "out-of-scope-qa"
            elif intent == "greeting": trace_name = "simple-qa"
            else: trace_name = "rag-qa"

        # 4. Build Final "Ideal" Schema
        trace = {
            "id": trace_id,
            "partitionKey": trace_id,
            "trace_id": trace_id,
            "trace_name": trace_name,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp or int(time.time() * 1000),
            "environment": result_data.get("environment", self.environment),
            "intent": intent,
            "routing_decision": routing_decision,
            "provider": result_data.get("provider", self.provider),
            "model": result_data.get("model", self.model),
            "input": {"query": query},
            "output": {"answer": answer},
            "latency_ms": result_data.get("latency_ms") or (collected_spans[-1]['end_time'] - collected_spans[0]['start_time'] if collected_spans else 0),
            "documents_found": documents_found,
            "retrieval_executed": result_data.get("retrieval_executed", intent == "search_query"),
            "retrieval_confidence": retrieval_confidence,
            "rag_data": rag_data,
            "spans": collected_spans,
            "provider_raw": provider_raw if isinstance(provider_raw, dict) else self._safe_serialize(provider_raw, max_length=500),
            "status": "success",
            "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time()*1000)%1000:03d}Z"
        }

        # Clear context after export to prevent leak between requests
        if hasattr(_context, 'spans'):
            _context.spans = []

        self.telemetry.log_trace(trace)