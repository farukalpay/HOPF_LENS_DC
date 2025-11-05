"""
HOPF_LENS_DC: Self-correcting LLM orchestrator with dynamic tool creation
Ready-to-run implementation using OpenAI function calling
"""

import json
import ast
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
import sys
from io import StringIO
import builtins
import bs4

import openai
import requests
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = None  # Set via environment variable or function argument
MODEL = "gpt-4-0613"

# Convergence parameters
TAU_A = 0.02  # semantic drift threshold
TAU_C = 0.01  # confidence improvement threshold
TAU_NU = 0.15  # max allowed fragility
K_ATTACK = 3  # counterfactual probes per round
K_EXEC = 4  # tasks per batch
T_MAX = 10  # max iterations
TIME_BUDGET_MS = 60000  # 60 seconds

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Task:
    task: str
    args: Dict[str, Any]
    priority: float = 0.5

    def key(self):
        """Canonical key for deduplication"""
        return (self.task, json.dumps(self.args, sort_keys=True))


@dataclass
class Evidence:
    claims: List[Dict[str, Any]]
    sources: List[str]
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# DYNAMIC TOOL REGISTRY
# ============================================================================

# Storage for dynamically created tools
DYNAMIC_TOOLS: Dict[str, Callable] = {}
DYNAMIC_TOOL_SOURCES: Dict[str, str] = {}
DYNAMIC_TOOL_DESCRIPTIONS: Dict[str, str] = {}
TOOL_DEBUG_DATA: Dict[str, Dict[str, Any]] = defaultdict(dict)

DUCKDUCKGO_HTML_ENDPOINTS = {
    "https://duckduckgo.com/html",
    "https://duckduckgo.com/html/",
    "https://lite.duckduckgo.com/lite/",
}

DUCKDUCKGO_BOOTSTRAP_URL = "https://duckduckgo.com/"

DEFAULT_DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
}


def _looks_like_duckduckgo_html(url: Optional[str]) -> bool:
    if not isinstance(url, str):
        return False
    return any(url.startswith(prefix) for prefix in DUCKDUCKGO_HTML_ENDPOINTS)


def _ensure_ddg_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(headers or {})
    for key, value in DEFAULT_DDG_HEADERS.items():
        merged.setdefault(key, value)
    return merged


def _normalise_duckduckgo_url(raw_url: Optional[str]) -> Optional[str]:
    if not raw_url:
        return None
    if raw_url.startswith("//"):
        return "https:" + raw_url
    if raw_url.startswith("/l/?"):
        return "https://duckduckgo.com" + raw_url
    return raw_url


def _extract_duckduckgo_results(html: Optional[str], max_results: int = 8) -> List[Dict[str, str]]:
    if not html or not html.strip():
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    results: List[Dict[str, str]] = []

    def add_entry(title_tag, snippet_tag):
        title = title_tag.get_text(" ", strip=True) if title_tag else None
        url = _normalise_duckduckgo_url(title_tag.get("href") if title_tag else None)
        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else None
        if any([title, url, snippet]):
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })

    for container in soup.select("div.result"):
        title_tag = container.select_one("a.result__a") or container.find("a", class_="result__a")
        snippet_tag = (
            container.select_one("a.result__snippet")
            or container.select_one("div.result__snippet")
            or container.find("a", class_="result__snippet")
            or container.find("div", class_="result__snippet")
        )
        add_entry(title_tag, snippet_tag)
        if len(results) >= max_results:
            break

    if results:
        return results[:max_results]

    # Fallback for lite or alternative layouts (table based)
    for table in soup.select("table.result"):
        title_tag = table.select_one("a") or table.find("a")
        snippet_tag = table.find(class_="result__snippet") or table.find("td", class_="result-snippet")
        add_entry(title_tag, snippet_tag)
        if len(results) >= max_results:
            break

    return results[:max_results]

def _should_attempt_tool_repair(tool_name: str, error: Exception) -> bool:
    if tool_name != "search_web":
        return False
    message = str(error).lower()
    return "get_text" in message or "attributeerror" in message


def _harden_search_web_source(source: str) -> Optional[str]:
    class GetTextGuard(ast.NodeTransformer):
        def __init__(self):
            self.modified = False

        def visit_FunctionDef(self, node):
            if node.name == "_safe_get_text":
                return node
            return self.generic_visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get_text":
                base = node.func.value
                new_args = [base] + list(node.args)
                new_call = ast.Call(
                    func=ast.Name(id="_safe_get_text", ctx=ast.Load()),
                    args=new_args,
                    keywords=node.keywords,
                )
                self.modified = True
                return ast.copy_location(new_call, node)
            return node

    class SearchWebHardener(ast.NodeTransformer):
        def __init__(self):
            self.modified = False

        def visit_FunctionDef(self, node):
            if node.name != "search_web":
                return self.generic_visit(node)

            guard = GetTextGuard()
            new_body = []
            for stmt in node.body:
                new_body.append(guard.visit(stmt))
            node.body = new_body

            if not guard.modified:
                return node

            helper_exists = any(
                isinstance(stmt, ast.FunctionDef) and stmt.name == "_safe_get_text"
                for stmt in node.body
            )

            if not helper_exists:
                helper_def = ast.FunctionDef(
                    name="_safe_get_text",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="node")],
                        vararg=ast.arg(arg="args"),
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=ast.arg(arg="kwargs"),
                        defaults=[],
                    ),
                    body=[
                        ast.If(
                            test=ast.Call(
                                func=ast.Name(id="hasattr", ctx=ast.Load()),
                                args=[
                                    ast.Name(id="node", ctx=ast.Load()),
                                    ast.Constant(value="get_text"),
                                ],
                                keywords=[],
                            ),
                            body=[
                                ast.Try(
                                    body=[
                                        ast.Return(
                                            value=ast.Call(
                                                func=ast.Attribute(
                                                    value=ast.Name(id="node", ctx=ast.Load()),
                                                    attr="get_text",
                                                    ctx=ast.Load(),
                                                ),
                                                args=[
                                                    ast.Starred(
                                                        value=ast.Name(id="args", ctx=ast.Load()),
                                                        ctx=ast.Load(),
                                                    )
                                                ],
                                                keywords=[
                                                    ast.keyword(
                                                        arg=None,
                                                        value=ast.Name(id="kwargs", ctx=ast.Load()),
                                                    )
                                                ],
                                            )
                                        )
                                    ],
                                    handlers=[
                                        ast.ExceptHandler(
                                            type=ast.Name(id="Exception", ctx=ast.Load()),
                                            name=None,
                                            body=[
                                                ast.Return(value=ast.Constant(value=None))
                                            ],
                                        )
                                    ],
                                    orelse=[],
                                    finalbody=[],
                                )
                            ],
                            orelse=[ast.Return(value=ast.Constant(value=None))],
                        ),
                        ast.Return(value=ast.Constant(value=None)),
                    ],
                    decorator_list=[],
                )
                node.body.insert(0, helper_def)

            self.modified = True
            return node

    try:
        module = ast.parse(source)
    except SyntaxError:
        return None

    hardener = SearchWebHardener()
    module = hardener.visit(module)

    if not hardener.modified:
        return None

    ast.fix_missing_locations(module)

    try:
        return ast.unparse(module)
    except AttributeError:
        return None


def _repair_search_web_tool(error_message: str) -> bool:
    description = DYNAMIC_TOOL_DESCRIPTIONS.get(
        "search_web",
        "DuckDuckGo HTML search tool with BeautifulSoup parsing",
    )

    print("  Attempting automatic repair for 'search_web'...")

    if "search_web" not in DYNAMIC_TOOLS:
        print("  No existing 'search_web' tool to repair; skipping automatic rebuild.")
        return False

    original_source = DYNAMIC_TOOL_SOURCES.get("search_web")
    if not isinstance(original_source, str):
        print("  Unable to access original tool source for 'search_web'.")
        return False

    hardened_source = _harden_search_web_source(original_source)
    if not hardened_source:
        print("  Automatic hardening produced no changes; leaving tool as-is.")
        return False

    result = update_tool("search_web", hardened_source, description)

    if result.get("success"):
        debug_entry = TOOL_DEBUG_DATA.setdefault("search_web", {})
        debug_entry["last_repair_trigger"] = error_message
        return True

    print(f"  Repair failed: {result.get('error')}")
    return False

def _escape_json_string(value: str) -> str:
    """Escape characters so the string can live inside JSON quotes."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _sanitize_tool_code_blocks(raw: str) -> str:
    """
    Replace common non-JSON code fences (`, ```...```) in tool_code with proper quoted strings.
    """
    if not isinstance(raw, str):
        return raw

    sanitized = raw

    patterns = [
        re.compile(r'"tool_code"\s*:\s*```(?:python)?\s*\n(.*?)```', re.DOTALL),
        re.compile(r'"tool_code"\s*:\s*`{1}\s*(.*?)\s*`{1}', re.DOTALL),
    ]

    for pattern in patterns:
        def replacer(match):
            code = match.group(1)
            escaped = _escape_json_string(code.strip("\n"))
            return f'"tool_code": "{escaped}"'
        sanitized = pattern.sub(replacer, sanitized)

    return sanitized


def _parse_tool_arguments(args_text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse tool call arguments from possibly malformed JSON emitted by the LLM.
    """
    if not isinstance(args_text, str):
        return args_text if isinstance(args_text, dict) else None

    candidates = []
    cleaned = args_text.strip()
    if cleaned:
        candidates.append(cleaned)
        sanitized = _sanitize_tool_code_blocks(cleaned)
        if sanitized != cleaned:
            candidates.append(sanitized)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(candidate, strict=False)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            continue
    return None


def _create_requests_proxy(tool_name: str):
    """Provide a proxy around requests so we can capture raw HTML payloads."""
    original_requests = requests

    def capture_response(response):
        debug_store = TOOL_DEBUG_DATA.setdefault(tool_name, {})
        debug_store["last_http_call"] = {
            "url": getattr(response, "url", None),
            "status": getattr(response, "status_code", None),
            "reason": getattr(response, "reason", None),
            "headers": dict(getattr(response, "headers", {})),
            "elapsed_ms": (
                getattr(getattr(response, "elapsed", None), "total_seconds", lambda: None)()
                if getattr(response, "elapsed", None)
                else None
            ),
            "timestamp": time.time(),
        }
        try:
            text_preview = response.text
        except Exception:
            text_preview = None
        if text_preview is not None:
            debug_store["last_raw_html"] = text_preview[:50000]
            debug_store["raw_html_char_length"] = len(text_preview)
        return response

    def perform_duckduckgo_html_request(callable_get, args, kwargs, session_obj=None, original_session_get=None):
        local_kwargs = dict(kwargs or {})
        url = local_kwargs.pop("url", None)
        if args:
            url = args[0] if url is None else url

        if not _looks_like_duckduckgo_html(url):
            return callable_get(*args, **kwargs)

        params = local_kwargs.pop("params", None)
        if params is None and len(args) > 1 and isinstance(args[1], dict):
            params = args[1]

        headers = _ensure_ddg_headers(local_kwargs.pop("headers", None))
        timeout = local_kwargs.get("timeout", 10)
        local_kwargs.setdefault("timeout", timeout)

        debug_store = TOOL_DEBUG_DATA.setdefault(tool_name, {})
        debug_store["duckduckgo_bootstrapped"] = True

        try:
            if session_obj is not None and original_session_get is not None:
                try:
                    original_session_get(
                        DUCKDUCKGO_BOOTSTRAP_URL,
                        headers=headers,
                        timeout=timeout,
                    )
                except Exception as bootstrap_error:
                    debug_store["duckduckgo_bootstrap_error"] = str(bootstrap_error)
                response = original_session_get(
                    url,
                    params=params,
                    headers=headers,
                    **local_kwargs,
                )
            else:
                session = original_requests.Session()
                try:
                    session.get(
                        DUCKDUCKGO_BOOTSTRAP_URL,
                        headers=headers,
                        timeout=timeout,
                    )
                except Exception as bootstrap_error:
                    debug_store["duckduckgo_bootstrap_error"] = str(bootstrap_error)
                response = session.get(
                    url,
                    params=params,
                    headers=headers,
                    **local_kwargs,
                )
        except Exception as request_error:
            debug_store["duckduckgo_request_error"] = str(request_error)
            response = callable_get(*args, **kwargs)
        return response

    class RequestsProxy:
        def __init__(self, module):
            self._module = module

        def get(self, *args, **kwargs):
            response = perform_duckduckgo_html_request(
                self._module.get,
                args,
                kwargs,
            )
            return capture_response(response)

        def Session(self, *args, **kwargs):
            session = self._module.Session(*args, **kwargs)
            original_get = session.get
            original_request = session.request

            def instrumented_get(*g_args, **g_kwargs):
                response = perform_duckduckgo_html_request(
                    original_get,
                    g_args,
                    g_kwargs,
                    session_obj=session,
                    original_session_get=original_get,
                )
                return capture_response(response)

            def instrumented_request(method, url, *r_args, **r_kwargs):
                if isinstance(method, str) and method.lower() == "get":
                    get_args = (url,) + r_args
                    response = perform_duckduckgo_html_request(
                        lambda *ga, **gk: original_request(method, *ga, **gk),
                        get_args,
                        r_kwargs,
                        session_obj=session,
                        original_session_get=original_get,
                    )
                    return capture_response(response)
                response = original_request(method, url, *r_args, **r_kwargs)
                if isinstance(method, str) and method.lower() == "get":
                    capture_response(response)
                return response

            session.get = instrumented_get
            session.request = instrumented_request
            return session

        def __getattr__(self, item):
            return getattr(self._module, item)

    return RequestsProxy(original_requests)


def _wrap_tool_with_debug(tool_name: str, tool_func: Callable) -> Callable:
    """Wrap dynamic tools to capture execution metadata for later inspection."""

    def wrapper(*args, **kwargs):
        debug_store = TOOL_DEBUG_DATA.setdefault(tool_name, {})
        debug_store["last_call_args"] = {
            "args": [repr(a) for a in args],
            "kwargs": {k: repr(v) for k, v in kwargs.items()}
        }
        start_time = time.time()
        result = tool_func(*args, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000
        debug_store["last_call_duration_ms"] = round(elapsed_ms, 3)

        processed_result = result
        fallback_applied = False

        if isinstance(result, dict):
            if tool_name == "search_web":
                parsed = result.get("results") if isinstance(result.get("results"), list) else []
                if not parsed:
                    fallback_entries = _extract_duckduckgo_results(
                        debug_store.get("last_raw_html"),
                        max_results=10,
                    )
                    if fallback_entries:
                        processed_result = dict(result)
                        processed_result["results"] = fallback_entries
                        processed_result["result_count"] = len(fallback_entries)
                        processed_result.setdefault(
                            "status",
                            debug_store.get("last_http_call", {}).get("status"),
                        )
                        processed_result["_fallback_parser"] = "duckduckgo_html_v1"
                        fallback_applied = True
                        debug_store["fallback_parser_used"] = True
                        debug_store["parsed_entries"] = fallback_entries
                        debug_store["parsed_entry_count"] = len(fallback_entries)
                else:
                    debug_store["parsed_entries"] = parsed
                    debug_store["parsed_entry_count"] = len(parsed)
            debug_store["last_result"] = processed_result
        else:
            debug_store["last_result"] = processed_result

        if tool_name == "search_web":
            raw_len = len(debug_store.get("last_raw_html") or "")
            parsed_count = debug_store.get("parsed_entry_count", 0)
            message = f"    [debug:{tool_name}] raw_html_len={raw_len}, parsed_entries={parsed_count}"
            if fallback_applied:
                message += " (fallback parser applied)"
            print(message)

        return processed_result

    return wrapper


def safe_exec_tool(code: str, tool_name: str) -> Callable:
    """
    Safely execute tool code and return the function.
    The code must define a function with the same name as tool_name.
    """
    allowed_module_names = {"requests", "json", "re", "time", "bs4"}
    original_import = builtins.__import__
    instrumented_requests = _create_requests_proxy(tool_name)

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_name = name.split('.')[0]
        if root_name not in allowed_module_names:
            raise ImportError(f"Import of '{name}' is not allowed in dynamic tools")
        if root_name == "requests":
            return instrumented_requests
        return original_import(name, globals, locals, fromlist, level)
    
    # Create a restricted namespace for execution
    namespace = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'dict': dict,
            'list': list,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'print': print,
            'isinstance': isinstance,
            'type': type,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'any': any,
            'all': all,
            'getattr': getattr,
            'Exception': Exception,
            '__import__': restricted_import,
        },
        'requests': instrumented_requests,
        'BeautifulSoup': BeautifulSoup,
        'bs4': bs4,
        'json': json,
        're': re,
        'time': time,
    }
    
    try:
        exec(code, namespace)
        if tool_name not in namespace:
            raise ValueError(f"Tool code must define a function named '{tool_name}'")
        return namespace[tool_name]
    except Exception as e:
        print(f"  [Tool Creation Error] Failed to create tool '{tool_name}': {e}")
        raise


def create_tool(tool_name: str, tool_code: str, description: str) -> Dict[str, Any]:
    """
    Create a new temporary tool and register it in memory.
    
    Args:
        tool_name: Name of the tool (must match function name in code)
        tool_code: Python code defining the tool function
        description: Description of what the tool does
    
    Returns:
        Status dictionary
    """
    print(f"  [create_tool] Creating tool: {tool_name}")
    print(f"  [create_tool] Description: {description}")
    print(f"  [create_tool] Code preview:\n{tool_code.strip()[:500]}\n")

    # Reset stored metadata so fresh diagnostics are captured
    TOOL_DEBUG_DATA.pop(tool_name, None)
    
    try:
        # Create the tool function
        tool_func = safe_exec_tool(tool_code, tool_name)
        wrapped_tool = _wrap_tool_with_debug(tool_name, tool_func)

        # Register in dynamic tools
        DYNAMIC_TOOLS[tool_name] = wrapped_tool
        DYNAMIC_TOOL_SOURCES[tool_name] = tool_code
        DYNAMIC_TOOL_DESCRIPTIONS[tool_name] = description
        
        print(f"  [create_tool] Successfully created and registered '{tool_name}'")
        print(f"  [create_tool] Total dynamic tools: {len(DYNAMIC_TOOLS)}")
        
        return {
            "success": True,
            "tool_name": tool_name,
            "message": f"Tool '{tool_name}' created successfully",
            "total_dynamic_tools": len(DYNAMIC_TOOLS)
        }
    except Exception as e:
        print(f"  [create_tool] Failed to create tool '{tool_name}': {e}")
        return {
            "success": False,
            "tool_name": tool_name,
            "error": str(e)
        }


def list_tools() -> Dict[str, Any]:
    """List all available tools (static and dynamic)"""
    return {
        "static_tools": list(STATIC_TOOLS.keys()),
        "dynamic_tools": list(DYNAMIC_TOOLS.keys()),
        "total_tools": len(STATIC_TOOLS) + len(DYNAMIC_TOOLS)
    }


def inspect_tool(tool_name: str, include_source: bool = True, include_debug: bool = True) -> Dict[str, Any]:
    """
    Retrieve metadata about a tool, including the latest captured debug data.
    """
    info: Dict[str, Any] = {"tool_name": tool_name}
    if tool_name in DYNAMIC_TOOLS:
        info["tool_type"] = "dynamic"
        info["description"] = DYNAMIC_TOOL_DESCRIPTIONS.get(tool_name, "")
        if include_source:
            info["source"] = DYNAMIC_TOOL_SOURCES.get(tool_name, "")
    elif tool_name in STATIC_TOOLS:
        info["tool_type"] = "static"
        info["description"] = (STATIC_TOOLS[tool_name].__doc__ or "").strip()
        if include_source:
            info["source"] = None
    else:
        return {"success": False, "error": f"Tool '{tool_name}' not found"}

    if include_debug:
        info["debug"] = TOOL_DEBUG_DATA.get(tool_name, {})

    info["success"] = True
    return info


def update_tool(tool_name: str, tool_code: str, description: Optional[str] = None) -> Dict[str, Any]:
    """
    Replace an existing dynamic tool implementation while keeping metadata up to date.
    """
    if tool_name not in DYNAMIC_TOOLS:
        return {
            "success": False,
            "tool_name": tool_name,
            "error": f"Tool '{tool_name}' does not exist and cannot be updated"
        }

    new_description = description if description is not None else DYNAMIC_TOOL_DESCRIPTIONS.get(tool_name, "")
    print(f"  [update_tool] Updating '{tool_name}'")
    return create_tool(tool_name=tool_name, tool_code=tool_code, description=new_description)


# ============================================================================
# STATIC TOOLS (Core tools always available)
# ============================================================================

def eval_math(expression: str) -> Dict[str, Any]:
    """Math evaluator with safe eval"""
    print(f"    [eval_math] Evaluating: {expression}")
    try:
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        print(f"    [eval_math] Result: {result}")
        return {"result": result, "success": True}
    except Exception as e:
        print(f"    [eval_math] Error: {e}")
        return {"result": None, "success": False, "error": str(e)}


def fetch_api(endpoint: str, params: Dict = None) -> Dict[str, Any]:
    """Generic API fetcher"""
    print(f"    [fetch_api] Fetching: {endpoint}")
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return {
            "endpoint": endpoint,
            "status": response.status_code,
            "data": response.text[:1000]  # Limit response size
        }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "error": str(e),
            "status": 0
        }


# Static tools that are always available
STATIC_TOOLS = {
    "create_tool": create_tool,
    "list_tools": list_tools,
    "inspect_tool": inspect_tool,
    "update_tool": update_tool,
    "eval_math": eval_math,
    "fetch_api": fetch_api,
}


def get_all_tools() -> Dict[str, Callable]:
    """Get combined dictionary of all available tools"""
    return {**STATIC_TOOLS, **DYNAMIC_TOOLS}


# ============================================================================
# FUNCTION CALLING SCHEMAS
# ============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "create_tool",
            "description": "Create a new temporary tool by providing Python code. The tool will be stored in memory and can be used immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool (must match function name in code)"
                    },
                    "tool_code": {
                        "type": "string",
                        "description": "Complete Python function code. Must define a function with name matching tool_name. Available imports: requests, BeautifulSoup, json, re, time"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the tool does"
                    }
                },
                "required": ["tool_name", "tool_code", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tools",
            "description": "List all available tools (static and dynamic)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "decompose_request",
            "description": "Decompose a query into atomic executable tasks",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string", "description": "Tool name to execute"},
                                "args": {"type": "object", "description": "Arguments for the tool"},
                                "priority": {"type": "number", "description": "Priority 0-1"}
                            },
                            "required": ["task", "args"]
                        }
                    },
                    "rationale": {"type": "string"}
                },
                "required": ["plan", "rationale"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compose_response",
            "description": "Compose final answer from task results",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Final answer to the query"},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "claims": {"type": "array", "items": {"type": "object"}},
                            "sources": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "confidence": {"type": "number", "description": "Confidence 0-1"}
                },
                "required": ["answer", "evidence", "confidence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "antipode_counterfactual",
            "description": "Generate counterfactual attack plan to test answer robustness",
            "parameters": {
                "type": "object",
                "properties": {
                    "attack_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "args": {"type": "object"}
                            },
                            "required": ["task", "args"]
                        }
                    },
                    "target_claims": {"type": "array", "items": {"type": "string"}},
                    "vulnerability": {"type": "number", "description": "Fragility score 0-1"}
                },
                "required": ["attack_plan", "target_claims", "vulnerability"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "putback_repair",
            "description": "Repair plan to satisfy policy constraints",
            "parameters": {
                "type": "object",
                "properties": {
                    "repaired_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "args": {"type": "object"}
                            }
                        }
                    },
                    "delta": {"type": "integer", "description": "Edit distance"}
                },
                "required": ["repaired_plan", "delta"]
            }
        }
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def evidence_oplus(w1: Optional[Dict], w2: Optional[Dict]) -> Dict:
    """Merge two evidence objects (monoid operation)"""
    if w1 is None:
        return w2 or {}
    if w2 is None:
        return w1
    
    combined_claims = w1.get("claims", []) + w2.get("claims", [])
    deduped_claims = []
    seen = set()
    for claim in combined_claims:
        if not isinstance(claim, dict):
            continue
        key = (claim.get("claim"), claim.get("source"))
        if key in seen:
            continue
        seen.add(key)
        deduped_claims.append(claim)
    
    merged = {
        "claims": deduped_claims,
        "sources": list(set(w1.get("sources", []) + w2.get("sources", [])))
    }
    return merged


def evidence_nu(w: Optional[Dict]) -> float:
    """Compute evidence fragility (higher = more fragile)"""
    if not w or "claims" not in w:
        return float('inf')
    
    claims = w.get("claims", [])
    if not claims:
        return float('inf')
    
    supports = [claim.get("support", 0.0) for claim in claims]
    if not supports:
        return 1.0
    
    return 1.0 - min(supports)


def semantic_drift(a1: str, a2: str) -> float:
    """Compute semantic drift between answers (simplified)"""
    if not a1 or not a2:
        return 1.0
    
    words1 = set(a1.lower().split())
    words2 = set(a2.lower().split())
    
    if not words1 or not words2:
        return 1.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return 1.0 - (intersection / union if union > 0 else 0)


def normalize_plan(tasks: List[Task]) -> List[Task]:
    """Deduplicate tasks, keeping highest priority"""
    task_map = {}
    for task in tasks:
        key = task.key()
        if key not in task_map or task.priority > task_map[key].priority:
            task_map[key] = task
    
    return sorted(task_map.values(), key=lambda t: -t.priority)


def violates_policy(answer: str, policy: Dict) -> bool:
    """Check if answer violates policy (simplified)"""
    if not answer:
        return True
    if len(answer) > policy.get("max_length", 10000):
        return True
    return False


def budget_remaining(budget_ms: int, start_time: float) -> bool:
    """Check if time budget remains"""
    elapsed_ms = (time.time() - start_time) * 1000
    return elapsed_ms < budget_ms


def summarize_search_results(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a text summary from search results when LLM synthesis fails"""
    print("  [Fallback] Attempting to summarize search results...")
    
    if not isinstance(results, dict):
        print("  [Fallback] Results is not a dict")
        return None
    
    aggregated: List[Dict[str, Any]] = []
    
    for key, payload in results.items():
        if not isinstance(key, str):
            continue
            
        if isinstance(payload, dict) and "results" in payload:
            entries = payload.get("results") or []
            print(f"  [Fallback] Found {len(entries)} entries in {key}")
            aggregated.extend([entry for entry in entries if isinstance(entry, dict)])
    
    if not aggregated:
        print("  [Fallback] No results to aggregate")
        return None
    
    top_entries = aggregated[:3]
    summary_parts: List[str] = []
    claims: List[Dict[str, Any]] = []
    sources: List[str] = []
    
    for entry in top_entries:
        title = (entry.get("title") or "Result").strip()
        snippet = (entry.get("snippet") or "").strip()
        url = entry.get("url", "")
        
        if snippet:
            summary_parts.append(f"{snippet}")
            claims.append({
                "claim": snippet,
                "source": url or title,
                "support": 0.5
            })
        elif title:
            summary_parts.append(title)
            claims.append({
                "claim": title,
                "source": url or title,
                "support": 0.3
            })
        
        if url:
            sources.append(url)
    
    summary_text = " ".join(summary_parts).strip()
    
    if not summary_text:
        print("  [Fallback] No text to summarize")
        return None
    
    evidence = {
        "claims": claims,
        "sources": list(dict.fromkeys([s for s in sources if s]))
    }
    
    result = {
        "answer": summary_text,
        "evidence": evidence,
        "confidence": 0.4
    }
    
    print(f"  [Fallback] Created summary: {summary_text[:100]}...")
    return result


# ============================================================================
# LLM INTERACTION
# ============================================================================

def call_llm(messages: List[Dict], tools: List[Dict], tool_choice: str = "auto") -> Dict:
    """Call OpenAI with function calling"""
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.7
        )
        return response
    except Exception as e:
        print(f"  [LLM Error] {e}")
        return None


def extract_tool_call(response) -> Optional[Dict]:
    """Extract tool call from LLM response"""
    if not response or not response.choices:
        return None
    
    message = response.choices[0].message
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_call = message.tool_calls[0]
        args_text = tool_call.function.arguments
        if not isinstance(args_text, str):
            args_text = json.dumps(args_text or {})
        else:
            args_text = args_text or ""
        arguments = _parse_tool_arguments(args_text)
        if arguments is None:
            preview = args_text[:200].replace("\n", "\\n")
            print(
                "  [Tool Parse Error] Unable to parse tool arguments after sanitization. "
                f"Preview: {preview}"
            )
            return None

        return {
            "name": tool_call.function.name,
            "arguments": arguments
        }
    return None


# ============================================================================
# TOOL BOOTSTRAPPING
# ============================================================================

def bootstrap_tools_with_llm(query: str) -> bool:
    """
    Ask LLM to create necessary tools for the query.
    Returns True if tools were created successfully.
    """
    print("\n=== BOOTSTRAPPING PHASE ===")
    print("Asking LLM to create necessary tools...")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool creator. Analyze the user's query and create any necessary tools "
                "that don't exist yet. You have access to create_tool function.\n\n"
                "For web searches, you should create a tool that:\n"
                "1. Uses DuckDuckGo's HTML search (NOT the Instant Answer API)\n"
                "2. Parses the HTML results using BeautifulSoup\n"
                "3. Returns a structured dictionary with results\n\n"
                "You can inspect existing tools with inspect_tool and refine them in-place with "
                "update_tool to iteratively improve their behavior.\n\n"
                "Example tool code structure:\n"
                "```python\n"
                "def search_web(query: str):\n"
                "    # Your implementation here\n"
                "    return {'query': query, 'results': [...]}\n"
                "```\n\n"
                f"Available static tools: {list(STATIC_TOOLS.keys())}\n"
                f"Currently available dynamic tools: {list(DYNAMIC_TOOLS.keys())}\n\n"
                "Create any missing tools needed to answer the query."
            )
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nCreate the necessary tools to answer this query."
        }
    ]
    
    # Allow multiple tool creations
    tools_created = 0
    max_bootstrap_calls = 5
    
    for i in range(max_bootstrap_calls):
        response = call_llm(messages, TOOL_SCHEMAS[:2], tool_choice="auto")
        tool_call = extract_tool_call(response)
        
        if not tool_call:
            print(f"  No more tools to create (iteration {i+1})")
            break
        
        if tool_call["name"] == "create_tool":
            args = tool_call["arguments"]
            result = create_tool(
                tool_name=args["tool_name"],
                tool_code=args["tool_code"],
                description=args["description"]
            )
            
            if result["success"]:
                tools_created += 1
                # Add the result to conversation history
                messages.append({
                    "role": "assistant",
                    "content": f"Created tool: {args['tool_name']}"
                })
                messages.append({
                    "role": "user",
                    "content": "Are there any other tools needed? If not, respond with 'done'."
                })
            else:
                print(f"  Failed to create tool: {result.get('error')}")
                break
        elif tool_call["name"] == "list_tools":
            result = list_tools()
            print(f"  Available tools: {result}")
            break
        else:
            break
    
    print(f"Bootstrap complete. Created {tools_created} new tools.")
    print(f"Total available tools: {len(get_all_tools())}")
    print("=" * 60 + "\n")
    
    return tools_created > 0


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def hopf_lens_dc(query: str, api_key: str, time_budget_ms: int = TIME_BUDGET_MS) -> Dict[str, Any]:
    """
    HOPF_LENS_DC orchestrator with dynamic tool creation and self-correction loop

    Args:
        query: The query to process
        api_key: OpenAI API key
        time_budget_ms: Time budget in milliseconds

    Returns:
        Dictionary containing answer, evidence, confidence, and metadata
    """
    if not api_key:
        raise ValueError("OpenAI API key is required")

    # Initialize OpenAI
    openai.api_key = api_key
    
    # Bootstrap: Let LLM create necessary tools
    bootstrap_tools_with_llm(query)
    
    # State
    P = []  # plan (list of Tasks)
    R = {}  # results
    a = ""  # answer
    w = {}  # evidence
    c = 0.0  # confidence
    t = 0  # iteration
    start_time = time.time()
    
    print(f"Starting HOPF_LENS_DC for query: {query}\n")
    
    while budget_remaining(time_budget_ms, start_time) and t < T_MAX:
        print(f"=== Iteration {t + 1} ===")
        
        # Get current tool list
        available_tools = get_all_tools()
        
        # ===== STEP 1: DECOMPOSE =====
        print("Step 1: Decomposing request...")
        messages = [
            {
                "role": "system",
                "content": (
                "You are a planner. Decompose the query into atomic executable tasks. "
                "Be minimal and efficient. Use the available tools. "
                f"Available tools: {', '.join(sorted(available_tools.keys()))}. "
                "If a dynamic tool underperforms, schedule inspect_tool to review its debug data "
                "and update_tool to refine its implementation. "
                "Always include required arguments."
            )
        },
            {
                "role": "user",
                "content": json.dumps({
                    "query": query,
                    "current_answer": a if a else None,
                    "evidence": w if w else None,
                    "available_tools": list(available_tools.keys())
                })
            }
        ]
        
        response = call_llm(messages, [TOOL_SCHEMAS[2]], tool_choice="auto")
        tool_call = extract_tool_call(response)
        
        added_tasks = False
        if tool_call and tool_call["name"] == "decompose_request":
            plan_data = tool_call["arguments"]["plan"]
            new_tasks = []
            for task_dict in plan_data:
                task_name = task_dict.get("task")
                if not task_name:
                    continue
                if task_name not in available_tools:
                    print(f"  Planner suggested unknown tool '{task_name}'; skipping")
                    continue
                task_args = task_dict.get("args", {})
                if not isinstance(task_args, dict):
                    continue
                priority = task_dict.get("priority", 0.5)
                new_tasks.append(Task(task=task_name, args=task_args, priority=priority))
            
            if new_tasks:
                added_tasks = True
                P = normalize_plan(P + new_tasks)
                print(f"  Added {len(new_tasks)} tasks. Total plan size: {len(P)}")
        
        # Fallback: if no tasks planned and we have search_web tool, use it
        if not added_tasks and not P and "search_web" in available_tools:
            fallback_task = Task(task="search_web", args={"query": query}, priority=0.6)
            P = normalize_plan(P + [fallback_task])
            print("  Planner fallback added 1 heuristic search task")
        
        # ===== STEP 2: EXECUTE =====
        print("Step 2: Executing tasks...")
        batch = P[:K_EXEC]
        executed_count = 0
        for task in batch:
            if task.task in available_tools:
                print(f"  Executing: {task.task}({json.dumps(task.args)[:100]}...)")
                result = None
                success = False
                try:
                    result = available_tools[task.task](**task.args)
                    success = True
                except Exception as e:
                    print(f"  Execution error: {e}")
                    if _should_attempt_tool_repair(task.task, e):
                        if _repair_search_web_tool(str(e)):
                            available_tools = get_all_tools()
                            print(f"  Re-executing {task.task} after repair...")
                            try:
                                result = available_tools[task.task](**task.args)
                                success = True
                            except Exception as post_error:
                                print(f"  Post-repair execution error: {post_error}")
                        else:
                            print("  Automatic repair attempt failed to update the tool.")
                    else:
                        print(f"  No repair attempt registered for '{task.task}'.")

                if success:
                    if task.task in DYNAMIC_TOOLS:
                        debug_snapshot = TOOL_DEBUG_DATA.get(task.task, {})
                        if isinstance(result, dict):
                            debug_summary = {
                                "parsed_entry_count": debug_snapshot.get("parsed_entry_count"),
                                "raw_html_char_length": debug_snapshot.get("raw_html_char_length"),
                                "last_http_call": debug_snapshot.get("last_http_call"),
                            }
                            raw_html_preview = (debug_snapshot.get("last_raw_html") or "")[:500]
                            if raw_html_preview:
                                debug_summary["raw_html_preview"] = raw_html_preview
                            result["_debug"] = debug_summary
                            if task.task in DYNAMIC_TOOL_SOURCES:
                                result["_tool_source_preview"] = DYNAMIC_TOOL_SOURCES[task.task][:500]
                    result_key = f"{task.task}_{len(R)}"
                    R[result_key] = result
                    print(f"  Stored result in key: {result_key}")
                    executed_count += 1
                P.remove(task)
            else:
                print(f"  Skipping unknown tool '{task.task}'")
                P.remove(task)
        
        print(f"  Executed {executed_count} tasks, {len(R)} total results stored")
        
        # ===== STEP 3: COMPOSE =====
        print("Step 3: Composing response...")
        
        # Prepare results summary for LLM
        results_summary = {}
        for key, value in R.items():
            if isinstance(value, dict):
                if "results" in value:
                    entries = []
                    for r in value.get("results", [])[:5]:
                        if not isinstance(r, dict):
                            continue
                        title = (r.get("title") or "")[:100]
                        snippet = (r.get("snippet") or "")[:200]
                        url = r.get("url", "")
                        entries.append({
                            "title": title,
                            "snippet": snippet,
                            "url": url
                        })
                    results_summary[key] = {
                        "type": "search_results",
                        "query": value.get("query", ""),
                        "results": entries
                    }
                else:
                    results_summary[key] = value
            else:
                results_summary[key] = value
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Synthesize a final answer from the task results. "
                    "Extract relevant information and provide a clear, direct answer. "
                    "Include evidence and confidence score."
                )
            },
            {
                "role": "user",
                "content": json.dumps({
                    "query": query,
                    "results": results_summary
                })
            }
        ]
        
        response = call_llm(messages, [TOOL_SCHEMAS[3]], tool_choice="auto")
        tool_call = extract_tool_call(response)
        
        a_new, w_new, c_new = a, w, c
        compose_succeeded = False
        
        if tool_call and tool_call["name"] == "compose_response":
            args = tool_call["arguments"]
            a_new = args.get("answer", "")
            w_new = args.get("evidence", {})
            c_new = args.get("confidence", 0.0)
            
            if a_new and str(a_new).strip() and c_new > 0.1:
                compose_succeeded = True
                print(f"  New answer: {a_new[:150]}...")
                print(f"  Confidence: {c_new}")
            else:
                print(f"  LLM compose returned low-quality answer (conf={c_new})")
        
        # Apply fallback if compose failed
        if not compose_succeeded:
            print("  Compose step failed, trying fallback summary...")
            fallback = summarize_search_results(R)
            if fallback:
                a_new = fallback["answer"]
                w_new = evidence_oplus(w_new, fallback["evidence"])
                c_new = max(c_new, fallback["confidence"])
                print(f"  Fallback answer: {a_new[:150]}...")
                print(f"  Fallback confidence: {c_new}")
            else:
                print("  Fallback also failed")
        
        # ===== STEP 4: ANTIPODE =====
        if c_new > 0.3:
            print("Step 4: Generating counterfactual attacks...")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Generate adversarial tasks to test the robustness of the answer. "
                        "Create tasks that would find contradictory information or expose weaknesses."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "answer": a_new,
                        "evidence": w_new,
                        "available_tools": list(available_tools.keys()),
                        "k": K_ATTACK
                    })
                }
            ]
            
            response = call_llm(messages, [TOOL_SCHEMAS[4]], tool_choice="auto")
            tool_call = extract_tool_call(response)
            
            if tool_call and tool_call["name"] == "antipode_counterfactual":
                attack_plan = tool_call["arguments"].get("attack_plan", [])
                attack_tasks = []
                for ap in attack_plan:
                    task_name = ap.get("task")
                    if not task_name or task_name not in available_tools:
                        continue
                    task_args = ap.get("args", {})
                    if not isinstance(task_args, dict):
                        continue
                    priority = ap.get("priority", 0.8)
                    attack_tasks.append(Task(task=task_name, args=task_args, priority=priority))
                
                if attack_tasks:
                    P = normalize_plan(P + attack_tasks)
                    print(f"  Added {len(attack_tasks)} attack tasks")
        else:
            print("Step 4: Skipping attacks (low confidence answer)")
        
        # ===== STEP 5: PUTBACK (if needed) =====
        policy = {"max_length": 5000}
        if violates_policy(a_new, policy):
            print("Step 5: Repairing plan due to policy violation...")
            messages = [
                {
                    "role": "system",
                    "content": "Repair the plan to satisfy constraints."
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "previous_answer": a,
                        "new_answer": a_new,
                        "plan": [{"task": task.task, "args": task.args} for task in P],
                        "policy": policy
                    })
                }
            ]
            
            response = call_llm(messages, [TOOL_SCHEMAS[5]], tool_choice="auto")
            tool_call = extract_tool_call(response)
            
            if tool_call and tool_call["name"] == "putback_repair":
                repaired = tool_call["arguments"]["repaired_plan"]
                P = normalize_plan([Task(**rp) for rp in repaired])
        
        # ===== STEP 6: CONVERGENCE =====
        print("Step 6: Checking convergence...")
        da = semantic_drift(a, a_new)
        dc = c_new - c
        nu = evidence_nu(w_new)
        
        print(f"  Drift: {da:.4f}, ΔConfidence: {dc:.4f}, Fragility: {nu:.4f}")
        
        # Accept step
        a = a_new
        w = evidence_oplus(w, w_new)
        c = max(c, c_new)
        t += 1
        
        # Check convergence
        if not P and da <= TAU_A and dc <= TAU_C and nu <= TAU_NU and c > 0.3:
            print("  CONVERGED!")
            break
        
        # Early exit if we have a good answer and no more tasks
        if not P and c > 0.5:
            print("  Good answer achieved, no more tasks!")
            break
        
        print()
    
    elapsed = time.time() - start_time
    print(f"\n=== FINAL RESULT (after {t} iterations, {elapsed:.2f}s) ===")
    print(f"Answer: {a}")
    print(f"Confidence: {c:.3f}")
    print(f"Evidence claims: {len(w.get('claims', []))}")
    print(f"Evidence sources: {len(w.get('sources', []))}")
    
    return {
        "answer": a,
        "evidence": w,
        "confidence": c,
        "steps": t,
        "results": R,
        "elapsed_seconds": elapsed,
        "dynamic_tools_created": list(DYNAMIC_TOOLS.keys())
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HOPF_LENS_DC: Dynamic LLM Orchestrator"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Who is Faruk Alpay?",
        help="Query to process"
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=TIME_BUDGET_MS,
        help="Time budget in milliseconds"
    )

    args = parser.parse_args()

    print(f"Testing HOPF_LENS_DC system with dynamic tool creation\n")
    print("=" * 80)

    result = hopf_lens_dc(args.query, args.api_key, args.time_budget)

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print(f"Total steps: {result['steps']}")
    print(f"Time taken: {result['elapsed_seconds']:.2f}s")
    print(f"Dynamic tools created: {result['dynamic_tools_created']}")
    print(f"\nFinal Answer: {result['answer']}")
