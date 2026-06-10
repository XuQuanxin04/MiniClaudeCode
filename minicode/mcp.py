from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from minicode.tooling import ToolDefinition, ToolResult

# 安全常量：禁止在命令参数中出现的危险字符
DANGEROUS_SHELL_CHARS = set('|&;`$(){}<>\n\r')

# MCP payload 大小上限（防止恶意服务端制造 OOM）
MAX_MCP_PAYLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# 允许的命令白名单（常见的 MCP 服务器命令）
ALLOWED_COMMANDS = {
    'node', 'npm', 'npx', 'python', 'python3', 'pip', 'pip3',
    'uv', 'deno', 'bun', 'cargo', 'go', 'java', 'javac',
    'ruby', 'gem', 'dotnet', 'curl', 'wget',
}


JsonRpcProtocol = str


@dataclass(slots=True)
class McpServerSummary:
    name: str
    command: str
    status: str
    toolCount: int
    error: str | None = None
    protocol: str | None = None
    resourceCount: int | None = None
    promptCount: int | None = None


def _sanitize_tool_segment(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    normalized = normalized.strip("_")
    return normalized or "tool"


def _validate_mcp_command(command: str) -> None:
    """验证 MCP 命令的合法性"""
    from pathlib import Path
    
    # Step 1: 先规范化命令路径，后面的白名单和路径检查都基于同一种表示。
    normalized = Path(command).resolve().as_posix()
    
    # Step 2: 禁止路径遍历，避免配置把 MCP 进程指向不可预期的位置。
    if '..' in normalized or '~' in normalized:
        raise RuntimeError("Invalid MCP command: contains path traversal characters")
    
    # Step 3: 白名单按命令名判断，Windows 下还要去掉 .exe 后缀。
    base_command = Path(command).name.lower()
    if base_command.endswith('.exe'):
        base_command = base_command[:-4]
    
    # Step 4: 绝对路径允许系统目录里的常见运行时，但禁止随便指向任意可执行文件。
    if Path(command).is_absolute():
        home_posix = str(Path.home().as_posix())
        allowed_system_dirs = [
            '/usr/bin', '/usr/local/bin', '/usr/local/sbin', '/usr/sbin', '/opt',
            # macOS Homebrew
            '/opt/homebrew/bin', '/opt/homebrew/sbin',  # Apple Silicon
            '/usr/local/Cellar',  # Intel
            # Linux extras
            '/snap/bin',  # Ubuntu Snap
            '/home/linuxbrew/.linuxbrew/bin',  # Homebrew on Linux
            # User-level tool directories (pip --user, pipx, cargo, nvm, etc.)
            f'{home_posix}/.local/bin',
            f'{home_posix}/.cargo/bin',
            f'{home_posix}/.nvm',
        ]
        if os.name == 'nt':
            allowed_system_dirs.extend([
                'C:\\Program Files',
                'C:\\Program Files (x86)',
                'C:\\Windows\\System32',
            ])
        
        is_in_allowed_dir = any(normalized.lower().startswith(d.lower()) for d in allowed_system_dirs)
        
        # Step 5: 不在安全目录且不在白名单，就拒绝启动，降低恶意 MCP 配置风险。
        if not is_in_allowed_dir and base_command not in ALLOWED_COMMANDS:
            raise RuntimeError(
                f"MCP command \"{command}\" is not in the allowed list. "
                f"Use a whitelisted command or place the executable in a standard system directory."
            )
        
        # Step 6: 即使路径合法，也不允许直接把系统 shell 当 MCP server 启动。
        dangerous_shells = ['cmd.exe', 'command.com', 'powershell.exe', 'pwsh.exe']
        if any(normalized.lower().endswith(d) for d in dangerous_shells):
            raise RuntimeError(
                f"MCP command \"{command}\" is a dangerous system shell. "
                f"Direct execution of shells is not allowed for security reasons."
            )
        return
    
    # Step 7: 相对命令必须来自白名单，防止当前目录下同名可执行文件被误启动。
    if base_command not in ALLOWED_COMMANDS:
        raise RuntimeError(
            f"MCP command \"{command}\" is not in the allowed list. "
            f"Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}. "
            f"Use absolute paths for custom commands."
        )


def _validate_mcp_args(args: list[str]) -> None:
    """验证 MCP 参数不包含危险的 shell 元字符"""
    for arg in args:
        for char in arg:
            # Step 1: MCP 启动不用 shell 拼接，所以参数里出现管道/重定向等元字符一律视为风险。
            if char in DANGEROUS_SHELL_CHARS:
                raise RuntimeError(
                    f"Invalid MCP argument: contains dangerous shell character '{char}'. "
                    f"MCP server arguments cannot contain shell metacharacters for security reasons."
                )


def _normalize_input_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    return schema if isinstance(schema, dict) else {"type": "object", "additionalProperties": True}


def _format_content_block(block: Any) -> str:
    if not isinstance(block, dict):
        return json.dumps(block, indent=2, ensure_ascii=False)
    if block.get("type") == "text" and "text" in block:
        return str(block["text"])
    return json.dumps(block, indent=2, ensure_ascii=False)


def _format_tool_call_result(result: Any) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=True, output=json.dumps(result, indent=2, ensure_ascii=False))
    parts: list[str] = []
    content = result.get("content")
    if isinstance(content, list) and content:
        parts.append("\n\n".join(_format_content_block(block) for block in content))
    if "structuredContent" in result:
        parts.append("STRUCTURED_CONTENT:\n" + json.dumps(result["structuredContent"], indent=2, ensure_ascii=False))
    if not parts:
        parts.append(json.dumps(result, indent=2, ensure_ascii=False))
    return ToolResult(ok=not bool(result.get("isError")), output="\n\n".join(parts).strip())


def _format_read_resource_result(result: Any) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=False, output=json.dumps(result, indent=2, ensure_ascii=False))
    contents = result.get("contents", [])
    if not contents:
        return ToolResult(ok=True, output="No resource contents returned.")
    rendered = []
    for item in contents:
        header_lines = [f"URI: {item.get('uri', '(unknown)')}"]
        if item.get("mimeType"):
            header_lines.append(f"MIME: {item['mimeType']}")
        header = "\n".join(header_lines) + "\n\n"
        if isinstance(item.get("text"), str):
            rendered.append(header + item["text"])
        elif isinstance(item.get("blob"), str):
            rendered.append(header + "BLOB:\n" + item["blob"])
        else:
            rendered.append(header + json.dumps(item, indent=2, ensure_ascii=False))
    return ToolResult(ok=True, output="\n\n".join(rendered))


def _format_prompt_result(result: Any) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=False, output=json.dumps(result, indent=2, ensure_ascii=False))
    header = f"DESCRIPTION: {result['description']}\n\n" if result.get("description") else ""
    body_parts = []
    for message in result.get("messages", []):
        role = message.get("role", "unknown")
        content = message.get("content")
        if isinstance(content, str):
            rendered = content
        elif isinstance(content, list):
            rendered = "\n".join(
                str(part["text"]) if isinstance(part, dict) and "text" in part else json.dumps(part, indent=2, ensure_ascii=False)
                for part in content
            )
        else:
            rendered = json.dumps(content, indent=2, ensure_ascii=False)
        body_parts.append(f"[{role}]\n{rendered}")
    output = (header + "\n\n".join(body_parts)).strip()
    return ToolResult(ok=True, output=output or json.dumps(result, indent=2, ensure_ascii=False))


class StdioMcpClient:
    """MCP client with lazy initialization.
    
    The server process is not started until the first request is made,
    reducing startup time and resource usage when MCP servers are configured
    but not immediately needed.
    """
    def __init__(self, server_name: str, config: dict[str, Any], cwd: str) -> None:
        self.server_name = server_name
        self.config = config
        self.cwd = cwd
        self.process: subprocess.Popen[bytes] | None = None
        self.protocol: JsonRpcProtocol | None = None
        self.next_id = 1
        self._pending: dict[int, Queue[Any]] = {}
        self._lock = threading.Lock()
        self.stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        # Lazy init state
        self._started = False
        self._start_error: str | None = None
        self._tools_cache: list[dict[str, Any]] | None = None
        self._resources_cache: list[dict[str, Any]] | None = None
        self._prompts_cache: list[dict[str, Any]] | None = None
    
    @property
    def is_started(self) -> bool:
        return self._started
    
    @property
    def start_error(self) -> str | None:
        return self._start_error

    def _protocol_candidates(self) -> list[JsonRpcProtocol]:
        configured = self.config.get("protocol")
        if configured == "content-length":
            return ["content-length"]
        if configured == "newline-json":
            return ["newline-json"]
        return ["content-length", "newline-json"]

    def start(self) -> None:
        """Start the MCP server process (idempotent).
        
        If already started, returns immediately.
        If previously failed, retries the connection.
        """
        if self._started:
            # Step 1: start 是幂等的，已经启动过就不重复拉起子进程。
            return
        
        if self._start_error is not None and self.process is None:
            # Step 2: 上次失败后首次使用时允许重试，避免启动阶段的一次错误永久禁用服务。
            self._start_error = None
        
        last_error: Exception | None = None
        for protocol in self._protocol_candidates():
            try:
                # Step 3: 同一个 MCP server 可能支持 content-length 或 newline-json，两种协议逐个尝试。
                self._spawn_process()
                self.protocol = protocol
                # Step 4: initialize 是握手请求，成功才说明这个子进程真的是可通信的 MCP server。
                self.request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "mini-code", "version": "0.1.0"},
                    },
                    timeout_seconds=2.0,
                )
                self.notify("notifications/initialized", {})
                self._started = True
                self._start_error = None
                return
            except Exception as error:  # noqa: BLE001
                # Step 5: 当前协议失败就关闭进程，换下一个协议重新试，避免半连接残留。
                last_error = error
                self.close()
        
        self._start_error = str(last_error or f'Failed to connect MCP server "{self.server_name}".')
        raise RuntimeError(self._start_error)
    
    def _ensure_started(self) -> None:
        """Ensure the server is started before making a request."""
        if self._started and not self._is_process_alive():
            # Step 1: 子进程意外退出时清理状态，下次请求会重新 start。
            self.close()
        if not self._started:
            # Step 2: 懒启动发生在真正需要工具/资源/Prompt 的时刻。
            self.start()

    def _is_process_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _spawn_process(self) -> None:
        command = str(self.config.get("command", "")).strip()
        if not command:
            raise RuntimeError(f'MCP server "{self.server_name}" has no command configured.')

        # Step 1: 启动外部进程前先做命令和参数安全验证。
        _validate_mcp_command(command)
        _validate_mcp_args(list(self.config.get("args", []) or []))

        process_cwd = Path(self.cwd)
        if self.config.get("cwd"):
            # Step 2: MCP server 可以有自己的工作目录，但必须相对当前项目解析。
            process_cwd = (process_cwd / str(self.config["cwd"])).resolve()
        env = os.environ.copy()
        for key, value in dict(self.config.get("env", {}) or {}).items():
            # Step 3: 每个 server 单独注入环境变量，不污染整个 Python 进程环境。
            env[str(key)] = str(value)

        popen_kwargs: dict = {}
        if os.name == "nt":
            # Step 4: Windows 下隐藏子进程窗口，避免每个 MCP server 弹出控制台。
            CREATE_NO_WINDOW = 0x08000000
            popen_kwargs["creationflags"] = CREATE_NO_WINDOW
        try:
            self.process = subprocess.Popen(  # noqa: S603
                [command, *list(self.config.get("args", []) or [])],
                cwd=str(process_cwd),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **popen_kwargs,
            )
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {command}. Install it first and ensure it is available in PATH.") from None

        self.stderr_lines = []
        with self._lock:
            self._pending = {}
        # Step 5: stderr 单独消费，超时/启动失败时能把最近日志带给用户。
        self._stderr_thread = threading.Thread(target=self._consume_stderr, daemon=True)
        self._stderr_thread.start()

    def _consume_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            try:
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self.stderr_lines.append(text)
                    self.stderr_lines = self.stderr_lines[-8:]
            except Exception:
                continue

    def _ensure_stdout_thread(self) -> None:
        if self._stdout_thread is not None:
            return
        self._stdout_thread = threading.Thread(target=self._consume_stdout, daemon=True)
        self._stdout_thread.start()

    def _consume_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                line_bytes = self.process.stdout.readline()
                if not line_bytes:
                    break

                try:
                    line = line_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                stripped = line.strip()
                if not stripped:
                    continue

                if len(line_bytes) > MAX_MCP_PAYLOAD_BYTES:
                    self.stderr_lines.append(
                        f"MCP payload too large: {len(line_bytes)} bytes (limit {MAX_MCP_PAYLOAD_BYTES})"
                    )
                    continue

                # Auto-detect protocol if not determined yet
                if self.protocol is None:
                    # Step 1: 如果握手前没确定协议，就根据第一行输出自动判断。
                    if line.lower().startswith("content-length:"):
                        self.protocol = "content-length"
                    else:
                        self.protocol = "newline-json"

                if self.protocol == "newline-json":
                    try:
                        # Step 2: newline-json 模式是一行一个 JSON-RPC 消息。
                        self._handle_message(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
                else:
                    # Step 3: content-length 模式先读 header，再按长度读取 JSON body。
                    header_lines = [line.rstrip("\r\n")]
                    while True:
                        next_line_bytes = self.process.stdout.readline()
                        if not next_line_bytes:
                            return
                        try:
                            next_line = next_line_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            return
                        h_stripped = next_line.rstrip("\r\n")
                        if h_stripped == "":
                            break
                        header_lines.append(h_stripped)

                    content_length = 0
                    for header in header_lines:
                        if header.lower().startswith("content-length:"):
                            try:
                                content_length = int(header.split(":", 1)[1].strip())
                            except ValueError:
                                pass
                            break

                    if content_length > MAX_MCP_PAYLOAD_BYTES:
                        self.stderr_lines.append(
                            f"MCP payload too large: {content_length} bytes (limit {MAX_MCP_PAYLOAD_BYTES})"
                        )
                        continue

                    if content_length > 0:
                        body_bytes = self.process.stdout.read(content_length)
                        if len(body_bytes) < content_length:
                            return
                        try:
                            self._handle_message(json.loads(body_bytes.decode("utf-8")))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
        finally:
            # Step 4: 子进程退出时唤醒所有等待中的 request，避免调用方一直阻塞。
            if self.process:
                exit_code = self.process.poll()
                error_msg = {"error": {"code": -1, "message": f"MCP server process exited (code={exit_code})"}}
                with self._lock:
                    for req_id, q in list(self._pending.items()):
                        q.put(error_msg)
                    self._pending.clear()

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_id = message.get("id")
        if not isinstance(message_id, int):
            # Step 1: notification 没有 id，不对应 pending request，这里直接忽略。
            return
        with self._lock:
            # Step 2: 用 id 找回对应等待队列，把响应交还给 request()。
            queue = self._pending.pop(message_id, None)
            if queue is not None:
                queue.put(message)

    def send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f'MCP server "{self.server_name}" is not running.')
        
        payload_bytes = json.dumps(message, ensure_ascii=False).encode("utf-8")
        
        if self.protocol == "newline-json":
            # Step 1: newline-json 协议直接写一行 JSON。
            self.process.stdin.write(payload_bytes + b"\n")
            self.process.stdin.flush()
            self._ensure_stdout_thread()
            return
        
        # Step 2: content-length 协议必须先写长度头，再写 JSON body。
        header = f"Content-Length: {len(payload_bytes)}\r\n\r\n".encode("utf-8")
        self.process.stdin.write(header + payload_bytes)
        self.process.stdin.flush()
        self._ensure_stdout_thread()

    def notify(self, method: str, params: Any) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout_seconds: float = 5.0) -> Any:
        # Step 1: 每个 request 分配递增 id，响应回来时靠 id 找到等待者。
        message_id = self.next_id
        self.next_id += 1
        response_queue: Queue[Any] = Queue(maxsize=1)
        with self._lock:
            self._pending[message_id] = response_queue
        # Step 2: 发送后阻塞等待对应响应；stdout 线程收到响应后会 put 到这个队列。
        self.send({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params})
        try:
            message = response_queue.get(timeout=timeout_seconds)
        except Empty as error:
            # Step 3: 超时要清掉 pending，并把最近 stderr 附上，方便诊断 MCP server 为什么没回。
            with self._lock:
                self._pending.pop(message_id, None)
            stderr = "\n".join(self.stderr_lines)
            raise RuntimeError(
                f"MCP {self.server_name}: request timed out for {method}" + (f"\n{stderr}" if stderr else "")
            ) from error
        if message.get("error"):
            # Step 4: JSON-RPC error 转成 RuntimeError，由上层包装成工具失败结果。
            details = message["error"].get("data")
            suffix = f"\n{json.dumps(details, indent=2, ensure_ascii=False)}" if details else ""
            raise RuntimeError(f"MCP {self.server_name}: {message['error']['message']}{suffix}")
        return message.get("result")

    def list_tools(self) -> list[dict[str, Any]]:
        """List tools with caching. Starts server lazily if not started."""
        if self._tools_cache is not None:
            # Step 1: 工具描述通常不变，缓存能避免每次构建工具列表都请求 MCP server。
            return self._tools_cache
        self._ensure_started()
        result = self.request("tools/list", {})
        self._tools_cache = list(result.get("tools", []) if isinstance(result, dict) else [])
        return self._tools_cache

    def list_resources(self) -> list[dict[str, Any]]:
        """List resources with caching. Starts server lazily if not started."""
        if self._resources_cache is not None:
            return self._resources_cache
        self._ensure_started()
        result = self.request("resources/list", {}, timeout_seconds=3.0)
        self._resources_cache = list(result.get("resources", []) if isinstance(result, dict) else [])
        return self._resources_cache

    def read_resource(self, uri: str) -> ToolResult:
        self._ensure_started()
        return _format_read_resource_result(self.request("resources/read", {"uri": uri}, timeout_seconds=5.0))

    def list_prompts(self) -> list[dict[str, Any]]:
        """List prompts with caching. Starts server lazily if not started."""
        if self._prompts_cache is not None:
            return self._prompts_cache
        self._ensure_started()
        result = self.request("prompts/list", {}, timeout_seconds=3.0)
        self._prompts_cache = list(result.get("prompts", []) if isinstance(result, dict) else [])
        return self._prompts_cache

    def get_prompt(self, name: str, args: dict[str, str] | None = None) -> ToolResult:
        self._ensure_started()
        return _format_prompt_result(
            self.request("prompts/get", {"name": name, "arguments": args or {}}, timeout_seconds=5.0)
        )

    def call_tool(self, name: str, input_data: Any) -> ToolResult:
        # Step 1: MCP 工具真正调用前确保 server 存活；懒启动失败会自然变成工具错误。
        self._ensure_started()
        return _format_tool_call_result(self.request("tools/call", {"name": name, "arguments": input_data or {}}))

    def close(self) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
            for queue in pending:
                queue.put({"error": {"message": f'MCP server "{self.server_name}" closed before completing the request.'}})
        
        if self.process is not None:
            try:
                # 跨平台进程终止
                if os.name == "nt":
                    # Windows: 使用 taskkill 终止进程树
                    try:
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                            capture_output=True,
                            timeout=5
                        )
                    except subprocess.TimeoutExpired:
                        # taskkill 本身超时，强制 kill
                        try:
                            self.process.kill()
                        except OSError:
                            pass
                    except Exception:
                        try:
                            self.process.kill()
                        except OSError:
                            pass
                else:
                    # Unix: 先 SIGTERM，超时后 SIGKILL
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            self.process.kill()
                        except OSError:
                            pass

                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            except OSError:
                pass  # 进程可能已经退出
            finally:
                self.process = None
        
        self.protocol = None
        self._stdout_thread = None
        self._stderr_thread = None
        # Reset lazy init state
        self._started = False
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None


def create_mcp_backed_tools(*, cwd: str, mcp_servers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Create MCP-backed tools with lazy server initialization.
    
    Instead of starting all MCP servers at startup (which is slow and
    resource-intensive), this function creates lightweight client objects
    that defer server startup until the first tool call.
    
    Benefits:
    - Faster startup: no waiting for MCP server processes to initialize
    - Lower resource usage: servers that aren't needed never start
    - Resilience: a failed server doesn't block other servers
    - Auto-retry: servers are retried on first use after failure
    """
    clients: list[StdioMcpClient] = []
    tools: list[ToolDefinition] = []
    servers: list[dict[str, Any]] = []
    resource_index: dict[str, dict[str, Any]] = {}
    prompt_index: dict[str, dict[str, Any]] = {}

    for server_name, config in mcp_servers.items():
        if config.get("enabled") is False:
            # Step 1: disabled server 仍写入摘要，让用户知道它被配置了但没有启用。
            servers.append(asdict(McpServerSummary(name=server_name, command=config.get("command", ""), status="disabled", toolCount=0, protocol=config.get("protocol"))))
            continue

        # Step 2: 为每个 server 建一个懒启动 client；此时不一定真的已经启动进程。
        client = StdioMcpClient(server_name, config, cwd)
        clients.append(client)
        
        # Step 3: 先登记 pending 状态，Prompt 可以展示“有这个 MCP server”。
        servers.append(
            asdict(
                McpServerSummary(
                    name=server_name,
                    command=config.get("command", ""),
                    status="pending",
                    toolCount=0,
                    protocol=config.get("protocol"),
                )
            )
        )
        
        # Step 4: 尝试发现工具/资源/Prompt；失败不阻塞整个 Agent，只把 server 状态标成 error。
        try:
            descriptors = client.list_tools()
            try:
                resources = client.list_resources()
            except Exception:  # noqa: BLE001
                resources = []
            try:
                prompts = client.list_prompts()
            except Exception:  # noqa: BLE001
                prompts = []

            for resource in resources:
                resource_index[f"{server_name}:{resource.get('uri')}"] = {"serverName": server_name, "resource": resource}
            for prompt in prompts:
                prompt_index[f"{server_name}:{prompt.get('name')}"] = {"serverName": server_name, "prompt": prompt}

            for descriptor in descriptors:
                # Step 5: MCP 工具名包成 mcp__server__tool，避免不同 server 的同名工具冲突。
                wrapped_name = f"mcp__{_sanitize_tool_segment(server_name)}__{_sanitize_tool_segment(str(descriptor.get('name', 'tool')))}"
                descriptor_name = str(descriptor.get("name", "tool"))
                input_schema = _normalize_input_schema(descriptor.get("inputSchema"))

                def _validator(value: Any) -> Any:
                    # Step 6: MCP server 自带 schema，这里先透传参数，具体校验交给 server。
                    return value

                def _run(input_data: Any, _context, *, _client=client, _descriptor_name=descriptor_name):
                    # Step 7: ToolRegistry 调用这个包装函数时，实际会转成 MCP tools/call 请求。
                    return _client.call_tool(_descriptor_name, input_data)

                tools.append(
                    ToolDefinition(
                        name=wrapped_name,
                        description=str(descriptor.get("description") or f"Call MCP tool {descriptor_name} from server {server_name}."),
                        input_schema=input_schema,
                        validator=_validator,
                        run=_run,
                    )
                )

            # Step 8: 发现成功后更新 server 摘要，system prompt 会显示工具/资源/Prompt 数量。
            for i, s in enumerate(servers):
                if s["name"] == server_name:
                    servers[i] = asdict(
                        McpServerSummary(
                            name=server_name,
                            command=config.get("command", ""),
                            status="connected",
                            toolCount=len(descriptors),
                            protocol=client.protocol,
                            resourceCount=len(resources),
                            promptCount=len(prompts),
                        )
                    )
                    break
        except Exception as error:  # noqa: BLE001
            # Step 9: MCP 失败只影响该 server，不影响本地文件工具和其他 MCP server。
            for i, s in enumerate(servers):
                if s["name"] == server_name:
                    servers[i] = asdict(
                        McpServerSummary(
                            name=server_name,
                            command=config.get("command", ""),
                            status="error",
                            toolCount=0,
                            error=str(error)[:200],
                            protocol=config.get("protocol"),
                        )
                    )
                    break

    if resource_index:
        tools.append(
            ToolDefinition(
                name="list_mcp_resources",
                description="List available MCP resources exposed by connected MCP servers.",
                input_schema={"type": "object", "properties": {"server": {"type": "string"}}},
                validator=lambda value: {"server": value.get("server")} if isinstance(value, dict) else {"server": None},
                run=lambda input_data, _context: ToolResult(
                    ok=True,
                    output="\n".join(
                        f"{entry['serverName']}: {entry['resource'].get('uri')}"
                        + (f" ({entry['resource'].get('name')})" if entry["resource"].get("name") else "")
                        + (f" - {entry['resource'].get('description')}" if entry["resource"].get("description") else "")
                        for entry in resource_index.values()
                        if not input_data.get("server") or entry["serverName"] == input_data["server"]
                    )
                    or "No MCP resources available.",
                ),
            )
        )

        def _read_resource(input_data: dict, _context) -> ToolResult:
            # Step 10: resource 工具按 server 名找到对应 client，再转发 resources/read。
            client = next((item for item in clients if item.server_name == input_data["server"]), None)
            if client is None:
                return ToolResult(ok=False, output=f"Unknown MCP server: {input_data['server']}")
            return client.read_resource(input_data["uri"])

        tools.append(
            ToolDefinition(
                name="read_mcp_resource",
                description="Read a specific MCP resource by server and URI.",
                input_schema={"type": "object", "properties": {"server": {"type": "string"}, "uri": {"type": "string"}}, "required": ["server", "uri"]},
                validator=lambda value: value,
                run=_read_resource,
            )
        )

    if prompt_index:
        tools.append(
            ToolDefinition(
                name="list_mcp_prompts",
                description="List available MCP prompts exposed by connected MCP servers.",
                input_schema={"type": "object", "properties": {"server": {"type": "string"}}},
                validator=lambda value: {"server": value.get("server")} if isinstance(value, dict) else {"server": None},
                run=lambda input_data, _context: ToolResult(
                    ok=True,
                    output="\n".join(
                        f"{entry['serverName']}: {entry['prompt'].get('name')}"
                        + (
                            " args=["
                            + ", ".join(
                                f"{arg.get('name')}{'*' if arg.get('required') else ''}"
                                for arg in entry["prompt"].get("arguments", [])
                            )
                            + "]"
                            if entry["prompt"].get("arguments")
                            else ""
                        )
                        + (f" - {entry['prompt'].get('description')}" if entry["prompt"].get("description") else "")
                        for entry in prompt_index.values()
                        if not input_data.get("server") or entry["serverName"] == input_data["server"]
                    )
                    or "No MCP prompts available.",
                ),
            )
        )

        def _get_prompt(input_data: dict, _context) -> ToolResult:
            # Step 11: prompt 工具同样是包装层，把 MiniCode 工具调用转成 MCP prompts/get。
            client = next((item for item in clients if item.server_name == input_data["server"]), None)
            if client is None:
                return ToolResult(ok=False, output=f"Unknown MCP server: {input_data['server']}")
            return client.get_prompt(input_data["name"], input_data.get("arguments"))

        tools.append(
            ToolDefinition(
                name="get_mcp_prompt",
                description="Fetch a rendered MCP prompt by server, prompt name, and optional arguments.",
                input_schema={"type": "object", "properties": {"server": {"type": "string"}, "name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["server", "name"]},
                validator=lambda value: value,
                run=_get_prompt,
            )
        )

    return {
        # Step 12: 返回值被 tools/__init__.py 合并进 ToolRegistry；dispose 用于程序退出时关闭子进程。
        "tools": tools,
        "servers": servers,
        "dispose": lambda: [client.close() for client in clients],
    }
