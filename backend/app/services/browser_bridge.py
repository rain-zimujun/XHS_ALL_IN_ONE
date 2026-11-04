"""Drive a real, logged-in browser for XHS creator signing.

Xiaohongshu signs creator API calls with a JS algorithm that changes with every
web build. Re-implementing it locally rots quickly, so this module keeps a real
browser profile (one per account) and lets the page itself do the signing:

* the browser owns the login state (the user scans a QR once per profile);
* requests are issued from the page context, so the site's own request layer
  attaches valid ``x-s`` / ``x-s-common`` / ``x-t`` headers;
* when the page context cannot sign a bare ``fetch`` the UI path is used instead
  (type into the composer, read the response the page itself produced).

Everything here is best-effort: callers must keep a plain-text fallback.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

CREATOR_ORIGIN = "https://creator.xiaohongshu.com"
PUBLISH_URL = f"{CREATOR_ORIGIN}/publish/publish?source=official"
TOPIC_API = "/api/galaxy/v2/creator/servicegw/v2/search/topics"

_BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


class BrowserUnavailable(RuntimeError):
    """Raised when the signing browser cannot be started or is not logged in."""


def find_browser() -> str:
    for path in _BROWSER_CANDIDATES:
        if os.path.exists(path):
            return path
    raise BrowserUnavailable("没有找到 Edge/Chrome，无法启动签名浏览器")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class CdpTarget:
    id: str
    ws_url: str
    url: str


class Cdp:
    """Minimal CDP client over the JSON endpoint + websocket."""

    def __init__(self, port: int) -> None:
        self.port = port

    def _http_json(self, path: str) -> Any:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    def version(self) -> dict:
        return self._http_json("/json/version")

    def targets(self) -> list[CdpTarget]:
        items = self._http_json("/json/list")
        return [CdpTarget(item["id"], item["webSocketDebuggerUrl"], item.get("url", "")) for item in items if item.get("type") == "page"]

    def page(self) -> Optional[CdpTarget]:
        pages = [target for target in self.targets() if not target.url.startswith("edge://") and not target.url.startswith("devtools://")]
        for target in pages:
            if "creator.xiaohongshu.com" in target.url:
                return target
        if pages:
            return pages[0]
        all_pages = self.targets()
        return all_pages[0] if all_pages else None

    def call(self, ws_url: str, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
        from websockets.sync.client import connect

        with connect(ws_url, open_timeout=5, close_timeout=1, max_size=64 * 1024 * 1024) as socket_:
            socket_.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
            deadline = time.time() + timeout
            while time.time() < deadline:
                message = json.loads(socket_.recv(timeout=timeout))
                if message.get("id") == 1:
                    if "error" in message:
                        raise RuntimeError(f"CDP {method} failed: {message['error']}")
                    return message.get("result") or {}
        raise TimeoutError(f"CDP {method} timed out")

    def capture_typing(self, ws_url: str, url_fragment: str, focus_js: str, text: str, timeout: float = 25.0) -> list[dict]:
        """Focus an element, type real keystrokes, and collect matching response bodies."""
        from websockets.sync.client import connect

        with connect(ws_url, open_timeout=5, close_timeout=1, max_size=64 * 1024 * 1024) as socket_:
            next_id = 0

            def send(method: str, params: Optional[dict] = None) -> int:
                nonlocal next_id
                next_id += 1
                socket_.send(json.dumps({"id": next_id, "method": method, "params": params or {}}))
                return next_id

            send("Network.enable")
            send("Runtime.enable")
            send("Runtime.evaluate", {"expression": focus_js, "awaitPromise": True, "returnByValue": True})
            time.sleep(0.4)
            # Type character by character: the composer only queries the topic API
            # with the text typed so far, so a single bulk insert queries just "#".
            for index in range(len(text)):
                send("Input.insertText", {"text": text[index]})
                time.sleep(0.08)
            deadline = time.time() + timeout
            request_ids: set[str] = set()
            collected: list[dict] = []
            settle_until = time.time() + 3.0
            while time.time() < deadline:
                if collected and time.time() > settle_until:
                    break
                try:
                    message = json.loads(socket_.recv(timeout=min(1.5, max(0.2, deadline - time.time()))))
                except Exception:
                    continue
                method = message.get("method")
                if method == "Network.requestWillBeSent":
                    params = message.get("params") or {}
                    if url_fragment in (params.get("request") or {}).get("url", ""):
                        request_ids.add(params.get("requestId", ""))
                elif method == "Network.loadingFinished":
                    params = message.get("params") or {}
                    if params.get("requestId") in request_ids:
                        body_id = send("Network.getResponseBody", {"requestId": params["requestId"]})
                        while True:
                            payload = json.loads(socket_.recv(timeout=5))
                            if payload.get("id") == body_id:
                                raw = (payload.get("result") or {}).get("body") or ""
                                try:
                                    collected.append(json.loads(raw))
                                except json.JSONDecodeError:
                                    collected.append({"_raw": raw})
                                settle_until = time.time() + 2.0
                                break
            return collected


class BrowserBridge:
    """One browser process per profile directory."""

    def __init__(self, profile_dir: Path, port: Optional[int] = None) -> None:
        # Edge ignores a relative --user-data-dir and falls back to the daily
        # profile, which then just forwards the URL to the running instance.
        self.profile_dir = Path(profile_dir).expanduser().resolve()
        self.port = port or self._default_port()
        self.process: Optional[subprocess.Popen] = None
        self.cdp = Cdp(self.port)

    def _default_port(self) -> int:
        """Deterministic port per profile so a restarted process can re-attach."""
        import hashlib

        digest = hashlib.sha1(str(self.profile_dir).lower().encode("utf-8")).hexdigest()
        return 20000 + int(digest[:4], 16) % 20000

    @property
    def active_port_file(self) -> Path:
        return self.profile_dir / "DevToolsActivePort"

    def discover_port(self) -> Optional[int]:
        """Read the devtools port Edge writes into the profile directory."""
        try:
            lines = self.active_port_file.read_text(encoding="utf-8").splitlines()
            return int(lines[0].strip())
        except (OSError, ValueError, IndexError):
            pass
        return self._port_from_processes()

    def _profile_processes(self) -> list[tuple[int, str]]:
        """(pid, command line) of Edge processes using this profile (Windows)."""
        script = (
            "Get-CimInstance Win32_Process -Filter \"name='msedge.exe'\" | "
            f"Where-Object {{ $_.CommandLine -like '*{self.profile_dir}*' }} | "
            "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"
        )
        try:
            output = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=25,
            ).stdout
        except Exception:
            return []
        rows = []
        for line in output.splitlines():
            pid_text, _, command = line.partition("|")
            if pid_text.strip().isdigit():
                rows.append((int(pid_text.strip()), command))
        return rows

    def _port_from_processes(self) -> Optional[int]:
        import re

        for _pid, command in self._profile_processes():
            match = re.search(r"--remote-debugging-port=(\d+)", command)
            if match and match.group(1) != "0":
                return int(match.group(1))
        return None

    def kill_profile_browsers(self) -> None:
        for pid, command in self._profile_processes():
            if "--type=" in command:
                continue
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20)
            except Exception:
                pass
        time.sleep(2)

    def attach(self) -> bool:
        candidates = [self.port, self.discover_port()]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                if Cdp(candidate).version():
                    self.port = candidate
                    self.cdp = Cdp(candidate)
                    return True
            except Exception:
                continue
        return False

    # -- lifecycle ---------------------------------------------------------
    def profile_in_use(self) -> bool:
        try:
            self.cdp.version()
            return True
        except Exception:
            return False

    def launch(self, visible: bool = False, url: str = "about:blank") -> None:
        if self.attach() or self.profile_in_use():
            self.process = None
            return
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.active_port_file.unlink()
        except OSError:
            pass
        log_path = self.profile_dir / "edge.log"
        last_error = ""
        for offset in range(5):
            port = self.port + offset
            args = [
                find_browser(),
                f"--user-data-dir={self.profile_dir}",
                f"--remote-debugging-port={port}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-features=Translate",
                "--window-size=1280,900",
                url,
            ]
            if not visible:
                # Off-screen instead of headless: automation detection is stricter on headless.
                args.insert(1, "--window-position=-2000,-2000")
            with open(log_path, "ab") as log:
                self.process = subprocess.Popen(args, stdout=log, stderr=log)
            deadline = time.time() + 25
            while time.time() < deadline:
                if self.attach():
                    return
                time.sleep(0.4)
                if self.process.poll() is not None:
                    last_error = f"进程退出码 {self.process.returncode}"
                    # A browser with this profile is already running but we could
                    # not attach; close it once and retry.
                    if self.process.returncode == 0 and self.attach():
                        return
                    break
            if self.process.poll() is None:
                last_error = "启动超时"
                break
            if offset == 0:
                self.kill_profile_browsers()
        raise BrowserUnavailable(f"签名浏览器启动失败（{last_error}），详见 {log_path}")

    def close(self) -> None:
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None

    def status(self) -> dict:
        info: dict[str, Any] = {"running": False, "logged_in": False, "port": self.port}
        try:
            self.cdp.version()
            info["running"] = True
        except Exception:
            info["reason"] = "浏览器未运行"
            return info
        info["logged_in"] = self.is_logged_in()
        if not info["logged_in"]:
            info["reason"] = "浏览器未登录"
        return info

    # -- page helpers ------------------------------------------------------
    def evaluate(self, expression: str, timeout: float = 60.0) -> Any:
        page = self.cdp.page()
        if page is None:
            raise BrowserUnavailable("浏览器没有可用页面")
        result = self.cdp.call(page.ws_url, "Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": True,
            "returnByValue": True,
        }, timeout=timeout)
        return result.get("result", {}).get("value")

    def navigate(self, url: str, wait: float = 2.0) -> None:
        page = self.cdp.page()
        if page is None:
            raise BrowserUnavailable("浏览器没有可用页面")
        self.cdp.call(page.ws_url, "Page.navigate", {"url": url})
        time.sleep(wait)

    def cookies(self) -> list[dict]:
        page = self.cdp.page()
        if page is None:
            return []
        result = self.cdp.call(page.ws_url, "Network.getAllCookies", {})
        return result.get("cookies") or []

    def is_logged_in(self) -> bool:
        names = {cookie.get("name") for cookie in self.cookies()}
        return {"galaxy_creator_session_id", "access-token-creator.xiaohongshu.com"} <= names

    def ensure_login(self, timeout: float = 180.0) -> bool:
        """Open the creator home page, letting the user scan when needed."""
        if not self.profile_in_use():
            self.launch(visible=True, url=f"{CREATOR_ORIGIN}/creator/home")
        if self.is_logged_in():
            return True
        self.navigate(f"{CREATOR_ORIGIN}/login", wait=1.5)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_logged_in():
                self.navigate(f"{CREATOR_ORIGIN}/creator/home", wait=2.0)
                return True
            time.sleep(2.0)
        return False

    # -- signing -----------------------------------------------------------
    def fetch_in_page(self, path: str, body: Optional[dict] = None, method: str = "POST") -> dict:
        """Issue the request from the page context so the site's layer signs it."""
        script = """
        (async () => {
          try {
            const response = await fetch(%s, {
              method: %s,
              credentials: 'include',
              headers: { 'content-type': 'application/json' },
              body: %s,
            });
            const text = await response.text();
            return { status: response.status, body: text };
          } catch (error) {
            return { status: 0, body: String(error) };
          }
        })()
        """ % (json.dumps(path), json.dumps(method), json.dumps(json.dumps(body or {}) if body is not None else "null"))
        return self.evaluate(script) or {"status": 0, "body": ""}

    def set_file_input(self, ws_url: str, selector: str, files: list[str], timeout: float = 30.0) -> bool:
        """Attach local files to a file input (DOM node ids are session-bound)."""
        from websockets.sync.client import connect

        with connect(ws_url, open_timeout=5, close_timeout=1, max_size=64 * 1024 * 1024) as socket_:
            next_id = 0

            def send(method: str, params: Optional[dict] = None) -> dict:
                nonlocal next_id
                next_id += 1
                socket_.send(json.dumps({"id": next_id, "method": method, "params": params or {}}))
                deadline = time.time() + timeout
                while time.time() < deadline:
                    message = json.loads(socket_.recv(timeout=timeout))
                    if message.get("id") == next_id:
                        if "error" in message:
                            raise RuntimeError(f"CDP {method} failed: {message['error']}")
                        return message.get("result") or {}
                raise TimeoutError(f"CDP {method} timed out")

            send("DOM.enable")
            document = send("DOM.getDocument", {"depth": 1})
            found = send("DOM.querySelector", {"nodeId": document["root"]["nodeId"], "selector": selector})
            if not found.get("nodeId"):
                return False
            send("DOM.setFileInputFiles", {"nodeId": found["nodeId"], "files": files})
            return True

    def visible_window(self) -> None:
        """Bring the window on-screen (used before asking the user to scan)."""
        try:
            page = self.cdp.page()
            if page is None:
                return
            self.cdp.call(page.ws_url, "Browser.setWindowBounds", {
                "windowId": self.cdp.call(page.ws_url, "Browser.getWindowForTarget", {"targetId": page.id}).get("windowId"),
                "bounds": {"left": 60, "top": 60, "width": 1280, "height": 900, "windowState": "normal"},
            })
        except Exception:
            pass

    # -- topic lookup ------------------------------------------------------
    _FOCUS_EDITOR_JS = """
    (() => {
      const editors = [...document.querySelectorAll('.tiptap.ProseMirror, [contenteditable="true"]')]
        .filter((el) => el.offsetParent !== null);
      const editor = editors[0];
      if (!editor) return null;
      editor.focus();
      const range = document.createRange();
      range.selectNodeContents(editor);
      range.collapse(false);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      return editor.className || editor.tagName;
    })()
    """

    _CLICK_IMAGE_TAB_JS = """
    (() => {
      const nodes = [...document.querySelectorAll('div,span,button,a')];
      const target = nodes.find((el) => el.childElementCount === 0 && el.textContent.trim() === '上传图文');
      if (!target) return false;
      target.click();
      return true;
    })()
    """

    _SAMPLE_PNG = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00"
        b"\x00IEND\xaeB`\x82"
    )

    def sample_image(self) -> str:
        path = self.profile_dir / "composer-sample.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._SAMPLE_PNG)
        return str(path)

    def _prepare_composer(self) -> bool:
        """Make sure the image-text composer (with at least one image) exists."""
        if self._wait_editor(4):
            return True
        self.evaluate(self._CLICK_IMAGE_TAB_JS)
        time.sleep(2.5)
        if self._wait_editor(5):
            return True
        page = self.cdp.page()
        if page is None:
            return False
        try:
            self.set_file_input(page.ws_url, "input.upload-input", [self.sample_image()])
        except Exception:
            return False
        return self._wait_editor(30)

    def _wait_editor(self, timeout: float = 20.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.evaluate(self._FOCUS_EDITOR_JS):
                return True
            time.sleep(1.0)
        return False

    def search_topics(self, keyword: str, timeout: Optional[float] = None) -> list[dict]:
        """Type a hashtag into the composer and harvest the topic suggestions."""
        keyword = (keyword or "").strip().lstrip("#")
        if not keyword:
            return []
        page = self.cdp.page()
        if page is None:
            raise BrowserUnavailable("浏览器没有可用页面")
        if not self.is_logged_in():
            raise BrowserUnavailable("签名浏览器未登录")
        if "publish" not in (page.url or "") or not self._prepare_composer():
            self.navigate(PUBLISH_URL, wait=4.0)
            page = self.cdp.page()
            if page is None:
                raise BrowserUnavailable("打开发布页失败")
            if not self._prepare_composer():
                raise BrowserUnavailable("没有找到发布页的正文编辑器")
        responses = self.cdp.capture_typing(page.ws_url, TOPIC_API, self._FOCUS_EDITOR_JS, f"#{keyword}", timeout=timeout or 25.0)
        # Leave the composer clean so the next lookup starts from an empty state.
        self.evaluate("(() => { document.execCommand('selectAll'); document.execCommand('delete'); return true })()")
        topics: list[dict] = []
        seen: set[str] = set()
        for response in responses:
            if not isinstance(response, dict):
                continue
            for item in ((response.get("data") or {}).get("topics") or []):
                if not isinstance(item, dict):
                    continue
                dto = item.get("topic_info_dto") if isinstance(item.get("topic_info_dto"), dict) else item
                topic_id, name = str(dto.get("id") or ""), str(dto.get("name") or "")
                if not topic_id or not name or topic_id in seen:
                    continue
                seen.add(topic_id)
                topics.append({"id": topic_id, "name": name, "link": str(dto.get("link") or "")})
        # Prefer exact/partial name matches; keep the rest as suggestions.
        needle = keyword.lower()
        topics.sort(key=lambda topic: (topic["name"].lower() != needle, needle not in topic["name"].lower()))
        return topics
