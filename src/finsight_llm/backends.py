"""Text-generation backends used by the evaluation: MLX (in-process) and llama.cpp (llama-server)."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


class MLXBackend:
    """Runs a model (optionally with a LoRA adapter) through mlx-lm with greedy decoding."""

    def __init__(self, model_path: str, adapter_path: str | None = None):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tok = load(model_path, adapter_path=adapter_path)
        self.sampler = make_sampler(temp=0.0)
        self.model_path = model_path
        self.adapter_path = adapter_path

    def generate(self, messages: list[dict], max_tokens: int) -> dict:
        from mlx_lm import stream_generate

        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.perf_counter()
        text, last = "", None
        for resp in stream_generate(self.model, self.tok, prompt, max_tokens=max_tokens, sampler=self.sampler):
            text += resp.text
            last = resp
        seconds = time.perf_counter() - t0
        return {
            "text": text,
            "tokens": last.generation_tokens if last else 0,
            "prompt_tokens": last.prompt_tokens if last else 0,
            "seconds": seconds,
            "decode_tps": last.generation_tps if last else 0.0,
            "peak_mem_gb": last.peak_memory if last else 0.0,
        }

    def close(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServerBackend:
    """Serves a GGUF with llama.cpp's `llama-server` (Metal, all layers offloaded) and queries it greedily."""

    def __init__(self, gguf_path: str, ctx: int = 4096, extra_args: list[str] | None = None):
        exe = shutil.which("llama-server")
        if not exe:
            raise RuntimeError("llama-server not found on PATH (brew install llama.cpp)")
        self.port = _free_port()
        self.gguf_path = gguf_path
        self.log = open(Path(gguf_path).with_suffix(".server.log"), "w")
        self.proc = subprocess.Popen(
            [exe, "-m", gguf_path, "--port", str(self.port), "-c", str(ctx), "-ngl", "99", "--jinja",
             "--no-webui", "--parallel", "1", *(extra_args or [])],
            stdout=self.log, stderr=subprocess.STDOUT)
        try:
            self._wait_healthy()
        except BaseException:
            self.close()
            raise

    def _wait_healthy(self) -> None:
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log.name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def generate(self, messages: list[dict], max_tokens: int) -> dict:
        body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
                           "stream": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        seconds = time.perf_counter() - t0
        timings = out.get("timings", {})
        usage = out.get("usage", {})
        return {
            "text": out["choices"][0]["message"]["content"] or "",
            "tokens": usage.get("completion_tokens", timings.get("predicted_n", 0)),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "seconds": seconds,
            "decode_tps": timings.get("predicted_per_second", 0.0),
            "peak_mem_gb": 0.0,
        }

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
