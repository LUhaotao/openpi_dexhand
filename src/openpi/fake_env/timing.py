from collections.abc import Mapping


def format_latency_report(timing: Mapping[str, Mapping[str, float]]) -> str:
    """Render the fixed three-layer latency report in Chinese."""
    inference = timing.get("inference", {})
    server = timing.get("server", {})
    e2e = timing.get("e2e", {})
    lines = [
        "[推理层]",
        f"VLM 推理: {inference.get('vlm_ms', 0.0):.3f} ms",
        f"FM 推理: {inference.get('fm_ms', 0.0):.3f} ms",
        "[Server层]",
        f"VLM 服务: {server.get('vlm_ms', 0.0):.3f} ms",
        f"KVCache 刷新: {server.get('kv_refresh_ms', 0.0):.3f} ms",
        f"FM 服务: {server.get('fm_ms', 0.0):.3f} ms",
        f"Server 总计: {server.get('total_ms', 0.0):.3f} ms",
        "[端到端]",
        f"Observation 上传到 Action 接收: {e2e.get('total_ms', 0.0):.3f} ms",
    ]
    return "\n".join(lines)
