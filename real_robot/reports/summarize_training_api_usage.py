"""Per-task response usage, including every implementation/recovery session."""
import argparse
from collections import Counter
import json
import sqlite3
import time

from appl.io import ROOT, atomic, digest, read


def summarize(name):
    cfg = read(ROOT / f"real_robot/configs/{name}.json")
    root = ROOT / cfg["run"]
    sessions = []
    for folder in sorted(root.glob("design_[0-9][0-9]")):
        path = folder / "cost_ledger.json"
        ledger = read(path)
        counts = Counter()
        usd = unknown = 0.
        for record in ledger["records"]:
            if "usd" not in record:
                counts["usage_unresolved"] += 1
                unknown += record["reserved_usd"]
                continue
            usage = record["usage"]
            counts["usage_reported_calls"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                counts[key] += usage[key]
            for key in ("cached_tokens", "cache_write_tokens"):
                counts[key] += usage.get("input_tokens_details", {}).get(key, 0)
            counts["reasoning_tokens"] += usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
            usd += record["usd"]
        db = sqlite3.connect(f"file:{folder / '_session/journal.sqlite'}?mode=ro", uri=True)
        states = Counter()
        for status, request, response in db.execute("SELECT status,request,response FROM api"):
            states[status] += 1
            q = json.loads(request)
            assert q["model"] == "gpt-6-astra" and q["reasoning"]["effort"] == "xhigh"
            if response:
                r = json.loads(response)
                assert r["model"] == "gpt-6-astra" and r["reasoning"]["effort"] == "xhigh"
        db.close()
        sessions.append(dict(session=folder.name, calls=sum(states.values()), states=dict(states),
            tokens_and_counts=dict(counts), estimated_usd=usd, unresolved_reserved_usd=unknown,
            ledger=str(path.relative_to(ROOT)), ledger_sha256=digest(path),
            saved_rates_usd_per_million=ledger["rates_usd_per_million"],
            pricing_reference=ledger["pricing"]))
    total = Counter()
    for session in sessions:
        total.update(session["tokens_and_counts"])
    result = dict(run=name, measured_at=time.time(), workflow_status=read(root / "workflow.json")["status"],
        scope="This training implementation workflow only, including blocked and repaired API revisions; prior cut and historical training costs excluded.",
        model="gpt-6-astra", reasoning_effort="xhigh", sessions=sessions,
        totals=dict(total), API_calls=sum(s["calls"] for s in sessions),
        estimated_usd=sum(s["estimated_usd"] for s in sessions),
        unresolved_reserved_usd=sum(s["unresolved_reserved_usd"] for s in sessions),
        cost_is_invoice=False, GPU_compute_included=False,
        token_accounting="Cached/cache-write tokens are input subsets; reasoning tokens are an output subset. Do not add these subsets again to total_tokens.")
    atomic(root / "API_USAGE.json", result)
    lines = [f"# {name} API usage", "", "核对日期：2026-09-22。",
        "", "仅统计本轮模型实现、预处理检查、失败诊断与修订的 API 调用；不含此前 cut、历史训练或 GPU 费用。",
        "所有实际请求与已返回响应均核对为 gpt-6-astra/xhigh。费用按每个会话保存的计价规则估算，不是账单。",
        "", f"当前流程状态：`{result['workflow_status']}`。",
        "", "| 会话 | 调用数 | Input tokens | 缓存读取 | 缓存写入 | Output tokens | 其中 reasoning | 估算 USD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in sessions:
        t = s["tokens_and_counts"]
        lines.append(f"| {s['session']} | {s['calls']} | {t.get('input_tokens',0):,} | {t.get('cached_tokens',0):,} | {t.get('cache_write_tokens',0):,} | {t.get('output_tokens',0):,} | {t.get('reasoning_tokens',0):,} | {s['estimated_usd']:.6f} |")
    lines += [f"| 总计 | {result['API_calls']} | {total['input_tokens']:,} | {total['cached_tokens']:,} | {total['cache_write_tokens']:,} | {total['output_tokens']:,} | {total['reasoning_tokens']:,} | {result['estimated_usd']:.6f} |", "",
        f"API 报告总 tokens：{total['total_tokens']:,}。尚未结清 usage 的记录：{total['usage_unresolved']}；对应预留 ${result['unresolved_reserved_usd']:.6f}，不计作已确认费用。",
        "缓存读取/写入已包含在 input 中；reasoning 已包含在 output 中，不能重复相加。",
        "若流程仍在运行，这是一份时间点快照；最终训练报告需使用完成后重新生成的统计。",
        "", f"[逐会话来源、哈希与完整统计](../{root.relative_to(ROOT / 'real_robot')}/API_USAGE.json)", ""]
    report = ROOT / f"real_robot/reports/{name.upper()}_API_USAGE.md"
    report.write_text("\n".join(lines))
    return dict(run=name, API_calls=result["API_calls"], estimated_usd=result["estimated_usd"],
                total_tokens=total["total_tokens"], unresolved=total["usage_unresolved"], report=str(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Configuration basenames without .json")
    for name in parser.parse_args().configs:
        print(summarize(name), flush=True)


if __name__ == "__main__":
    main()
