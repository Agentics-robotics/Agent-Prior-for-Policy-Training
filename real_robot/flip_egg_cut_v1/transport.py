"""Official Responses transport with recorded costs and no retry/fallback."""

from http.client import HTTPSConnection
from pathlib import Path
import json
import os
import time

from appl.io import atomic, read
from appl.journal import encode

RATES = dict(input=10.0, cached=1.0, cache_write=12.5, output=50.0)
PRICING = "https://developers.openai.com/api/docs/models/gpt-6-astra"


def usage_cost(usage):
    n = usage["input_tokens"]
    out = usage["output_tokens"]
    d = usage.get("input_tokens_details") or {}
    cached = d.get("cached_tokens", 0)
    write = d.get("cache_write_tokens")
    assumed = write is None
    if assumed:
        write = n - cached
    if min(n, out, cached, write) < 0 or cached + write > n:
        raise ValueError("Invalid provider token accounting")
    multiplier = 2 if n > 272000 else 1
    usd = (
        (n - cached - write) * RATES["input"]
        + cached * RATES["cached"]
        + write * RATES["cache_write"]
    ) * multiplier
    usd += out * RATES["output"] * (1.5 if n > 272000 else 1)
    return usd / 1e6, assumed


class Client:
    provenance = "official_openai_responses_real_robot"

    def __init__(self, cfg, journal, credential_path=None):
        self.cfg = cfg
        self.j = journal
        self.model = cfg["model"]
        self.reasoning = cfg["reasoning_effort"]
        path = Path(
            credential_path
            or f"/tmp/exp2_new_credentials_{os.getuid()}/credential.json"
        )
        if path.stat().st_mode & 0o077:
            raise ValueError("Credential file must be private")
        self._key = read(path)["OPENAI_API_KEY"]
        self.ledger = self.j.root.parent / "cost_ledger.json"
        if not self.ledger.exists():
            atomic(
                self.ledger,
                dict(
                    schema="real_robot.api_cost.v1",
                    pricing=PRICING,
                    rates_usd_per_million=RATES,
                    execution_guard_usd=cfg["execution_guard_usd"],
                    guard_scope="Developer execution guard for this authorized Flip egg cut stage; not a claimed user dollar budget.",
                    records=[],
                    stopped=None,
                ),
            )

    def configuration(self):
        return dict(
            base_url="https://api.openai.com/v1",
            model=self.model,
            reasoning_effort=self.reasoning,
            service_tier=self.cfg["service_tier"],
            timeout_seconds=self.cfg["request_timeout_seconds"],
            automatic_retries=0,
        )

    def post(self, suffix, body):
        connection = HTTPSConnection(
            "api.openai.com", timeout=self.cfg["request_timeout_seconds"]
        )
        try:
            connection.request(
                "POST",
                "/v1/responses" + suffix,
                encode(body),
                {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self._key,
                },
            )
            response = connection.getresponse()
            status = response.status
            raw = response.read()
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError):
                value = dict(non_json_body=raw.decode(errors="replace"))
            return status, value
        finally:
            connection.close()

    def respond(self, request):
        if (request["model"], request.get("reasoning", {}).get("effort")) != (
            "gpt-6-astra",
            "xhigh",
        ):
            raise ValueError("Actual request model/effort mismatch")
        seq = self.j.db.execute("SELECT MAX(seq) FROM api").fetchone()[0]
        request["service_tier"] = self.cfg["service_tier"]
        with self.j.db:
            self.j.db.execute(
                "UPDATE api SET request=? WHERE seq=?", (encode(request), seq)
            )
        atomic(self.j.root / "api" / f"{seq:04d}.request.json", request)
        folder = self.j.root / "transport" / f"{seq:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        sent = False
        reserved = False
        try:
            ledger = read(self.ledger)
            if ledger["stopped"]:
                raise RuntimeError("Recorded transport stop; no automatic continuation")
            counter = {
                k: request[k]
                for k in (
                    "model",
                    "instructions",
                    "input",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "reasoning",
                )
            }
            atomic(folder / "count_request.json", counter)
            status, count = self.post("/input_tokens", counter)
            atomic(
                folder / "count_response.json", dict(http_status=status, response=count)
            )
            if status != 200 or type(count.get("input_tokens")) is not int:
                raise RuntimeError("Input token count failed; no generation sent")
            n = count["input_tokens"]
            # Bound context growth explicitly; no silent history projection or summarization.
            if n > 240000:
                raise RuntimeError("Declared 240k input context guard reached")
            reservation = (
                n * RATES["cache_write"]
                + request["max_output_tokens"] * RATES["output"]
            ) / 1e6
            used = sum(r.get("usd", r["reserved_usd"]) for r in ledger["records"])
            if used + reservation > ledger["execution_guard_usd"]:
                raise RuntimeError(
                    "Insufficient execution guard for the full next request"
                )
            entry = dict(
                seq=seq,
                reserved_usd=reservation,
                input_tokens_counted=n,
                status="reserved",
                time=time.time(),
            )
            ledger["records"].append(entry)
            atomic(self.ledger, ledger)
            reserved = True
            atomic(
                folder / "generation_attempt.json",
                dict(time=time.time(), automatic_retry=False),
            )
            sent = True
            status, response = self.post("", request)
            atomic(
                folder / "response.json", dict(http_status=status, response=response)
            )
            if response.get("usage"):
                cost, assumed = usage_cost(response["usage"])
                entry.update(
                    usd=cost,
                    usage=response["usage"],
                    conservative_cache_write_accounting=assumed,
                    status="usage_reported",
                )
                if cost > reservation:
                    ledger["stopped"] = "Provider cost exceeded preflight reservation"
            else:
                entry["status"] = "charge_unknown_reserved"
                ledger["stopped"] = "Missing provider usage"
            if status != 200:
                ledger["stopped"] = "Provider HTTP error; no automatic retry"
            returned = (
                response.get("model"),
                response.get("reasoning", {}).get("effort"),
            )
            if status == 200 and returned != (self.model, self.reasoning):
                ledger["stopped"] = "Returned model/effort mismatch"
            if status == 200 and response.get("service_tier") not in ("default", None):
                ledger["stopped"] = "Unexpected service tier"
            atomic(self.ledger, ledger)
            if status == 200 and ledger["stopped"]:
                raise RuntimeError(ledger["stopped"])
            return status, response
        except Exception as error:
            reason = (
                str(error)
                if isinstance(error, (RuntimeError, ValueError))
                else type(error).__name__
            )
            atomic(
                folder / "interruption.json",
                dict(
                    reason=reason,
                    generation_attempted=sent,
                    reservation_retained=reserved,
                    automatic_retry=False,
                ),
            )
            ledger = read(self.ledger)
            ledger["stopped"] = reason
            atomic(self.ledger, ledger)
            if not sent:
                with self.j.db:
                    self.j.db.execute(
                        "UPDATE api SET status='not_sent' WHERE seq=?", (seq,)
                    )
            raise
