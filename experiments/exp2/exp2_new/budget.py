"""Official-only transport: reserve costs before generation, never retry."""
import json
import os
import time
from decimal import Decimal
from http.client import HTTPSConnection
from pathlib import Path

from appl.io import atomic, read
from appl.journal import encode, lock
from .common import BASE

NANO = 1_000_000_000
# Nanodollars per token. Pricing reviewed 2026-09-19; standard service only.
RATES = dict(input=10000, cached=1000, cache_write=12500, output=50000)
PRICING = 'https://developers.openai.com/api/docs/models/gpt-6-astra'


class BudgetStop(RuntimeError):
    pass


def reserve_cost(input_tokens, max_output_tokens):
    long = input_tokens > 272000
    return input_tokens*RATES['cache_write']*(2 if long else 1) + max_output_tokens*RATES['output']*(3 if long else 2)//2


def usage_cost(usage):
    n = usage['input_tokens']; out = usage['output_tokens']
    details = usage.get('input_tokens_details') or {}
    cached = details.get('cached_tokens', 0)
    write = details.get('cache_write_tokens')
    # If write accounting is absent, retain the conservative write tariff for
    # every noncached token rather than assert a false exact invoice estimate.
    assumed = write is None
    if assumed: write = n-cached
    if min(n, out, cached, write)<0 or cached+write>n:
        raise ValueError('Invalid provider token accounting')
    long = n > 272000
    cost = ((n-cached-write)*RATES['input'] + cached*RATES['cached'] + write*RATES['cache_write'])*(2 if long else 1)
    cost += out*RATES['output']*(3 if long else 2)//2
    return cost, assumed


class Ledger:
    def __init__(self, root=BASE/'budget'):
        self.root = Path(root)
        self.path = self.root/'ledger.json'

    def initialize(self, usd, authorization):
        if self.path.exists(): raise ValueError('Budget already initialized')
        cap = int(Decimal(str(usd))*NANO)
        if cap<=0: raise ValueError('Positive explicit budget required')
        atomic(self.path, dict(cap_nano_usd=cap, authorization=authorization,
                              rates_nano_usd_per_token=RATES, pricing_source=PRICING,
                              records=[], stopped=None, created=time.time()))

    def reserve(self, key, n, maximum):
        with lock(self.root/'ledger.lock'):
            v=read(self.path); amount=reserve_cost(n, maximum)
            if v['stopped']: raise BudgetStop(v['stopped'])
            if any(r['key']==key for r in v['records']): raise ValueError('Request already reserved; no retry')
            used=sum(r.get('settled_nano_usd',r['reserved_nano_usd']) for r in v['records'])
            if used+amount>v['cap_nano_usd']:
                v['stopped']='Insufficient remaining budget for the full next request reservation'
                atomic(self.path,v); raise BudgetStop(v['stopped'])
            v['records'].append(dict(key=key,input_tokens_counted=n,max_output_tokens=maximum,
                                     reserved_nano_usd=amount,status='reserved',time=time.time()))
            atomic(self.path,v)

    def settle(self, key, body, http_status):
        with lock(self.root/'ledger.lock'):
            v=read(self.path); r=next(r for r in v['records'] if r['key']==key)
            r.update(http_status=http_status, response_id=body.get('id'), usage=body.get('usage'))
            if body.get('usage'):
                amount, conservative=usage_cost(body['usage'])
                r.update(settled_nano_usd=amount, conservative_cache_write_accounting=conservative, status='usage_reported')
                if amount>r['reserved_nano_usd']:
                    v['stopped']='Reported usage exceeds preflight reservation; admission stopped'
            else:
                r['status']='charge_unknown_reserved'; v['stopped']='Provider did not report generation usage'
            if http_status!=200: v['stopped']='Provider HTTP error; no automatic retry'
            atomic(self.path,v)

    def stop(self, reason):
        with lock(self.root/'ledger.lock'):
            v=read(self.path);v['stopped']=reason;atomic(self.path,v)


class BudgetClient:
    provenance='official_responses_api_budgeted'
    model='gpt-6-astra'
    reasoning='xhigh'

    def __init__(self, journal):
        self.j=journal;self.ledger=Ledger()
        path=Path('/tmp')/f'exp2_new_credentials_{os.getuid()}'/'credential.json'
        if path.stat().st_mode & 0o077: raise ValueError('Credential permissions must be private')
        self._key=read(path)['OPENAI_API_KEY']

    def configuration(self):
        return dict(base_url='https://api.openai.com/v1',model=self.model,reasoning_effort=self.reasoning,
                    service_tier='default',timeout_seconds=600,credential='private out-of-repository file')

    def post(self, suffix, body):
        connection=HTTPSConnection('api.openai.com',timeout=600)
        try:
            connection.request('POST','/v1/responses'+suffix,encode(body),
                               {'Content-Type':'application/json','Authorization':'Bearer '+self._key})
            reply=connection.getresponse();status=reply.status;raw=reply.read()
            try: value=json.loads(raw)
            except (ValueError,UnicodeError): value=dict(non_json_body=raw.decode(errors='replace'))
            return status,value
        finally: connection.close()

    def respond(self, request):
        seq=self.j.db.execute('SELECT MAX(seq) FROM api').fetchone()[0]
        out=self.j.root/'transport'/f'{seq:04d}';out.mkdir(parents=True,exist_ok=False)
        if (request['model'],request['reasoning']['effort'])!=(self.model,self.reasoning):
            raise ValueError('Model or reasoning effort changed')
        request['service_tier']='default'
        # Save the exact wire request before network I/O, including standard tier.
        with self.j.db:
            self.j.db.execute('UPDATE api SET request=? WHERE seq=?',(encode(request),seq))
        atomic(self.j.root/'api'/f'{seq:04d}.request.json',request)
        key=str(self.j.root.resolve().relative_to(BASE.resolve()))+f'/{seq:04d}'
        reserved=False;sent=False
        try:
            current=read(self.ledger.path)
            if current['stopped']: raise BudgetStop(current['stopped'])
            count_request={k:request[k] for k in ('model','instructions','input','tools','tool_choice','parallel_tool_calls','reasoning')}
            atomic(out/'count_request.json',count_request)
            status,count=self.post('/input_tokens',count_request)
            atomic(out/'count_response.json',dict(http_status=status,response=count))
            if status!=200 or not isinstance(count.get('input_tokens'),int):
                raise RuntimeError('Input-token preflight failed; no generation request sent')
            self.ledger.reserve(key,count['input_tokens'],request['max_output_tokens']);reserved=True
            atomic(out/'generation_attempt.json',dict(time=time.time(),request_key=key,automatic_retry=False))
            sent=True
            status,body=self.post('',request)
            atomic(out/'response.json',dict(http_status=status,response=body))
            self.ledger.settle(key,body,status)
            if status==200 and (body.get('model')!=self.model or (body.get('reasoning') or {}).get('effort')!=self.reasoning):
                self.ledger.stop('Returned model/effort mismatch')
                raise RuntimeError('Unexpected provider model/effort; response preserved, not executed')
            return status,body
        except Exception as error:
            reason=str(error) if isinstance(error,(BudgetStop,RuntimeError)) else type(error).__name__
            atomic(out/'interruption.json',dict(reason=reason,generation_attempted=sent,
                                              reservation_retained=reserved,automatic_retry=False))
            if not sent:
                with self.j.db:
                    self.j.db.execute("UPDATE api SET status='not_sent' WHERE seq=?",(seq,))
            self.ledger.stop(reason)
            raise
