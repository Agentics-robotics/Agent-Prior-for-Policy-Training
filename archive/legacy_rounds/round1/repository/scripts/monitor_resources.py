"""Read-only device telemetry; has no access to policy RNGs or data loaders."""
import argparse
import json
import os
import subprocess
import time
from relative_dp.utils import ROOT, atomic_json

parser=argparse.ArgumentParser()
parser.add_argument('--pid',type=int,required=True)
parser.add_argument('--gpu',type=int,default=1)
args=parser.parse_args()
samples=[]
while True:
    try:
        os.kill(args.pid,0)
    except ProcessLookupError:
        break
    result=subprocess.run(['nvidia-smi',f'--id={args.gpu}',
        '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True)
    if result.returncode:
        record={'timestamp':time.time(),'error':result.stderr}
    else:
        values=[float(x.strip()) for x in result.stdout.strip().split(',')]
        record=dict(timestamp=time.time(),gpu_utilization_percent=values[0],
                    memory_used_mib=values[1],device_power_watts=values[2],temperature_c=values[3])
        samples.append(record)
    with (ROOT/'logs/resources.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    if samples:
        energy=sum((a['device_power_watts']+b['device_power_watts'])/2*(b['timestamp']-a['timestamp'])/3600
                   for a,b in zip(samples,samples[1:]))
        atomic_json(ROOT/'artifacts/resource_usage_summary.json',dict(
            gpu_physical_index=args.gpu,pipeline_pid=args.pid,samples=len(samples),
            monitoring_started_timestamp=samples[0]['timestamp'],last_timestamp=samples[-1]['timestamp'],
            sampled_peak_memory_mib=max(s['memory_used_mib'] for s in samples),
            sampled_peak_power_watts=max(s['device_power_watts'] for s in samples),
            sampled_mean_gpu_utilization_percent=sum(s['gpu_utilization_percent'] for s in samples)/len(samples),
            approximate_device_energy_wh=energy,
            note='10-second device telemetry; began after first training started. Power includes device idle baseline; incomplete-window estimate, not a billing or incremental energy measurement.'))
    time.sleep(10)
print(f'Resource monitor finished after {len(samples)} samples')
