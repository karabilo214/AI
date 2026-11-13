"""Ограниченная по абсолютному UTC-дедлайну проверка. Не запускает Ollama."""
import argparse,base64,datetime,json,os,subprocess,sys,threading,time
from pathlib import Path

def powershell(script):
    cmd=['powershell','-NoProfile','-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode()]
    result=subprocess.run(cmd,capture_output=True,timeout=15)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
    return result.stdout.decode('utf-8-sig',errors='replace').strip()

def available():
    result=powershell("Get-Process | Where-Object { $_.ProcessName -match '^(WorldOfTanks.*|ollama.*|llama-server)$' } | Select-Object ProcessName,Id | ConvertTo-Json -Compress")
    if result:raise RuntimeError('device busy: '+result)

def kill_tree(pid):
    subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True,timeout=10)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',required=True);ap.add_argument('--start',type=int,default=0);args=ap.parse_args()
    deadline=datetime.datetime.fromisoformat(args.deadline).timestamp()
    remaining=deadline-time.time()
    if remaining<=0:raise SystemExit('deadline expired')
    available()
    watchdog=threading.Timer(remaining,lambda:kill_tree(os.getpid()));watchdog.daemon=True;watchdog.start()
    out=Path('llm_app/results/audit_fix_gpu');out.mkdir(parents=True,exist_ok=True)
    commands=[['-m','unittest','discover','-s','llm_app/tests','-t','.'],
              ['-m','llm_app.check_audit_gpu'],['-m','llm_app.check_pool','64'],
              ['-m','llm_app.check_audit_api'],
              ['-m','llm_app.check_gspec','qwen3:30b','qwen3:0.6b','7','cpu','--tokens=16'],
              ['-m','llm_app.check_switch','qwen3:14b','qwen3:0.6b','--tokens=16'],
              ['-m','llm_app.check_audit_recovery'],
              ['-m','llm_app.check_audit_api','--extended'],
              ['-m','unittest','discover','-s','llm_app/tests','-t','.'],
              ['-m','llm_app.check_audit_gpu'],
              ['-m','llm_app.selftest','--model','Qwen3-0.6B','--n','48'],
              ['-m','llm_app.check_audit_moe_recovery'],
              ['-m','llm_app.selftest','--model','Qwen3-0.6B','--resident','0','--draft','Qwen3-0.6B','--n','48'],
              ['-m','llm_app.check_audit_api','--extended'],
              ['-m','llm_app.check_audit_metadata'],
              ['-m','llm_app.check_audit_sampling'],
              ['-m','llm_app.check_audit_recovery'],
              ['-m','unittest','discover','-s','llm_app/tests','-t','.'],
              ['-m','llm_app.check_audit_api','--extended'],['-m','llm_app.check_audit_sampling','--gguf']]
    status=out/'status.json'
    results=json.loads(status.read_text()) if status.exists() else []
    env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1','NUMBA_NUM_THREADS':'2'}
    try:
        for index,command in enumerate(commands):
            if index<args.start:continue
            available()
            remaining=deadline-time.time()
            if remaining<20:break
            log=out/f'{index}_{time.time_ns()}.log'
            print('START',index,command,flush=True)
            with log.open('w',encoding='utf-8') as handle:
                child=subprocess.Popen([sys.executable,*command],stdout=handle,stderr=subprocess.STDOUT,env=env,stdin=subprocess.DEVNULL)
                try:rc=child.wait(timeout=min(remaining-5,600))
                except subprocess.TimeoutExpired:
                    kill_tree(child.pid);child.wait(timeout=10);rc=-1
            results.append({'command':command,'returncode':rc,'log':str(log)})
            (out/'status.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(log.read_text(encoding='utf-8',errors='replace')[-10000:],flush=True)
            if rc:raise SystemExit(rc)
    finally:
        watchdog.cancel();watchdog.join(timeout=2)
    print('AUDIT_RUN_FINISHED',flush=True)
if __name__=='__main__':main()
