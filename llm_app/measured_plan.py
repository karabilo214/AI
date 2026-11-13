"""Явная калибровка GGUF; измерения отделены от обслуживания запросов."""
import os
import platform
import time
import statistics
import torch
from .plan import measured_schedule, ram_info


def signature(eng):
    return dict(model=os.path.basename(eng.g.path),size=eng.g.size,layers=eng.L,
                d=eng.D,ff=eng.FF,maxt=eng.maxt,device=torch.cuda.get_device_name(),
                threads=os.environ.get('LLM_CPP_THREADS','16'),cpu=platform.processor(),
                cpu_count=os.cpu_count(),ram_total=ram_info()[0],model_mtime_ns=os.stat(eng.g.path).st_mtime_ns)


def configure(eng, profile):
    if profile.get('version')!=1 or profile.get('signature')!=signature(eng):
        raise ValueError('профиль не соответствует модели, устройству или настройкам')
    saved=profile.get('place',[])
    if len(saved)!=eng.L or any(p!=q and {p,q}!={'stream','hyb'} for p,q in zip(saved,eng.place)):
        raise ValueError('профиль не соответствует текущему размещению в памяти')
    routes={}
    for key,sample in profile['batches'].items():
        t=int(key)
        if not 1<=t<=16:raise ValueError('профиль пачки вне 1..16')
        initial=sample['baseline_route']
        if len(initial)!=eng.L or len(sample['layers'])!=eng.L:
            raise ValueError('не совпадает число слоёв профиля')
        if any((p=='gpu')!=(r=='gpu') for p,r in zip(eng.place,initial)):
            raise ValueError('профиль меняет GPU-резидентность')
        for p,row in zip(eng.place,sample['layers']):
            if p=='cpu' and ('gpu_s' in row or 'copy_s' in row):
                raise ValueError('CPU-слой не имеет GPU-представления')
        route,estimate,base=measured_schedule(sample['layers'],initial)
        if 'validated_route' in sample:
            tested=sample['validated_route']
            if len(tested)!=eng.L or any(r not in ('gpu','cpu','stream') or (r=='gpu')!=(p=='gpu')
                                         for r,p in zip(tested,eng.place)):
                raise ValueError('неверный проверенный маршрут')
            if tested not in (route,initial):
                raise ValueError('проверенный маршрут не соответствует измерениям')
            route=tested
        routes[t]=route
    # Все представления создаются до публикации маршрутов; ctx после калибровки
    # инвалидируется. Слои CPU без GPU-представления остаются CPU.
    need_cpu=[l for l,p in enumerate(eng.place) if p=='stream' and any(r[l]=='cpu' for r in routes.values())]
    for l in need_cpu:
        ensure_hybrid(eng,l)
    eng._measured_routes=routes
    eng._measured_context_bucket=profile.get('context_bucket',0)
    eng._active_routes=None;eng._reset_loader();eng.graphs={};eng.ctx=[]


def ensure_hybrid(eng,l):
    if eng.place[l]!='stream':return
    from .gcpu import GCpuLayer
    # CpuMat хранит развёрнутые масштабы; KV — fp32. Не резервируем их вслепую.
    estimate=4*eng.KV*eng.maxt*eng.HD*2+sum(eng.g.raw(f'blk.{l}.{name}.weight')[0].size for name in
        ('attn_q','attn_k','attn_v','attn_output','ffn_gate','ffn_up','ffn_down'))
    if ram_info()[1]<estimate+256*2**20:raise MemoryError('недостаточно RAM для калибровки CPU-слоя')
    cl=GCpuLayer(eng.g,l,eng.small[l],eng.H,eng.KV,eng.HD,eng.FF,eng.eps,eng.maxt)
    # Сохраняем валидный префикс при добавлении второго представления.
    n=len(eng.ctx)
    if n:
        cl.K[:,:n]=eng.Kc[l][:,:n].float().cpu().numpy()
        cl.V[:,:n]=eng.Vc[l][:,:n].float().cpu().numpy()
    eng.HW[l]=eng.W[l];eng.W[l]=cl;eng.place[l]='hyb';eng._ids()
    if not hasattr(eng,'hx'):
        eng.hx=torch.empty(16,eng.D).pin_memory();eng.xs=torch.cuda.Stream()


def calibrate(eng,batches=(1,4,16),repeats=3):
    """Медианы трёх реальных проходов, отдельно вычисление GPU, CPU и DMA.

    Вызывать на свободном движке. Изменяет только валидность KV, добавляет
    CPU-представления при достаточной RAM. В стоимость CPU входят D2H/H2D.
    """
    if repeats<3:raise ValueError('нужно не менее трёх повторов')
    if not batches or any(t not in range(1,17) for t in batches):raise ValueError('пачки вне 1..16')
    eng._reset_loader();torch.cuda.synchronize();eng.ctx=[]
    original_place=list(eng.place)
    previous_routes=getattr(eng,'_measured_routes',None)
    try:
        # Подкачиваемые слои должны иметь оба пути для честного измерения.
        for l in list(eng.stream_ids):ensure_hybrid(eng,l)
        result=dict(version=1,signature=signature(eng),place=list(eng.place),context_bucket=0,batches={})
        for t in batches:
            # Проверка предыдущей пачки оставляет две предварительные загрузки.
            # Освободить их до ручных запросов калибровки следующей пачки.
            eng._reset_loader()
            eng._active_routes=None
            rows=[]
            for l,p in enumerate(eng.place):
                x=torch.ones((t,eng.D),device='cuda',dtype=torch.float32)*.01
                eng.pos.fill_(0)
                row={}
                def median(fn):
                    fn();torch.cuda.synchronize()  # compile/warm-up outside samples
                    samples=[]
                    for _ in range(repeats):
                        x.fill_(.01);torch.cuda.synchronize();start=time.perf_counter()
                        fn();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
                    return statistics.median(samples)
                if p in ('cpu','hyb'):row['cpu_s']=median(lambda:eng._cpu_layer(l,x,0))
                if p=='gpu':row['gpu_s']=median(lambda:eng._gpu_layer(l,eng.W[l],x,t))
                elif p=='hyb':
                    eng._request(l);weights=eng._acquire(l);torch.cuda.synchronize()
                    row['gpu_s']=median(lambda:eng._gpu_layer(l,weights,x,t))
                    # Отдельные синхронные DMA, в тот же буфер и из той же RAM.
                    pk=eng._pk(l);slot=eng.cur['s']
                    def copy():
                        for name in pk:eng.cbuf[slot][name][:pk[name][0].numel()].copy_(pk[name][0],non_blocking=True)
                    row['copy_s']=median(copy)
                    # _release запросил бы следующий слой; калибровка не крутит цикл.
                    ev=torch.cuda.Event();ev.record();eng.cur['rel']['ev']=ev;eng.cur['rel']['done'].set()
                rows.append(row)
            initial=['cpu' if p in ('cpu','hyb') and t==1 else 'stream' if p=='hyb' else p for p in original_place]
            if previous_routes is not None:
                initial=previous_routes.get(t,initial)
            route,estimate,base=measured_schedule(rows,initial)
            sample=dict(layers=rows,route=route,baseline_route=initial,estimate_s=estimate,baseline_estimate_s=base)
            # Модель DMA — только предложение. Принимаем маршрут после проверки
            # полного прохода, включая синхронизации и копирование KV.
            timings={'baseline':[],'candidate':[]};answers={}
            for rep in range(repeats):
                order=('baseline','candidate') if rep%2 else ('candidate','baseline')
                for label in order:
                    eng._measured_routes={t: initial if label=='baseline' else route}
                    eng._active_routes=None;eng._reset_loader()
                    tokens=[min(1000,eng.V-1)]*t
                    eng._decode(tokens,0);torch.cuda.synchronize()
                    started=time.perf_counter();eng._decode(tokens,0);torch.cuda.synchronize()
                    timings[label].append(time.perf_counter()-started)
                    answers[label]=eng._pick(t,0.,1.,None)
            old_s=statistics.median(timings['baseline']);new_s=statistics.median(timings['candidate'])
            equal=answers['baseline']==answers['candidate']
            sample['validation']=dict(baseline_s=old_s,candidate_s=new_s,tokens_equal=equal)
            sample['validated_route']=route if equal and new_s<old_s*.95 else initial
            result['batches'][str(t)]=sample
    finally:
        eng._reset_loader()
        eng._measured_routes=(previous_routes if previous_routes is not None else
            {1:['cpu' if p in ('cpu','hyb') else p for p in original_place]})
        eng._active_routes=None;eng.ctx=[];eng.phase=None;eng.graphs={}
    return result
