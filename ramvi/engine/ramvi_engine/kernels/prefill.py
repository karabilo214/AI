"""Специализированный prefill: ограниченная распаковка, GEMM и блочное внимание.

Включение: LLM_PREFILL=blocked (legacy — прежний путь).
Короткие запросы остаются на legacy; порог LLM_PREFILL_MIN_TOKENS=256.
Временные веса: не более ROWS * K чисел fp32, по одной матрице.
"""
import os
import torch

ROWS = 512


def enabled(n=None):
    mode = os.environ.get('LLM_PREFILL', 'legacy')
    if mode not in ('legacy', 'blocked'):
        raise ValueError('LLM_PREFILL должен быть legacy или blocked')
    minimum=int(os.environ.get('LLM_PREFILL_MIN_TOKENS','256'))
    if minimum<0:raise ValueError('LLM_PREFILL_MIN_TOKENS должен быть неотрицательным')
    return mode == 'blocked' and (n is None or n>=minimum)


def weight_rows(w, first, last):
    """Точное восстановление представленных весов, без повторной квантизации."""
    if torch.is_tensor(w):
        return w[first:last].float()
    device=w['m8'].device if 'm8' in w else w['W'].device if w['t'] in ('F16','F32') else w['D'].device
    if device.type=='cuda' and ('m8' in w or w['t'] in ('Q4_K','Q6_K')):
        from ramvi_engine.kernels.prefill_kernels import decode_rows
        return decode_rows(w,first,last)
    if 'm8' in w:
        ch, k = w['CH'], w['K']
        e = w['e4'][first:last].int().reshape(last-first, k//ch, ch//2)
        code = torch.stack((e & 15, e >> 4), 2).reshape(last-first, k)
        m = w['m8'][first:last].int()
        bits = ((m >> 7) << 31) | ((code+w['base']) << 23) | ((m & 127) << 16)
        result = torch.where(code == 15, 0, bits).view(torch.float32)
        # rp ограничивает исключения выбранными строками; скалярное чтение здесь
        # синхронизирует только один блок, не создаёт копию полной матрицы.
        a, b = int(w['rp'][first]), int(w['rp'][last])
        result[w['er'][a:b].long()-first, w['ec'][a:b].long()] = w['ev'][a:b]
        return result
    if w['t'] in ('F16', 'F32'):
        return w['W'][first:last].float()
    k = w['K']; dev = w['D'].device
    col = torch.arange(k, device=dev)
    block = col//256
    if w['t'] == 'Q4_K':
        sub = col//32
        off = (col//64)*32 + col%32
        q = w['Q'][first:last, off].int()
        q = (q >> (((col%64)//32)*4)) & 15
        sc = w['SC'][first:last, sub].float()
        mn = w['MN'][first:last, sub].float()
        d = w['D'][first:last, block].float()
        return d[..., 0]*sc*q.float() - d[..., 1]*mn
    if w['t'] == 'Q6_K':
        within = col%256; half = within//128; j = within%128
        lo = w['QL'][first:last, block*128+half*64+j%64].int()
        hi = w['QH'][first:last, block*64+half*32+j%32].int()
        q = ((lo >> ((j//64)*4)) & 15) | (((hi >> ((j//32)*2)) & 3) << 4)
        return w['D'][first:last, block].float()*w['SC'][first:last, col//16].float()*(q-32).float()
    raise NotImplementedError(w['t'])


def matrix(x, w, y, add=False):
    """GEMM блоков строк на все токены; fp32 IEEE, без неявного TF32."""
    if x.device.type == 'cuda':
        from ramvi_engine.kernels.prefill_kernels import matmul
    else:
        matmul = lambda a,b: a @ b.T
    n = w.shape[0] if torch.is_tensor(w) else w['N']
    for first in range(0, n, ROWS):
        last = min(first+ROWS, n)
        dense = weight_rows(w, first, last)
        value = matmul(x, dense)
        if add: y[:, first:last].add_(value)
        else: y[:, first:last].copy_(value)


def layer(eng, l, weights, x, start, kc, vc, gguf=False):
    """Общий prefill слоя; постоянные decode-буферы и графы не меняются."""
    import triton
    from ramvi_engine.kernels.tkernels import rmsg_k, qk_rope_cache_k
    from ramvi_engine.kernels.prefill_kernels import attention
    n, d = x.shape
    h, kv, hd = eng.H, eng.KV, eng.HD
    qkv_dim = (h+2*kv)*hd
    # Буферы на один блок токенов; веса распаковываются один раз на весь X.
    norm = torch.empty_like(x)
    qkv = torch.empty((n,qkv_dim),device=x.device,dtype=torch.float32)
    q = torch.empty((n,h*hd),device=x.device,dtype=torch.float32)
    att = torch.empty_like(q)
    sm = eng.small[l]
    rmsg_k[(n,)](x,sm['ln1'],norm,eng.eps,d,DP=triton.next_power_of_2(d))
    if gguf:
        a,b = h*hd,(h+kv)*hd
        matrix(norm,weights['q'],qkv[:,:a])
        matrix(norm,weights['k'],qkv[:,a:b])
        matrix(norm,weights['v'],qkv[:,b:])
    else:
        matrix(norm,weights['qkv'],qkv)
    # Записываем все K/V до attention. Каузальная маска исключает будущие строки.
    eng.pos.fill_(start)
    qk_rope_cache_k[(h+2*kv,n)](qkv,sm['qn'],sm['kn'],eng.inv,eng.pos,q,kc,vc,
        eng.maxt,eng.eps,NH=h,NKV=kv,HDIM=hd,half=hd//2)
    attention(q,kc,vc,att,start,h,kv,hd)
    del qkv,q
    matrix(att,weights['o'],x,add=True)
    del att
    rmsg_k[(n,)](x,sm['ln2'],norm,eng.eps,d,DP=triton.next_power_of_2(d))
    gu = torch.empty((n,2*eng.FF),device=x.device,dtype=torch.float32)
    if gguf:
        matrix(norm,weights['gate'],gu[:,:eng.FF])
        matrix(norm,weights['up'],gu[:,eng.FF:])
    else:
        matrix(norm,weights['gu'],gu)
    # Та же формула, что у decode, без изменения dtype.
    mid = gu[:,:eng.FF] / (1.0+torch.exp(-gu[:,:eng.FF])) * gu[:,eng.FF:]
    del gu,norm
    matrix(mid,weights['down'],x,add=True)
