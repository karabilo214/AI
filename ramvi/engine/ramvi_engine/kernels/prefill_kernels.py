"""Блочные FP32 ядра prefill; не используют autotune или буферы decode."""
import torch
import triton
import triton.language as tl


@triton.jit
def mm_k(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,XS:tl.constexpr,
         WS:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=32,BK:tl.constexpr=32):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    c=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for base in range(0,K,BK):
        k=base+kk
        a=tl.load(X+r[:,None]*XS+k[None,:],(r[:,None]<M)&(k[None,:]<K),0)
        b=tl.load(W+c[None,:]*WS+k[:,None],(c[None,:]<N)&(k[:,None]<K),0)
        acc=tl.dot(a,b,acc,input_precision='ieee')
    tl.store(Y+r[:,None]*N+c[None,:],acc,(r[:,None]<M)&(c[None,:]<N))


def matmul(x,w):
    m,k=x.shape;n=w.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=torch.float32)
    mm_k[(triton.cdiv(m,32),triton.cdiv(n,32))](x,w,out,m,n,k,x.stride(0),w.stride(0))
    return out


@triton.jit
def att_k(Q,K,V,O,N:tl.constexpr,START:tl.constexpr,MAXT:tl.constexpr,
          H:tl.constexpr,G:tl.constexpr,HD:tl.constexpr,BQ:tl.constexpr=16,BK:tl.constexpr=32):
    h=tl.program_id(0);qi=tl.program_id(1)*BQ+tl.arange(0,BQ)
    j=tl.arange(0,BK);d=tl.arange(0,HD)
    q=tl.load(Q+qi[:,None]*(H*HD)+h*HD+d[None,:],qi[:,None]<N,0)
    maximum=tl.full((BQ,),-float('inf'),tl.float32)
    denom=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,HD),tl.float32)
    # Последний блок query может быть неполным; маска store убирает хвост.
    end=tl.minimum(START+N,START+(tl.program_id(1)+1)*BQ)
    for base in range(0,end,BK):
        pos=base+j
        keys=tl.load(K+((h//G)*MAXT+pos[None,:])*HD+d[:,None],pos[None,:]<START+N,0).to(tl.float32)
        score=tl.dot(q,keys,input_precision='ieee')*(HD**-0.5)
        valid=(pos[None,:]<=START+qi[:,None])&(pos[None,:]<START+N)
        score=tl.where(valid,score,-float('inf'))
        newmax=tl.maximum(maximum,tl.max(score,1))
        prob=tl.exp(score-newmax[:,None]);alpha=tl.exp(maximum-newmax)
        vals=tl.load(V+((h//G)*MAXT+pos[:,None])*HD+d[None,:],pos[:,None]<START+N,0).to(tl.float32)
        acc=acc*alpha[:,None]
        acc=tl.dot(prob,vals,acc,input_precision='ieee')
        denom=denom*alpha+tl.sum(prob,1);maximum=newmax
    tl.store(O+qi[:,None]*(H*HD)+h*HD+d[None,:],acc/denom[:,None],qi[:,None]<N)


def attention(q,k,v,out,start,h,kv,hd):
    att_k[(h,triton.cdiv(q.shape[0],16))](q,k,v,out,q.shape[0],start,k.shape[1],h,h//kv,hd)


@triton.jit
def dequant_k(A,B,C,D,E,OUT,FIRST,ROWS:tl.constexpr,K:tl.constexpr,
              TYPE:tl.constexpr,BASE=0,CH:tl.constexpr=256):
    r=tl.program_id(0)*16+tl.arange(0,16)
    col=tl.program_id(1)*256+tl.arange(0,256)
    row=FIRST+r[:,None];cc=col[None,:]
    mask=(r[:,None]<ROWS)&(cc<K)
    if TYPE==0:  # lossless compressed bf16
        m=tl.load(A+row*K+cc,mask,0).to(tl.int32)
        ep=(cc//CH)*(CH//2)+cc%(CH//2)
        byte=tl.load(B+row*(K//2)+ep,mask,0).to(tl.int32)
        code=(byte>>((cc%CH)//(CH//2)*4))&15
        bits=((m>>7)<<31)|((code+BASE)<<23)|((m&127)<<16)
        value=tl.where(code==15,0,bits).to(tl.float32,bitcast=True)
    elif TYPE==4:
        off=(cc//64)*32+cc%32
        byte=tl.load(A+row*(K//2)+off,mask,0).to(tl.int32)
        q=(byte>>((cc%64)//32*4))&15
        sc=tl.load(B+row*(K//32)+cc//32,mask,0).to(tl.float32)
        mn=tl.load(C+row*(K//32)+cc//32,mask,0).to(tl.float32)
        scale=tl.load(D+(row*(K//256)+cc//256)*2,mask,0).to(tl.float32)
        minimum=tl.load(D+(row*(K//256)+cc//256)*2+1,mask,0).to(tl.float32)
        value=(scale*sc)*q.to(tl.float32)-minimum*mn
    else:  # Q6_K
        block=cc//256;within=cc%256;half=within//128;j=within%128
        lo=tl.load(A+row*(K//2)+block*128+half*64+j%64,mask,0).to(tl.int32)
        hi=tl.load(B+row*(K//4)+block*64+half*32+j%32,mask,0).to(tl.int32)
        q=((lo>>((j//64)*4))&15)|(((hi>>((j//32)*2))&3)<<4)
        scale=tl.load(C+row*(K//16)+cc//16,mask,0).to(tl.float32)
        dd=tl.load(D+row*(K//256)+block,mask,0).to(tl.float32)
        value=(dd*scale)*(q-32).to(tl.float32)
    tl.store(OUT+r[:,None]*K+cc,value,mask)


@triton.jit
def exceptions_k(ER,EC,EV,OUT,COUNT,FIRST,ROWS:tl.constexpr,K:tl.constexpr):
    j=tl.program_id(0)*256+tl.arange(0,256)
    r=tl.load(ER+j,j<COUNT,0);c=tl.load(EC+j,j<COUNT,0)
    v=tl.load(EV+j,j<COUNT,0)
    tl.store(OUT+(r-FIRST)*K+c,v,(j<COUNT)&(r>=FIRST)&(r<FIRST+ROWS))


def decode_rows(w,first,last):
    k=w['K'];rows=last-first
    if 'm8' in w:
        args=[w['m8'],w['e4'],w['m8'],w['m8'],w['m8']];typ=0
    elif w['t']=='Q4_K':args=[w['Q'],w['SC'],w['MN'],w['D'],w['D']];typ=4
    else:args=[w['QL'],w['QH'],w['SC'],w['D'],w['D']];typ=6
    out=torch.empty((rows,k),device=args[0].device,dtype=torch.float32)
    dequant_k[(triton.cdiv(rows,16),triton.cdiv(k,256))](*args,out,first,rows,k,typ,w.get('base',0),w.get('CH',256),enable_fp_fusion=False)
    if typ==0 and w['ev'].numel():
        exceptions_k[(triton.cdiv(w['ev'].numel(),256),)](w['er'],w['ec'],w['ev'],out,w['ev'].numel(),first,rows,k)
    return out
