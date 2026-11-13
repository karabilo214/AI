"""Проверки форм и чисел до токенизации и выделения рабочих буферов."""
import math

def validate_request(req):
    if not isinstance(req, dict):
        raise ValueError('тело запроса должно быть JSON-объектом')
    if 'messages' in req:
        if not isinstance(req['messages'],list) or not req['messages']:
            raise ValueError('messages должен быть непустым списком')
        for msg in req['messages']:
            if not isinstance(msg,dict) or msg.get('role') not in ('system','developer','user','assistant','tool'):
                raise ValueError('неверная форма сообщения или role')
            content=msg.get('content')
            if content is not None and not isinstance(content,(str,list)):
                raise ValueError('content должен быть строкой или списком частей')
            if isinstance(content,list) and any(not isinstance(p,(str,dict)) for p in content):
                raise ValueError('неверная часть content')
    if 'options' in req and not isinstance(req['options'],dict):
        raise ValueError('options должен быть объектом')
    for obj in (req,req.get('options',{})):
        for key,lower,upper in (('temperature',0,2),('top_p',0,1)):
            if obj.get(key) is not None:
                value=obj[key]
                if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
                    raise ValueError(f'{key} должен быть конечным числом')
                if value<lower or value>upper or key=='top_p' and value==0:
                    raise ValueError(f'неверный диапазон {key}')
        for key in ('max_tokens','max_completion_tokens','num_predict'):
            if key in obj and (isinstance(obj[key],bool) or not isinstance(obj[key],int) or obj[key]<0):
                raise ValueError(f'{key} должен быть неотрицательным целым')
        if 'seed' in obj and obj['seed'] is not None and (isinstance(obj['seed'],bool) or not isinstance(obj['seed'],int) or obj['seed']<0):
            raise ValueError('seed должен быть неотрицательным целым')
        for key in ('stop','stop_sequences'):
            value=obj.get(key)
            if value is not None and not (isinstance(value,str) or isinstance(value,list) and all(isinstance(s,str) for s in value)):
                raise ValueError(f'{key} должен быть строкой или списком строк')
    for key in ('tools',):
        if key in req and (not isinstance(req[key],list) or any(not isinstance(t,dict) for t in req[key])):
            raise ValueError(f'{key} должен быть списком объектов')
    return req

def validate_capabilities(req, model, maxt, endpoint):
    if req.get('model') not in (None,model):
        raise ValueError(f'загружена только модель {model}')
    if req.get('n',1) != 1:
        raise NotImplementedError('поддерживается только n=1')
    for obj in (req,req.get('options',{})):
        for key in ('frequency_penalty','presence_penalty'):
            if obj.get(key,0) != 0:
                raise NotImplementedError(f'{key} пока не поддерживается')
        if obj.get('repeat_penalty',1) != 1:
            raise NotImplementedError('repeat_penalty пока не поддерживается')
        if 'top_k' in obj and obj['top_k'] not in (0,-1):
            if not (obj.get('temperature',req.get('temperature',1))==0 and obj['top_k']==1):
                raise NotImplementedError('top_k пока не поддерживается при sampling')
        if 'num_ctx' in obj and obj['num_ctx'] != maxt:
            raise ValueError(f'контекст сервера фиксирован: {maxt}')
    for key in ('logprobs','logit_bias','suffix','echo'):
        if req.get(key):
            raise NotImplementedError(f'{key} пока не поддерживается')
    for msg in req.get('messages',[]):
        for part in msg.get('content') or [] if isinstance(msg.get('content'),list) else []:
            if isinstance(part,dict) and part.get('type') not in ('text','input_text','output_text','tool_use','tool_result','thinking'):
                raise NotImplementedError('поддерживается только текст и инструменты')
