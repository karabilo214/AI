"""Сравнивать можно только совпадающие списки задач и определённые скорости."""
def paired_rows(a,b,key):
    if not a or len(a)!=len(b):
        raise ValueError('наборы задач пусты или имеют разную длину')
    if any(x.get(key)!=y.get(key) for x,y in zip(a,b)):
        raise ValueError('порядок или тексты задач не совпадают')
    return list(zip(a,b))

def validate_speed(rows):
    for row in rows:
        if row.get('n',0)<=0 or row.get('gen_s',0)<=0 or row.get('tps',0)<=0:
            raise ValueError('ответ не содержит измеримой скорости; сравнение отклонено')
        if not row.get('out'):
            raise ValueError('пустой ответ нельзя засчитывать как ускорение')


def compare_run(a,b):
    if not isinstance(a,dict) or not isinstance(b,dict) or a.get('version')!=1 or b.get('version')!=1:
        raise ValueError('нет манифеста запуска: исторические файлы требуют отдельной проверки параметров')
    if a.get('parameters')!=b.get('parameters'):
        raise ValueError('параметры генерации не совпадают')
    sa,sb=a.get('server',{}),b.get('server',{})
    if sa.get('weights_id') and sb.get('weights_id'):
        if sa['weights_id']!=sb['weights_id']:raise ValueError('разные веса модели')
    elif a.get('model')!=b.get('model'):
        raise ValueError('разные модели и нет общего идентификатора весов')
    if sa.get('tokenizer') and sb.get('tokenizer') and sa['tokenizer']!=sb['tokenizer']:
        raise ValueError('разные токенизаторы')
