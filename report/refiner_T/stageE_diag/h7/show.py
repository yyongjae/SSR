import json, sys
R = json.load(open('/home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/perception.json'))
def pr(o, ind=0):
    for k, v in o.items():
        if isinstance(v, dict):
            print(' ' * ind + str(k) + ':'); pr(v, ind + 2)
        else:
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list): v = '%.3f [%.3f, %.3f] n=%d' % (*v[0], v[1])
            elif isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v): v = '%.3f [%.3f, %.3f]' % tuple(v)
            print(' ' * ind + str(k) + ': ' + str(v))
for k in sys.argv[1:]:
    print('==', k); pr(R[k]) if isinstance(R[k], dict) else print(R[k])
