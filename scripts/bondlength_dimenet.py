import argparse, json, os, sys, time, inspect, numpy as np, torch
sys.path.insert(0, os.getcwd())
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import benchmarks.models as M
from benchmarks.benchmark_config import cfg
from benchmarks.twobody_dataloader import TwoBodyDataset

def build(name):
    cls = {'dimenet': M.DimeNetRegressor, 'schnet': M.SchNetRegressor}[name]
    c = cfg.model_configs[name]
    ok = set(inspect.signature(cls.__init__).parameters) - {'self'}
    return cls(**{k: v for k, v in c.items() if k in ok})

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--mode', required=True, choices=['leaked','leakage_free'])
p.add_argument('--seed', type=int, required=True)
p.add_argument('--epochs', type=int, default=50)
p.add_argument('--data', default='dataset_combined.npz')
a = p.parse_args()
od = f"bl/{a.model}_{a.mode}_seed{a.seed}"; os.makedirs(od, exist_ok=True)
rf = os.path.join(od, 'result.json')
if os.path.exists(rf) and json.load(open(rf)).get('done'):
    print('already complete'); sys.exit(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(a.seed); np.random.seed(a.seed)
base = TwoBodyDataset(a.data, target_label='distance_ang', normalize_labels=False, verbose=False)
n = len(base); g = np.random.RandomState(a.seed); idx = g.permutation(n)
ntr, nva = int(.8*n), int(.1*n)
tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
def prep(i):
    d = base[int(i)]
    z, pos, y = d.z, d.pos, d.y.view(-1)[0]
    if a.mode == 'leakage_free':
        k = z.shape[0]
        pos = torch.zeros_like(pos)
        pos[:, 0] = torch.arange(k, dtype=pos.dtype) * 1.5
    return Data(z=z, pos=pos, y=y.view(1))
TR = [prep(i) for i in tr]; VA = [prep(i) for i in va]; TE = [prep(i) for i in te]
ys = torch.tensor([float(d.y) for d in TR]); mu, sd = float(ys.mean()), float(ys.std())
trl = DataLoader(TR, batch_size=32, shuffle=True)
val = DataLoader(VA, batch_size=32); tel = DataLoader(TE, batch_size=32)
model = build(a.model).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = torch.nn.L1Loss()
def ev(loader):
    model.eval(); s=0.0; c=0
    with torch.no_grad():
        for d in loader:
            d=d.to(dev)
            o=model(d.z, d.pos, d.batch).squeeze()
            t=((d.y.view(-1)-mu)/sd).to(dev)
            e=(o-t).abs()*sd
            s+=float(e.sum()); c+=int(e.numel())
    return s/max(c,1)
best, bestep, hist = float('inf'), 0, []
sf = os.path.join(od,'state.pt'); ep0=1
if os.path.exists(sf):
    st=torch.load(sf, map_location=dev, weights_only=False)
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
    ep0, best, bestep, hist = st['epoch']+1, st['best'], st['bestep'], st['hist']
for ep in range(ep0, a.epochs+1):
    model.train(); t0=time.time(); tot=0.0; nb=0
    for d in trl:
        d=d.to(dev); opt.zero_grad()
        o=model(d.z, d.pos, d.batch).squeeze()
        t=((d.y.view(-1)-mu)/sd).to(dev)
        loss=crit(o,t)
        if not torch.isfinite(loss):
            json.dump({'done':False,'error':'nonfinite','epoch':ep}, open(rf,'w')); sys.exit(2)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tot+=float(loss); nb+=1
    v=ev(val); hist.append({'epoch':ep,'train':tot/max(nb,1),'val_mae':v,'secs':round(time.time()-t0,2)})
    if v<best:
        best,bestep=v,ep; torch.save({'model':model.state_dict()}, os.path.join(od,'best.pt'))
    torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'epoch':ep,'best':best,'bestep':bestep,'hist':hist}, sf)
    json.dump({'done':False,'model':a.model,'mode':a.mode,'seed':a.seed,'hist':hist,'best_val_mae':best}, open(rf,'w'), indent=2)
    print(f"ep {ep} train {tot/max(nb,1):.5f} val {v:.5f} {hist[-1]['secs']}s", flush=True)
    if ep-bestep>20: break
model.load_state_dict(torch.load(os.path.join(od,'best.pt'), map_location=dev, weights_only=False)['model'])
tm=ev(tel)
triv=float(np.mean([abs(float(d.y)-mu) for d in TE]))
json.dump({'done':True,'model':a.model,'mode':a.mode,'seed':a.seed,'test_mae':tm,
           'trivial_mean_mae':triv,'best_val_mae':best,'best_epoch':bestep,
           'n_params':sum(q.numel() for q in model.parameters()),
           'n_train':len(TR),'n_test':len(TE),'hist':hist}, open(rf,'w'), indent=2)
assert 0.40 < triv < 0.60, f'trivial baseline {triv} outside expected raw-Angstrom range 0.40-0.60'
print('TEST_MAE', tm, 'TRIVIAL', triv)
