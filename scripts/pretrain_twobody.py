import argparse, json, os, sys, time, inspect, numpy as np, torch
sys.path.insert(0, os.getcwd())
from torch_geometric.loader import DataLoader
import benchmarks.models as M
from benchmarks.twobody_dataloader import TwoBodyDataset
from benchmarks.benchmark_config import cfg

def build(name, c):
    cls = {'schnet': M.SchNetRegressor, 'gotennet': M.GotenNetRegressor}[name]
    ok = set(inspect.signature(cls.__init__).parameters) - {'self'}
    return cls(**{k: v for k, v in c.items() if k in ok})

p = argparse.ArgumentParser()
p.add_argument('--target', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--epochs', type=int, default=50)
p.add_argument('--clip', type=float, default=1.0)
p.add_argument('--data', default='dataset_combined.npz')
p.add_argument('--model', default='schnet')
a = p.parse_args()
od = f"results_twobody/{a.target}/{a.model}/seed_{a.seed}"; os.makedirs(od, exist_ok=True)
rf = os.path.join(od, 'results.json')
if os.path.exists(rf) and json.load(open(rf)).get('done'):
    print('already complete'); sys.exit(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(a.seed); np.random.seed(a.seed)
ds = TwoBodyDataset(a.data, target_label=a.target)
n = len(ds); g = np.random.RandomState(a.seed); idx = g.permutation(n)
tr, va = idx[:int(.8*n)], idx[int(.8*n):int(.9*n)]
ys = torch.tensor([float(ds[int(i)].y.view(-1)[0]) for i in tr])
mu, sd = float(ys.mean()), float(ys.std()); sd = sd if sd > 1e-8 else 1.0
trl = DataLoader([ds[int(i)] for i in tr], batch_size=64, shuffle=True)
val = DataLoader([ds[int(i)] for i in va], batch_size=64)
mcfg = cfg.model_configs[a.model]
model = build(a.model, mcfg).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = torch.nn.L1Loss()
best, bestep, hist = float('inf'), 0, []
sf = os.path.join(od, 'state.pt'); ep0 = 1
if os.path.exists(sf):
    st = torch.load(sf, map_location=dev, weights_only=False)
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
    ep0, best, bestep, hist = st['epoch']+1, st['best'], st['bestep'], st['hist']
for ep in range(ep0, a.epochs+1):
    model.train(); t0=time.time(); tot=0.0; nb=0
    for d in trl:
        d = d.to(dev); opt.zero_grad()
        out = model(d.z, d.pos, d.batch).squeeze()
        y = ((d.y.view(-1) - mu)/sd).to(dev)
        loss = crit(out, y)
        if not torch.isfinite(loss):
            json.dump({'done':False,'error':'nonfinite'}, open(rf,'w')); sys.exit(2)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); tot += float(loss); nb += 1
    model.eval(); se=0.0; cnt=0
    with torch.no_grad():
        for d in val:
            d = d.to(dev)
            out = model(d.z, d.pos, d.batch).squeeze()
            y = ((d.y.view(-1)-mu)/sd).to(dev)
            e = (out-y).abs()*sd
            se += float(e.sum()); cnt += int(e.numel())
    vmae = se/max(cnt,1)
    hist.append({'epoch':ep,'train':tot/max(nb,1),'val_mae':vmae,'secs':round(time.time()-t0,2)})
    if vmae < best:
        best, bestep = vmae, ep
        torch.save({'model_state_dict':model.state_dict(),'epoch':ep,'val_mae':vmae,
                    'target':a.target,'seed':a.seed,'norm':[mu,sd]}, os.path.join(od,'best_model.pt'))
    torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'epoch':ep,
                'best':best,'bestep':bestep,'hist':hist}, sf)
    json.dump({'done':False,'target':a.target,'seed':a.seed,'hist':hist,'best_val_mae':best,
               'n_params':sum(q.numel() for q in model.parameters()),'norm':[mu,sd]}, open(rf,'w'), indent=2)
    print(f"ep {ep} train {tot/max(nb,1):.5f} val_mae {vmae:.5f} {hist[-1]['secs']}s", flush=True)
    if ep - bestep > 15: break
json.dump({'done':True,'target':a.target,'seed':a.seed,'hist':hist,'best_val_mae':best,
           'best_epoch':bestep,'n_params':sum(q.numel() for q in model.parameters()),
           'norm':[mu,sd],'n_train':len(tr),'n_val':len(va)}, open(rf,'w'), indent=2)
print('BEST_VAL_MAE', best)
