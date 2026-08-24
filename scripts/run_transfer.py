import argparse, json, os, sys, time, inspect, numpy as np, torch
sys.path.insert(0, os.getcwd())
from torch_geometric.loader import DataLoader
import benchmarks.models as M

def build(name, cfg):
    cls = {'schnet': M.SchNetRegressor, 'gotennet': M.GotenNetRegressor}[name]
    ok = set(inspect.signature(cls.__init__).parameters) - {'self'}
    return cls(**{k: v for k, v in cfg.items() if k in ok})

def split(n, seed, fr=(0.8, 0.1)):
    g = np.random.RandomState(seed); idx = g.permutation(n)
    a, b = int(fr[0]*n), int((fr[0]+fr[1])*n)
    return idx[:a], idx[a:b], idx[b:]

def get_case(case, seed, mname='schnet'):
    fam, key = case.split(':')
    if fam == 'qm9':
        from torch_geometric.datasets import QM9
        from benchmarks.qm9_config import MODEL_CONFIGS, QM9_TARGETS, TRAINING_CONFIG
        ds = QM9('./data/QM9'); ti = QM9_TARGETS[key]
        tr, va, te = split(len(ds), seed)
        tr, va, te = tr[:110000], va[:10000], te[:10000]
        gy = lambda d: d.y[:, ti]
        return ds, tr, va, te, MODEL_CONFIGS[mname], TRAINING_CONFIG, gy, None
    if fam == 'md17':
        from torch_geometric.datasets import MD17
        from benchmarks.md17_config import MODEL_CONFIGS, TRAINING_CONFIG
        ds = MD17('./data/MD17', name=key)
        tr, va, te = split(len(ds), seed)
        tr, va, te = tr[:950], va[:50], te[:1000]
        ys = torch.tensor([float(ds[int(i)].energy.view(-1)[0]) for i in tr])
        mu, sd = float(ys.mean()), float(ys.std())
        gy = lambda d: (d.energy.view(-1) - mu) / sd
        return ds, tr, va, te, MODEL_CONFIGS[mname], TRAINING_CONFIG, gy, (mu, sd)
    from benchmarks.crysmtm.regression_dataloader import RegressionLoader
    from benchmarks.crysmtm_config import MODEL_CONFIGS, TRAINING_CONFIG
    pi = ['HOMO','LUMO','Eg','Ef','Et','Eta','disp','vol','bond'].index(key)
    ds = RegressionLoader(label_dir='./data/CrysMTM', modalities=['xyz','element'], as_pyg_data=True)
    tr, va, te = split(len(ds), seed, (0.7, 0.1))
    ys = torch.tensor([float(ds[int(i)].y.view(-1)[pi]) for i in tr])
    mu, sd = float(ys.mean()), float(ys.std())
    gy = lambda d: (d.y.view(d.num_graphs, -1)[:, pi] - mu) / sd
    return ds, tr, va, te, MODEL_CONFIGS[mname], TRAINING_CONFIG, gy, (mu, sd)

def evaluate(model, loader, dev, gy, norm):
    model.eval(); se = 0.0; n = 0
    with torch.no_grad():
        for d in loader:
            d = d.to(dev)
            out = model(d.z, d.pos, d.batch).squeeze()
            y = gy(d).squeeze().to(dev)
            e = (out - y).abs()
            if norm is not None: e = e * norm[1]
            se += float(e.sum()); n += int(e.numel())
    return se / max(n, 1)

p = argparse.ArgumentParser()
p.add_argument('--case', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--arm', choices=['scratch','pretrained'], required=True)
p.add_argument('--epochs', type=int, default=50)
p.add_argument('--ckpt', default=None)
p.add_argument('--clip', type=float, default=1.0)
p.add_argument('--load', choices=['full','encoder'], default='encoder')
p.add_argument('--lr', type=float, default=None)
p.add_argument('--model', default='schnet')
p.add_argument('--outdir', default='runs')
a = p.parse_args()
tag = f"{a.case.replace(':','_')}__{a.model}__seed{a.seed}__{a.arm}"
od = os.path.join(a.outdir, tag); os.makedirs(od, exist_ok=True)
sf, rf = os.path.join(od,'state.pt'), os.path.join(od,'result.json')
if os.path.exists(rf) and json.load(open(rf)).get('done'):
    print('already complete', tag); sys.exit(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(a.seed); np.random.seed(a.seed)
ds, tr, va, te, mcfg, tcfg, gy, norm = get_case(a.case, a.seed, a.model)
bs = tcfg.get('batch_size', 32)
trl = DataLoader([ds[int(i)] for i in tr], batch_size=bs, shuffle=True)
val = DataLoader([ds[int(i)] for i in va], batch_size=bs)
tel = DataLoader([ds[int(i)] for i in te], batch_size=bs)
model = build(a.model, mcfg).to(dev)
lr = a.lr if a.lr is not None else tcfg.get('lr_finetune' if a.arm=='pretrained' else 'lr_scratch', tcfg.get('lr', 1e-4))
loaded = False
if a.arm == 'pretrained' and a.ckpt and os.path.exists(a.ckpt):
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd_ = ck.get('model_state_dict', ck)
    if a.load == 'encoder':
        heads = {'schnet': ('schnet.lin1.', 'schnet.lin2.'), 'gotennet': ('regressor.',)}[a.model]
        head = [k for k in sd_ if any(k.startswith(h) for h in heads)]
        sd_ = {k: v for k, v in sd_.items() if k not in head}
        meta_head_dropped = head
    else:
        meta_head_dropped = []
    info = model.load_state_dict(sd_, strict=False)
    loaded = True
opt = torch.optim.Adam(model.parameters(), lr=lr)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=10, min_lr=1e-6)
crit = torch.nn.L1Loss()
ep0, best, bestep, hist = 1, float('inf'), 0, []
if os.path.exists(sf):
    st = torch.load(sf, map_location=dev, weights_only=False)
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sch.load_state_dict(st['sch'])
    ep0, best, bestep, hist = st['epoch']+1, st['best'], st['bestep'], st['hist']
    print('resumed at epoch', ep0)
meta = {'case':a.case,'seed':a.seed,'arm':a.arm,'model':a.model,'device':dev,
        'gpu':torch.cuda.get_device_name(0) if dev=='cuda' else None,
        'n_params':sum(q.numel() for q in model.parameters()),
        'n_train':len(tr),'n_val':len(va),'n_test':len(te),'lr':lr,'clip':a.clip,
        'epochs_budget':a.epochs,'batch_size':bs,'ckpt_loaded':loaded,'norm':norm,
        'load_mode':a.load,'head_dropped':(meta_head_dropped if loaded else [])}
for ep in range(ep0, a.epochs+1):
    model.train(); t0 = time.time(); tot = 0.0; nb = 0; bad = False
    for d in trl:
        d = d.to(dev); opt.zero_grad()
        out = model(d.z, d.pos, d.batch).squeeze()
        loss = crit(out, gy(d).squeeze().to(dev))
        if not torch.isfinite(loss): bad = True; break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); tot += float(loss); nb += 1
    if bad:
        json.dump({**meta,'done':False,'error':'nonfinite_loss','epoch':ep}, open(rf,'w'), indent=2)
        print('ABORT nonfinite', tag); sys.exit(2)
    vmae = evaluate(model, val, dev, gy, norm)
    sch.step(vmae)
    hist.append({'epoch':ep,'train_loss':tot/max(nb,1),'val_mae':vmae,
                 'lr':opt.param_groups[0]['lr'],'secs':round(time.time()-t0,2)})
    if vmae < best:
        best, bestep = vmae, ep
        torch.save({'model':model.state_dict(),'epoch':ep,'val_mae':vmae}, os.path.join(od,'best.pt'))
    torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'sch':sch.state_dict(),
                'epoch':ep,'best':best,'bestep':bestep,'hist':hist}, sf)
    json.dump({**meta,'done':False,'hist':hist,'best_val_mae':best,'best_epoch':bestep}, open(rf,'w'), indent=2)
    print(f"ep {ep} train {tot/max(nb,1):.5f} val {vmae:.5f} lr {opt.param_groups[0]['lr']:.2e} {hist[-1]['secs']}s", flush=True)
    if ep - bestep > tcfg.get('early_stopping_patience', 30): print('early stop'); break
bk = torch.load(os.path.join(od,'best.pt'), map_location=dev, weights_only=False)
model.load_state_dict(bk['model'])
tmae = evaluate(model, tel, dev, gy, norm)
json.dump({**meta,'done':True,'hist':hist,'best_val_mae':best,'best_epoch':bestep,
           'test_mae':tmae,'epochs_run':len(hist)}, open(rf,'w'), indent=2)
print('TEST_MAE', tmae)
