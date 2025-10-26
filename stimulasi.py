"""
meta planner: dreamer + cem + predictive novelty
belajar dari surprise, pilih goal sendiri, planning optimal
"""
import numpy as np
from collections import deque

cfg = {
    'v_max': 2.0, 'dt': 0.05, 'd': 48, 'horizon': 10,
    'lr': 0.001, 'rays': 12, 'cem_n': 40, 'cem_top': 8, 'cem_iter': 3
}

# simulator fisika + sensor
class Sim:
    def __init__(self):
        self.pos = np.zeros(3)  # x, y, theta
        self.obs = []
    
    def reset(self, init, obs):
        self.pos = init.copy()
        self.obs = obs
    
    def step(self, v, delta):
        self.pos[0] += v * np.cos(self.pos[2]) * cfg['dt']
        self.pos[1] += v * np.sin(self.pos[2]) * cfg['dt']
        self.pos[2] += v * np.tan(delta) * cfg['dt']
        return self.lidar()
    
    def lidar(self):
        angles = np.linspace(0, 2*np.pi, cfg['rays'], endpoint=False)
        scan = np.full(cfg['rays'], 50.0)
        for i, a in enumerate(angles):
            ray = self.pos[2] + a
            for o in self.obs:
                dx = o['x'] - self.pos[0]
                dy = o['y'] - self.pos[1]
                if dx*np.cos(ray) + dy*np.sin(ray) > 0:
                    d = np.hypot(dx, dy)
                    scan[i] = min(scan[i], max(0, d - o['r']))
        return scan

# optimizer adam buat semua weight
class Opt:
    def __init__(self, lr=cfg['lr']):
        self.lr = lr
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, name, w, grad):
        if name not in self.m:
            self.m[name] = np.zeros_like(w)
            self.v[name] = np.zeros_like(w)
        
        self.t += 1
        self.m[name] = 0.9*self.m[name] + 0.1*grad
        self.v[name] = 0.999*self.v[name] + 0.001*grad**2
        
        m = self.m[name] / (1 - 0.9**self.t)
        v = self.v[name] / (1 - 0.999**self.t)
        
        return w - self.lr * m / (np.sqrt(v) + 1e-8)

# world model rekursif
class Brain:
    def __init__(self, in_dim):
        d = cfg['d']
        # encoder
        self.enc = np.random.randn(in_dim, d) * 0.02
        # dynamics gru-style
        self.wz = np.random.randn(d+2, d) * 0.02
        self.wr = np.random.randn(d+2, d) * 0.02
        self.wh = np.random.randn(d+2, d) * 0.02
        # output heads
        self.w_act = np.random.randn(d, 2) * 0.02
        self.w_rew = np.random.randn(d, 1) * 0.02
        # memory & optimizer
        self.errors = deque(maxlen=1000)
        self.opt = Opt()
    
    def encode(self, x):
        return np.tanh(x @ self.enc)
    
    def next(self, h, a):
        # prediksi state berikutnya (rekursif)
        ha = np.concatenate([h, a])
        z = 1 / (1 + np.exp(-(ha @ self.wz)))
        r = 1 / (1 + np.exp(-(ha @ self.wr)))
        h_new = np.tanh(np.concatenate([r*h, a]) @ self.wh)
        h_out = (1-z)*h + z*h_new
        rew = float(h_out @ self.w_rew)
        return h_out, rew
    
    def rollout(self, h0, actions):
        # imagine trajectory di kepala
        states = [h0]
        rews = []
        h = h0
        for a in actions:
            h, r = self.next(h, a)
            states.append(h)
            rews.append(r)
        return states, rews
    
    def act(self, h):
        return np.tanh(h @ self.w_act)
    
    def novelty(self, h):
        # novelty dari prediction error (bukan jarak)
        if len(self.errors) < 10:
            return 1.0
        avg = np.mean(list(self.errors))
        std = np.std(list(self.errors)) + 1e-8
        uncertainty = np.sum(h**2)
        return float(np.clip((uncertainty - avg) / std, 0, 2))
    
    def belajar(self, obs, act, obs_next, rew):
        # training kontinyu (bukan cuma pas surprise)
        h = self.encode(obs)
        h_target = self.encode(obs_next)
        h_pred, r_pred = self.next(h, act)
        
        # loss
        loss_dyn = np.sum((h_pred - h_target)**2)
        loss_rew = (r_pred - rew)**2
        total = loss_dyn + loss_rew
        
        self.errors.append(loss_dyn)
        
        # update weights
        grad = 2 * (h_pred - h_target)
        ha = np.concatenate([h, act])
        self.wh = self.opt.step('wh', self.wh, np.outer(ha, grad))
        
        grad_r = 2 * (r_pred - rew) * h_pred
        self.w_rew = self.opt.step('wr', self.w_rew, grad_r.reshape(-1,1))
        
        return total

# planner cem
class Planner:
    def __init__(self, brain):
        self.brain = brain
    
    def plan(self, h, goal, lidar):
        # cari action sequence terbaik pake cem
        mean = np.zeros((cfg['horizon'], 2))
        std = np.ones((cfg['horizon'], 2)) * 0.5
        
        for _ in range(cfg['cem_iter']):
            # sample kandidat
            samples = [mean + std*np.random.randn(cfg['horizon'],2) 
                      for _ in range(cfg['cem_n'])]
            samples = [np.clip(s, -1, 1) for s in samples]
            
            # evaluasi
            costs = [self._cost(h, s, goal, lidar) for s in samples]
            
            # ambil yang terbaik
            idx = np.argsort(costs)[:cfg['cem_top']]
            elites = [samples[i] for i in idx]
            
            # update distribusi
            mean = np.mean(elites, axis=0)
            std = np.std(elites, axis=0) + 0.1
        
        return mean[0]
    
    def _cost(self, h, actions, goal, lidar):
        states, rews = self.brain.rollout(h, actions)
        
        total = 0
        for i, (s, r) in enumerate(zip(states[1:], rews)):
            # jarak ke goal
            c_goal = np.sum((s[:2] - goal[:2]/50)**2)
            # collision
            c_col = 50 * np.sum(np.maximum(0, 2.0 - s[2:6]))
            # curiosity reward
            nov = self.brain.novelty(s)
            
            discount = 0.95**i
            total += discount * (c_goal + c_col - nov - r)
        
        return total

# agent utama
class Agent:
    def __init__(self, sim):
        self.sim = sim
        self.brain = Brain(in_dim=cfg['rays'])
        self.planner = Planner(self.brain)
    
    def jalan(self, steps=500):
        coverage = set()
        nov_total = 0
        
        for t in range(steps):
            lidar = self.sim.lidar()
            h = self.brain.encode(lidar)
            
            # pilih goal sendiri
            goal = self._pilih_goal(h, lidar)
            
            # planning
            act = self.planner.plan(h, goal, lidar)
            v = (act[0]+1) * cfg['v_max']/2
            delta = act[1] * 0.5
            
            # eksekusi
            lidar_next = self.sim.step(v, delta)
            
            # belajar
            rew = -np.hypot(self.sim.pos[0]-goal[0], self.sim.pos[1]-goal[1])
            loss = self.brain.belajar(lidar, act, lidar_next, rew)
            
            # tracking
            h_next = self.brain.encode(lidar_next)
            nov = self.brain.novelty(h_next)
            nov_total += nov
            
            cell = (int(self.sim.pos[0]), int(self.sim.pos[1]))
            coverage.add(cell)
            
            if t % 100 == 0:
                print(f"  step {t}: coverage={len(coverage)}, nov={nov:.2f}, loss={loss:.3f}")
        
        return {'coverage': len(coverage), 'novelty': nov_total/steps}
    
    def _pilih_goal(self, h, lidar):
        # cari arah dengan novelty tertinggi
        best = None
        best_score = -999
        
        for i in range(8):
            angle = i * np.pi/4
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # simulasi ke arah itu
            h_future, _ = self.brain.next(h, direction*0.5)
            nov = self.brain.novelty(h_future)
            
            # cek openness dari lidar
            ray_idx = int(i * cfg['rays']/8) % cfg['rays']
            openness = lidar[ray_idx] / 50
            
            score = nov * openness * (1 + np.random.rand()*0.2)
            if score > best_score:
                best_score = score
                best = direction * 15
        
        return best if best is not None else np.array([10, 10])

# musuh yang bikin level
class Musuh:
    def __init__(self):
        self.diff = 0.3
    
    def bikin_level(self, history):
        n_obs = int(3 + 7*self.diff)
        obs = []
        for i in range(n_obs):
            x = 10 + i*5 + np.random.randn()*3
            y = 10 + i*5 + np.random.randn()*3
            obs.append({'x': float(x), 'y': float(y), 'r': 1.5})
        return {'init': np.zeros(3), 'obstacles': obs}
    
    def adaptasi(self, hasil):
        if hasil['coverage'] > 40:
            self.diff = min(1.0, self.diff + 0.15)
        else:
            self.diff = max(0.1, self.diff - 0.05)

# main loop
def main(rounds=15):
    sim = Sim()
    agent = Agent(sim)
    musuh = Musuh()
    history = []
    
    for r in range(rounds):
        level = musuh.bikin_level(history)
        sim.reset(level['init'], level['obstacles'])
        
        print(f"round {r+1} (diff {musuh.diff:.2f}, obs {len(level['obstacles'])})")
        
        hasil = agent.jalan(steps=300)
        history.append(hasil)
        musuh.adaptasi(hasil)
        
        print(f"  -> coverage={hasil['coverage']}, novelty={hasil['novelty']:.3f}\n")
    
    avg_cov = np.mean([h['coverage'] for h in history])
    print(f"rata-rata coverage: {avg_cov:.1f}")

if __name__ == "__main__":
    main(15)
