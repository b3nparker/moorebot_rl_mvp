"""Small 2D continuous-control obstacle navigation environment.
This is a simulator, NOT a calibrated Moorebot/Gazebo model.
Observation: 9 idealized range rays, goal distance, goal bearing, last action.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WORLD = 6.0
WALLS = [
    (1.9, 0.0, 2.15, 3.8),
    (3.85, 2.2, 4.1, 6.0),
    (0.0, 0.0, 6.0, 0.08),
    (0.0, 5.92, 6.0, 6.0),
    (0.0, 0.0, 0.08, 6.0),
    (5.92, 0.0, 6.0, 6.0),
]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_steps=300):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.radius = 0.13
        self.dt = 0.15
        self.max_range = 2.0
        self.ray_angles = np.linspace(-math.pi * 0.75, math.pi * 0.75, 9)
        self.action_space = spaces.Box(low=np.array([-1., -1.], dtype=np.float32), high=np.ones(2, dtype=np.float32))
        # ranges normalized to [0,1]; goal distance normalized to [0,1]; bearing normalized to [-1,1]; prior action
        self.observation_space = spaces.Box(low=np.array([0.]*10+[-1.,-1.,-1.], dtype=np.float32),
                                            high=np.ones(13, dtype=np.float32), dtype=np.float32)
        self.pos = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.theta = 0.
        self.steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.fig = self.ax = None

    def _collision(self, pos):
        x,y = pos
        for x0,y0,x1,y1 in WALLS:
            cx = max(x0, min(x1, x))
            cy = max(y0, min(y1, y))
            if (x-cx)**2 + (y-cy)**2 <= self.radius**2:
                return True
        return False

    def _sample_point(self):
        for _ in range(10000):
            p = self.np_random.uniform(.35, WORLD-.35, size=2)
            if not self._collision(p):
                return p
        raise RuntimeError('No free spawn point')

    def _ranges(self):
        # Raycast against axis aligned rectangles with an explicit max range.
        out=[]
        for offset in self.ray_angles:
            angle=self.theta+offset
            dx,dy=math.cos(angle),math.sin(angle)
            best=self.max_range
            for x0,y0,x1,y1 in WALLS:
                if abs(dx)>1e-10:
                    for xx in (x0,x1):
                        t=(xx-self.pos[0])/dx
                        yy=self.pos[1]+t*dy
                        if t>=0 and y0<=yy<=y1: best=min(best,t)
                if abs(dy)>1e-10:
                    for yy in (y0,y1):
                        t=(yy-self.pos[1])/dy
                        xx=self.pos[0]+t*dx
                        if t>=0 and x0<=xx<=x1: best=min(best,t)
            out.append(np.clip(best,0,self.max_range))
        return np.asarray(out,dtype=np.float32)

    def _obs(self):
        delta=self.goal-self.pos
        goal_dist=min(float(np.linalg.norm(delta))/8.5,1.)
        bearing=wrap(math.atan2(delta[1],delta[0])-self.theta)/math.pi
        return np.concatenate((self._ranges()/self.max_range,
                np.array([goal_dist,bearing],dtype=np.float32),self.last_action)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos=self._sample_point()
        self.goal=self._sample_point()
        while np.linalg.norm(self.goal-self.pos)<2.0:
            self.goal=self._sample_point()
        self.theta=float(self.np_random.uniform(-math.pi,math.pi))
        self.last_action[:]=0
        self.steps=0
        return self._obs(), {'position':self.pos.copy(),'goal':self.goal.copy()}

    def step(self, action):
        a=np.clip(np.asarray(action,dtype=np.float32),-1,1)
        old_dist=float(np.linalg.norm(self.goal-self.pos))
        # Forward-only speed 0..0.45m/s, heading control +/- 1.8rad/s
        v=float((a[0]+1)*0.225)
        omega=float(a[1]*1.8)
        new_theta=wrap(self.theta+omega*self.dt)
        next_pos=self.pos+v*self.dt*np.array([math.cos(new_theta),math.sin(new_theta)])
        hit=self._collision(next_pos)
        if not hit:
            self.pos=next_pos
        self.theta=new_theta
        self.last_action=a.copy()
        self.steps+=1
        dist=float(np.linalg.norm(self.goal-self.pos))
        success=dist<0.28
        # Progress potential + arrival bonus. Reward values are illustrative.
        reward=6*(old_dist-dist)-0.012-0.006*abs(float(a[1]))
        if hit: reward-=10
        if success: reward+=40
        terminated=bool(hit or success)
        truncated=bool(self.steps>=self.max_steps)
        return self._obs(), float(reward), terminated, truncated, {
            'success': bool(success), 'collision':bool(hit), 'distance':dist,
            'position':self.pos.copy(), 'goal':self.goal.copy()}

    def render(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        if self.fig is None:
            plt.ion()
            self.fig,self.ax=plt.subplots(figsize=(6,6))
        ax=self.ax; ax.clear()
        for x0,y0,x1,y1 in WALLS:
            ax.add_patch(patches.Rectangle((x0,y0),x1-x0,y1-y0,color='#596779'))
        ax.plot(self.goal[0],self.goal[1],marker='*',markersize=18,color='tab:green')
        ax.add_patch(patches.Circle(self.pos,self.radius,color='tab:blue'))
        ax.arrow(self.pos[0],self.pos[1],.28*math.cos(self.theta),.28*math.sin(self.theta),
                 head_width=.08,color='tab:orange')
        ax.set(xlim=(0,WORLD),ylim=(0,WORLD),aspect='equal',title=f'Step {self.steps} | distance {np.linalg.norm(self.goal-self.pos):.2f} m')
        plt.pause(.001)

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig=None
