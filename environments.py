import gym
from gym.spaces import Space, MultiBinary, Discrete
from collections import namedtuple
import random
import pandas as pd
from tabular import TabularMemory
import numpy as np



State=namedtuple('State',['barometer','pressure','weather'])
VState=namedtuple('VState',['barometer','weather'])


class DogBarometer(gym.Env):
    
    actions={
        0:'wait',
        1: 'press',
        2: 'exit_with_coat',
        3: 'exit_no_coat',      
        }
    
    def __init__(self,b_accuracy=(0.9,0.9),weather_predict=(0.9,0.9),p_pressure=(0.5,0.5),rain_coat_rw=8,rain_no_coat_rw=-8,
                    sun_no_coat_rw=8,sun_coat_rw=-8,wait_rw=0,hidden=False,init_p_pressure_high=0.5,time_limit=100):
        self.time=0
        self.time_limit=time_limit
        self.inside=True
        self.coat=0
        self.hidden=hidden
        
        self.init_p_pressure=init_p_pressure_high
        
        self.b_accuracy_low=b_accuracy[0]
        self.b_accuracy_high=b_accuracy[1]
        
        
        self.weather_predict_low=weather_predict[0]
        self.weather_predict_high=weather_predict[1]
        
        self.p_pressure_low_low=p_pressure[0]
        self.p_pressure_high_high=p_pressure[1]
        


        self.rain_coat_rw=rain_coat_rw
        self.rain_no_coat_rw=rain_no_coat_rw
        self.sun_no_coat_rw=sun_no_coat_rw
        self.sun_coat_rw=sun_coat_rw
        self.wait_rw=wait_rw
                
        self.initiate_state()
        self.number_of_actions=len(self.actions)
        self.memory=TabularMemory(self.number_of_actions)
        
        self.observation_space=ObservationSpace(3)
        self.State=State
        
        if hidden:
            self.observation_space=ObservationSpace(2)
            self.State=VState
            
        self.action_space=Discrete(4)
        
    def __repr__(self):
        return  str({'hidden':self.hidden,**self.rewards,**self.transitions})

    @property
    def rewards(self):

        return  dict(rain_coat_rw=self.rain_coat_rw,
       rain_no_coat_rw=self.rain_no_coat_rw,
        sun_no_coat_rw=self.sun_no_coat_rw,
        sun_coat_rw=self.sun_coat_rw,
        wait_rw=self.wait_rw)
        
    @property
    def transitions(self):
    
        transitions=dict(
        init_p_pressure=self.init_p_pressure,
        b_accuracy_low=self.b_accuracy_low,
        b_accuracy_high=self.b_accuracy_high,       
        weather_predict_low=self.weather_predict_low,
        weather_predict_high=self.weather_predict_high,
        p_pressure_low_low=self.p_pressure_low_low,
        p_pressure_high_high=self.p_pressure_high_high)
        return transitions
        
    
        
    def step(self,action):
    
        self.time+=1
    
        #check action is valid
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        #assume haven't pressed the barometer
        press=False
        
        #assume not finished
        done=False
         
        if action<=1: #have not exited
            if self.actions[action]=='press':
                press=True
                        
        else:
            self.inside=False
            done=True
            if self.actions[action]=='exit_with_coat':
                self.coat=True
            elif self.actions[action]=='exit_no_coat':
                self.coat=False
            
        self._evolve_state(press=press)
        reward=self.reward_get(self.inside,self.coat)       
        info={'time':self.time,'inside':self.inside,'coat':self.coat,'weather':self.weather}
        
        return self._return_array(),reward,done,info
            
    
    def reset(self):
        self.initiate_state()

        return self._return_array()
        
    def _return_array(self):
        if self.hidden:
            arr=np.array(self.obscure_state(self.state))
        else:
            arr=np.array(self.state)
        return arr
        
    @staticmethod
    def obscure_state(state):
        return VState(**{k:val for k, val in state._asdict().items() if k in VState._fields})  
    
    def reward_get(self,inside,coat):
        
        reward=self.wait_rw
        
        # if the dog has left the kennel
        if inside==False:
            if coat:
                if self.state.weather==0:
                    reward=self.rain_coat_rw
                else:
                    reward=self.rain_no_coat_rw
            else:
                if self.state.weather==1:
                    reward=self.sun_no_coat_rw
                else:
                    reward=self.sun_coat_rw
        return reward
                
    
    def initiate_state(self):
        self.time=0
        self.inside=True
        self.coat=0
        
        pressure=0
        if random.random()<self.init_p_pressure: pressure=1
            
        
        weather=0
        #if random.random()<self.p_sun: weather=1
        
        barometer=0
        #barometer=(pressure+1)%2
        #if random.random()<self.b_accuracy: barometer=pressure
         
        
        
        self.state=State(weather=weather,barometer=barometer,
                         pressure=pressure)
            
        self.weather=weather
        self.pressure=pressure
        self._evolve_state()
        
   
        
    def _evolve_state(self,press=False):
  

        
        #pressure depends on previous pressure
        if self.state.pressure==0:
            #set marginal prob for weather based on prev pressure
            weather_predict=self.weather_predict_low
            
            pressure=1 # assume  pressure is high
            b_accuracy=self.b_accuracy_high #set barometer accuracy accordingly
            if random.random()<self.p_pressure_low_low:
                pressure=0 #it is low pressure
                b_accuracy=self.b_accuracy_low #adjust barometer accuracy

        elif self.state.pressure==1:
            #set marginal prob for weather based on prev pressure
            weather_predict=self.weather_predict_high
            
            
            pressure=0 #assume  pressure is low
            b_accuracy=self.b_accuracy_low #set barometer accuracy accordingly
            if random.random()<self.p_pressure_high_high:
                pressure=1 #it is high pressure
                b_accuracy=self.b_accuracy_high #adjust barometer accuracy
        else:
            print(self.state)
            raise ValueError
                
        #weather is a function of previous pressure
        if random.random()<weather_predict:
            weather=self.state.pressure #weather is doing what you would expect
        else:
            weather=(1+self.state.pressure)%2 #weather is misbehaving
        
        #barometer driven by current pressure
        if random.random()<b_accuracy:
            barometer=pressure #barometer is correct
        else:
            barometer=(1+pressure)%2 #barometer is incorrect
          
        #except when the barometer is pressed
        if press:
            barometer=1 #barometer will say high pressure
        
        self.weather=weather
        self.pressure=pressure
        
        self.state=State(barometer=barometer,weather=weather,
                         pressure=pressure)
                         
class ObservationSpace(MultiBinary):
        def contains(self, x):
            if isinstance(x, (list,State)):
                x = np.array(x)  # Promote list to array for contains check
            return ((x==0) | (x==1)).all()

    