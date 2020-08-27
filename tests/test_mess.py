from pytest import approx
from pytest_cases import fixture, parametrize_with_cases, parametrize, case
import pytest
from environments2 import DogBarometer, State
from DB_testing import Test_obj as Tes_obj #careful importing anything with the word test in it.
import pandas as pd
import random

#A list of scenarios to test over

#symmetric',
db_one=dict(
rain_coat_rw=4,
wait_rw=-1,
p_pressure=(0.6,0.6))

# pressure_asym',
db_two=dict(
rain_coat_rw=4,
wait_rw=-1,
p_pressure=(0.6,0.4))

# barrom_asym',
db_three=dict(
b_accuracy=(0.7,0.8)
)

#'start_pres'
db_four=dict(
init_p_pressure_high=0.3,
p_pressure=(0.6,0.4))


        
#these objects will generate test data which is reused by the tests before being dumped
        
class T_obj(Tes_obj):
    def __init__(self,db,trials=10000):
        super().__init__(db,e_state=State,decider=Tes_obj.make_same_action(0),max_per=trials)
        
class T_M_obj(Tes_obj):
        def __init__(self,db,trials=10000):
            super().__init__(db,e_state=State,decider=Tes_obj.random_action,trials=trials)

#this  is from pytest_cases.
#we want to parameterize fixtures, and feed them into test functions

def case_list():
        return [(obj,param) for param in [db_one,db_two,db_three,db_four] for obj in [T_obj,T_M_obj]]
        
def generate_list_names():
    return [obj+'_'+param for param in ['symmetric','pressure_asym','barrom_asym','start_pres'] for obj in ['Simple 0 repeat action','random_actions']]


class CasesFoo:
    @parametrize(who=case_list(),ids=generate_list_names(),idgen=None,scope="module")
    def case_generator(self,who):
        return who[0],who[1]
    


@fixture(scope="session")
@parametrize_with_cases("param",cases=CasesFoo)
def db_obj(param):
    
    db=DogBarometer(**param[1])
    #print('setup', request.param)
    obj=param[0]
    db_obj=obj(db)
    yield db_obj
    #print('tear down',param)

    
def test_pressure_auto_correlate(db_obj):
    db=db_obj.db
    df=db_obj.df
    trials=db_obj.trials
    

    #test pressure is autocorrelates
    pressure_high_high=((df.pressure_==df.pressure)&(df.pressure==1)).sum()/(df.pressure==1).sum()
    assert pressure_high_high==approx(db.p_pressure_high_high,abs=0.02)

    pressure_low_low=((df.pressure_==df.pressure)&(df.pressure==0)).sum()/(df.pressure==0).sum()
    assert pressure_high_high==approx(db.p_pressure_high_high,abs=0.02)
    
def test_pres_cause_weather(db_obj):
    db=db_obj.db
    df=db_obj.df
    trials=db_obj.trials
    

    #test pressure causes weather high pressure sun
    wp_high=((df.weather_==df.pressure)&(df.pressure==1)).sum()/(df.pressure==1).sum()
    assert wp_high==approx(db.weather_predict_high,abs=0.02)

    #test pressure causes weather low pressure= rain
    wp_high=((df.weather_==df.pressure)&(df.pressure==0)).sum()/(df.pressure==0).sum()
    assert wp_high==approx(db.weather_predict_low,abs=0.02)
    
def test_barom_accuracy(db_obj):
    db=db_obj.db
    df=db_obj.df
    trials=db_obj.trials
    

    #test barometer accuracy high-high case
    barrom_accuracy_high=((df.barometer_==df.pressure_)&(df.pressure_==1)&(df.action!=1)).sum()/((df.pressure_==1)&(df.action!=1)).sum()
    assert barrom_accuracy_high==approx(db.b_accuracy_high,abs=0.03)

    #test barometer accuracy low low case
    #trickier because we need to ex-out the effect of the barometer having been pressed.
    barrom_accuracy_low=((df.barometer_==df.pressure_)&(df.pressure_==0)&(df.action!=1)).sum()/((df.pressure_==0)&(df.action!=1)).sum()
    assert barrom_accuracy_low==approx(db.b_accuracy_low,abs=0.03)
    
    
@pytest.fixture(scope="module",params=[db_one,db_two])
def db(request):
    return DogBarometer(**request.param)


def test_do_nothing(db):
    #test not done doing nothing

    s,r,d,info=db.step(0)

    s=State(*s)
    assert not d
	
def test_press_works(db):
    #test baromter press works

    s,r,d,info=db.step(1)

    s=State(*s)
    assert s.barometer==1
    assert not d
	
def test_start_inside(db):
    #test always start inside

    s=db.reset()
    s=State(*s)
    assert db.inside
	
def test_go_outside_ends(db):
    #test going outside ends

    db.reset()
    s,r,d,info=db.step(2)
    s=State(*s)
    assert not db.inside
    assert db.coat
    assert d
    
def test_go_outside_with_coat_end(db):
    #test going outside without coat ends

    db.reset()
    s,r,d,info=db.step(3)
    s=State(*s)
    assert not db.inside
    assert not db.coat
    assert d