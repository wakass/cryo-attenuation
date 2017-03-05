from scipy.optimize import minimize
from numpy import power, divide, log10
import numpy
from operator import mul

##Calculator to find the optimum attenuators for a high-frequency line in a cryogenic system
##This takes into account: 
##  * cooling power availabe at each separate stage 
##  * strengths of available attenatuors (Gs)
##  * desired power output at the end of the high-frequency lines (DUT_maxpower)
##
## Losses in coaxial lines that count towards heating are not included in the model

## Should a coaxial line be included, you should specify 
## Example data
####     Lengths between plates, used in coaxial
####     147mm PT1
####     113mm PT2
####     123 mm STill
####     Nbti:
####     126mm
####     172mm

#### Coaxial frequency dependent attenuation
##### sc 219 becu, according to spec has attenuation @ freq of db/m
####  FREQ		Atten
##### 0.5GHz	0.92
##### 1.0GHz	1.32
##### 5.0GHz	3.03
##### 10.0GHz	4.39
##### 20.0GHz	6.33
####     
####     Temperature of each stage with coaxes accounted for
####     Ts = [22e-3,   #mixing chamber
####     		0.09,   #100 mk
####     		0.8,    #still
####     		2.0,    #becu coax
####     		3.2,    #pt2
####     		24.6,   #becu coax
####     		46.0,   #pt1
####     		173.,   #becu coax 
####     		300.    #Room temperature
####     		]
####     Gs = [-13.,  #mixing chamber 
####     		-7., #100mk
####     		-20.,#still
####     		-1.62, #becu coax
####     		-3., 
####     		-2.97, #becu coax
####     		-3., 
####     		-1.98 #becu coax
####     		]
#no attenuation assumed from NbTi cables

coaxBeCuAtten = -6.33 / 1000. #example attenuation in db/mm (for a BeCu coax)

##desired characteristics
options = {
	'DUT_power_max': -37, #desired minimum power output (dBm) (at the end of the chain)
	'source_power': 24, #power of microwave source at the beginning of the transmission line (dBm)
	'attenuation_max' : 55, #maximum attenuation of the full transmission line (dB)
	
	#defined in order of end of chain (at DUT) to the "beginning" (usually room temperature)
	#stages also have a cooling power (CP), defined in Watt
	'Ts': [ {'type':'stage'	, 'T': 22e-3, 	'CP': 3e-6 			}, #type atten indicates an attenuator can be placed here and optimized, 'stage' means no thermalization (ignored)
			{'type':'stage'	, 'T': 0.09 , 	'CP': 1e-5  		}, 
			{'type':'stage'	, 'T': 0.8  ,	'CP': 3e-3			},
			{'type':'coax'	, 'T': 2.0  , 	'length':  123., 	'atten': coaxBeCuAtten }, #attenuation given in dB per mm, length in mm
			{'type':'stage'	, 'T': 3.2  , 	'CP': 1e-1 			}, 
			{'type':'coax' 	, 'T': 24.6 ,  	'length':  213., 	'atten': coaxBeCuAtten },
			{'type':'stage'	, 'T': 46.0 ,	'CP':10. 			},
			{'type':'coax' 	, 'T': 173. ,	'length': 147., 	'atten': coaxBeCuAtten }, #coaxes attenuate at the temperature halfway between stages, assuming flat temperature profile.
			{'type':'RT'	, 'T': 300.0 ,    						},
			], #temperatures of all stages and coaxial lines (K)
	
	#list of available attenuators (dB)
	'Gs': numpy.append(-numpy.arange(0,10,0.5),[-10.,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20.,-25,-30.,-40.]) 
}



## Now define the constraints for the minimization procedure.
##
##

CPs = []
for x in options['Ts']:
	if x['type'] == 'stage':
		CPs.append(x['CP'])
	else:
		CPs.append(1.)

Ts = [ x['T'] for x in options['Ts'] ] #collect temperatures of all stages

#ineq means fun() should be non-negative >=0
const = ({'type': 'ineq',
		'fun': lambda g: numpy.array(rmp(CPs[0],g[0]) - options['DUT_power_max'])}, #1.a power output at end should be at least..
		
		{'type': 'ineq',
		'fun': lambda g: numpy.array(dBmtoWatt(options['source_power'])*reduce(mul,g) - options['DUT_power_max'])}, #1.b and should be attainable with all the attenuation
		
		{'type': 'ineq',
		'fun': lambda g: numpy.array(GtodB(reduce(mul,g)) + options['attenuation_max']) }, #2. no more than total x dB attenuation
		
		# {'type': 'ineq',
		# 'fun': lambda g: numpy.array( rmp(CPs[1],g[1]) - mip(CPs[0],g[0]) )}, 	#3. what power comes out of previous chain should be bigger maximally than the next chain. 
		# {'type': 'ineq',														# Otherwise, power would be lost

)


for i,x in enumerate(options['Ts']):
	if x['type'] == 'stage':
		#3.
		#find the next stage, that has thermalization at an attenuator
		#and constrain it so that whatever power comes out of the attenuator higher up in the chain 
		#is never more than the maximum dissipation at the next stage
		for j,x_next in enumerate(options['Ts'][i+1:]):
			if x_next['type'] == 'stage':
				if ('CP' in x_next) and ('CP' in x):
					k = i + j
					# print 'k %d' % k
					const += ({
								'type':'ineq',
								'fun': lambda g,*args: numpy.array(rmp(args[0],g[args[3]]) - mip(args[1],g[args[2]])),
								'args': (x_next['CP'], x['CP'],i,k)
								},)
					break
	if x['type'] == 'coax':
		const += ({
					'type':'eq',
					'fun': lambda g,*args: numpy.array(g[args[2]] - dBtoG( args[0]*args[1] )),
					'args':(x['atten'],x['length'],i)
				},)
# print 'constraints:'
# print len(const)

const_ext = ()
for i,x in enumerate(options['Ts']):
	if x['type'] == 'stage':
		const_ext = const_ext + ({
					'type': 'eq',
					'fun': lambda g,*i: numpy.array( min(abs(dBtoG(options['Gs']) - g[i[0]] )) ), #make sure the attens are in the range of available to buy attens
					'args': (i,)
					},)



#conversion functions
def dBmtoWatt(dbm):
	return power(10,dbm/10.)*1e-3
def GtodB(G):
	return log10(G)*10.0
def dBtoG(dB):
	return 10.**(dB/10.0)
	
	
def electronTemperature(Ts,Gs):
	#b is the vector containing attenuations
	Te = Ts[0];
	for n, T in enumerate(Ts):
		G_t = Gs[0]
		for i in range(1, n): #calculate total attenuation up to n
			G_t *= Gs[i]
		
		Te += T*G_t #electron temperature increased by ..
	return Te


##Cooling power related functions
#the maximum input power to an attenuator, before cooling power is exceeded.
def mip(cp, g):
	return cp / (1 - g)
#the max output power of a stage, based on not overheating.
def rmp(cp,g):
	return mip(cp,g)*g

#get the initial guess for the parameters of the minimization problem
def get_g0():
	g0 = numpy.array([])
	for x in options['Ts']:
		if x['type'] == 'stage':
			g0 = numpy.append(g0,.5)
		if x['type'] == 'coax':
			atten = x['atten'] * x['length']
			g0 = numpy.append(g0,dBtoG(atten))
	return g0
g0=get_g0()

#bounds for the parameters
bnds = []
for x in options['Ts']:
	if x['type'] == 'stage' or  x['type'] == 'coax':
		bnds.append((1e-6, .9))

def et(gs):
	#condition everything to go into electronTemperature
	

	#gs is the attenuation at each stage
	return electronTemperature(Ts,gs)


def print_stats(res):
	atten = GtodB(res.x)
	print 'Attenuations (dB): %s' % atten
	print 'Te (mK): %g' % (electronTemperature(Ts,res.x)*1000.)
	print 'Total attenuation in attenuators only: %s' % sum(atten) 
	print 'Total attenuation in attenuators and coaxes: %s' % (sum(atten) )
	print 'Max Power on DUT uncooled: %s' % (24 + sum(atten) )
	print 'Max Power on DUT cooled: %s' % (10*log10(rmp(CPs[0],res.x[0]) /1e-3) )


res = minimize(et, g0 ,constraints=const,method='SLSQP',bounds=bnds,options={'ftol':1e-12,'maxiter':10000,'disp':True})
print_stats(res)

#then get the values and restrain them further to the available attenuation values.
Gs = res.x
res = minimize(et, Gs, constraints=const_ext+const,method='SLSQP',bounds=bnds,options={'ftol':1e-12,'maxiter':10000,'disp':True})

print_stats(res)