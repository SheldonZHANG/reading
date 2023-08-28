# *-----------------------------------------------------------------
# | PROGRAM NAME: iptw demo.py
# | DATE: 11/27/19
# | CREATED BY: MATT BOGARD 
# | PROJECT FILE:           
# *----------------------------------------------------------------
# | PURPOSE:  inverse probablilty of treatment weighted regression (IPTW)  
# *----------------------------------------------------------------


# illustrating results from IPTW regression as an alternative to propensity score matching in python


# attempt to override sci notation in output 
pd.set_option('display.float_format', lambda x: '%3.f' % x)

# Data: lalonde - used by Dehejia and Wahba (1999) to evaluate propensity score matching.

# Ref: Dehejia, Rajeev and Sadek Wahba. 1999.`Causal Effects in Non-Experimental Studies: Re-Evaluating 
# the Evaluation of Training Programs.'' Journal of the American Statistical Association 94 (448):
# 1053-1062

# Ref: Austin PC. An Introduction to Propensity Score Methods for Reducing the Effects of 
# Confounding in Observational Studies. Multivariate Behav Res. 2011;46(3):399–424.
#  doi:10.1080/00273171.2011.568786

# Descriptions: 

# treat - an indicator variable for treatment status.
# age - age in years.
# educ - years of schooling.
# black - indicator variable for blacks.
# hisp- indicator variable for Hispanics.
# married - indicator variable for martial status.
# nodegr - indicator variable for high school diploma.
# re74 - real earnings in 1974.
# re75 - real earnings in 1975.
# re78  - real earnings in 1978.


# Import pandas  and numpy
import pandas as pd
import numpy as np

#--------------------------------------
# Loading & examining our data
#--------------------------------------
 
# read data (also available at: https://github.com/BioSciEconomist/econometricsense/blob/master/lalonde.csv
df = pd.read_csv('/Users/amandabogard/Google Drive/R Training/lalonde.csv')

### inspecting data

df.columns
df.head()
df.tail()
df.info()
df.shape
df.index

df.describe()

#-----------------------------------
# estimate propensity scores and calculate weights
#----------------------------------

import statsmodels.api as sm # import stastmodels
import statsmodels.formula.api as smf # this allows us to use an explicit formulation
 

model = smf.glm('treat ~ age + educ + black + hispan + married + nodegree + re74 + re75', data=df, family=sm.families.Binomial(link = sm.genmod.families.links.logit))
result = model.fit()
result.summary()

# add propensity scores 
df['ps'] =  result.fittedvalues

# explore
df.ps.plot( kind='hist', normed=True, bins=30)
df.ps.describe()

# assess common support
df.boxplot(column='ps', by='treat', rot=90)

# calculate propensity score weights for ATT (Austin, 2011)
df['wt'] = np.where(df['treat']==1, 1,df.ps/(1-df.ps))

df.wt.describe() # check

# create weighted data frame (controls are weighted up by 'wt' to look like treatments)

from statsmodels.stats.weightstats import DescrStatsW

wdf = DescrStatsW(df, weights=df.wt, ddof=1) 

#----------------------------------
# balance diagnostics
#----------------------------------

#### descriptives unmatched or unweighted  data

# split data and use numpy to deal with pandas issues elated to using group by with means on integer 0/1 values

trt = df[df['treat'].isin([1])]  
ctrl =df[df['treat'].isin([0])] 

# descriptives for unmatched or unweighted data
np.average(trt.black)
np.average(ctrl.black)

np.average(trt.re74)
np.average(ctrl.re74)

np.average(trt.age)
np.average(ctrl.age)

# descriptives with weighted data
np.average(trt.black, weights =  trt.wt)
np.average(ctrl.black, weights = ctrl.wt)

np.average(trt.re74, weights =  trt.wt)
np.average(ctrl.re74, weights = ctrl.wt)

np.average(trt.age, weights =  trt.wt)
np.average(ctrl.age, weights = ctrl.wt)

#----------------------------------------
# estimation of treatment effects
#----------------------------------------

### unweighted comparisons

# regression using smf
lmresults = smf.ols('re78 ~ treat', data=df).fit()

# inspect results
lmresults.summary()  # b_est = -635

### weighted comparisons

# weighted differences in outcome
np.average(trt.re78, weights =  trt.wt)
np.average(ctrl.re78, weights = ctrl.wt)

# iptw regression
iptwResults = smf.wls('re78 ~ treat', data=df, weights = df.wt).fit()
iptwResults.summary() # b_est =  1214


# this result if very similar to the result using R Matchit 
# see: https://gist.github.com/BioSciEconomist/6e8b3fba57d00761527215128bdbf11a 


