
# *-----------------------------------------------------------------
# | PROGRAM NAME: ex matching.py
# | DATE: 6/25/21
# | CREATED BY: MATT BOGARD
# | PROJECT FILE:
# *----------------------------------------------------------------
# | PURPOSE: very basic matching and IPTW analysis with balance diagnostics
# *----------------------------------------------------------------

# see: https://stats.stackexchange.com/questions/206832/matched-pairs-in-python-propensity-score-matching

# see also: https://gist.github.com/BioSciEconomist/826556977841cbf966f03542b2be0c55

# https://gist.github.com/BioSciEconomist/d97a720c8b58fded158b9cb51acb5a70

# see also: https://medium.com/@bmiroglio/introducing-the-pymatch-package-6a8c020e2009 

# !!!!!!! WARNING - THIS IMPLEMENTATION OF MATCHING HAS A LOT OF PROBLEMS
#         - IT ESSENTIALLY MATCHES WITH REPLACEMENT AND HAS NO WAY OF 
#           SPECIFYING CALIPERS 

# results are not so different from using R MatchIt using matching with replacement

import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def get_matching_pairs(treated_df, non_treated_df, scaler=True):

    treated_x = treated_df.values
    non_treated_x = non_treated_df.values

    if scaler == True:
        scaler = StandardScaler()

    if scaler:
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(non_treated_x)
    distances, indices = nbrs.kneighbors(treated_x)
    indices = indices.reshape(indices.shape[0])
    matched = non_treated_df.iloc[indices]
    return matched



# read lalonde data

df = pd.read_csv("/Users/mattbogard/Google Drive/Data/lalonde.csv")

df['ID'] = range(0,614) # create ID

# create propensity score
import statsmodels.api as sm # import stastmodels
import statsmodels.formula.api as smf # this allows us to use an explicit formulation
 

model = smf.glm('treat ~ age + educ + race + married + nodegree + re74 + re75', data=df, family=sm.families.Binomial(link = sm.genmod.families.links.logit))
result = model.fit()
result.summary()

# add propensity scores 
df['ps'] =  result.fittedvalues

# explore
# df.ps.plot( kind='hist', normed=True, bins=30)
df.ps.describe()

# assess common support
df.boxplot(column='ps', by='treat', rot=90)
df.boxplot(column='ps', by='married', rot=90)

# data prep for matching

mask = df['treat'] == 1
trt = df[mask]
trt = trt[['ps']]

mask = df['treat'] == 0
ctrl = df[mask]
ctrl = ctrl[['ps']]

matched_df = get_matching_pairs(trt, ctrl)




#
# TO DO: modify algo to allow caliper based matching
#


# def get_matching_pairs(treated_df, non_treated_df, caliper, scaler=True):

#     treated_x = treated_df.values
#     non_treated_x = non_treated_df.values

#     if scaler == True:
#         scaler = StandardScaler()

#     if scaler:
#         scaler.fit(treated_x)
#         treated_x = scaler.transform(treated_x)
#         non_treated_x = scaler.transform(non_treated_x)

#     nbrs = NearestNeighbors(n_neighbors=1, radius = caliper, algorithm='ball_tree').fit(non_treated_x)
#     distances, indices = nbrs.kneighbors(treated_x)
#     indices = indices.reshape(indices.shape[0])
#     matched = non_treated_df.iloc[indices]
#     return matched


# df.ps.std()


# matched_df = get_matching_pairs(trt, ctrl,1)


#
# stack data
#

df_match = pd.concat([trt,matched_df])

# convert index back to ID
df_match['ID'] = df_match.index 

# add outcomes and tmatching variables
xvars = ['ID','treat', 'age', 'educ', 'race', 'married', 'nodegree', 're74',
       're75', 're78']

df_match = df_match.merge(df[xvars], on = ['ID'], how = 'left')

# check
df_match.groupby(['treat']).size().reset_index(name='count')



# double check
chk = df_match.merge(df[['ID','ps','treat']], on = ['ID'], how = 'left')

# check duplicates
tmp = df_match.groupby(['ID']).size().reset_index(name='count') # count duplicates
tmp = tmp.sort_values(['count'], ascending=[False]) # sort
print(tmp) # check


# the duplicates here verify this amounts to matching with replacement


#--------------------------------
# balance disgnostics 
#-------------------------------

#
# before matching
#

# create copy of analysis file with covariates of interest
tmp = df[['treat','age','educ','race','married','nodegree','re74','re75']]

tmp =pd.get_dummies(tmp,columns = ['race'])

 
# test group means
mask = tmp['treat'] == 1
tmp1A = tmp[mask]
tmp2A =  pd.DataFrame(tmp1A.mean())
tmp2A.reset_index(inplace=True)
tmp2A.columns = ['varname','Mean A']
 
# control group means
mask = tmp['treat'] == 0
tmp1B = tmp[mask]
tmp2B =  pd.DataFrame(tmp1B.mean())
tmp2B.reset_index(inplace=True)
tmp2B.columns = ['varname','Mean B']
 
# bring data togehter in a wide file
tmp3 = pd.merge(tmp2A, tmp2B[['varname','Mean B']], on='varname', how='left')
tmp3['diff'] = tmp3['Mean B'] - tmp3['Mean A']
 
# note: According to Ho et al.'s article (http://imai.princeton.edu/research/files/matchit.pdf)
# "the standardize = TRUE option will print out standardized versions of the balance measures,
# where the mean difference is standardized (divided) by the standard deviation in the original
# treated group."
 
# test group sd
mask = tmp['treat'] == 1
tmp1A = tmp[mask]
tmp2A =  pd.DataFrame(tmp1A.std())
tmp2A.reset_index(inplace=True)
tmp2A.columns = ['varname','StdDev A']
 
# add sd
tmp4 = pd.merge(tmp3, tmp2A[['varname','StdDev A']], on='varname', how='left')
 
# calculate absolute standardized mean difference
tmp4['ASMD'] = abs(tmp4['diff']/tmp4['StdDev A'])
 
balance_unmatched = tmp4 
 
#
# after matching
#

 
# create copy of analysis file with covariates of interest
tmp = df_match[['treat','age','educ','race','married','nodegree','re74','re75']]
 
# test group means
mask = tmp['treat'] == 1
tmp1A = tmp[mask]
tmp2A =  pd.DataFrame(tmp1A.mean())
tmp2A.reset_index(inplace=True)
tmp2A.columns = ['varname','Mean A']
 
# control group means
mask = tmp['treat'] == 0
tmp1B = tmp[mask]
tmp2B =  pd.DataFrame(tmp1B.mean())
tmp2B.reset_index(inplace=True)
tmp2B.columns = ['varname','Mean B']
 
tmp3 = pd.merge(tmp2A, tmp2B[['varname','Mean B']], on='varname', how='left')
tmp3['diff'] = tmp3['Mean B'] - tmp3['Mean A']
 
# note: According to Ho et al.'s article (http://imai.princeton.edu/research/files/matchit.pdf)
# "the standardize = TRUE option will print out standardized versions of the balance measures,
# where the mean difference is standardized (divided) by the standard deviation in the original
# treated group."
 
# test group sd (this is the original unmatched treatment group data)
 
# create copy of original analysis file with covariates of interest
tmp = df[['treat','age','educ','race','married','nodegree','re74','re75']]
 
mask = tmp['treat'] == 1
tmp1A = tmp[mask]
tmp2A =  pd.DataFrame(tmp1A.std())
tmp2A.reset_index(inplace=True)
tmp2A.columns = ['varname','StdDev A']
 
tmp4 = pd.merge(tmp3, tmp2A[['varname','StdDev A']], on='varname', how='left')
tmp4['ASMD'] = abs(tmp4['diff']/tmp4['StdDev A'])

balance_matched = tmp4


#-------------------------------
# estimation of treatment effects
#--------------------------------

# regression using smf
results = smf.ols('re78 ~ treat', data=df).fit()

results.summary()  # b_est = -635

# matched results
results = smf.ols('re78 ~ treat', data=df_match).fit()


results.summary()  # b_est = 1788 - this is much higer than ~ 1200 we get from R matchit


#-------------------------------
# compare to regression with controls
#--------------------------------

# regression using smf
results = smf.ols('re78 ~ treat + age + educ + race + married + nodegree + re74 + re75', data=df).fit()

results.summary()  # b_est = 1548


#-----------------------------------
# compare to IPTW regression
#----------------------------------


# trim weights
perc =[.10, .90]
df.ps.describe(percentiles = perc)

tmp = df

tmp['ps'] = np.where(tmp['ps'] < .025, .025,tmp.ps)
tmp['ps'] = np.where(tmp['ps'] > .72, .72,tmp.ps)


# check
tmp.groupby(['treat']).size().reset_index(name='count')
tmp.ps.describe()


# calculate propensity score weights for ATT (Austin, 2011)
tmp['wt'] = np.where(tmp['treat']==1, 1,tmp.ps/(1-tmp.ps))

tmp.wt.describe() # check

tmp =pd.get_dummies(tmp,columns = ['race'])


# create weighted data frame (controls are weighted up by 'wt' to look like treatments)

# from statsmodels.stats.weightstats import DescrStatsW

# wdf = DescrStatsW(tmp, weights=tmp.wt, ddof=1) 


# descriptives/balance with weighted data

trt = tmp[tmp.treat == 1]
ctrl = tmp[tmp.treat == 0]

np.average(trt.race_black, weights =  trt.wt)
np.average(ctrl.race_black, weights = ctrl.wt)

np.average(trt.re74, weights =  trt.wt)
np.average(ctrl.re74, weights = ctrl.wt)

np.average(trt.age, weights =  trt.wt)
np.average(ctrl.age, weights = ctrl.wt)

# iptw regression
iptwResults = smf.wls('re78 ~ treat', data=tmp, weights = tmp.wt).fit()
iptwResults.summary() # b_est =  1193.77 

# this result is closer to the matched results when using R matchit
# vs the matched analysis above


#----------------------------------
#  try pymatch
#----------------------------------


# https://medium.com/@bmiroglio/introducing-the-pymatch-package-6a8c020e2009

from pymatch.Matcher import Matcher

trt = df[df.treat == 1]
ctrl = df[df.treat == 0]


m = Matcher(trt, ctrl, yvar="re78", exclude=['ID'])

# this package starts to get buggy very fast

m.predict_scores()

m.tune_threshold(method='random')


#----------------------------------
#  try causal inference package
#----------------------------------

# https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html

from causalinference import CausalModel

import graphviz as gr

g = gr.Digraph()
g.edge("T", "Y")
g.edge("X", "Y")
g.edge("X", "P(x)")
g.edge("P(x)", "T")
g


# fit propensity score model
model = smf.glm('treat ~ age + educ + race + married + nodegree + re74 + re75', data=df, family=sm.families.Binomial(link = sm.genmod.families.links.logit))
ps_model = model.fit()
ps_model.summary()

# add propensity scores 
df['ps'] =  ps_model.fittedvalues

df[["treat", "educ", "race","ps"]].head() # check

# ps matching 
cm = CausalModel(
    Y=df["re78"].values, 
    D=df["treat"].values, 
    X=df[["ps"]].values
)

cm.est_via_matching(matches=1, bias_adj=True)

print(cm.estimates) # the ATT is very similar to the poorly implemented nearest neighbors matching above

# documentation 
# est_via_matching(self, weights='inv', matches=1, bias_adj=False)
# Estimates average treatment effects using nearest- neighborhood matching.

# Matching is done with replacement. Method supports multiple matching. Correcting bias that arise due to imperfect matches is also supported. For details on methodology, see [1].

# Parameters
# weights: str or positive definite square matrix
# Specifies weighting matrix used in computing distance measures. Defaults to string ‘inv’, which does inverse variance weighting. String ‘maha’ gives the weighting matrix used in the Mahalanobis metric.

# matches: int
# Number of matches to use for each subject.

# bias_adj: bool
# Specifies whether bias adjustments should be attempted.

# References

# 1
# Imbens, G. & Rubin, D. (2015). Causal Inference in Statistics, Social, and Biomedical Sciences: An Introduction.











