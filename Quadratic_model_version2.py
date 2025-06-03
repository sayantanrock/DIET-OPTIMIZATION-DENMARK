######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
#####################################     QUADRATIC OPTIMIZATION ( UN-NORMALIZED )       #############################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################


import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn import linear_model
import time
start_time = time.time()
#import matplotlib.pyplot as plt

tmrel_wholegrain = 75
tmrel_dairy = 250
tmrel_fish = 50
tmrel_fruit = 300
tmrel_vegetable = 300
tmrel_meat = 15
tmrel_nut = 30
tmrel_legume = 100

daly_crc = 772.56
daly_ihd = 1460.46
daly_t2d = 725.9
daly_stroke = 1200.5
daly_hypertension = 71.47

cf_reduction_factor = 0
daly_reduction_factor = 0
###############################################################################
###################    DOSE RESPONSE MODEL ####################################
#######################     WHOLE GRAIN    ####################################
def dose_response_wholegrain_t2d(x):
    X=[[0], [30] ]
    Y=[[1], [0.87]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 100:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[100]]))    
    return rr_pred  
def dose_response_wholegrain_crc(x):
    X=[[0], [30] ]
    Y=[[1], [0.95]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 374:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[374]]))    
    return rr_pred 
def dose_response_wholegrain_ihd(x):
    X=[[0], [90] ]
    Y=[[1], [0.81]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 225:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[225]]))    
    return rr_pred
#######################     DAIRY    ##########################################
def dose_response_dairy_t2d(x):
    X=[[0], [400] ]
    Y=[[1], [0.93]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 600:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[600]]))    
    return rr_pred
def dose_response_dairy_hyper(x):
    X=[[0], [200] ]
    Y=[[1], [0.95]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 800:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[800]]))    
    return rr_pred
def dose_response_dairy_crc(x):
    X=[[0], [400] ]
    Y=[[1], [0.87]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 900:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[900]]))    
    return rr_pred
#####################   FISH   ################################################
def dose_response_fish_ihd(x):
    X=[[0], [100] ]
    Y=[[1], [0.88]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 320:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[320]]))    
    return rr_pred
def dose_response_fish_stroke(x):
    X=[[0], [100] ]
    Y=[[1], [0.86]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 2000:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[2000]]))    
    return rr_pred
###################### FRUIT ##################################################
def dose_response_fruit_ihd(x):
    X=[[0], [200] ]
    Y=[[1], [0.9]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 750:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[750]]))    
    return rr_pred
def dose_response_fruit_stroke(x):
    X=[[0], [100] ]
    Y=[[1], [0.9]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 595:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[595]]))    
    return rr_pred
def dose_response_fruit_t2d(x):
    X=[[0], [100] ]
    Y=[[1], [0.98]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 560:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[560]]))    
    return rr_pred
##################  VEGETABLE  ################################################
def dose_response_vegetable_stroke(x):
    X=[[0], [200] ]
    Y=[[1], [0.87]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 500:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[500]]))    
    return rr_pred
def dose_response_vegetable_crc(x):
    X=[[0], [100] ]
    Y=[[1], [0.97]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 480:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[480]]))    
    return rr_pred
def dose_response_vegetable_ihd(x):
    X=[[0], [200] ]
    Y=[[1], [0.84]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 600:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[600]]))    
    return rr_pred
##############################  MEAT  #########################################
def dose_response_meat_crc(x):
    X=[[0], [100] ]
    Y=[[1], [1.12]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 100:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[100]]))    
    return rr_pred
def dose_response_meat_t2d(x):
    X=[[0], [100] ]
    Y=[[1], [1.17]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 170:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[170]]))    
    return rr_pred
############################### NUTS ##########################################
def dose_response_nut_ihd(x):
    X=[[0], [28] ]
    Y=[[1], [0.71]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 28:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[28]]))    
    return rr_pred
############################## LEGUMES ########################################
def dose_response_legume_ihd(x):
    X=[[0], [50] ]
    Y=[[1], [0.96]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 230:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[230]]))    
    return rr_pred
###############################################################################
###############################################################################

constraintmatrix = pd.read_excel('C:\\Users\\says\\Documents\\DROSH\\Data\\data_matrix.xlsx')  # assuming the file is an Excel file
limitsdata = pd.read_excel("C:\\Users\\says\\Documents\\DROSH\\Data\\Constraint limits.xlsx")
data_composition = pd.read_excel('C:\\Users\\says\\Documents\\DROSH\\Data\\additionalconstraintmatrix.xlsx')
intercept_wg = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==1].unique()[0]
intercept_dairy = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==2].unique()[0]
intercept_fish = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==3].unique()[0]
intercept_fruit = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==4].unique()[0]
intercept_veg = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==5].unique()[0]
intercept_meat = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==6].unique()[0]
intercept_nut = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==7].unique()[0]
intercept_legume = constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==8].unique()[0]

# DALY CONSTRAINTS 
limitsdata.loc[29, 'rhs'] = 4373 - 4373*cf_reduction_factor
limitsdata.loc[30, 'rhs'] = 986.5 - 986.5*daly_reduction_factor

# 4373
# 986.5

data_wg = constraintmatrix.loc[constraintmatrix['Group']==1]
data_dairy = constraintmatrix.loc[constraintmatrix['Group']==2]
data_fish = constraintmatrix.loc[constraintmatrix['Group']==3]
data_fruit = constraintmatrix.loc[constraintmatrix['Group']==4]
data_veg = constraintmatrix.loc[constraintmatrix['Group']==5]
data_meat = constraintmatrix.loc[constraintmatrix['Group']==6]
data_nut = constraintmatrix.loc[constraintmatrix['Group']==7]
data_legume = constraintmatrix.loc[constraintmatrix['Group']==8]
data_misc = constraintmatrix.loc[constraintmatrix['Group']==0]



consumption_wg = data_wg['Baseline']
consumption_dairy = data_dairy['Baseline']
consumption_fish = data_fish['Baseline']
consumption_fruit = data_fruit['Baseline']
consumption_veg = data_veg['Baseline']
consumption_meat = data_meat['Baseline']
consumption_nut = data_nut['Baseline']
consumption_legume = data_legume['Baseline']
consumption_misc = data_misc['Baseline']

upper_columns = [ 'Protein, g', 'Added sugar, g', 'Fat, upper, g',
                      'Saturated fatty acids, g', 'Sodium, mg', 'Alcohol, g', 'CF']

G_wg = data_wg[upper_columns]
G_dairy = data_dairy[upper_columns]
G_fish = data_fish[upper_columns]
G_fruit = data_fruit[upper_columns]
G_veg = data_veg[upper_columns]
G_meat = data_meat[upper_columns]
G_nut = data_nut[upper_columns]
G_legume = data_legume[upper_columns]
G_misc = data_misc[upper_columns]

equal_columns = [ 'Energy (kJ)']
A_wg =  data_wg[equal_columns]
A_dairy =  data_dairy[equal_columns]
A_fish =  data_fish[equal_columns]
A_fruit =  data_fruit[equal_columns]
A_veg =  data_veg[equal_columns]
A_meat =  data_meat[equal_columns]
A_nut =  data_nut[equal_columns]
A_legume =  data_legume[equal_columns]
A_misc =  data_misc[equal_columns]

lower_columns = [ 'Protein, g',
       'Carbohydrates, g',  'Dietary fibre, g',
       'Fat, lower, g', 
       'n-3 fatty acids, g', 'Vitamin A, RE µg', 'Vitamin E, alfa-TE',
       'Thiamin (Vitamin B1), mg', 'Riboflavin (Vitamin B2), mg', 'Niacin, NE',
       'Vitamin B6, mg', 'Folate, µg', 'Vitamin B12, µg', 'Vitamin C, mg',
        'Potassium, mg', 'Calcium, mg',
       'Magnesium, mg', 'Phosphorus, mg', 'Iron, mg', 'Zinc, mg', 'Iodine, µg',
       'Selenium, µg']
M_wg = data_wg[lower_columns]
M_dairy = data_dairy[lower_columns]
M_fish = data_fish[lower_columns]
M_fruit = data_fruit[lower_columns]
M_veg = data_veg[lower_columns]
M_meat = data_meat[lower_columns]
M_nut = data_nut[lower_columns]
M_legume = data_legume[lower_columns]
M_misc = data_misc[lower_columns]

slope_wg = data_wg['Daly_slope'].unique()
slope_dairy = data_dairy['Daly_slope'].unique()
slope_fish = data_fish['Daly_slope'].unique()
slope_fruit = data_fruit['Daly_slope'].unique()
slope_veg = data_veg['Daly_slope'].unique()
slope_meat = data_meat['Daly_slope'].unique()
slope_nut = data_nut['Daly_slope'].unique()
slope_legume = data_legume['Daly_slope'].unique()

h = limitsdata['rhs'].iloc[[2,4,7,8,19,28,29]]
b = limitsdata['rhs'].iloc[[0]]
n = limitsdata['rhs'].iloc[[1,3,5,6,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27]]
daly_limit = limitsdata['rhs'].iloc[[30]]


P_wg = np.identity(len(consumption_wg))
P_dairy = np.identity(len(consumption_dairy))
P_fish = np.identity(len(consumption_fish))
P_fruit = np.identity(len(consumption_fruit))
P_veg = np.identity(len(consumption_veg))
P_meat = np.identity(len(consumption_meat))
P_nut = np.identity(len(consumption_nut))
P_legume = np.identity(len(consumption_legume))
P_misc = np.identity(len(consumption_misc))

Q_wg = -(2*consumption_wg)
Q_dairy = -(2*consumption_dairy)
Q_fish = -(2*consumption_fish)
Q_fruit = -(2*consumption_fruit)
Q_veg = -(2*consumption_veg)
Q_meat = -(2*consumption_meat)
Q_nut = -(2*consumption_nut)
Q_legume = -(2*consumption_legume)
Q_misc = -(2*consumption_misc)

G_wg= G_wg.to_numpy()
G_dairy= G_dairy.to_numpy()
G_fish= G_fish.to_numpy()
G_fruit= G_fruit.to_numpy()
G_veg= G_veg.to_numpy()
G_meat= G_meat.to_numpy()
G_nut= G_nut.to_numpy()
G_legume= G_legume.to_numpy()
G_misc= G_misc.to_numpy()

A_wg= A_wg.to_numpy()
A_dairy= A_dairy.to_numpy()
A_fish= A_fish.to_numpy()
A_fruit= A_fruit.to_numpy()
A_veg= A_veg.to_numpy()
A_meat= A_meat.to_numpy()
A_nut= A_nut.to_numpy()
A_legume= A_legume.to_numpy()
A_misc= A_misc.to_numpy()

M_wg= M_wg.to_numpy()
M_dairy= M_dairy.to_numpy()
M_fish= M_fish.to_numpy()
M_fruit= M_fruit.to_numpy()
M_veg= M_veg.to_numpy()
M_meat= M_meat.to_numpy()
M_nut= M_nut.to_numpy()
M_legume= M_legume.to_numpy()
M_misc= M_misc.to_numpy()

h= h.to_numpy()
b= b.to_numpy()
n= n.to_numpy()

wg_content = data_composition['Whole_grain'].loc[data_composition['Whole_grain']!=0]
meat_content = data_composition['Red_meat_cooked'].loc[data_composition['Red_meat_cooked']!=0]
legume_content = 2.5

Q_wg= Q_wg.to_numpy()
Q_dairy= Q_dairy.to_numpy()
Q_fish= Q_fish.to_numpy()
Q_fruit= Q_fruit.to_numpy()
Q_veg= Q_veg.to_numpy()
Q_meat= Q_meat.to_numpy()
Q_nut= Q_nut.to_numpy()
Q_legume= Q_legume.to_numpy()
Q_misc= Q_misc.to_numpy()

x1 = cp.Variable(len(consumption_wg), nonneg=True)
x2 = cp.Variable(len(consumption_dairy), nonneg=True)
x3 = cp.Variable(len(consumption_fish), nonneg=True)
x4 = cp.Variable(len(consumption_fruit), nonneg=True)
x5 = cp.Variable(len(consumption_veg), nonneg=True)
x6 = cp.Variable(len(consumption_meat), nonneg=True)
x7 = cp.Variable(len(consumption_nut), nonneg=True)
x8 = cp.Variable(len(consumption_legume), nonneg=True)
x9 = cp.Variable(len(consumption_misc), nonneg=True)




objective = cp.Minimize(cp.quad_form(x1, P_wg) + Q_wg.T @ x1 + 
                        cp.quad_form(x2, P_dairy) + Q_dairy.T @ x2 +
                        cp.quad_form(x3, P_fish) + Q_fish.T @ x3 +
                        cp.quad_form(x4, P_fruit) + Q_fruit.T @ x4 +
                        cp.quad_form(x5, P_veg) + Q_veg.T @ x5 +
                        cp.quad_form(x6, P_meat) + Q_meat.T @ x6 +
                        cp.quad_form(x7, P_nut) + Q_nut.T @ x7 +
                        cp.quad_form(x8, P_legume) + Q_legume.T @ x8 +
                        cp.quad_form(x9, P_misc) + Q_misc.T @ x9   )

constraints = [G_wg.T @ x1 + G_dairy.T @ x2 + G_fish.T @ x3 + G_fruit.T @ x4 + G_veg.T @ x5 + G_meat.T @ x6 + G_nut.T @ x7 + G_legume.T @ x8 + G_misc.T @ x9 <= h,
               A_wg.T @ x1 + A_dairy.T @ x2 + A_fish.T @ x3 + A_fruit.T @ x4 + A_veg.T @ x5 + A_meat.T @ x6 + A_nut.T @ x7 + A_legume.T @ x8 + A_misc.T @ x9 == b,
               M_wg.T @ x1 + M_dairy.T @ x2 + M_fish.T @ x3 + M_fruit.T @ x4 + M_veg.T @ x5 + M_meat.T @ x6 + M_nut.T @ x7 + M_legume.T @ x8 + M_misc.T @ x9 >= n]


aux_meat = cp.Variable() 
constraints += [
    aux_meat == slope_meat * x6@meat_content + intercept_meat,
    aux_meat >= 0,
    aux_meat <= slope_meat * 100 + intercept_meat
]
wg_daly_contrib = cp.pos(slope_wg * x1@ wg_content  + intercept_wg)
dairy_daly_contrib = cp.pos(slope_dairy * x2.sum() +intercept_dairy)
fish_daly_contrib = cp.pos(slope_fish*x3.sum() + intercept_fish)
fruit_daly_contrib = cp.pos(slope_fruit*x4.sum() + intercept_fruit)
veg_daly_contrib = cp.pos(slope_veg*x5.sum() + intercept_veg)
meat_daly_contrib = aux_meat
nut_daly_contrib = cp.pos(slope_nut*x7.sum() + intercept_nut)
legume_daly_contrib = cp.pos(slope_legume*x8.sum()*legume_content + intercept_legume)
daly_expr =  wg_daly_contrib + dairy_daly_contrib + fish_daly_contrib + fruit_daly_contrib + veg_daly_contrib  + meat_daly_contrib + nut_daly_contrib + legume_daly_contrib

constraints.append( daly_expr <= daly_limit )
# constraints.append( x2[4] == 0)
# constraints.append( x2[5] == 0)
# constraints.append( x6[0] == 0)
# constraints.append( x6[1] == 0)
# constraints.append( x6[2] == 0)
prob = cp.Problem(objective,constraints)
prob.solve(solver=cp.SCS)
print("Status:", prob.status)

print(np.round(wg_daly_contrib.value,1), np.round(dairy_daly_contrib.value,1), 
      np.round(fish_daly_contrib.value,1), np.round(fruit_daly_contrib.value,1), 
      np.round(veg_daly_contrib.value,1), np.round(meat_daly_contrib.value,1), 
      np.round(nut_daly_contrib.value,1), np.round(legume_daly_contrib.value,1))

###################################### PRINT ##########################################################
food_group = pd.concat([data_wg['Foodgroup'] , data_dairy['Foodgroup'], data_fish['Foodgroup'],
                        data_fruit['Foodgroup'],data_veg['Foodgroup'],data_meat['Foodgroup'],
                        data_nut['Foodgroup'],data_legume['Foodgroup'],data_misc['Foodgroup']])
baseline = pd.concat([data_wg['Baseline'] , data_dairy['Baseline'], data_fish['Baseline'],
                        data_fruit['Baseline'],data_veg['Baseline'],data_meat['Baseline'],
                        data_nut['Baseline'],data_legume['Baseline'],data_misc['Baseline']])
optimal = np.concat([x1.value,x2.value,x3.value,x4.value,x5.value,x6.value,x7.value,x8.value,x9.value])
optimized_diet = pd.DataFrame({
    'Foodgroup': food_group,
    'Baseline': np.round(baseline.values, 1),
    'Optimized': np.round(optimal, 1)  # optimized food sub-group amounts
})
print("Optimized Diet:")
print(optimized_diet)
optimized_diet.to_excel('C:\\Users\\says\\OneDrive - Danmarks Tekniske Universitet\Dokumenter\\drosh\\Result\\optimised_diet.xlsx')

# ###################################### PRINT ##########################################################
baseline_group = [data_wg['Baseline'].sum() , data_dairy['Baseline'].sum() , data_fish['Baseline'].sum() ,
                        data_fruit['Baseline'].sum() ,data_veg['Baseline'].sum() ,data_meat['Baseline'].sum() ,
                        data_nut['Baseline'].sum() ,data_legume['Baseline'].sum() ,data_misc['Baseline'].sum()]
optimal_group = [x1.value.sum(),x2.value.sum(),x3.value.sum(),x4.value.sum(),x5.value.sum(),x6.value.sum(),x7.value.sum(),x8.value.sum(),x9.value.sum()]
optimized_foodgroup = pd.DataFrame({
    'Foodgroup': ['Whole Grain', 'Dairy', 'Fish', 'Fruit', 'Vegetable', 'Meat', 'Nut', 'Legume', 'Miscellanous'],
    'Baseline': np.round(baseline_group, 1),
    'Optimized': np.round(optimal_group, 1)  # optimized food sub-group amounts
})
print("Optimized Food Group:")
print(optimized_foodgroup)
optimized_foodgroup.to_excel('C:\\Users\\says\\OneDrive - Danmarks Tekniske Universitet\Dokumenter\\drosh\\Result\\optimised_food_group.xlsx')

###################################### NUTRIENT SLACK ##########################################################
nutrients_u = pd.DataFrame({
    "Nutrient": limitsdata["Nutrient"].iloc[[2,4,7,8,19,28,29]],
    "Target Value": h,
    "Baseline Value" : G_wg.T @ data_wg['Baseline'] + G_dairy.T @ data_dairy['Baseline'] + G_fish.T @ data_fish['Baseline'] + G_fruit.T @ data_fruit['Baseline'] + G_veg.T @ data_veg['Baseline'] + G_meat.T @ data_meat['Baseline'] + G_nut.T @ data_nut['Baseline'] + G_legume.T @ data_legume['Baseline'] + G_misc.T @ data_misc['Baseline'],
    "Nutrient Value": G_wg.T @ x1.value + G_dairy.T @ x2.value + G_fish.T @ x3.value + G_fruit.T @ x4.value + G_veg.T @ x5.value + G_meat.T @ x6.value + G_nut.T @ x7.value + G_legume.T @ x8.value + G_misc.T @ x9.value,
    "Slack": np.round(prob.constraints[0].dual_value, 6)
})
nutrients_e = pd.DataFrame({
    "Nutrient": limitsdata["Nutrient"].iloc[[0]],
    "Target Value": b,
    "Baseline Value" : A_wg.T @ data_wg['Baseline'] + A_dairy.T @ data_dairy['Baseline'] + A_fish.T @ data_fish['Baseline'] + A_fruit.T @ data_fruit['Baseline'] + A_veg.T @ data_veg['Baseline'] + A_meat.T @ data_meat['Baseline']  + A_nut.T @ data_nut['Baseline'] + A_legume.T @ data_legume['Baseline'] + A_misc.T @ data_misc['Baseline'],
    "Nutrient Value": A_wg.T @ x1.value + A_dairy.T @ x2.value + A_fish.T @ x3.value + A_fruit.T @ x4.value + A_veg.T @ x5.value + A_meat.T @ x6.value + A_nut.T @ x7.value + A_legume.T @ x8.value + A_misc.T @ x9.value,
    "Slack": np.round(prob.constraints[1].dual_value, 6)
})
nutrients_l = pd.DataFrame({
    "Nutrient": limitsdata["Nutrient"].iloc[[1,3,5,6,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27]],
    "Target Value": n,
    "Baseline Value" : M_wg.T @ data_wg['Baseline'] + M_dairy.T @ data_dairy['Baseline'] + M_fish.T @ data_fish['Baseline'] + M_fruit.T @ data_fruit['Baseline'] + M_veg.T @ data_veg['Baseline'] + M_meat.T @ data_meat['Baseline']  + M_nut.T @ data_nut['Baseline'] + M_legume.T @ data_legume['Baseline'] + M_misc.T @ data_misc['Baseline'],
    "Nutrient Value": M_wg.T @ x1.value + M_dairy.T @ x2.value + M_fish.T @ x3.value + M_fruit.T @ x4.value + M_veg.T @ x5.value + M_meat.T @ x6.value + M_nut.T @ x7.value + M_legume.T @ x8.value + M_misc.T @ x9.value ,
    "Slack": np.round(prob.constraints[2].dual_value, 6)
})
print("\nNutrient Constraints and Slack:")
print(nutrients_u)
print(nutrients_e)
print(nutrients_l)
nutrient_slack = pd.concat([nutrients_u, nutrients_e, nutrients_l], axis=0)
nutrient_slack.to_excel('C:\\Users\\says\\OneDrive - Danmarks Tekniske Universitet\Dokumenter\\drosh\\Result\\slack_variable.xlsx')
##############################################################################
#DALY CALCULATION FOR THE OUTPUT
###############################################################################
# PAF FOR BASELINE
# #WHOLEGRAIN
# baseline_wg = data_wg['Baseline'] @ data_composition['Whole_grain'].loc[data_composition['Whole_grain']!=0]
# paf1_wg = (dose_response_wholegrain_t2d(baseline_wg) - dose_response_wholegrain_t2d(tmrel_wholegrain))/dose_response_wholegrain_t2d(baseline_wg)
# paf2_wg = (dose_response_wholegrain_crc(baseline_wg) - dose_response_wholegrain_crc(tmrel_wholegrain))/dose_response_wholegrain_crc(baseline_wg)
# paf3_wg = (dose_response_wholegrain_ihd(baseline_wg) - dose_response_wholegrain_ihd(tmrel_wholegrain))/dose_response_wholegrain_ihd(baseline_wg)
# #DAIRY
# baseline_dairy = data_dairy['Baseline'].loc[data_dairy['Group']==2].sum()
# paf1_dairy = (dose_response_dairy_t2d(baseline_dairy) - dose_response_dairy_t2d(tmrel_dairy))/dose_response_dairy_t2d(baseline_dairy)
# paf2_dairy = (dose_response_dairy_hyper(baseline_dairy) - dose_response_dairy_hyper(tmrel_dairy))/dose_response_dairy_hyper(baseline_dairy)
# paf3_dairy = (dose_response_dairy_crc(baseline_dairy) - dose_response_dairy_crc(tmrel_dairy))/dose_response_dairy_crc(baseline_dairy)
# #FISH
# baseline_fish = data_fish['Baseline'].loc[data_fish['Group']==3].sum()
# paf1_fish = (dose_response_fish_ihd(baseline_fish) - dose_response_fish_ihd(tmrel_fish))/dose_response_fish_ihd(baseline_fish)
# paf2_fish = (dose_response_fish_stroke(baseline_fish) - dose_response_fish_stroke(tmrel_fish))/dose_response_fish_stroke(baseline_fish)
# #FRUIT
# baseline_fruit = data_fruit['Baseline'].loc[data_fruit['Group']==4].sum()
# paf1_fruit = (dose_response_fruit_ihd(baseline_fruit) - dose_response_fruit_ihd(tmrel_fruit))/dose_response_fruit_ihd(baseline_fruit)
# paf2_fruit = (dose_response_fruit_stroke(baseline_fruit) - dose_response_fruit_stroke(tmrel_fruit))/dose_response_fruit_stroke(baseline_fruit)
# paf3_fruit = (dose_response_fruit_t2d(baseline_fruit) - dose_response_fruit_t2d(tmrel_fruit))/dose_response_fruit_t2d(baseline_fruit)
# #VEGETABLE
# baseline_veg = data_veg['Baseline'].loc[data_veg['Group']==5].sum()
# paf1_veg = (dose_response_vegetable_ihd(baseline_veg) - dose_response_vegetable_ihd(tmrel_vegetable))/dose_response_vegetable_ihd(baseline_veg)
# paf2_veg = (dose_response_vegetable_crc(baseline_veg) - dose_response_vegetable_crc(tmrel_vegetable))/dose_response_vegetable_crc(baseline_veg)
# paf3_veg = (dose_response_vegetable_stroke(baseline_veg) - dose_response_vegetable_stroke(tmrel_vegetable))/dose_response_vegetable_stroke(baseline_veg)
# #MEAT
# baseline_meat = data_meat['Baseline'].loc[data_meat['Group']==6] @ data_composition['Red_meat_cooked'].loc[data_composition['Red_meat_cooked']!=0]
# paf1_meat = (dose_response_meat_crc(baseline_meat) - dose_response_meat_crc(tmrel_meat))/dose_response_meat_crc(baseline_meat)
# paf2_meat = (dose_response_meat_t2d(baseline_meat) - dose_response_meat_t2d(tmrel_meat))/dose_response_meat_t2d(baseline_meat)
# #NUTS
# baseline_nut = data_nut['Baseline'].loc[data_nut['Group']==7].sum()
# paf1_nut = (dose_response_nut_ihd(baseline_nut) - dose_response_nut_ihd(tmrel_nut))/dose_response_nut_ihd(baseline_nut)
# #LEGUME
# baseline_legume = data_legume['Baseline'].loc[data_legume['Group']==8].sum()
# paf1_legume = (dose_response_legume_ihd(baseline_legume) - dose_response_legume_ihd(tmrel_legume))/dose_response_legume_ihd(baseline_legume)
# #Total burden
# burden_ihd = (1-(1-paf3_wg[0][0])*(1-paf1_fish[0][0])*(1-paf1_fruit[0][0])*(1-paf1_veg[0][0])*(1-paf1_nut[0][0])*(1-paf1_legume[0][0]))*daly_ihd
# burden_crc = (1-(1-paf2_wg[0][0])*(1-paf3_dairy[0][0])*(1-paf2_veg[0][0])*(1-paf1_meat[0][0]))*daly_crc
# burden_t2d = (1-(1-paf1_wg[0][0])*(1-paf1_dairy[0][0])*(1-paf3_fruit[0][0])*(1-paf2_meat[0][0]))*daly_t2d
# burden_stroke = (1-(1-paf2_fish[0][0])*(1-paf2_fruit[0][0])*(1-paf3_veg[0][0]))*daly_stroke
# burden_hyper = paf2_dairy[0][0] * daly_hypertension
# burden_baseline = burden_ihd + burden_crc + burden_t2d + burden_stroke + burden_hyper

# burden_baseline_slope =  (baseline_wg*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==1].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==1].unique()[0] +
# baseline_dairy*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==2].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==2].unique()[0] +
# baseline_fish*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==3].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==3].unique()[0] +
# baseline_fruit*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==4].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==4].unique()[0] +
# baseline_veg*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==5].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==5].unique()[0] +
# baseline_meat*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==6].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==6].unique()[0] +
# baseline_nut*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==7].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==7].unique()[0] +
# baseline_legume*constraintmatrix['Daly_slope'].loc[constraintmatrix['Group']==8].unique()[0] + constraintmatrix['Daly_intercept'].loc[constraintmatrix['Group']==8].unique()[0] )

# #####################################################################################################################
# #####################################################################################################################
# PAF FOR OPTIMAL DIET
#WHOLEGRAIN
baseline_wg = x1.value @ data_composition['Whole_grain'].loc[data_composition['Whole_grain']!=0]
paf1_wg = (dose_response_wholegrain_t2d(baseline_wg) - dose_response_wholegrain_t2d(tmrel_wholegrain))/dose_response_wholegrain_t2d(baseline_wg)
if(paf1_wg <= 0):
    paf1_wg = 0
else : paf1_wg = (dose_response_wholegrain_t2d(baseline_wg) - dose_response_wholegrain_t2d(tmrel_wholegrain))/dose_response_wholegrain_t2d(baseline_wg)
paf2_wg = (dose_response_wholegrain_crc(baseline_wg) - dose_response_wholegrain_crc(tmrel_wholegrain))/dose_response_wholegrain_crc(baseline_wg)
if(paf2_wg <= 0):
    paf2_wg = 0
else : paf2_wg = (dose_response_wholegrain_crc(baseline_wg) - dose_response_wholegrain_crc(tmrel_wholegrain))/dose_response_wholegrain_crc(baseline_wg)
paf3_wg = (dose_response_wholegrain_ihd(baseline_wg) - dose_response_wholegrain_ihd(tmrel_wholegrain))/dose_response_wholegrain_ihd(baseline_wg)
if(paf3_wg <= 0):
    paf3_wg = 0
else : paf3_wg = (dose_response_wholegrain_ihd(baseline_wg) - dose_response_wholegrain_ihd(tmrel_wholegrain))/dose_response_wholegrain_ihd(baseline_wg)
#DAIRY
baseline_dairy = x2.value.sum()
paf1_dairy = (dose_response_dairy_t2d(baseline_dairy) - dose_response_dairy_t2d(tmrel_dairy))/dose_response_dairy_t2d(baseline_dairy)
if(paf1_dairy <= 0):
    paf1_dairy = 0
else :paf1_dairy = (dose_response_dairy_t2d(baseline_dairy) - dose_response_dairy_t2d(tmrel_dairy))/dose_response_dairy_t2d(baseline_dairy)
paf2_dairy = (dose_response_dairy_hyper(baseline_dairy) - dose_response_dairy_hyper(tmrel_dairy))/dose_response_dairy_hyper(baseline_dairy)
if(paf2_dairy <= 0):
    paf2_dairy = 0
else :paf2_dairy = (dose_response_dairy_hyper(baseline_dairy) - dose_response_dairy_hyper(tmrel_dairy))/dose_response_dairy_hyper(baseline_dairy)
paf3_dairy = (dose_response_dairy_crc(baseline_dairy) - dose_response_dairy_crc(tmrel_dairy))/dose_response_dairy_crc(baseline_dairy)
if(paf3_dairy <= 0):
    paf3_dairy = 0
else :paf3_dairy = (dose_response_dairy_crc(baseline_dairy) - dose_response_dairy_crc(tmrel_dairy))/dose_response_dairy_crc(baseline_dairy)
#FISH
baseline_fish = x3.value.sum()
paf1_fish = (dose_response_fish_ihd(baseline_fish) - dose_response_fish_ihd(tmrel_fish))/dose_response_fish_ihd(baseline_fish)
if(paf1_fish <= 0):
    paf1_fish = 0
else : paf1_fish = (dose_response_fish_ihd(baseline_fish) - dose_response_fish_ihd(tmrel_fish))/dose_response_fish_ihd(baseline_fish)
paf2_fish = (dose_response_fish_stroke(baseline_fish) - dose_response_fish_stroke(tmrel_fish))/dose_response_fish_stroke(baseline_fish)
if(paf2_fish <= 0):
    paf2_fish = 0
else :paf2_fish = (dose_response_fish_stroke(baseline_fish) - dose_response_fish_stroke(tmrel_fish))/dose_response_fish_stroke(baseline_fish)
#FRUIT
baseline_fruit = x4.value.sum()
paf1_fruit = (dose_response_fruit_ihd(baseline_fruit) - dose_response_fruit_ihd(tmrel_fruit))/dose_response_fruit_ihd(baseline_fruit)
if(paf1_fruit <= 0):
    paf1_fruit = 0
else :paf1_fruit = (dose_response_fruit_ihd(baseline_fruit) - dose_response_fruit_ihd(tmrel_fruit))/dose_response_fruit_ihd(baseline_fruit)
paf2_fruit = (dose_response_fruit_stroke(baseline_fruit) - dose_response_fruit_stroke(tmrel_fruit))/dose_response_fruit_stroke(baseline_fruit)
if(paf2_fruit <= 0):
    paf2_fruit = 0
else :paf2_fruit = (dose_response_fruit_stroke(baseline_fruit) - dose_response_fruit_stroke(tmrel_fruit))/dose_response_fruit_stroke(baseline_fruit)
paf3_fruit = (dose_response_fruit_t2d(baseline_fruit) - dose_response_fruit_t2d(tmrel_fruit))/dose_response_fruit_t2d(baseline_fruit)
if(paf3_fruit <= 0):
    paf3_fruit = 0
else :paf3_fruit = (dose_response_fruit_t2d(baseline_fruit) - dose_response_fruit_t2d(tmrel_fruit))/dose_response_fruit_t2d(baseline_fruit)
#VEGETABLE
baseline_veg = x5.value.sum()
paf1_veg = (dose_response_vegetable_ihd(baseline_veg) - dose_response_vegetable_ihd(tmrel_vegetable))/dose_response_vegetable_ihd(baseline_veg)
if(paf1_veg <= 0):
    paf1_veg = 0
else :paf1_veg = (dose_response_vegetable_ihd(baseline_veg) - dose_response_vegetable_ihd(tmrel_vegetable))/dose_response_vegetable_ihd(baseline_veg)
paf2_veg = (dose_response_vegetable_crc(baseline_veg) - dose_response_vegetable_crc(tmrel_vegetable))/dose_response_vegetable_crc(baseline_veg)
if(paf2_veg <= 0):
    paf2_veg = 0
else :paf2_veg = (dose_response_vegetable_crc(baseline_veg) - dose_response_vegetable_crc(tmrel_vegetable))/dose_response_vegetable_crc(baseline_veg)
paf3_veg = (dose_response_vegetable_stroke(baseline_veg) - dose_response_vegetable_stroke(tmrel_vegetable))/dose_response_vegetable_stroke(baseline_veg)
if(paf3_veg <= 0):
    paf3_veg = 0
else :paf3_veg = (dose_response_vegetable_stroke(baseline_veg) - dose_response_vegetable_stroke(tmrel_vegetable))/dose_response_vegetable_stroke(baseline_veg)
#MEAT
baseline_meat = x6.value @ data_composition['Red_meat_cooked'].loc[data_composition['Red_meat_cooked']!=0]
paf1_meat = (dose_response_meat_crc(baseline_meat) - dose_response_meat_crc(tmrel_meat))/dose_response_meat_crc(baseline_meat)
if(paf1_meat <= 0):
    paf1_meat = 0
else :paf1_meat = (dose_response_meat_crc(baseline_meat) - dose_response_meat_crc(tmrel_meat))/dose_response_meat_crc(baseline_meat)
paf2_meat = (dose_response_meat_t2d(baseline_meat) - dose_response_meat_t2d(tmrel_meat))/dose_response_meat_t2d(baseline_meat)
if(paf2_meat <= 0):
    paf2_meat = 0
else :paf2_meat = (dose_response_meat_t2d(baseline_meat) - dose_response_meat_t2d(tmrel_meat))/dose_response_meat_t2d(baseline_meat)
#NUTS
baseline_nut = x7.value.sum()
paf1_nut = (dose_response_nut_ihd(baseline_nut) - dose_response_nut_ihd(tmrel_nut))/dose_response_nut_ihd(baseline_nut)
if(paf1_nut <= 0):
    paf1_nut = 0
else :paf1_nut = (dose_response_nut_ihd(baseline_nut) - dose_response_nut_ihd(tmrel_nut))/dose_response_nut_ihd(baseline_nut)
#LEGUME
baseline_legume = x8.value.sum() * 2.5
paf1_legume = (dose_response_legume_ihd(baseline_legume) - dose_response_legume_ihd(tmrel_legume))/dose_response_legume_ihd(baseline_legume)
if(paf1_legume <= 0):
    paf1_legume = 0
else :paf1_legume = (dose_response_legume_ihd(baseline_legume) - dose_response_legume_ihd(tmrel_legume))/dose_response_legume_ihd(baseline_legume)
#Total burden

burden_ihd = (1-(1-paf3_wg)*(1-paf1_fish)*(1-paf1_fruit)*(1-paf1_veg)*(1-paf1_nut)*(1-paf1_legume))*daly_ihd
burden_crc = (1-(1-paf2_wg)*(1-paf3_dairy)*(1-paf2_veg)*(1-paf1_meat))*daly_crc
burden_t2d = (1-(1-paf1_wg)*(1-paf1_dairy)*(1-paf3_fruit)*(1-paf2_meat))*daly_t2d
burden_stroke = (1-(1-paf2_fish)*(1-paf2_fruit)*(1-paf3_veg))*daly_stroke
burden_hyper = paf2_dairy * daly_hypertension
burden_TOTAL = burden_ihd + burden_crc + burden_t2d + burden_stroke + burden_hyper

burden_optimal =  {
    "DISEASES": ["IHD", "CRC", "T2D", "STROKE", "HYPERTENSION"],
    "BURDEN": [burden_ihd[0][0], burden_crc[0][0], burden_t2d[0][0], burden_stroke[0][0], burden_hyper]
}
df = pd.DataFrame(burden_optimal)
print(df)
print('TOTAL BURDEN', burden_TOTAL[0][0])
df.to_excel('C:\\Users\\says\\OneDrive - Danmarks Tekniske Universitet\Dokumenter\\drosh\\Result\\DALY_OPTIMAL.xlsx')
# paf_baseline = {
#     "FOOD GROUP": ["W.G", "DAIRY", "FISH", "FRUIT", "VEGETABLE", "MEAT", "NUT", "LEGUME"],
#     "GROUP": [1, 2, 3, 4, 5, 6, 7, 8],
#     "IHD": [paf3_wg, None, paf1_fish, paf1_fruit, paf1_veg, None, paf1_nut, paf1_legume],
#     "CRC": [paf2_wg, paf3_dairy, None, None, paf2_veg, paf1_meat, None, None],
#     "T2D": [paf1_wg, paf1_dairy, None, paf3_fruit, None, paf2_meat, None, None],
#     "STROKE": [None, None, paf2_fish, paf2_fruit, paf3_veg, None, None, None],
#     "HYPERTENSION": [None, paf2_dairy, None, None, None, None, None, None]
# }
# df1 = pd.DataFrame(paf_baseline)
# food_groups = optimized_foodgroup['Foodgroup'].iloc[:-1]
# baseline_values = optimized_foodgroup ['Baseline'].iloc[:-1]
# optimal_values = optimized_foodgroup ['Optimized'].iloc[:-1]

# # Define bar width and positions
# x = np.arange(len(food_groups))  # Numeric positions for each food group
# width = 0.4  # Width of the bars

# fig, ax = plt.subplots(figsize=(10, 6))
# fig.patch.set_facecolor('white')  # Change figure background
# ax.set_facecolor('white')  # Change plot background

# # Plot side-by-side bars
# ax.bar(x - width/2, baseline_values, width, label='Baseline diet', color='cyan')
# ax.bar(x + width/2, optimal_values, width, label='Optimal diet', color='red')

# # Formatting the plot
# ax.set_xlabel('Food Group')
# ax.set_ylabel('Consumption (gm/day)')
# ax.set_title('Optimal Diet Composition of Food Groups')
# ax.legend()

# # Adjust x-axis labels
# ax.set_xticks(x)
# ax.set_xticklabels(food_groups, rotation=45, ha='right', color='white')

# # Change text and tick colors to white for better visibility
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# ax.xaxis.label.set_color('black')
# ax.yaxis.label.set_color('black')
# ax.tick_params(axis='x', colors='black')
# ax.tick_params(axis='y', colors='black')
# plt.savefig('OPTIMAL_FOOD_50_50.png') 

# optimized_foodgroup.to_excel('optimised_diet.xlsx')


print("--- %s seconds ---" % (time.time() - start_time))

