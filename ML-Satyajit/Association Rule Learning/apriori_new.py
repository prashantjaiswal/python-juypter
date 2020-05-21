# -*- coding: utf-8 -*-

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
#from apyori import apriori
  
df = pd.read_excel('Online Retail.xlsx')
df.head()

#There is a little cleanup, we need to do. 
#First, some of the descriptions have spaces that need to be removed.
#We’ll also drop the rows that don’t have invoice numbers and 
#remove the credit transactions (those with invoice numbers containing C). 
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]


#After the cleanup, we need to consolidate the items into 1 transaction per row 
#with each product 1 hot encoded. For the sake of keeping the data set small, 
#I’m only looking at sales for France. However, in additional code below, 
#I will compare these results to sales from Germany. Further country comparisons 
#would be interesting to investigate. 
basket = (df[df['Country'] =="France"]
  .groupby(['InvoiceNo', 'Description'])['Quantity']
  .sum().unstack().reset_index().fillna(0)
  .set_index('InvoiceNo'))

#There are a lot of zeros in the data but we also need to make sure any positive values
#are converted to a 1 and anything less the 0 is set to 0. This step will complete the 
#one hot encoding of the data and remove the postage column (since that charge is 
#not one we wish to explore): 
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

#Now that the data is structured properly, we can generate frequent item sets that 
#have a support of at least 7% (this number was chosen so that I could get enough 
#useful examples): 
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

#The final step is to generate the rules with their corresponding support, confidence, & lift.\
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
'''
That’s all there is to it! Build the frequent items using apriori then build the rules 
with association_rules.

Now, the tricky part is ﬁguring out what this tells us. For instance, we can see that 
there are quite a few rules with a high lift value which means that it occurs more 
frequently than would be expected given the number of transactions and product 
combinations. We can also see several where the conﬁdence is high as well. 
This part of the analysis is where the domain knowledge will come in handy. 
Since I do not have that, I’ll just look for a couple of illustrative examples.  
We can ﬁlter the dataframe using standard pandas code. In this case, look for a large 
lift (6) and high conﬁdence (.8):  
'''
rules[ (rules['lift'] >= 6) &
 (rules['confidence'] >= 0.8) ]

'''
In looking at the rules, it seems that the green and red alarm clocks are purchased 
together and the red paper cups, napkins and plates are purchased together in a manner 
that is higher than the overall probability would suggest.  
At this point, you may want to look at how much opportunity there is to use the popularity
 of one product to drive sales of another. For instance, we can see that we sell 340 
 Green Alarm clocks but only 316 Red Alarm Clocks so maybe we can drive more Red Alarm 
 Clock sales through recommendations?  
 
'''

basket['ALARM CLOCK BAKELIKE GREEN'].sum()


basket['ALARM CLOCK BAKELIKE RED'].sum()


'''
What is also interesting is to see how the combinations vary by country of purchase. 
Let’s check out what some popular combinations might be in Germany: 
  '''

basket2 = (df[df['Country'] =="Germany"]
  .groupby(['InvoiceNo', 'Description'])['Quantity']
  .sum().unstack().reset_index().fillna(0)
  .set_index('InvoiceNo'))

basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2 = rules2[ (rules2['lift'] >= 4) &
  (rules2['confidence'] >= 0.5)]


print(rules2)
rules.to_csv('rules2.csv')