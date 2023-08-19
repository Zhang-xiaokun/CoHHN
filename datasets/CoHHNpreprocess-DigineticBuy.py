'''

creat by kun at Sep 2021
Reference: 
https://github.com/xiaxin1998/DHCN https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch/blob/master/main.py

'''

import os
import pickle
import time
import pandas as pd
import numpy as np

products = 'products'
path_products = 'digineticaBuy/' + products + '.csv'



price_level_num = 100

items = pd.read_csv(path_products,error_bad_lines=False)


items = items.rename(columns={'itemId;pricelog2;product.name.tokens':'inf'})[['inf']]


def reg_itemID(strlist):
    results = int(strlist.split(';')[0])
    return results
def reg_price(strlist):
    price = int(strlist.split(';')[1])
    results = 2**price
    return results
items['itemID'] = items.inf.map(reg_itemID)
items['price'] = items.inf.map(reg_price)


items = items[['itemID', 'price']]

products_cate = 'product-categories'
path_products_cate = 'digineticaBuy/' + products_cate + '.csv'

item_cate = pd.read_csv(path_products_cate)

item_cate = item_cate.rename(columns={'itemId;categoryId':'inf'})[['inf']]
def re_categoryId(strlist):
    results = int(strlist.split(';')[1])
    return results
item_cate['itemID'] = item_cate.inf.map(reg_itemID)
item_cate['cate'] = item_cate.inf.map(re_categoryId)
item_cate = item_cate[['itemID', 'cate']]
item_all = pd.merge(item_cate, items, how='left', on = 'itemID')
item_all.drop_duplicates(subset=['itemID'],keep='first',inplace=True)


group_cate_num = pd.DataFrame(item_all.groupby(item_all['cate']).count())
group_num=group_cate_num.reset_index()[['cate','itemID']].rename(columns={'itemID':'count'})

group_cate_min = pd.DataFrame(item_all['price'].groupby(item_all['cate']).min())
group_min=group_cate_min.reset_index()[['cate','price']].rename(columns={'price':'min'})

group_cate_max = pd.DataFrame(item_all['price'].groupby(item_all['cate']).max())
group_max=group_cate_max.reset_index()[['cate','price']].rename(columns={'price':'max'})


group_cate_mean = pd.DataFrame(item_all['price'].groupby(item_all['cate']).mean())
group_mean=group_cate_mean.reset_index()[['cate','price']].rename(columns={'price':'mean'})


group_cate_std = pd.DataFrame(item_all['price'].groupby(item_all['cate']).std())
group_std=group_cate_std.reset_index()[['cate','price']].rename(columns={'price':'std'})


item_data1 = pd.merge(item_all, group_num, how='left', on = 'cate')
item_data2 = pd.merge(item_data1, group_min, how='left', on = 'cate')
item_data3 = pd.merge(item_data2, group_max, how='left', on = 'cate')
item_data4 = pd.merge(item_data3, group_mean, how='left', on = 'cate')
item_data5 = pd.merge(item_data4, group_std, how='left', on = 'cate')

item_data = item_data5[item_data5['count']>9]
item_data = item_data[item_data['std']!=0]


import math

def logistic(t, u, s):
    gama = s * 3**(0.5) / math.pi
    results = 1 / (1 + math.exp((t-u)/gama))
    return results
def get_price_level(price, p_min, p_max, mean, std):
    if std == 0:
        print('only one sample')
        return -1
    fenzi = logistic(price, mean, std) - logistic(p_min, mean, std)
    fenmu = logistic(p_max, mean, std) - logistic(p_min, mean, std)
    if fenmu == 0 or price==0:
        return -1
    results = int(fenzi / fenmu * price_level_num) + 1
    return results

item_data['price_level'] = item_data.apply(lambda row: get_price_level(row['price'], row['min'], row['max'], row['mean'], row['std']), axis=1)
item_final = item_data[item_data['price_level'] != -1]

group_price_num = pd.DataFrame(item_final.groupby(item_final['price_level']).count())
group_num=group_price_num.reset_index()[['price_level','itemID']].rename(columns={'itemID':'count'})

item_views = 'train-item-views'
path_item_views = 'digineticaBuy/' + item_views + '.csv'

interaction = pd.read_csv(path_item_views)
def re_sess(strlist):
    results = int(strlist.split(';')[0])
    return results

def re_itemID(strlist):
    results = int(strlist.split(';')[2])
    return results

def re_time(strlist):
    results = int(strlist.split(';')[3])
    return results
interaction= interaction.rename(columns={'sessionId;userId;itemId;timeframe;eventdate':'inf'})[['inf']]

interaction['sessionID'] = interaction.inf.map(re_sess)
interaction['itemID'] = interaction.inf.map(re_itemID)
interaction['time'] = interaction.inf.map(re_time)
interaction = interaction[['sessionID', 'itemID', 'time']]

user_item1 = pd.merge(interaction, item_final, how='left', on = 'itemID')
user_item2 = user_item1.dropna(axis=0)

user_item2.sort_values(by=["sessionID","time"],inplace=True,ascending=[True,True])

user_click_num = pd.DataFrame(user_item2.groupby(user_item2['sessionID']).count())
click_num=user_click_num.reset_index()[['sessionID','itemID']].rename(columns={'itemID':'click_num'})
item_data3 = pd.merge(user_item2, click_num, how='left', on = 'sessionID')
item_data4 = item_data3[item_data3['click_num']>1]
data_all = item_data4[['sessionID', 'itemID', 'time', 'price', 'cate', 'price_level']]




data_all = data_all.rename(columns={'price_level':'priceLevel', 'cate':'category'})
data_all = data_all[['sessionID', 'itemID', 'time', 'price', 'priceLevel', 'category']]

reviewerID2sessionID = {}
asin2itemID = {}
category2categoryID = {}

sessionNum = 0
itemNum = 0
categoryNum = 0


for _, row in data_all.iterrows():
    if row['sessionID'] not in reviewerID2sessionID:
        sessionNum += 1
        reviewerID2sessionID[row['sessionID']] = sessionNum
    if row['itemID'] not in asin2itemID:
        itemNum += 1
        asin2itemID[row['itemID']] = itemNum
    if row['category'] not in category2categoryID:
        categoryNum += 1
        category2categoryID[row['category']] = categoryNum
print('#session: ', sessionNum)
print('&item: ', itemNum)
print('#category: ', categoryNum)

def reSession(reviewerID):
    if reviewerID in reviewerID2sessionID:
        return reviewerID2sessionID[reviewerID]
    else:
        print('session is not recorded')
        return 'none'
    
def reItem(asin):
    if asin in asin2itemID:
        return asin2itemID[asin]
    else:
        print('item is not recorded')
        return 'none'
    
def reCate(category):
    if category in category2categoryID:
        return category2categoryID[category]
    else:
        print('category is not recorded')
        return 'none'
    

def priceInt(price):
    return int(price)

data_all['sessionID']=data_all.sessionID.map(reSession)
data_all['itemID']=data_all.itemID.map(reItem)
data_all['priceLevel']=data_all.priceLevel.map(priceInt)
data_all['category']=data_all.category.map(reCate)

item2price = {}
for _, row in data_all.iterrows():
    if row['itemID'] not in item2price:
        item2price[row['itemID']] = row['priceLevel']


data = data_all[['sessionID', 'itemID', 'priceLevel', 'category']]

item_inter_num = pd.DataFrame(data.groupby(data['itemID']).count())
item_inter_num = item_inter_num.reset_index()[['sessionID', 'itemID']]
item_num=item_inter_num.rename(columns={'sessionID':'item_num'})
data = pd.merge(data, item_num, how='left', on = 'itemID')

data = data[data['item_num']>9]
data = data[['sessionID', 'itemID', 'priceLevel', 'category']]

# dict (sessionID:[itemID,itemID])
sess_all = {}
# dict (sessionID:[priceLevel, priceLevel])
price_all = {}
# dict (sessionID:[cate, cate])
cate_all = {}
# dict (sessionID:[brand, brand])
# brand_all = {}
for _, row in data.iterrows():
    sess_id = row['sessionID']
    item_id = row['itemID']
    price = row['priceLevel']
    cate = row['category']
    if sess_id in sess_all:
        sess_all[sess_id].append(item_id)
        price_all[sess_id].append(price)
        cate_all[sess_id].append(cate)
    else:
        sess_all[sess_id] = []
        sess_all[sess_id].append(item_id)
        price_all[sess_id] = []
        price_all[sess_id].append(price)
        cate_all[sess_id] = []
        cate_all[sess_id].append(cate)

sess_total = data['sessionID'].max()
split_num = int(sess_total/10*9)

tra_sess = dict() # dict(session_id:[item_id,item_id])
tes_sess = dict() 
tra_price = dict()# dict(session_id:[price,price])
tes_price = dict()
tra_cate = dict()# dict(session_id:[cate,cate])
tes_cate = dict()
for sess_temp in sess_all.keys():
    all_seqs = sess_all[sess_temp]
    all_price = price_all[sess_temp]
    all_cate = cate_all[sess_temp]
    if len(all_seqs) < 2:
        continue
    if len(all_seqs) > 20:
        all_seqs = all_seqs[:20]
        all_price = all_price[:20]
        all_cate = all_cate[:20]
        
        
    if int(sess_temp) < split_num:
        tra_sess[sess_temp] = all_seqs
        tra_price[sess_temp] = all_price
        tra_cate[sess_temp] = all_cate
    else:
        tes_sess[sess_temp] = all_seqs
        tes_price[sess_temp] = all_price
        tes_cate[sess_temp] = all_cate

item_dict = {} #dict(old_itemID: new_itemID)
cate_dict = {} #dict(old_cate: new_cate)
price_dict = {} #dict(old_price: new_price)

item_price = {} #dict[new_itemID: priceLevel]
item_cate = {} #dict[new_itemID: cate]


# tra_sess tra_price tra_cate
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_seqs = []
    train_price = []
    train_cate = []
    item_ctr = 1
    price_ctr = 1
    cate_ctr = 1
    for s in tra_sess:
        seq = tra_sess[s]
        price_seq = tra_price[s]
        cate_seq = tra_cate[s]
        outseq = []
        pri_outseq = []
        cate_outseq = []
        for i, p, c in zip(seq, price_seq, cate_seq):
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
            if p in price_dict:
                pri_outseq += [price_dict[p]]
            else:
                pri_outseq += [price_ctr]
                price_dict[p]= price_ctr
                price_ctr += 1
            if c in cate_dict:
                cate_outseq += [cate_dict[c]]
            else:
                cate_outseq += [cate_ctr]
                cate_dict[c] = cate_ctr
                cate_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            print('session length is 1')
            continue
        train_seqs += [outseq]
        train_price += [pri_outseq]
        train_cate += [cate_outseq]
    print("#train_session",len(train_seqs))
    print("#train_items",item_ctr-1)
    print("#train_price",price_ctr-1)
    print("#train_category",cate_ctr-1)
    return train_seqs, train_price, train_cate


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_seqs = []
    test_price = []
    test_cate = []
    for s in tes_sess:
        outseq = []
        out_price = []
        out_cate = []
        for i, j, k in zip(tes_sess[s], tes_price[s], tes_cate[s]):
            if i in item_dict:
                outseq += [item_dict[i]]
                out_price += [price_dict[j]]
                out_cate += [cate_dict[k]]
        if len(outseq) < 2:
            print('obtain test session length is 1')
            continue
        test_seqs += [outseq]
        test_price += [out_price]
        test_cate += [out_cate]
    return test_seqs, test_price, test_cate

# data augmentation
def process_seqs(iseqs, iprice, icate):
    out_seqs = []
    out_price = []
    out_cate = []
    labs = []
    max_length = 19
    for seq, pri, cat in zip(iseqs, iprice, icate):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_price += [pri[:-i]]
            out_cate += [cat[:-i]]
    return out_seqs, out_price, out_cate, labs

# non data augmentation
def process_seqs_no(iseqs, iprice, icate):
    print("no data augment")
    out_seqs = []
    out_price = []
    out_cate = []
    labs = []
    max_length = 19
    for seq, pri, cat in zip(iseqs, iprice, icate):     
        labs += [seq[-1]]
        out_seqs += [seq[:-1]]
        out_price += [pri[:-1]]
        out_cate += [cat[:-1]]
    return out_seqs, out_price, out_cate, labs

tra_seqs, tra_pri, tra_cat = obtian_tra()
tes_seqs, tes_pri, tes_cat = obtian_tes()

tr_seqs, tr_pri, tr_cat, tr_labs = process_seqs_no(tra_seqs, tra_pri, tra_cat)
te_seqs, te_pri, te_cat, te_labs = process_seqs_no(tes_seqs, tes_pri, tes_cat)

print('train sequence: ',tr_seqs[:5])
print('train price: ',tr_pri[:5])
print('train category: ',tr_cat[:5])
print('train lab: ',tr_labs[:5])

# construct all matrics whose shape is similar as session-items [[],[]]
def tomatrix(all_seqs, all_pri, all_cate):

    price_item_dict = {}
    price_item = []

    price_category_dict = {}
    price_category = []


    category_item_dict = {}
    category_item = []


    # price-item dict -> {price_id:[1, 3, 4]}

    for s_seq, p_seq, c_seq in zip(all_seqs, all_pri, all_cate):
        for i_temp, p_temp, c_temp in zip(s_seq, p_seq, c_seq):
            if p_temp not in price_item_dict:
                price_item_dict[p_temp] = []
            if p_temp not in price_category_dict:
                price_category_dict[p_temp] = []
            if c_temp not in category_item_dict:
                category_item_dict[c_temp] = []
            price_item_dict[p_temp].append(i_temp)
            price_category_dict[p_temp].append(c_temp)
            category_item_dict[c_temp].append(i_temp)

    price_item_dict = dict(sorted(price_item_dict.items()))
    
    price_category_dict = dict(sorted(price_category_dict.items()))
    
    category_item_dict = dict(sorted(category_item_dict.items()))
    print("#price",len(price_item_dict))
    print("#category",len(category_item_dict))
    
    
    price_item = list(price_item_dict.values())
    price_category = list(price_category_dict.values())
    category_item = list(category_item_dict.values())
    return price_item, price_category, category_item

def data_masks(all_sessions):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) 
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1) # all weights in graph are set as 1
    results = (data, indices, indptr)
    return results

tra_pi, tra_pc, tra_ci = tomatrix(tra_seqs+tes_seqs, tra_pri+tes_pri, tra_cat+tes_cat)

tra = (tr_seqs, tr_pri, data_masks(tr_seqs), data_masks(tr_pri), data_masks(tra_pi), data_masks(tra_pc), data_masks(tra_ci), tr_labs)
tes = (te_seqs, te_pri, data_masks(te_seqs), data_masks(te_pri), data_masks(tra_pi), data_masks(tra_pc), data_masks(tra_ci), te_labs)

all = 0
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('#interactions: ',all)
print('#session: ',(len(tra_seqs) + len(tes_seqs)))
print('sequence average length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

train_data_path = 'CoHHN/digineticaBuy'

if not os.path.exists(train_data_path):
    os.makedirs(train_data_path)
path_data_train = train_data_path + "/train.txt"
path_data_test = train_data_path + "/test.txt"

pickle.dump(tra, open(path_data_train, 'wb'))
pickle.dump(tes, open(path_data_test, 'wb'))
print("dataset: digineticaBuy")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("done")