import imp
from multiprocessing import context
from unittest import result
from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from testmysql.models import UsersDetails
import sys

def home_page(Request):
    #return HttpResponse("Hello jamal for the first project")
    query1=UsersDetails.objects.raw('SELECT name, namefamily, email, mobile, bank, account FROM users where id in (2066, 2067)')
    return render(Request, "showAIresult/checkresult.html", {"UsersDetails":query1})
def check_gui(request):
    
    if(request.POST.get('check1')):
        var1=request.POST.get('check1')
        rfm_new=buys_ai()
        result1=rfm_new[rfm_new['M']==int(var1)]
        result1=result1.reset_index(level=0)
        user_id_list = result1.user.tolist()
        user_id_list=str(tuple(user_id_list))
        query1=UsersDetails.objects.raw('SELECT name, namefamily, email, mobile, bank, account FROM users where id in'+ user_id_list)
        return render(request, "showAIresult/checkresult.html",{"UsersDetails":query1})
    elif request.POST.get('check2'):
        var2=request.POST.get('check2')
        rfm_new=buys_ai()
        result1=rfm_new[rfm_new['F']==int(var2)]
        result1=result1.reset_index(level=0)
        user_id_list = result1.user.tolist()
        user_id_list=str(tuple(user_id_list))
        query1=UsersDetails.objects.raw('SELECT name, namefamily, email, mobile, bank, account FROM users where id in'+ user_id_list)
        return render(request, "showAIresult/checkresult.html",{"UsersDetails":query1})
    elif request.POST.get('check3'):
        var3=request.POST.get('check3')
        rfm_R=buys_ai()
        result1=rfm_R[rfm_R['R']==int(var3)]
        result1=result1.reset_index(level=0)
        user_id_list = result1.user.tolist()
        user_id_list=str(tuple(user_id_list))
        query1=UsersDetails.objects.raw('SELECT name, namefamily, email, mobile, bank, account FROM users where id in'+ user_id_list)
        return render(request, "showAIresult/checkresult.html",{"UsersDetails":query1})
    elif (request.POST.get('check1')) and (request.POST.get('check3')):
        var4_M=request.POST.get('check1')
        Var4_R=request.POST.get('check3')
        title1="مشتریانی که خرید بالا و اخیرا مارجعه نموده اند"
        rfm_R=buys_ai()
        result1=rfm_R[rfm_R['R']==int(Var4_R)]
        result=rfm_R[rfm_R['M']==int(var4_M)]
        result1=result1.reset_index(level=0)
        user_id_list = result1.user.tolist()
        user_id_list=str(tuple(user_id_list))
        query1=UsersDetails.objects.raw('SELECT name, namefamily, email, mobile, bank, account FROM users where id in'+ user_id_list)
        return render(request, "showAIresult/checkresult.html",{"UsersDetails":query1, "title_response":"title jamal"})
    elif (request.POST.get('check1')) and (request.POST.get('check2')):
        var5_M=request.POST.get('check1')
        Var5_F=request.POST.get('check2')
        title1="مشتریانی که حجم خرید بالا و زیاد مراجعه نموده اند"
        rfm_R=buys_ai()
        result1=rfm_R[rfm_R['F']==int(Var5_F)]
        result=rfm_R[rfm_R['M']==int(var5_M)]
        result1=result1.reset_index(level=0)
        user_id_list = result1.user.tolist()
        user_id_list=str(tuple(user_id_list))
        query1=UsersDetails.objects.raw('SELECT name, namefamily, email, mobile, bank, account FROM users where id in'+ user_id_list)
        return render(request, "showAIresult/checkresult.html",{"UsersDetails":query1, "title_response":"title jamal"})
    else:
        msg="please select one of the checkbox for assess your data"
        return render(request, "showAIresult/checkresult.html",{"var1":msg})

    
def admin_dashbord(Request):
     return render(Request, "admin/adminpanel.html",{})
def buys_ai():
    data=pd.read_csv('buys.csv')
    data['total_buy']=data['price']*data['size']
    data['updated_at2'] = data['updated_at'].apply(lambda x: pd.to_datetime(x))
    pin_date = max(data['updated_at2']) + dt.timedelta(1)
    rfm = data.groupby('user').agg({
    'updated_at2': lambda x: (pin_date - x.max()).days,
    'order': 'count',
    'total_buy': 'sum'
        })
    rfm.rename(columns= {
    'updated_at2': 'Recency',
    'order': 'Frequency',
    'total_buy': 'Monetary'
    }, inplace=True)
    r_labels = range(4, 0, -1)
    r_groups = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
    f_labels = range(1, 5)
    f_groups = pd.qcut(rfm['Frequency'], q=4, labels=f_labels)
    m_labels = range(1, 5)
    m_groups = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
    rfm['R'] = r_groups.values
    rfm['F'] = f_groups.values
    rfm['M'] = m_groups.values
    X = rfm[['R', 'F', 'M']]
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300)
    kmeans.fit(X)
    rfm['kmeans_cluster'] = kmeans.labels_
    #Uid_m=rfm[rfm['M']==4]
    return rfm
    #return render(Request, "admin/adminpanel.html",{"data_buy":Uid_m})

    
def adminAI(Request):
    data = pd.read_csv('pima-indians-diabetes.csv', names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
    data.head(10)
    skines = data[data['skin'] == 0]
    x=data[data['skin'] != 0]['skin'].mean()
    skin_mean = data[data['skin'] != 0]['skin'].mean()
    data.replace({'skin': 0}, skin_mean, inplace=True)
    x2=data.describe()
    test_mean = data[data['test'] != 0]['test'].mean()
    data.replace({'test': 0}, test_mean, inplace=True)
    data = pd.read_csv('pima-indians-diabetes.csv', names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
    data.replace({'test': 0, 'skin': 0}, np.nan, inplace=True)
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=3)
    data = imputer.fit_transform(data)
    np.set_printoptions(threshold=sys.maxsize)
    data = data[:, :-1]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    from sklearn.preprocessing import Normalizer
    norm = Normalizer()
    data = norm.fit_transform(data)
    np.sum(data[0]**2)
    from sklearn.preprocessing import Binarizer
    binarizer = Binarizer(threshold=0.0)
    data_b = binarizer.fit_transform(data)
    x3=np.concatenate((data, data_b), axis=1)
    return render(Request, "admin/adminpanel.html",{"data_AI":x3})