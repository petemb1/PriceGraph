#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import math
import json
import argparse #pmb
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool


PWD = os.path.dirname(os.path.realpath(__file__))


def load_stock(s):
    df = pd.read_csv(os.path.join('../data', s), index_col=0)
    df.set_index(df.index.astype('str'), inplace=True)
    return df


def load_ci(f, xi='close'):
    with open(os.path.join('../CI', xi, '%s.json' % f[:-4])) as fp:
        return json.load(fp)


def load_embedding(f, xi='close', ti=None):
    with open(os.path.join('../Struc2vec', xi, '%s.json' % f[:-4])) as fp:
        j = json.load(fp)
    if ti is not None:
        return {d: j[d] for d in ti if d in j}
    return j


def z_score(df):
    #pmbmean = df.mean() #pmb
    pmbstd = df.std() #pmb
    #pmb = (df - pmbmean) / pmbstd #pmb
    #return (df - df.mean()) / df.std()
    return (df - df.mean()) / pmbstd if pmbstd else df #pmb do not divide by zero
    #return pmb #pmb


def stock_sample(input_):
    s, d = input_
    #T = 20
    T = time_step #PMB changed to variable - 5 from 20 for new timestep
    df = global_df[s]
    #df = global_df[s].astype(float) #pmb to remove type errors
    if d not in df.index:
        return
    iloc = list(df.index).index(d) + 1
    if iloc < T:  # not enough history data
        return
    xss = {}
    for xi in x_column:
        # t - this says whether the target is reached for close or next bar is higher for others = 0 or 1
        # t = 1 if df.iloc[iloc+target-1,:][xi] > df.loc[d, xi] else 0
        
        # t = 1 if df.iloc[iloc+target-1,:][xi] > df.loc[d, xi] else 0 #pmb first assign the value as before, then change it if it is close
        t = 1 # pmb initiallise with 1
        # if xi == 'close': #and (df.loc[d, 'high'] > df.loc[d, 'low']): #pmb only check bar targets for 'close' prices - other data points only look at next bar + check divide by zero.
        if (df.loc[d, 'close']-df.loc[d, 'low'])/(df.loc[d, 'high']-df.loc[d, 'low']) >= 0.5:  #pmb if a bull bar
            for i in range(iloc, len(df)):
                if df.iloc[i,:]['high'] > (df.loc[d, 'close']*2 - df.loc[d, 'low']):
                    t=1 #buy
                    break
                elif df.iloc[i,:]['low'] < df.loc[d, 'low']:
                    t=0 #sell
                    break
        else: #pmb - it is a bear bar
            for i in range(iloc, len(df)):
                if df.iloc[i,:]['low'] < (df.loc[d, 'close']*2 - df.loc[d, 'high']):
                    t=0 #sell
                    break
                elif df.iloc[i,:]['high'] < df.loc[d, 'high']:
                    t=1 #buy
                    break
        if np.isnan(t).any(): #pmb return if not a number
            return #pmb

        # else: #pmb
        #     t = 1 if df.iloc[iloc+target-1,:][xi] > df.loc[d, xi] else 0 #pmb if no value is found resort to previous method of using next bar
            
        # y - list of dates in timestep.
        y = df.iloc[iloc-T:iloc][xi].copy()
        yz = np.array(z_score(y))
        if np.isnan(yz).any():
            return
        # ems - struc2vec data
        ems = global_ems[s][xi]
        if d not in ems:
            return
        keys = ['%s' % i for i in range(T)]
        emd = np.array([ems[d][k] for k in keys])
        if len(emd) < T:
            return
        # ci - CI - collective influence data
        cis = global_ci[s][xi]
        if d not in cis:
            return
        cid = cis[d]
        cid = [cid[str(i)] for i in range(T)]
        ciz = np.array(z_score(np.array(cid)))
        if np.isnan(ciz).any():
            ciz = np.array(cid)
        xss['%s_ems' % xi] = emd
        xss['%s_ys' % xi] = yz
        xss['%s_cis' % xi] = ciz
        xss['%s_t' % xi] = t
    return s, d, \
           xss['close_t'], xss['close_ems'], xss['close_ys'], xss['close_cis'], \
           xss['open_t'], xss['open_ems'], xss['open_ys'], xss['open_cis'], \
           xss['high_t'], xss['high_ems'], xss['high_ys'], xss['high_cis'], \
           xss['low_t'], xss['low_ems'], xss['low_ys'], xss['low_cis'], \
           xss['vol_t'], xss['vol_ems'], xss['vol_ys'], xss['vol_cis'], \
           xss['amount_t'], xss['amount_ems'], xss['amount_ys'], xss['amount_cis']


def sample_by_dates(dates):
    files = os.listdir('../data')
    fds = [(f, d) for d in dates for f in files]
    pool = Pool()
    samples = pool.map(stock_sample, fds)
    pool.close()
    pool.join()

    samples = filter(lambda s: s is not None, samples)
    stocks, days, \
    close_t, close_ems, close_ys, close_cis, \
    open_t, open_ems, open_ys, open_cis, \
    high_t, high_ems, high_ys, high_cis, \
    low_t, low_ems, low_ys, low_cis, \
    vol_t, vol_ems, vol_ys, vol_cis, \
    amount_t, amount_ems, amount_ys, amount_cis = zip(*samples)
    return {'stock': np.array(stocks), 'day': np.array(days),
            'close_t': np.array(close_t), 'close_ems': np.array(close_ems), 'close_ys': np.array(close_ys), 'close_cis': np.array(close_cis),
            'open_t': np.array(open_t), 'open_ems': np.array(open_ems), 'open_ys': np.array(open_ys), 'open_cis': np.array(open_cis),
            'high_t': np.array(high_t), 'high_ems': np.array(high_ems), 'high_ys': np.array(high_ys), 'high_cis': np.array(high_cis),
            'low_t': np.array(low_t), 'low_ems': np.array(low_ems), 'low_ys': np.array(low_ys), 'low_cis': np.array(low_cis),
            'vol_t': np.array(vol_t), 'vol_ems': np.array(vol_ems), 'vol_ys': np.array(vol_ys), 'vol_cis': np.array(vol_cis),
            'amount_t': np.array(amount_t), 'amount_ems': np.array(amount_ems), 'amount_ys': np.array(amount_ys), 'amount_cis': np.array(amount_cis),
            }


def generate_data_year(year):
    global global_ems
    start_date = datetime(year, 1, 1)
    days = [(start_date+timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    days = [d for d in days if '%s0101' % year <= d <= '%s1231' % year]
    global_ems = {f: {xc: load_embedding(f, xc, days) for xc in x_column} for f in files}
    dataset = sample_by_dates(days)
    with open(os.path.join('../dataset', '%s.pickle' % year), 'wb') as fp:
        pickle.dump(dataset, fp)


def generate_data_season(year, season):
    global global_ems
    sm = (season - 1) * 3 + 1 #PMB added start month
    start_date = datetime(year, sm, 1)
    days = [(start_date+timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    sm, em = str((season - 1) * 3 + 1).zfill(2), str(season * 3).zfill(2)
    days = [d for d in days if '%s%s01' % (year, sm) <= d <= '%s%s31' % (year, em)]
    global_ems = {f: {xc: load_embedding(f, xc, days) for xc in x_column} for f in files}
    dataset = sample_by_dates(days)
    with open(os.path.join('../dataset', '%s_S%s.pickle' % (year, season)), 'wb') as fp:
        pickle.dump(dataset, fp)


def getArgParser():
    parser = argparse.ArgumentParser(description='Train the price graph model on stock') #pmb
    parser.add_argument( #pmb
        '-ts', '--timestep', type=int, default=20, #pmb
        help='the length of time_step') #pmb
    parser.add_argument( #pmb
        '-tg', '--target', type=int, default=1, #pmb
        help='price is higher this number of bars in the future') #pmb    
    return parser #pmb


if __name__ == '__main__':
    args = getArgParser().parse_args() #pmb
    time_step = args.timestep #pmb
    target = args.target #pmb    
    print(args) #pmb

    files = os.listdir('../data')
    if not os.path.exists('../dataset'):
        os.makedirs('../dataset')
    x_column = ['close', 'open', 'high', 'low', 'vol', 'amount']
    y_column = 'close'
    #target = 1 #pmb removed - added as argument
    global_ems = None
    global_df = {f: load_stock(f) for f in files}
    global_ci = {f: {xc: load_ci(f, xc) for xc in x_column} for f in files}

    #for y in range(2018, 2009, -1):
    for y in range(2022, 1961, -1): #PMB data back to 1962 (new) 2005 (old)
        print(y)
        generate_data_year(y)
    #for m in range(1, 5):
    for m in range(1, 5): #PMB I only have 3 seasons of 2023 data in the old file - now I have 4 seasons
        print(m)
        #generate_data_season(2019, m) #PMB
        generate_data_season(2023, m)
