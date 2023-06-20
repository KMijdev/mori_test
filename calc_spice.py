#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 計算やデータ処理のライブラリ
import numpy as np
import pandas as pd
#バージョン情報表示
print('pandas ver:',pd.__version__)
# データ可視化のライブラリ(#jupyternotebook内でmatplotlibで図を描写するときの必須のおまじない)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print('seaborn ver:',sns.__version__)
# データセットの取得&処理のライブラリ
from mpl_toolkits.mplot3d import Axes3D
# GUI用のライブラリ
import tkinter
from tkinter import filedialog as tkFileDialog
from tkinter import filedialog
#csv保存用ライブラリ
import csv
#osライブラリ
import os
#ピーク算出
from scipy.signal import find_peaks
from scipy.signal import argrelmax
from scipy.signal import argrelmin


# In[2]:


def drawfig2_1(fk1,spa1,fk2,spa2):

    fig=plt.figure(figsize=(5,5),facecolor='w')#figureオブジェクトを作成
    ax=fig.add_subplot(1,1,1)#グラフを描画するsubplot領域を作成
    plt.style.use('fast')
    
    for iii in [1,2]:
        if iii==1:
            tstr='Original'
            x=1/fk1[1:]; y=spa1[1:len(x)+1]            
            ax.plot(x,y,color='Red',label=tstr,linewidth = 2)#Axesオブジェクトにグラフの線を追加
            
            #極大値の算出
            Peak_ori=argrelmax(y,order=10)
            Peak_ori_5=Peak_ori[0:4]
            ax.scatter(x[Peak_ori_5],y[Peak_ori_5],color='Magenta')#Axesオブジェクトにグラフの線を追加
           
        if iii==2:
            tstr='After waveform processing'
            x=1/fk2[1:]; y=spa2[1:len(x)+1]
            ax.plot(x,y,color='Blue',label=tstr,linewidth = 2,linestyle = "--")#Axesオブジェクトにグラフの線を追加

            #極大値の算出
            Peak_ori=argrelmax(y,order=10)
            Peak_ori_5=Peak_ori[0:4]
            ax.scatter(x[Peak_ori_5],y[Peak_ori_5],color='Blue')#Axesオブジェクトにグラフの線を追加
            
    ax.set_xlabel('Period [μs]',size=14,weight='light')
    ax.set_ylabel('Fourier Spectrunm [V*μs]',size=14,weight='light')

    # x軸に補助目盛線を設定
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)

    # y軸に目盛線を設定
    ax.grid(which = "major", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)
    
    ax.legend()#凡例表示
    plt.xscale('log')
    plt.title("メニスカス振動のFFT", fontname="MS Gothic")
    #plt.show()#グラフを描画 
    return x[Peak_ori_5],plt


# In[3]:


#FFT関数
def nfft_n(x):
    nd=len(x)
    nn=2
    while nn<nd:
        nn=nn*2
    xx=np.zeros(nn)
    xx[0:nd]=xx[0:nd]+x[0:nd]
    sp=np.fft.fft(xx)/nn # complex number
    return nn,sp

#IFFT関数
def ifft_n(nn,sp):
    wv=np.fft.ifft(sp*nn) # complex number
    return wv.real


# In[4]:


def LTSpice(name):
    import matplotlib.pyplot as plt
    from PyLTSpice.LTSpice_RawRead import LTSpiceRawRead

    ##必須
    #.rawファイルを読み込む。
    #LTR = LTSpiceRawRead(r'C:\Users\e11437\Desktop\測定ツール_Spice\DR_AL4.2.txt_KM800_HLF_DR1d.raw')
    LTR = LTSpiceRawRead(name)
    #LTspiceで使っているラベル類を抜きだす。
    print(LTR.get_trace_names())
    #.rawファイルの上のほうの情報を抜き出す。
#    print(LTR.get_raw_property())

    ##必須##
    #ラベル値を指定してrawファイルからデータを抜き出す。
    V_waveform_Original    = LTR.get_trace("V(waveform)")
    V_meniscus_vol_Original    = LTR.get_trace("V(meniscus_volume)")
    I_Noz_flow_Original    = LTR.get_trace("I(Noz_flow)")
    Time_Original       = LTR.get_trace('time')
    V_out_volume_Original=LTR.get_trace('V(out_volume)')
    V_cham_press_Original = LTR.get_trace('V(cham_press)') #普通にALを出すために追加

    ##必須
    #step実行命令がある場合step番号を取得する。
    steps   = LTR.get_steps()
#    print('steps:',steps)
    #データの整形(_Original→Non_Original ※SpiceのStepとはなにか？　→最大ステップ時間)

    for step in range(len(steps)):
        Time=Time_Original.get_time_axis(step)*10**6
        V_meniscus_vol=V_meniscus_vol_Original.get_wave(step)*10**9
        V_waveform=V_waveform_Original.get_wave(step)
        I_Noz_flow=I_Noz_flow_Original.get_wave(step)
        V_out_volume=V_out_volume_Original.get_wave(step)*10**9
        V_cham_press=V_cham_press_Original.get_wave(step)*10**9
        
##必須
##データの補間の実施
#def Hokan(Time,V_meniscus_vol,I_Noz_flow,V_waveform)
    from scipy.interpolate import interp1d

    #測定データを補間する関数を出力
    HokanF_V_meniscus = interp1d(Time, V_meniscus_vol,kind="linear") 
    HokanF_I_Noz_flow = interp1d(Time, I_Noz_flow,kind="linear") 
    HokanF_V_waveform = interp1d(Time, V_waveform,kind="linear") 
    HokanF_V_out_volume = interp1d(Time, V_out_volume,kind="linear")
    HokanF_V_cham_press = interp1d(Time, V_cham_press,kind="linear")

    Start=0 #開始時間　データから
    End=max(Time) #終了時間　データから
    num=100001 #補間後のデータ数
    dT=(End-Start)/(num-1) #補間後のデータ間隔

    #print(dT)
    #補間後の時間を設定
    tt=np.linspace(Start,End,num,endpoint=True) #補間後の時間

    #補間後のy(yy)を出力
    V_meniscus_vol_Hokan=HokanF_V_meniscus(tt)
    I_Noz_flow_Hokan=HokanF_I_Noz_flow(tt)
    V_waveform_Hokan=HokanF_V_waveform(tt)
    V_out_volume_Hokan=HokanF_V_out_volume(tt)
    V_cham_press_Hokan=HokanF_V_cham_press(tt)
    
    return tt,dT,V_meniscus_vol_Hokan,I_Noz_flow_Hokan,V_waveform_Hokan,V_out_volume_Hokan,V_cham_press_Hokan


# In[5]:


#任意の値に最も近い値を探す
def find_nearest_value(array, value):
    idx = np.abs(array - value).argmin()
    return array[idx]


# In[6]:


#描画関数(final
import math
import itertools
import operator
from scipy.optimize import curve_fit

def drawfig1_3(tt,dt,xx1,xx2,wf,vov,vcp,drop_num):
    D_list=[]
    plt.rcParams['font.family']='sans-serif'
    plt.rcParams['axes.facecolor']='White'
    plt.style.use('fast')

    fig=plt.figure(figsize=(15,10))#,facecolor='white')#figureオブジェクトを作成
    ax1=fig.add_subplot(2,1,1)#グラフを描画するsubplot領域を作成
    ax2=fig.add_subplot(2,1,2)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([-30, 5])
    ax2.set_xlim([0, 100])
    ax2.set_ylim([-7, 7])

    drop_num=drop_num
    
    for iii in [1,2]:
        if iii==1:
            tstr='Original wave'
            x=tt
            y=xx1[0:len(tt)]
            yw=wf[0:len(tt)]
            ov=vov[0:len(tt)]
            #グラフで表示する
            ax1.plot(x,yw,color='navy',label='Waveform',linewidth = 2)#Axesオブジェクトにグラフの線を追加
            ax2.plot(x,y,color='Red',label=tstr,linewidth = 2)#Axesオブジェクトにグラフの線を追加
            ax2.plot(x,ov,color='pink',label='v_out_volume',linewidth = 2)#Axesオブジェクトにグラフの線を追加

     
            #実際には使っていない##
            ##----ここから-----##
            #極大値の算出
            Peak_orimax=argrelmax(y,order=10)#ピークインデックスの取得
            x_orimax=x[Peak_orimax]#ピークのインデックス番号から値を取り出す(x) 
            y_orimax=y[Peak_orimax]#ピークのインデックス番号から値を取り出す(y)  
            x_orimax_chk=x_orimax[y_orimax>1.5]#y_orimaxが基準以上のもののみ残す
            y_orimax_chk=y_orimax[y_orimax>1.5]#y_orimaxが基準以上のもののみ残す        
            x_orimax_trim=x_orimax_chk[0:drop_num]#スライスする
            y_orimax_trim=y_orimax_chk[0:drop_num]#スライスする
            ax2.scatter(x_orimax_trim,y_orimax_trim,color='Black')#Axesオブジェクトにグラフの線を追加

            #極小値の算出
            Peak_orimin=argrelmin(y,order=10)
            x_orimin=x[Peak_orimin]#ピークのインデックス番号から値を取り出す(x)
            y_orimin=y[Peak_orimin]#ピークのインデックス番号から値を取り出す(y)
            x_orimin_chk=x_orimin[y_orimin<-2.5]#y_oriminが基準以上のもののみ残す
            y_orimin_chk=y_orimin[y_orimin<-2.5]#y_oriminが基準以上のもののみ残す        
            x_orimin_trim=x_orimin_chk[0:drop_num]#スライスする
            y_orimin_trim=y_orimin_chk[0:drop_num]#スライスする
            ax2.scatter(x_orimin_trim,y_orimin_trim,color='Gray')#Axesオブジェクトにグラフの線を追加
            ##----ここまで-----##

            ##吐出タイミングの算出
            #変化率を算出
            Delta_y=np.diff(y)
            #xの補正(差分を取って1減る分の補正、Endを-1で末端要素を削除)
            D_x=x[:-1]
            D_y=y[:-1]
            xy_Delta_hairetu1=np.vstack((D_x,D_y))#xyを同じ配列(2次元)に入れる
            xy_Delta_hairetu=np.vstack((xy_Delta_hairetu1,Delta_y))#xyを同じ配列(2次元)に入れる
#            print('xy_Delta_hairetu.shape',xy_Delta_hairetu.shape)
            #ソートを行う [1]行を基準にソート
            xy_Delta_hairetu_sort=xy_Delta_hairetu[:,np.argsort(xy_Delta_hairetu[2])]
#            print('xy_Delta_haitretu_sort',xy_Delta_hairetu_sort)
            #ピーク値の小さい順に6つをスライス（ピーク値が小さい順に並んでいる）
            xy_Delta_sort_trim=xy_Delta_hairetu_sort[:,:drop_num]   
#            print('xy_Delta_sort_trim',xy_Delta_sort_trim)
            #時間が早い順に改めてソートする
            jetting_vol=xy_Delta_sort_trim[:,np.argsort(xy_Delta_sort_trim[0])]
            
            total_jetting_vol=sum(jetting_vol[1])
            ruikei_jetting_vol=list(itertools.accumulate(jetting_vol[1]))
#            ruikei_jetting_vol=np.array(itertools.accumulate(jetting_vol[1]))

#            print('累計',np.shape(ruikei_jetting_vol))
            ruikei_jetting_vol_t=list(jetting_vol[0])
#            print('累計t',np.shape(ruikei_jetting_vol_t))
#            ax3.bar(ruikei_jetting_vol_t,ruikei_jetting_vol,color="#FF5B70", edgecolor="black",width=5,alpha=0.5)#Axesオブジェクトにグラフの線を追加
                        
            ax2.plot(xy_Delta_hairetu[0],xy_Delta_hairetu[2],color='seagreen')#吐出タイミング
#            ax2.plot(xy_Delta_sort_trim[0],xy_Delta_sort_trim[2],color='Cyan')#Axesオブジェクトにグラフの線を追加
#            ax2.plot(jetting_vol[0],jetting_vol[1],color='Magenta')#吐出量
#            ax2.plot(jetting_vol[0],jetting_vol[2],color='Orange')#吐出量
            print('吐出時間[μs]',jetting_vol[0])
            print('吐出量＿[pl]',jetting_vol[1])
            print('合計吐出量[pl]','{:.3g}'.format(total_jetting_vol))
            print('累計吐出量[pl]',ruikei_jetting_vol)
            
        if iii==2:
            tstr='After waveform processing'
            x=tt
            y=xx2[0:len(tt)]
            ax2.plot(x,y,color='Blue',label=tstr,linewidth = 3,)#Axesオブジェクトにグラフの線を追加

            
            ####手動パラメータ設定の必要あり####
            ##22μs以降のデータに対して極大値、極小値を取得する##
            ##22μsのindexを取得、表示 #22
            # 最も近い値を探す
            target_value = 22.0
            nearest_value = find_nearest_value(x, target_value)
            trim_index=np.where(x==nearest_value)
            #print('index(30μs)',trim_index)
            trim_list=list(trim_index)#tuple→listに変換
            trim_int=int(trim_list[0])#list→intに変換
            
            x_trim=x[trim_int:]#15us以上のxをスライス
            y_trim=y[trim_int:]#15μs以上のyをスライス
            
            #print('x_trim:',x_trim)
            #print('y_trim:',y_trim)


        #極大値の算出
            Peak_aftmax=argrelmax(y_trim,order=10)
            x_aftmax=x_trim[Peak_aftmax]
            y_aftmax=y_trim[Peak_aftmax]
            x_aftmax_trim=x_aftmax[0:3]
            y_aftmax_trim=y_aftmax[0:3]
            ax2.plot(x_aftmax_trim,y_aftmax_trim,color='Cyan',marker='o',linestyle = "--")#Axesオブジェクトにグラフの線を追加
            #極小値の算出
            Peak_aftmin=argrelmin(y_trim,order=10)
            x_aftmin=x_trim[Peak_aftmin]
            y_aftmin=y_trim[Peak_aftmin]
            x_aftmin_trim=x_aftmin[0:3]
            y_aftmin_trim=y_aftmin[0:3]
            ax2.plot(x_aftmin_trim,y_aftmin_trim,color='lawngreen',marker='o',linestyle = "--")#Axesオブジェクトにグラフの線を追加
            
    np.set_printoptions(precision=1)
    print('----圧力室振動ピーク----') 
    print('x_orimax_trim:',x_orimax_trim) 
    print('y_orimax_trim:',y_orimax_trim)
    print('x_orimin_trim:',x_orimin_trim) 
    print('y_orimin_trim:',y_orimin_trim)

    print('----メニスカス振動ピーク----')     
    print('x_aftmax_trim:',x_aftmax_trim) 
    print('y_aftmax_trim:',y_aftmax_trim)
    print('x_aftmin_trim:',x_aftmin_trim) 
    print('y_aftmin_trim:',y_aftmin_trim)

    #最大値と最小値の算出
    Max_afure=max(y_trim)
    Min_afure=min(y_trim)
    Max_afure_index=np.argmax(y_trim)#ndarrayの最大値
    Min_afure_index=np.argmin(y_trim)#ndarrayの最小値        
    
    #最大値以上をスライスしたデータの生成
    Peak_slice=Peak_aftmin[0]            
    x_trim2=x_trim[Max_afure_index:]
    y_trim2=y_trim[Max_afure_index:]
    print('★peak_aftmax',Peak_aftmin)
    print('★Min_afure_index',Min_afure_index)
    print('★peak_slice[0]',Peak_slice[0])

#    print('----メニスカス溢れ関係指標----')         
#    print('最大溢れ体積[pl]','{:.3g}'.format(Max_afure))
#    print('最大後退体積[pl]','{:.3g}'.format(Min_afure)) 

    y_aftmin_trim_abs=abs(y_aftmin_trim)

#    print('y_aftmin_trim(abs):',y_aftmin_trim_abs)
    #合計吐出量に対する最大溢れ量の比率

    Afure_score=Max_afure/total_jetting_vol
#    print('溢れスコア:','{:.3g}'.format(Afure_score))            

    
    # モデル関数の定義
    def gensui_cos(t, gensuihi,A,isou_t,omega):        
    #"""減衰cos波"""
            kinji_wave=A*np.cos(omega*t-isou_t)*np.exp(-omega*gensuihi*t)
            return kinji_wave
   
    x_trim3=x_trim2-x_trim2[0]#x_trim3はx_trim2のindex0を0に補正したもの
    #print('x_trim3',x_trim3)
    #print('x_trim3.shape',x_trim3.shape)
    
    #推定値の算出 #bounds=値の範囲(overflow対策)
    param,cov=curve_fit(gensui_cos,x_trim3,y_trim2,bounds=(0,3))    
    #print('param',param)
    fit_gensuihi=param[0]#gensuihi
    fit_A=param[1]#A
    fit_isou_t=param[2]#isou_t
    fit_omega=param[3]#omega
    
    if fit_gensuihi<0:
        fit_gensuihi=-fit_gensuihi
        fit_omega=-fit_omega

    fit_MeniscusQ=-math.pi/math.log(fit_gensuihi)
        
    #print('fit_gensuihi,fit_A,fit_isou_t,fit_omega',fit_gensuihi,fit_A,fit_isou_t,fit_omega)
    y2=gensui_cos(x_trim3,fit_gensuihi,fit_A,fit_isou_t,fit_omega)
    
#    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax2.plot(x_trim2,y_trim2,color='Orange',label='xy_trim2',linewidth = 4)#Axesオブジェクトにグラフの線を追加
    ax2.plot(x_trim,y_trim,color='Green',label='xy_trim',linewidth = 2)#Axesオブジェクトにグラフの線を追加
        
    ax2.plot(x_trim2,y2,color='Black',linestyle = "--")#近似後の波形
#    ax2.plot(x_trim,y3,color='black',linestyle = "--")#トリミング後のオリジナル波形
#    ax2.plot(x_trim2,y_trim2,color='White',linestyle = "--")#トリミング後のオリジナル波形
    #print('x_trim',x_trim.shape)
    #print('y_trim',y_trim.shape)
    #print('x',x.shape)
    #print('y',y.shape)
    #print('x_trim2',x_trim2.shape)
    #print('y_trim2',y_trim2.shape)
    #print('x_trim3',x_trim3.shape)
#    print('y_trim3',y_trim3.shape)
    
    #グラフの軸の表示
    ax1.set_xlabel('Time[μs]',size=14,weight='light')
    ax2.set_xlabel('Time[μs]',size=14,weight='light')
    ax1.set_ylabel('Voltage[V]',size=14,weight='light')  
    ax2.set_ylabel('V_meniscus_vol[pl]',size=14,weight='light')  
    
    # x軸に補助目盛線を設定
    ax1.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)
    ax2.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)
    # y軸に目盛線を設定
    ax1.grid(which = "major", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)
    ax2.grid(which = "major", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)
    ax1.legend()
    ax2.legend()
#    plt.xlim([0,50])
#    plt.ylim([ymin,ymax])
    plt.title("駆動電圧とメニスカス振動", fontname="MS Gothic")
    #plt.show()#グラフを描画
        
    #メニスカスQ値の算出
    for a in range(2):
        Gensuiritu_p=y_aftmax_trim[a+1]/y_aftmax_trim[a]
        MeniscusQ_p=-math.pi/math.log(Gensuiritu_p)
        Gensuiritu_n=y_aftmin_trim[a+1]/y_aftmin_trim[a]
        
        # たまに減衰率がマイナスになるため対処（本質的にはダメな気がする）
        if Gensuiritu_n > 0:
            MeniscusQ_n=-math.pi/math.log(Gensuiritu_n)
        else:
            MeniscusQ_n=-math.pi/math.log((-1)*Gensuiritu_n)
            
        if a==0:
            Gensuiritu_p_list=Gensuiritu_p
            MeniscusQ_p_list=MeniscusQ_p
            Gensuiritu_n_list=Gensuiritu_n
            MeniscusQ_n_list=MeniscusQ_n
        else:
            Gensuiritu_p_list=np.hstack((Gensuiritu_p_list,Gensuiritu_p))
            MeniscusQ_p_list=np.hstack((MeniscusQ_p_list,MeniscusQ_p))
            Gensuiritu_n_list=np.hstack((Gensuiritu_n_list,Gensuiritu_n))
            MeniscusQ_n_list=np.hstack((MeniscusQ_n_list,MeniscusQ_n))

    #ALを計算
    AL , graph_AL = calc_AL(vcp,dt)
    
    print('----メニスカス振動関係指標----')                 
    print('減衰比(p):',Gensuiritu_p_list)
    print('メニスカスQ値(p):',MeniscusQ_p_list)
    print('減衰比(n):',Gensuiritu_n_list)
    print('メニスカスQ値(n):',MeniscusQ_n_list)
    MeniscusQ_ave=(MeniscusQ_p_list[0]+MeniscusQ_n_list[0])/2
    print('メニスカスQ値(ave):',MeniscusQ_ave)
    
    #print(Afure_score.shape)
    D_list.append(AL) #ALを追加
    D_list.append(total_jetting_vol) #トータル吐出量を追加
    D_list.append(Max_afure)
    D_list.append(Min_afure)
    D_list.append(Afure_score)
    D_list.append(MeniscusQ_ave)
    D_list.append(MeniscusQ_p_list[0])
    D_list.append(MeniscusQ_p_list[1])
    D_list.append(MeniscusQ_n_list[0])
    D_list.append(MeniscusQ_n_list[1])    
    D_list.append(total_jetting_vol)
    D_list.append(fit_gensuihi)
    D_list.append(fit_A)
    D_list.append(fit_isou_t)
    D_list.append(fit_omega)
    D_list.append(fit_MeniscusQ)
    
    return D_list,plt,graph_AL


# In[7]:


#train_data=main()
#train_data.head()
#pandas dfをcsv出力
#train_data.to_csv('csv_out.csv')


# In[8]:


def calc_AL(vcp,dt):    

    #FFT関数による周波数解析
    nn,sp1=nfft_n(vcp)
    spa1=np.sqrt(sp1.real[0:int(nn/2)+1]**2+sp1.imag[0:int(nn/2)+1]**2)*dt*nn
    fk1=np.arange(0,nn/2+1)/nn/dt

    fig=plt.figure(figsize=(5,5),facecolor='w')#figureオブジェクトを作成
    ax=fig.add_subplot(1,1,1)#グラフを描画するsubplot領域を作成
    plt.style.use('fast')

    tstr='Original'
    x=1/fk1[1:]; y=spa1[1:len(x)+1]            
    ax.plot(x,y,color='Red',label=tstr,linewidth = 2)#Axesオブジェクトにグラフの線を追加

    #極大値の算出
    Peak_ori=argrelmax(y,order=1000)
    Peak_ori_5=Peak_ori[0:4]
    ax.scatter(x[Peak_ori_5],y[Peak_ori_5],color='Magenta')#Axesオブジェクトにグラフの線を追加
    
    #ALを極大値から計算
    AL=1/x[1]*1000/2
    #print('AL=',AL)

    ax.set_xlabel('Period [μs]',size=14,weight='light')
    ax.set_ylabel('Fourier Spectrunm [V*μs]',size=14,weight='light')

    # x軸に補助目盛線を設定
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)

    # y軸に目盛線を設定
    ax.grid(which = "major", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 1)

    ax.legend()#凡例表示
    plt.xscale('log')
    plt.title("圧力振動のFFT", fontname="MS Gothic")
    #plt.show()#グラフを描画 
    
    return AL,plt


# In[9]:


def main(dirpath,fname,drop_num):
    Afure_list=[]
    #ディレクトリを指定してcsvからファイル名を呼び出す
    import os
    os.chdir(dirpath)
    name_list=pd.read_csv(fname)
    name_list_l=name_list.iloc[:,1]
    file_num=name_list_l.count()
    print('file_num:',file_num)
    
    #drop_numを表に出した
    drop_num=drop_num
    
    for p in range(file_num):
        name=name_list_l[p]
        #print('name',name)
        
        #LTspice関数による読み出し
        tt,dt,xx1,ni1,wf,vov,vcp=LTSpice(name)
        
        #FFT関数による周波数解析
        nn,sp1=nfft_n(xx1)
        spa1=np.sqrt(sp1.real[0:int(nn/2)+1]**2+sp1.imag[0:int(nn/2)+1]**2)*dt*nn
        fk1=np.arange(0,nn/2+1)/nn/dt

        sp=sp1
        for i in range(0,len(fk1)):
            if 0.05<=fk1[i]:#逆FFT範囲指定 0.05
                sp.real[i]=0; sp.real[nn-i]=0
                sp.imag[i]=0; sp.imag[nn-i]=0

        #IFFT関数によるメニスカス周期の切り出し
        xx2=ifft_n(nn,sp)
        nn,sp2=nfft_n(xx2)
        spa2=np.sqrt(sp2.real[0:int(nn/2)+1]**2+sp2.imag[0:int(nn/2)+1]**2)*dt*nn
        fk2=np.arange(0,nn/2+1)/nn/dt
            
        #溢れスコア、最大溢れ量、最小溢れ量
        Afure , graph_meniscus ,graph_AL=drawfig1_3(tt,dt,xx1,xx2,wf,vov,vcp,drop_num) # time history
        
        #一旦コメントアウト↓
        meniscus_peak , graph_meniscus_FFT=drawfig2_1(fk1,spa1,fk2,spa2) # Fourier spectrum
        print('Meniscus peak',meniscus_peak[0])
        
        #新規fitting用関数
#        meniscus_fitting(tt,xx2)
        
        Afure_list.append(Afure)
        #print('Afure',Afure)
        
#print(xx1.shape)
#print(xx2.shape)
#print(tt.shape)
    #print("Afure_list:",Afure_list)
    Afure_df=pd.DataFrame(Afure_list,columns=['AL','Volume','Afure_max','Afure_min','Afure_score','MeniscusQ_ave','Meniscus_Q_p[0]','Meniscus_Q_p[1]','Meniscus_Q_n[0]','Meniscus_Q_n[1]','total_jetting_vol','fit_gensuihi','A','isou_t','omega','fit_MeniscusQ'])
    Afure_df.head()
    train_data=pd.concat([name_list,Afure_df],axis=1)
    
    return train_data


# In[10]:


#train_data=main(dirpath,fname)
#train_data.head()
#pandas dfをcsv出力
#train_data.to_csv('DR2d_train_csv_0509.csv')


# In[13]:


#---------------
# Execute
#---------------
#dirpath="C:\\Users\\e11589\\Desktop\\SPICE_bo"
#fname="rawfile_list.csv"

if __name__ == '__main__': main(dirpath,fname,drop_num)


# In[ ]:




