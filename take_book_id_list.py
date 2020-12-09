from bs4 import BeautifulSoup
import requests
import os
import re
import sys
from pprint import pprint
import time
import csv
import pandas as pd

headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'}

def get_soup(url):
    r = requests.get(url, headers = headers).text
    return BeautifulSoup(r, 'html.parser')


adong_id_list=[]

adong = 'https://book.naver.com/bestsell/bestseller_list.nhn?cp=kyobo&cate=07&bestWeek=2020-11-1&indexCount=11&type=list&page='


for i in range(1,7):
    adongi = adong + str(i)
    for j in range(0,25):
        findingid = get_soup(adongi).findAll("dt", {"id": "book_title_" + str(j)})
        hrefs = [dt.find("a")['href'] for dt in findingid]
        real_id = "=".join(hrefs)[49:]
        adong_id_list.append(real_id)
    time.sleep(3)
    
data={'book_list':adong_id_list}
df=pd.DataFrame.from_dict(data)

df.to_csv('book_id_list(어린이영어).csv',encoding='utf-8-sig')